"""Frozen-result comparisons and cached native review; no acquisition."""

import copy
import hashlib
import json
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
from opencv_dataflow.payloads import NO_VALUE


def stable_result(result):
    return json.loads(
        json.dumps(
            dict(
                passed=bool(result.passed),
                score=result.score,
                message=result.message,
                fail_reasons=result.fail_reasons,
                outputs={
                    name: value
                    for name, value in result.outputs.items()
                    if name in ("report", "coverage", "valid_pixels")
                },
            ),
            allow_nan=False,
        )
    )


def array_hashes(result):
    def array(value):
        return dict(
            shape=list(value.shape),
            dtype=value.dtype.str,
            sha256=hashlib.sha256(value.tobytes()).hexdigest(),
        )

    return dict(
        maps={name: array(value) for name, value in result.outputs["maps"].items()},
        images={
            name: array(value)
            for name, value in result.outputs.items()
            if isinstance(value, np.ndarray)
        },
        graphics={name: array(value) for name, value in result.graphics.items()},
    )


def no_analysis():
    from opencv_simple_surface_enhance.solver import SurfaceEstimator

    stack = ExitStack()
    stack.enter_context(
        patch.object(SurfaceEstimator, "estimate", side_effect=AssertionError("Fit"))
    )
    for name in (
        "dataflow.execute",
        "dataflow.import_capture",
        "dataflow.load_recipe",
        "execution.execute",
        "execution.prepare_capture",
        "execution.prepare_stack",
        "recipe.import_capture",
        "recipe.load_recipe",
        "render.relight",
        "render.normal_change",
        "render.render_views",
    ):
        stack.enter_context(
            patch("opencv_simple_surface_enhance." + name, side_effect=AssertionError(name))
        )
    return stack


def native_review(app, window, block, target=None):
    before, arrays = stable_result(block.last_result), array_hashes(block.last_result)
    parameters = copy.deepcopy(block.parameters)
    inspector = window.inspector
    inspector.select(block)
    with (
        no_analysis(),
        patch.object(block, "create_plugin_dialog", side_effect=AssertionError("Opened training")),
    ):
        inspector.open_plugin_review_dialog()
        app.processEvents()
        popup = inspector._plugin_review_dialog
        assert popup.snapshot is not None and popup.selected_pixel is None
        frozen = popup.snapshot
        if block.last_result.outputs["overview"] is not NO_VALUE:
            np.testing.assert_array_equal(
                frozen.view("overview")[0], block.last_result.outputs["overview"]
            )
        reasons = frozen.record._output["maps"]["reason"]
        rows = np.argwhere((reasons != 0) & (reasons != 1))
        if not len(rows):
            rows = np.argwhere(reasons == 0)
        y, x = (int(v) for v in (rows[len(rows) // 2] if len(rows) else [0, 0]))
        popup.source_view.pixel_chosen.emit(x, y)
        popup.focus_pixel()
        zoom = popup.source_view.transform().m11()
        popup.map_view.pixel_chosen.emit(x, y)
        assert popup.source_view.transform().m11() == zoom
        assert popup.table.rowCount() == frozen.light_count
        popup.light.setCurrentIndex(min(1, frozen.light_count - 1))
        for name in ("observation_flags", "used", "light_residual", "quality"):
            popup.map.setCurrentIndex(popup.map.findData(name))
        if target is not None:
            popup.map.setCurrentIndex(popup.map.findData("normals"))
            popup.export_to(target)
        popup.source.setCurrentIndex(popup.source.findData("reference"))
        assert "Training reference" in popup.view_note.text()
        popup.source.setCurrentIndex(0)
        popup.overview()
        assert popup.selected_pixel is None and not popup.inspect.isChecked()
        assert stable_result(block.last_result) == before
        assert array_hashes(block.last_result) == arrays and block.parameters == parameters
        popup.reject()
        assert popup.snapshot is None and popup.source_view.image_item.pixmap().isNull()
        app.processEvents()
    return frozen
