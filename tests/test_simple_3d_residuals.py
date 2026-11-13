"""Residual graphics preserve old numerical results and full source-row evidence."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("opencv_simple_3d_measure.residuals")
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.plugin_dialog import PluginDialogContext
from opencv_dataflow.pointclouds import PointCloud

ROOT = Path(__file__).resolve().parents[1] / "examples/Simple Plugins/3d_measure"
CASES = ["good", "raised", "missing", "line_good", "line_nonparallel",
         "tless_frame13", "tless_frame24"]


@pytest.mark.local_examples
@pytest.mark.parametrize("name", CASES)
def test_residual_projects_reuse_recipes_and_keep_reports_with_and_without_graphics(name):
    folder = ROOT / "residuals" / name
    graph = Graph.load(folder / "project.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["measure"]
        assert block.state in ("pass", "fail"), block.error
        report = block.last_result.outputs["report"]
        identity = json.loads((folder / "source_identity.json").read_bytes())
        prior = json.loads((ROOT / identity["base_project"] / "expected.json").read_bytes())
        assert json.loads(json.dumps(report)) == prior
        assert block.preview_output == "residuals"
        cloud = block.last_result.outputs["residuals"]
        assert isinstance(cloud, PointCloud) and cloud.colors is not None
        assert cloud.metadata["view"] == "all_selected_fit_residuals"
        summary = block.last_result.outputs["residual_report"]
        assert summary == json.loads((folder / "residual_report.json").read_bytes())
        for row in summary["regions"].values():
            if row["histogram"] is None:
                assert row["minimum"] is None and row["residual_points"] == 0
            else:
                hist = row["histogram"]
                assert (sum(hist["counts"]) + hist["below_range"] + hist["above_range"]
                        == row["selected_points"])
        block.parameters["render_graphics"] = False
        with patch("opencv_simple_3d_measure.dataflow.ResidualReview", side_effect=AssertionError):
            graph.run()
        assert block.last_result.outputs["report"] == report
        assert block.last_result.outputs["residuals"] is NO_VALUE
        assert block.last_result.outputs["residual_report"] is NO_VALUE
    finally:
        graph.close()


@pytest.mark.local_examples
def test_saved_raw_residuals_are_full_population_with_explicit_missing_values():
    for name in CASES:
        folder = ROOT / "residuals" / name
        with np.load(folder / "residuals.npz", allow_pickle=False) as archive:
            manifest = json.loads(str(archive["manifest"].item()))
            assert manifest["row_index_scope"] == "zero_based_rows_of_scene_xyz_sha256"
            for row in manifest["regions"].values():
                prefix = row["array_prefix"]
                assert len(archive[prefix + "_rows"]) == row["selected_points"]
                if row["histogram"] is None:
                    assert prefix + "_values" not in archive.files
                else:
                    assert len(archive[prefix + "_values"]) == row["residual_points"]


@pytest.mark.local_examples
def test_real_native_residual_review_cache_and_histogram_restore_without_refitting(app):
    graph = Graph.load(ROOT / "residuals/tless_frame13/project.json", builtin_registry())
    context = dialog = None
    try:
        graph.run()
        block = graph.blocks["measure"]
        active = dict(block.parameters)
        context = PluginDialogContext(graph, block)
        dialog = block.create_plugin_dialog(None, dict(active), context)
        dialog._review()
        review = dialog._children[-1]
        review.display.setCurrentIndex(2)
        cached = review.residual_review()
        with patch("opencv_simple_3d_measure.qt_review.ResidualReview", side_effect=AssertionError):
            review.plane.setCurrentIndex(review.plane.findData("part_surface"))
            review.layers["planes"].setChecked(False)
            review.display.setCurrentIndex(0)
            review.display.setCurrentIndex(2)
            assert review.residual_review() is cached
        assert review.histogram.row["selected_points"] > 1000
        review._capture()
        frozen = review.residual_review()
        assert frozen is not cached
        review._outputs_changed()
        assert review.residual_review() is frozen
        dialog.reject()
        assert block.parameters == active
    finally:
        if dialog:
            dialog.done(0)
        if context:
            context._close()
        graph.close()
