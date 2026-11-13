import copy
import json
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE, metadata

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/Simple Plugins/reference_inspect"


def test_optional_assessment_preserves_results_and_uses_a_separate_display_grid(monkeypatch):
    integration = pytest.importorskip("opencv_simple_reference_inspect.dataflow")
    graph = Graph.load(EXAMPLE / "project.json", builtin_registry())
    try:
        block = graph.blocks["inspection"]
        original_renderer = integration.draw_assessment
        monkeypatch.setattr(
            integration,
            "draw_assessment",
            lambda *a, **k: pytest.fail("Disabled assessment must do no review work"),
        )
        graph.run()
        before = block.last_result.outputs
        source = dict(block.last_inputs)
        assert before["assessment"] is NO_VALUE
        monkeypatch.setattr(integration, "draw_assessment", original_renderer)
        block.parameters["render_assessment"] = True
        block.execute(source)
        assert block.state == "pass", block.error
        after = block.last_result.outputs
        assert after["inspection"] == before["inspection"]
        for name in ("overlay", "defect_mask", "inspected_mask", "deviation_map", "residual"):
            np.testing.assert_array_equal(after[name], before[name])
        assert after["assessment"].shape == (1050, 1500, 3)
        assert (
            metadata(after["assessment"])["spatial"]["root_id"]
            != metadata(source["image"])["spatial"]["root_id"]
        )
        assert block.filmstrip_output == "assessment"
        block.parameters.update(render_assessment=False, render_overlay=False)
        monkeypatch.setattr(
            integration,
            "draw_assessment",
            lambda *a, **k: pytest.fail("Disabled assessment renderer called"),
        )
        block.execute(source)
        assert block.last_result.outputs["assessment"] is NO_VALUE
        assert block.last_result.outputs["overlay"] is NO_VALUE
        assert block.filmstrip_output is None
        mask = np.zeros(source["image"].shape, np.uint8)
        block.execute({**source, "mask": mask})
        assert block.state == "fail" and "Not evaluated" in block.last_result.message
    finally:
        graph.close()


def test_training_review_reuses_cached_result_and_invalidates_after_new_input(
    app, monkeypatch, tmp_path
):
    from opencv_dataflow.app import MainWindow

    trainer_module = pytest.importorskip("opencv_simple_reference_inspect.qt_training")
    review_module = pytest.importorskip("opencv_simple_reference_inspect.qt_review")
    from PySide6.QtWidgets import QFileDialog

    graph = Graph.load(EXAMPLE / "project.json", builtin_registry())
    window = MainWindow()
    try:
        window.replace_graph(graph, EXAMPLE / "project.json")
        block = graph.blocks["inspection"]
        original = copy.deepcopy(block.parameters)
        window.inspector.select(block)
        window.inspector.open_plugin_dialog()
        trainer = window.inspector._plugin_dialog
        assert trainer.model is not None and not trainer.review_button.isEnabled()
        trainer.test_image = trainer_module.read_image(EXAMPLE / "assets/changed.png")
        trainer.preview_button.click()
        cached = trainer.inspection
        assert trainer.review_button.isEnabled()
        monkeypatch.setattr(
            type(trainer.model),
            "inspect",
            lambda *a, **k: pytest.fail("Review must reuse the inspection"),
        )
        seen = []

        def review(dialog):
            dialog.show()
            app.processEvents()
            assert dialog.review.inspection is cached[0]
            dialog._actual_size()
            assert dialog.left.transform() == dialog.right.transform()
            dialog.left.scale(2.0, 2.0)
            dialog._sync(dialog.left, dialog.right)
            scale = dialog.left.transform().m11()
            for mode in ("Signed residual", "Deviation / tolerance", "Inspected coverage"):
                dialog.mode.setCurrentText(mode)
                assert dialog.left.transform() == dialog.right.transform()
                assert dialog.left.transform().m11() == pytest.approx(scale)
            for tab in (1, 2, 0):
                dialog.tabs.setCurrentIndex(tab)
                app.processEvents()
            monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a: str(tmp_path))
            dialog._export()
            assert "Saved" in dialog.export_status.text()
            seen.append(dialog.review.summary())

        monkeypatch.setattr(review_module.ReferenceReviewDialog, "exec", review)
        trainer.review_button.click()
        assert len(seen) == 1 and seen[0]["inspection"]["defect_area"] == 754
        assert trainer.inspection is cached and block.parameters == original
        (exported,) = tmp_path.glob("reference-review-*/review.json")
        assert json.loads(exported.read_bytes())["inspection"] == cached[0].report
        monkeypatch.setattr(
            QFileDialog, "getOpenFileName", lambda *a: (str(EXAMPLE / "assets/nominal.png"), "")
        )
        trainer._open_test()
        assert trainer.inspection is None and not trainer.review_button.isEnabled()
        assert trainer.model is not None
        trainer.fields["absolute_threshold"].setText("14")
        assert trainer.model is None
        assert block.parameters == original
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
