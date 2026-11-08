import shutil
import time
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest

from opencv_dataflow.app import MainWindow
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples/Simple Plugins/path_width"


def test_relocated_bead_project_reports_gaps_and_publishes_overlay(tmp_path):
    pytest.importorskip("opencv_simple_path_width.dataflow")
    moved = tmp_path / "moved"
    shutil.copytree(_EXAMPLE, moved)
    graph = Graph.load(moved / "project.json", builtin_registry())
    graph.filmstrip_enabled = True
    try:
        inspection = graph.blocks["inspection"]
        graph.run()
        assert inspection.state == "pass", inspection.error
        assert inspection.last_result.outputs["mean_width"] == pytest.approx(12., abs=.2)
        graph.run()
        assert inspection.state == "fail", inspection.error
        result = inspection.last_result
        assert result.outputs["longest_gap"] == pytest.approx(24., abs=2.)
        assert {item["code"] for item in result.fail_reasons} == {
            "gap_above_maximum", "station_out_of_tolerance",
        }
        assert result.outputs["trend"].shape == (292, 630, 3)
        rows = graph.filmstrip.snapshot()
        assert [(row["filename"], row["state"]) for row in rows] == [
            ("reference.png", "pass"), ("bead.png", "fail"),
        ]
        assert all(row["override"] is not None and row["thumbnail"] for row in rows)
    finally:
        graph.close()


def test_real_weld_example_replays_with_edge_search_bands():
    pytest.importorskip("opencv_simple_path_width.dataflow")
    measurement = pytest.importorskip("opencv_simple_path_width.measurement")
    if not hasattr(measurement.PathWidthSettings, "nominal_width"):
        pytest.skip("Weld example requires SimplePathWidth with edge search bands")
    graph = Graph.load(_EXAMPLE / "weld_bead/project.json", builtin_registry())
    graph.filmstrip_enabled = True
    try:
        inspection = graph.blocks["inspection"]
        for state, stations in (("pass", 31), ("pass", 31), ("fail", 18)):
            graph.run()
            assert inspection.state == state, inspection.error
            assert inspection.last_result.outputs["coverage"] == pytest.approx(stations / 31)
        assert len(graph.filmstrip.snapshot()) == 3
    finally:
        graph.close()


def test_train_preview_stage_apply_and_restore(app, monkeypatch, tmp_path):
    trainer = pytest.importorskip("opencv_simple_path_width.qt_training")
    original = pytest.importorskip("opencv_simple_path_width.recipe")
    moved = tmp_path / "training"
    shutil.copytree(_EXAMPLE, moved)
    project = moved / "project.json"
    graph = Graph.load(project, builtin_registry())
    block = graph.blocks["inspection"]
    original_path = block.parameters["recipe_path"]
    reference = original.load_reference(moved / original_path)
    window = MainWindow()
    warnings = []
    monkeypatch.setattr(window, "warn", warnings.append)
    previews = []

    def preview(self):
        self.show()
        app.processEvents()
        position = self.trend.mapFromScene(QPointF(343., 90.))
        QTest.mouseClick(self.trend.viewport(), Qt.MouseButton.LeftButton, pos=position)
        assert self.marker.rect().width() == 12
        assert self.cursor.isVisible()
        assert self.windowTitle().startswith("Station ")
        previews.append(self.result)

    monkeypatch.setattr(trainer.MeasurementPreview, "exec", preview)
    try:
        window.replace_graph(graph, project)
        window.inspector.select(block)
        window.start_run(False)
        deadline = time.monotonic() + 10
        while window.runner is not None and time.monotonic() < deadline:
            app.processEvents()
            QTest.qWait(10)
        assert window.runner is None
        window.inspector.open_plugin_dialog()
        assert not warnings
        dialog = window.inspector._plugin_dialog
        app.processEvents()
        np.testing.assert_array_equal(dialog.reference, reference)
        dialog._capture_reference()
        np.testing.assert_array_equal(dialog.reference, reference)
        canvas = dialog.canvas
        vertices = np.asarray(canvas.vertices)
        index = int(np.argmax(np.linalg.norm(np.diff(vertices, axis=0), axis=1)))
        midpoint = (vertices[index] + vertices[index + 1]) / 2

        def viewport_point(point):
            return canvas.mapFromScene(QPointF(*point + .5))

        QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.ShiftModifier, viewport_point(midpoint))
        assert len(canvas.vertices) == len(vertices) + 1
        inserted = canvas.selected
        adjusted = midpoint + [0., 2.]
        QTest.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, pos=viewport_point(midpoint))
        QTest.mouseMove(canvas.viewport(), viewport_point(adjusted))
        QTest.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton, pos=viewport_point(adjusted))
        np.testing.assert_allclose(canvas.vertices[inserted], adjusted, atol=1.)
        QTest.keyClick(canvas, Qt.Key.Key_Delete)
        np.testing.assert_allclose(canvas.vertices, vertices)
        canvas.reverse()
        dialog.preview_button.click()
        assert len(previews) == 1
        dialog.stage_button.click()
        assert not warnings
        assert block.parameters["recipe_path"] == original_path
        window.inspector.apply()
        assert not warnings
        assert block.parameters["recipe_path"] != original_path
        graph.save(project)
        window.inspector.open_plugin_dialog()
        restored = window.inspector._plugin_dialog
        assert not warnings
        np.testing.assert_array_equal(restored.reference, reference)
        np.testing.assert_allclose(restored.canvas.vertices, np.vstack((vertices[0], vertices[:0:-1])))
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
