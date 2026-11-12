"""Offscreen mask drawing and persistence through the reference trainer."""

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog

from opencv_dataflow.app import MainWindow
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/Simple Plugins/reference_inspect"


def click(canvas, point, *, button=Qt.MouseButton.LeftButton, modifiers=Qt.KeyboardModifier.NoModifier):
    QTest.mouseClick(canvas, button, modifiers, canvas.to_widget(point).toPoint())


def drag(canvas, start, end):
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=canvas.to_widget(start).toPoint())
    target = canvas.to_widget(end).toPoint()
    QTest.mouseMove(canvas, target)
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=target)


def test_mask_shapes_combine_and_undo_without_mutating_the_imported_mask(app):
    editor_type = pytest.importorskip("opencv_simple_reference_inspect.qt_masks").MaskEditor
    image = np.zeros((80, 100), np.uint8)
    imported = np.ones_like(image, bool)
    imported[17:22, 18:24] = False
    before = imported.copy()
    editor = editor_type(image, imported)
    try:
        np.testing.assert_array_equal(editor.mask, imported)
        editor.show()
        app.processEvents()
        editor.clear_button.click()
        assert not editor.ok.isEnabled()
        editor.draw_button.click()
        drag(editor.canvas, [10, 10], [30, 30])
        editor.apply_button.click()
        expected = np.zeros_like(image)
        expected[10:31, 10:31] = 1
        np.testing.assert_array_equal(editor.mask, expected)
        assert editor.ok.isEnabled()

        editor.operation.setCurrentText("Add to mask")
        editor.shape.setCurrentText("circle")
        drag(editor.canvas, [50, 30], [55, 30])
        editor.apply_button.click()
        cv2.circle(expected, (50, 30), 5, 1, -1)
        np.testing.assert_array_equal(editor.mask, expected)

        editor.operation.setCurrentText("Exclude from mask")
        editor.shape.setCurrentText("rectangle")
        drag(editor.canvas, [14, 14], [16, 16])
        editor.apply_button.click()
        excluded = expected.copy()
        excluded[14:17, 14:17] = 0
        np.testing.assert_array_equal(editor.mask, excluded)
        editor.undo_button.click()
        np.testing.assert_array_equal(editor.mask, expected)
        editor.reject()
        np.testing.assert_array_equal(imported, before)
    finally:
        editor.close()


def test_polygon_points_remain_editable_while_zooming_and_panning(app):
    editor_type = pytest.importorskip("opencv_simple_reference_inspect.qt_masks").MaskEditor
    editor = editor_type(np.zeros((80, 100), np.uint8), np.zeros((80, 100), bool))
    try:
        editor.show()
        app.processEvents()
        canvas = editor.canvas
        editor.shape.setCurrentText("polygon")
        for point in ([10, 10], [50, 10], [30, 50]):
            click(canvas, point)
        QTest.keyClick(canvas, Qt.Key.Key_Return)
        assert not canvas.drawing and canvas.active
        click(canvas, [30, 10], modifiers=Qt.KeyboardModifier.ShiftModifier)
        assert len(canvas.parameters["vertices"]) == 4
        drag(canvas, [30, 10], [30, 17])
        assert [30, 17] in canvas.parameters["vertices"]
        click(canvas, [30, 17], button=Qt.MouseButton.RightButton)
        assert len(canvas.parameters["vertices"]) == 3

        position = canvas.to_widget([30, 25])
        prior = canvas.to_image(position)
        event = QWheelEvent(position, canvas.mapToGlobal(position.toPoint()), QPoint(), QPoint(0, 120),
                            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                            Qt.ScrollPhase.NoScrollPhase, False)
        app.sendEvent(canvas, event)
        assert canvas._zoom == pytest.approx(1.25) and canvas.to_image(position) == prior
        drag(canvas, [30, 20], [33, 24])
        assert canvas.parameters["vertices"] == [[13, 14], [53, 14], [33, 54]]

        rect = canvas.image_rect()
        start = canvas.to_widget([30, 25]).toPoint()
        QTest.mousePress(canvas, Qt.MouseButton.MiddleButton, pos=start)
        QTest.mouseMove(canvas, start + QPoint(40, -20))
        QTest.mouseRelease(canvas, Qt.MouseButton.MiddleButton, pos=start + QPoint(40, -20))
        assert canvas.image_rect().topLeft() == rect.topLeft() + QPointF(40, -20)
        editor.apply_button.click()
        expected = np.zeros((80, 100), np.uint8)
        cv2.fillPoly(expected, [np.array([[13, 14], [53, 14], [33, 54]], np.int32)], 1)
        np.testing.assert_array_equal(editor.mask, expected)
        editor.accept()
        assert editor.result() == QDialog.DialogCode.Accepted
    finally:
        editor.close()


def test_mask_dialog_cancel_and_accept_flow_through_retraining_and_apply(app, monkeypatch, tmp_path):
    editor_type = pytest.importorskip("opencv_simple_reference_inspect.qt_masks").MaskEditor
    model_io = pytest.importorskip("opencv_simple_reference_inspect")
    read_image = pytest.importorskip("opencv_simple_reference_inspect.cli").read_image
    moved = tmp_path / "mask_training"
    shutil.copytree(EXAMPLE, moved)
    project = moved / "project.json"
    graph = Graph.load(project, builtin_registry())
    window = MainWindow()
    warnings = []
    monkeypatch.setattr(window, "warn", warnings.append)
    try:
        window.replace_graph(graph, project)
        block = graph.blocks["inspection"]
        original = dict(block.parameters)
        original_bytes = (moved / original["recipe_path"]).read_bytes()
        window.inspector.select(block)
        window.inspector.open_plugin_dialog()
        trainer = window.inspector._plugin_dialog
        model = trainer.model
        old_inspection, old_stable = model.inspection_mask.copy(), model.normalization_mask.copy()

        def cancel(editor):
            editor.clear_button.click()
            editor.reject()
            return editor.result()

        monkeypatch.setattr(editor_type, "exec", cancel)
        trainer.inspection_mask_button.click()
        assert trainer.model is model
        np.testing.assert_array_equal(trainer.inspection_mask, old_inspection)

        monkeypatch.setattr(editor_type, "exec", lambda editor: QDialog.DialogCode.Accepted)
        trainer.normalization_mask_button.click()
        assert trainer.model is model

        def exclude_change(editor):
            editor.operation.setCurrentText("Exclude from mask")
            editor.draw_button.click()
            editor.canvas.parameters["rectangle"] = [157, 104, 26, 29]
            editor.canvas.drawing = False
            editor.apply_shape()
            editor.accept()
            return editor.result()

        monkeypatch.setattr(editor_type, "exec", exclude_change)
        trainer.inspection_mask_button.click()
        assert trainer.model is None
        np.testing.assert_array_equal(trainer.normalization_mask, old_stable)
        desired = old_inspection.copy()
        desired[104:133, 157:183] = False
        np.testing.assert_array_equal(trainer.inspection_mask, desired)
        trainer.train_button.click()
        trainer.test_image = read_image(moved / "assets/changed.png")
        trainer.preview_button.click()
        assert trainer.inspection[0].passed and trainer.inspection[0].report["defect_area"] == 0
        trainer.stage_button.click()
        assert block.parameters == original
        window.inspector.apply()
        assert not warnings and block.parameters["recipe_path"] != original["recipe_path"]
        saved = model_io.load_model(moved / block.parameters["recipe_path"], sha256=block.parameters["sha256"])
        np.testing.assert_array_equal(saved.inspection_mask, desired)
        np.testing.assert_array_equal(saved.normalization_mask, old_stable)
        assert (moved / original["recipe_path"]).read_bytes() == original_bytes
        graph.save(project)
        window.inspector.open_plugin_dialog()
        np.testing.assert_array_equal(window.inspector._plugin_dialog.inspection_mask, desired)
    finally:
        window.dirty = False
        window.close()
        app.processEvents()


def test_unset_normalization_mask_edits_the_effective_inspection_region(app, monkeypatch, tmp_path):
    editor_type = pytest.importorskip("opencv_simple_reference_inspect.qt_masks").MaskEditor
    trainer_type = pytest.importorskip("opencv_simple_reference_inspect.qt_training").ReferenceTrainingDialog
    from types import SimpleNamespace

    trainer = trainer_type(None, {"recipe_path": "", "sha256": ""}, SimpleNamespace(project_dir=tmp_path))
    try:
        trainer._append(np.zeros((80, 100), np.uint8), "Reference")
        trainer.inspection_mask = np.zeros((80, 100), bool)
        trainer.inspection_mask[20:40, 30:60] = True

        def unchanged(editor):
            np.testing.assert_array_equal(editor.mask, trainer.inspection_mask)
            return QDialog.DialogCode.Accepted

        monkeypatch.setattr(editor_type, "exec", unchanged)
        trainer.normalization_mask_button.click()
        assert trainer.normalization_mask is None
    finally:
        trainer.close()
