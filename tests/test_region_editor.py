"""Offscreen pointer and transactional checks for image-space region editing."""

import copy
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog

from opencv_dataflow.region_editor import RegionCanvas, RegionEditor
from opencv_dataflow.regions import ShapeRegion


def canvas(app, shape="rectangle"):
    params = copy.deepcopy(ShapeRegion.defaults)
    params["shape"] = shape
    widget = RegionCanvas(np.zeros((100, 200, 3), np.uint8), params)
    widget.resize(800, 600)
    widget.show()
    app.processEvents()
    return widget


def click(widget, point):
    QTest.mouseClick(widget, Qt.MouseButton.LeftButton, pos=widget.to_widget(point).toPoint())


def drag(widget, start, end):
    QTest.mousePress(widget, Qt.MouseButton.LeftButton, pos=widget.to_widget(start).toPoint())
    QTest.mouseMove(widget, widget.to_widget(end).toPoint())
    QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, pos=widget.to_widget(end).toPoint())


def test_scaled_letterboxed_mapping_and_rectangle_draw_resize_move(app):
    widget = canvas(app)
    try:
        assert widget.to_image(QPointF(1, 1)) is None
        assert widget.to_image(widget.to_widget([37, 62])) == [37, 62]
        widget.new_shape()
        drag(widget, [100, 70], [20, 10])
        assert not widget.drawing
        assert widget.parameters["rectangle"] == [20, 10, 81, 61]
        drag(widget, [100, 70], [120, 80])
        assert widget.parameters["rectangle"] == [20, 10, 101, 71]
        drag(widget, [60, 40], [70, 45])
        assert widget.parameters["rectangle"] == [30, 15, 101, 71]
        widget.resize(500, 800)
        app.processEvents()
        assert widget.to_image(widget.to_widget([37, 62])) == [37, 62]
    finally:
        widget.close()


def test_circle_draw_center_and_radius_handles(app):
    widget = canvas(app, "circle")
    try:
        widget.new_shape()
        drag(widget, [60, 40], [90, 40])
        assert widget.parameters["center"] == [60, 40]
        assert widget.parameters["radius"] == 30
        drag(widget, [60, 40], [65, 45])
        assert widget.parameters["center"] == [65, 45]
        drag(widget, [95, 45], [105, 45])
        assert widget.parameters["radius"] == 40
    finally:
        widget.close()


def test_polygon_finish_undo_vertex_edit_and_move(app):
    widget = canvas(app, "polygon")
    try:
        widget.new_shape()
        for point in [[20, 20], [100, 20], [100, 70], [10, 10]]:
            click(widget, point)
        QTest.keyClick(widget, Qt.Key.Key_Backspace)
        click(widget, [20, 70])
        QTest.keyClick(widget, Qt.Key.Key_Return)
        assert not widget.drawing
        assert widget.parameters["vertices"] == [[20, 20], [100, 20], [100, 70], [20, 70]]
        drag(widget, [20, 20], [25, 25])
        assert widget.parameters["vertices"][0] == [25, 25]
        drag(widget, [60, 45], [65, 50])
        assert widget.parameters["vertices"] == [[30, 30], [105, 25], [105, 75], [25, 75]]
    finally:
        widget.close()


def test_escape_restores_gesture_and_pointer_clamps_at_image_boundary(app):
    widget = canvas(app)
    try:
        original = copy.deepcopy(widget.parameters)
        widget.new_shape()
        QTest.mousePress(
            widget, Qt.MouseButton.LeftButton, pos=widget.to_widget([20, 20]).toPoint()
        )
        QTest.keyClick(widget, Qt.Key.Key_Escape)
        assert widget.parameters == original and not widget.drawing
        widget.new_shape()
        QTest.mousePress(
            widget, Qt.MouseButton.LeftButton, pos=widget.to_widget([20, 20]).toPoint()
        )
        QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, pos=widget.rect().bottomRight())
        assert widget.parameters["rectangle"] == [20, 20, 180, 80]
    finally:
        widget.close()


def test_dialog_drafts_are_private_and_incomplete_polygon_cannot_accept(app):
    params = copy.deepcopy(ShapeRegion.defaults)
    original = copy.deepcopy(params)
    dialog = RegionEditor(np.zeros((100, 200), np.uint8), params)
    dialog.show()
    app.processEvents()
    try:
        dialog.shape.setCurrentText("polygon")
        assert not dialog.ok.isEnabled()
        for point in [[20, 20], [100, 20]]:
            click(dialog.canvas, point)
        dialog.accept()
        assert dialog.result() != QDialog.DialogCode.Accepted
        click(dialog.canvas, [60, 70])
        QTest.mouseClick(
            dialog.canvas,
            Qt.MouseButton.RightButton,
            pos=dialog.canvas.to_widget([60, 70]).toPoint(),
        )
        assert dialog.ok.isEnabled()
        dialog.invert.setChecked(True)
        dialog.accept()
        assert dialog.result() == QDialog.DialogCode.Accepted
        assert dialog.parameters["vertices"] == [[20, 20], [100, 20], [60, 70]]
        assert dialog.parameters["invert"]
        assert params == original
    finally:
        dialog.close()


def test_inspector_accept_cancel_cache_and_saved_project(app, monkeypatch, tmp_path):
    from opencv_dataflow.app import MainWindow
    from opencv_dataflow.core import Graph

    window = MainWindow()
    window.load_regions_example()
    window.graph.run()
    block = window.graph.blocks["roi_circle"]
    window.scene.nodes[block.id].setSelected(True)
    original = window.graph.to_dict()
    window.inspector.fields["radius"].setText("61")
    calls = []

    def cancel(dialog):
        assert dialog.canvas.image_shape == (360, 480)
        assert dialog.parameters["radius"] == 61
        calls.append("cancel")
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(RegionEditor, "exec", cancel)
    window.inspector.edit_region()
    assert window.graph.to_dict() == original
    assert window.inspector.fields["radius"].text() == "61"

    def accept(dialog):
        dialog.canvas.parameters.update(center=[145, 181], radius=61)
        dialog.accept()
        return dialog.result()

    monkeypatch.setattr(RegionEditor, "exec", accept)
    window.inspector.edit_region()
    assert block.parameters["center"] == [145, 181]
    assert block.parameters["radius"] == 61
    assert window.dirty and block.last_result is None
    assert window.inspector.block is block
    # Reopen on the same captured frame after Apply invalidated the run records.
    monkeypatch.setattr(RegionEditor, "exec", cancel)
    window.inspector.edit_region()
    assert calls == ["cancel", "cancel"]
    window.path = tmp_path / "drawn-regions.json"
    assert window.save_project()
    restored = Graph.load(window.path, window.registry)
    assert restored.blocks[block.id].parameters == block.parameters
    restored.run()
    assert all(b.state == "pass" for b in restored.blocks.values())
    window.dirty = False
    window.close()


def test_inspector_missing_frame_and_active_run_do_not_change_graph(app, monkeypatch):
    from opencv_dataflow.app import MainWindow

    window = MainWindow()
    window.load_regions_example()
    window.scene.nodes["roi_circle"].setSelected(True)
    original = window.graph.to_dict()
    warnings = []
    monkeypatch.setattr(window, "warn", warnings.append)
    window.inspector.edit_region()
    assert len(warnings) == 1 and "Run once" in warnings[0]
    assert window.graph.to_dict() == original
    window.runner = object()
    try:
        window.inspector.edit_region()
        assert len(warnings) == 1
    finally:
        window.runner = None
        window.dirty = False
        window.close()
