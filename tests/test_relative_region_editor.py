"""Pointer editing in local coordinate frames, with no desktop automation."""

import copy

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog, QPushButton
from test_region_editor import click, drag

from opencv_dataflow.app import MainWindow
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.poses import PoseRegion, local_offset, make_pose
from opencv_dataflow.region_editor import RegionCanvas, RegionEditor


@pytest.mark.parametrize("shape", ["rectangle", "circle", "polygon"])
def test_draw_resize_move_in_rotated_local_coordinates_matches_runtime(app, shape):
    image = np.zeros((220, 300), np.uint8)
    matrix = local_offset(150, 110, 35) @ np.diag([1.5, 1.5, 1])
    anchor = make_pose(matrix, [300, 220], {})
    parameters = copy.deepcopy(PoseRegion.defaults)
    parameters["shape"] = shape
    widget = RegionCanvas(image, parameters, pose=anchor)
    widget.resize(950, 700)
    widget.show()
    app.processEvents()
    try:
        assert widget.to_image(widget.to_widget([-17, 12])) == [-17, 12]
        assert widget.to_image(QPointF(0, 0)) is None
        widget.new_shape()
        if shape == "rectangle":
            drag(widget, [-20, -15], [20, 15])
            assert widget.parameters["rectangle"] == [-20, -15, 41, 31]
            drag(widget, [20, 15], [25, 18])
            assert widget.parameters["rectangle"] == [-20, -15, 46, 34]
            drag(widget, [0, 0], [4, 3])
            assert widget.parameters["rectangle"] == [-16, -12, 46, 34]
        elif shape == "circle":
            drag(widget, [-10, 5], [10, 5])
            assert widget.parameters["center"] == [-10, 5] and widget.parameters["radius"] == 20
            drag(widget, [-10, 5], [-5, 2])
            drag(widget, [15, 2], [20, 2])
            assert widget.parameters["center"] == [-5, 2] and widget.parameters["radius"] == 25
        else:
            for point in [[-20, -15], [20, -15], [0, 20]]:
                click(widget, point)
            QTest.keyClick(widget, Qt.Key.Key_Return)
            drag(widget, [0, 20], [2, 24])
            assert widget.parameters["vertices"] == [[-20, -15], [20, -15], [2, 24]]
        block = builtin_registry().create("region.pose", parameters=widget.parameters)
        block.execute({"image": image, "pose": anchor})
        assert block.state == "pass", block.error
        from opencv_dataflow.selection import rasterize

        np.testing.assert_array_equal(
            rasterize(widget.region(), image.shape), block.last_result.outputs["mask"]
        )
        widget.resize(600, 850)
        app.processEvents()
        assert widget.to_image(widget.to_widget([-17, 12])) == [-17, 12]
    finally:
        widget.close()


def test_projective_editor_mapping_and_cancelled_gesture(app):
    image = np.zeros((200, 300), np.uint8)
    anchor = make_pose([[1.5, 0.1, 140], [0.1, 1.2, 90], [0.001, -0.001, 1]], [300, 200], {})
    widget = RegionCanvas(image, PoseRegion.defaults, pose=anchor)
    widget.resize(900, 600)
    widget.show()
    app.processEvents()
    try:
        for point in [[0, 0], [-20, -15], [18, 22]]:
            assert widget.to_image(widget.to_widget(point)) == point
        before = copy.deepcopy(widget.parameters)
        widget.new_shape()
        QTest.mousePress(
            widget, Qt.MouseButton.LeftButton, pos=widget.to_widget([-12, -8]).toPoint()
        )
        QTest.keyClick(widget, Qt.Key.Key_Escape)
        assert widget.parameters == before
    finally:
        widget.close()


@pytest.mark.parametrize("with_pose", [False, True])
def test_shape_mask_runtime_optional_frame_and_saved_project(with_pose, tmp_path):
    registry, graph = builtin_registry(), Graph()
    source = graph.add(registry.create("input.pattern_demo"))
    anchor = graph.add(
        registry.create("pose.define", parameters={"x": 250.0, "y": 180.0, "angle_degrees": 30.0})
    )
    mask = graph.add(registry.create("mask.shape", parameters={"rectangle": [0, 0, 20, 15]}))
    graph.connect(source.id, "image", anchor.id, "image")
    graph.connect(source.id, "image", mask.id, "image")
    if with_pose:
        graph.connect(anchor.id, "pose", mask.id, "pose")
    graph.run()
    assert mask.state == "pass", mask.error
    expected = mask.last_result.outputs["mask"].copy()
    assert expected.any() and mask.supports_region_editor
    assert bool(expected[0, 0]) is not with_pose
    path = tmp_path / "mask.json"
    graph.save(path)
    loaded = Graph.load(path, registry)
    loaded.run()
    np.testing.assert_array_equal(loaded.blocks[mask.id].last_result.outputs["mask"], expected)


@pytest.mark.parametrize("kind", ["region.pose", "mask.shape"])
def test_inspector_pose_editor_apply_cancel_cached_pair_and_missing_pose(
    app, monkeypatch, tmp_path, kind
):
    window = MainWindow()
    window.load_pose_inspection_example()
    graph = window.graph
    if kind == "mask.shape":
        block = graph.add(
            window.registry.create(kind, parameters={"rectangle": [-20, -20, 40, 40]})
        )
        graph.connect("pose_images", "image", block.id, "image")
        graph.connect("pose_offset", "pose", block.id, "pose")
        window.replace_graph(graph)
    else:
        block = graph.blocks["pose_region"]
    graph.run()
    assert block.state == "pass", block.error
    window.inspector.select(block)
    buttons = window.inspector.editor.findChildren(QPushButton)
    assert any(
        b.text() == f"Draw / edit {'mask' if kind == 'mask.shape' else 'region'}…" for b in buttons
    )
    before = graph.to_dict()
    warnings = []
    monkeypatch.setattr(window, "warn", warnings.append)

    def cancel(dialog):
        assert dialog.canvas.pose is not None
        assert "Local pose" in dialog.coordinates.text()
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(RegionEditor, "exec", cancel)
    window.inspector.edit_region()
    assert graph.to_dict() == before and not warnings

    def accept(dialog):
        dialog.canvas.parameters.update(shape="rectangle", rectangle=[-16, -12, 32, 24])
        dialog.accept()
        return dialog.result()

    monkeypatch.setattr(RegionEditor, "exec", accept)
    window.inspector.edit_region()
    assert block.parameters["rectangle"] == [-16, -12, 32, 24] and window.dirty
    # Apply invalidates outputs, but the captured image/pose pair can still be edited.
    monkeypatch.setattr(RegionEditor, "exec", cancel)
    window.inspector.edit_region()
    assert not warnings
    path = tmp_path / "relative_drawn.json"
    graph.save(path)
    loaded = Graph.load(path, window.registry)
    loaded.run()
    assert loaded.blocks[block.id].state == "pass"
    graph.run()
    # A fresh negative match must never fall back to the earlier captured pose.
    graph.blocks["pose_offset"].last_result.outputs["pose"] = {}
    window.inspector.edit_region()
    assert warnings and "valid pose" in warnings[-1]
    window.dirty = False
    window.close()
