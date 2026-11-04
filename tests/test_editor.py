"""Offscreen Qt interaction checks: ports, inspector, and worker lifecycle."""

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QInputDialog

from opencv_dataflow.app import MainWindow
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.canvas import CanvasScene, CanvasView
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.toolblock_editor import InterfaceDialog
from opencv_dataflow.toolblocks import encapsulate


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def pump_until(app, predicate, seconds=8):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        QTest.qWait(10)
    raise AssertionError("Qt condition timed out")


@pytest.mark.parametrize("reverse", [False, True])
def test_drag_ports_both_directions(app, reverse):
    registry = builtin_registry()
    graph = Graph()
    a = graph.add(registry.create("input.demo"))
    b = graph.add(registry.create("gapi.gray"))
    b.position = [350, 0]
    scene = CanvasScene(graph)
    view = CanvasView(scene)
    view.resize(1000, 500)
    view.show()
    view.fit_graph()
    app.processEvents()
    start = view.mapFromScene(scene.nodes[a.id].sockets[True, "image"].scenePos())
    end = view.mapFromScene(scene.nodes[b.id].sockets[False, "image"].scenePos())
    if reverse:
        start, end = end, start
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(view.viewport(), (start + end) / 2)
    QTest.mouseMove(view.viewport(), end)
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=end)
    assert len(graph.connections) == 1
    assert graph.connections[0].source == a.id
    assert graph.connections[0].target == b.id
    view.close()


def test_run_inspector_delete_and_continuous_stop(app):
    window = MainWindow()
    window.show()
    app.processEvents()
    last = list(window.graph.blocks.values())[-1]
    window.scene.nodes[last.id].setSelected(True)
    assert window.inspector.block is last
    window.start_run(False)
    assert window.scene.locked
    pump_until(app, lambda: window.runner is None)
    assert all(b.state == "pass" for b in window.graph.blocks.values())
    assert "2 contours" in window.inspector.details.toPlainText()
    assert window.inspector.preview_choice.count() == 4  # input, two outputs, one graphic
    assert not window.inspector.image.pixmap().isNull()
    # Applying exposure and thumbnail choices must update the node and persist.
    window.inspector.thumb.setChecked(False)
    window.inspector.fields["expected_count"].setText("3")
    window.inspector.apply()
    assert last.parameters["expected_count"] == 3
    assert not last.thumbnail
    assert window.dirty
    window.start_run(True)
    pump_until(app, lambda: last.state == "fail")
    window.stop_run()
    pump_until(app, lambda: window.runner is None)
    assert last.last_result.score == 1 - 1 / 3
    assert not window.scene.locked
    wire = window.scene.wires[-1]
    window.scene.clearSelection()
    wire.setSelected(True)
    window.scene.delete_selected()
    assert len(window.graph.connections) == 3
    assert last.state == "idle"
    window.dirty = False
    window.close()


def test_canvas_node_movement_updates_wires(app):
    window = MainWindow()
    node = next(iter(window.scene.nodes.values()))
    old = window.scene.wires[0].path().pointAtPercent(0)
    node.setPos(node.pos() + QPoint(40, 30))
    new = window.scene.wires[0].path().pointAtPercent(0)
    assert new.x() - old.x() == 40
    assert new.y() - old.y() == 30
    assert node.block.position == [40.0, 30.0]
    window.dirty = False
    window.close()


def test_toolblock_group_navigation_run_and_save(app, monkeypatch, tmp_path):
    window = MainWindow()
    window.show()
    app.processEvents()
    original = list(window.graph.blocks.values())
    for b in original[1:]:
        window.scene.nodes[b.id].setSelected(True)
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Inspect", True))
    window.create_toolblock()
    block = window.inspector.block
    assert block.name == "Inspect"
    assert len(window.graph.blocks) == 2
    assert len(window.toolblock_templates) == 1
    # Enter by double-clicking the actual node header.
    center = window.view.mapFromScene(window.scene.nodes[block.id].scenePos() + QPoint(80, 20))
    QTest.mouseDClick(window.view.viewport(), Qt.MouseButton.LeftButton, pos=center)
    app.processEvents()
    assert window.graph is block.graph
    assert "Inspect" in window.breadcrumb.text()
    inner_nodes = list(block.graph.blocks.values())
    inner = encapsulate(
        window.graph,
        [inner_nodes[0].id, inner_nodes[1].id],
        "Preprocess",
        window.registry,
        owner=block,
    )
    assert next(iter(block.input_bindings.values())).block == inner.id
    window.scene.rebuild(inner.id)
    window.enter_toolblock(inner)
    assert len(window.graph_stack) == 2
    window.start_run(False)  # Still runs the complete job, including its image source.
    pump_until(app, lambda: window.runner is None)
    assert block.state == "pass", block.error
    assert all(b.state == "pass" for b in window.graph.blocks.values())
    assert window.back_action.isEnabled()
    path = tmp_path / "job.json"
    window.path = path
    assert window.save_project()
    loaded = Graph.load(path, window.registry)
    assert len(loaded.blocks) == 2  # Save targets the root while editing nested graphs.
    window.leave_toolblock()
    assert window.graph is block.graph
    window.leave_toolblock()
    assert window.graph is window.root_graph and not window.graph_stack
    assert window.inspector.block is block
    window.dirty = False
    window.close()


def test_toolblock_interface_dialog_and_library_instances(app):
    window = MainWindow()
    original = list(window.graph.blocks.values())
    block = encapsulate(window.graph, [b.id for b in original[1:]], "Inspect", window.registry)
    dialog = InterfaceDialog(block, window.graph)
    assert dialog.table.rowCount() == len(block.inputs) + len(block.outputs)
    dialog.table.item(0, 1).setText("image")
    dialog.apply()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert window.graph.connections[0].input == "image"
    window.toolblock_templates.append(block.definition())
    window.populate_library()
    item = window.library.item(window.library.count() - 1)
    window.add_block(item)
    first = window.inspector.block
    window.add_block(item)
    second = window.inspector.block
    assert first.id != second.id
    assert first.graph is not second.graph
    first.graph.blocks[original[-1].id].parameters["expected_count"] = 9
    assert second.graph.blocks[original[-1].id].parameters["expected_count"] == 2
    window.dirty = False
    window.close()


def test_image_file_dropdowns_metadata_and_preview(app, tmp_path):
    import cv2
    import numpy as np

    window = MainWindow()
    window.show()
    path = tmp_path / "red16.png"
    image = np.zeros((20, 30, 3), np.uint16)
    image[..., 2] = 65535
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(encoded.tobytes())
    block = window.graph.add(window.registry.create("input.file"))
    window.scene.rebuild(block.id)
    window.inspector.fields["path"].setText(str(path))
    window.inspector.fields["color_format"].setCurrentText("rgb")
    window.inspector.fields["pixel_depth"].setCurrentText("uint16")
    window.inspector.apply()
    assert block.parameters["color_format"] == "rgb"
    window.start_run(False)
    pump_until(app, lambda: window.runner is None)
    assert block.state == "pass", block.error
    assert isinstance(block.last_output_images["image"], ImageArray)
    assert block.last_output_images["image"].color_format == "rgb"
    window.inspector.preview_choice.setCurrentText("Output / image")
    shown = window.inspector.image.pixmap().toImage()
    color = shown.pixelColor(shown.width() // 2, shown.height() // 2)
    assert color.red() == 255 and color.blue() == 0
    assert '"bits_per_channel": 16' in window.inspector.details.toPlainText()
    window.dirty = False
    window.close()
