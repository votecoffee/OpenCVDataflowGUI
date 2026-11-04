"""Offscreen Qt interaction checks: ports, inspector, and worker lifecycle."""

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from opencv_dataflow.app import MainWindow
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.canvas import CanvasScene, CanvasView
from opencv_dataflow.core import Graph


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
