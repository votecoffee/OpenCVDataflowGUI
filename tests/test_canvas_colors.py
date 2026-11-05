"""Semantic canvas colors, reusable terminal aliases and independent status styling."""

import numpy as np
import pytest
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QDialog, QLabel

from opencv_dataflow.app import MainWindow
from opencv_dataflow.blocks import builtin_registry, demo_graph
from opencv_dataflow.canvas import CanvasScene, Wire
from opencv_dataflow.core import Block, Graph, Port, Result
from opencv_dataflow.payloads import compatible
from opencv_dataflow.toolblocks import Binding, ToolBlock
from opencv_dataflow.visual_style import BLOCK_COLORS, PORT_COLORS, block_type, port_type


@pytest.mark.parametrize(
    "port,expected",
    [
        (Port("image"), "image"),
        (Port("mask"), "region"),
        (Port("search_mask"), "region"),
        (Port("region", "data"), "region"),
        (Port("region_a", "data"), "region"),
        (Port("training_region", "data"), "region"),
        (Port("pose", "data"), "pose"),
        (Port("transform", "data"), "pose"),
        (Port("coordinate_frame", "data"), "pose"),
        (Port("ticket", "data"), "data"),
        (Port("count", "number"), "number"),
        (Port("depth", "depth"), "depth"),
        (Port("images", "pyramid"), "pyramid"),
        (Port("value", "any"), "any"),
        (Port("custom", "data", visual_type="pose"), "pose"),
        (Port("custom", "data", visual_type="unknown"), "data"),
    ],
)
def test_semantic_ports_keep_runtime_contracts(port, expected):
    assert port_type(Block(), port) == expected
    assert compatible(port.kind, port.kind)


def test_requested_colors_and_aliases_survive_nested_toolblocks(app):
    registry = builtin_registry()
    inner = Graph()
    pose = inner.add(registry.create("pose.offset", block_id="offset"))
    wrapped = ToolBlock(
        inner, {"origin": Binding(pose.id, "pose")}, {"result": Binding(pose.id, "pose")}
    )
    outer = Graph()
    outer.add(wrapped)
    nested = ToolBlock(
        outer,
        {"reference": Binding(wrapped.id, "origin")},
        {"final": Binding(wrapped.id, "result")},
    )
    graph = Graph()
    graph.add(nested)
    restored = Graph.from_dict(graph.to_dict(), registry)
    scene = CanvasScene(restored)
    node = scene.nodes[nested.id]
    assert node.sockets[False, "reference"].brush().color().name() == "#66a9f2"
    assert node.sockets[True, "final"].visual_type == "pose"
    assert PORT_COLORS["image"] == "#e6b94f"
    assert PORT_COLORS["region"] == "#ba8aeb"


def test_wires_use_source_type_and_generic_target_hint(app):
    registry = builtin_registry()
    graph = Graph()
    region = graph.add(registry.create("region.shape"))
    inspect = graph.add(registry.create("inspect.contours"))
    graph.connect(region.id, "mask", inspect.id, "image")
    scene = CanvasScene(graph)
    wire = scene.wires[0]
    assert wire.pen().color() == scene.nodes[region.id].sockets[True, "mask"].brush().color()
    assert wire.visual_type == "region"
    assert scene.nodes[inspect.id].sockets[False, "image"].visual_type == "image"
    assert wire.pen().isCosmetic()
    source = scene.nodes[region.id].sockets[True, "region"]
    target = scene.nodes[inspect.id].sockets[False, "region"]
    scene.start_wire(source)
    assert scene.drag_path.pen().color() == source.brush().color()
    scene.finish_wire(None)
    source.visual_type = "any"
    typed_wire = Wire(wire.connection, source, target)
    assert typed_wire.visual_type == "region"


def test_explicit_plugin_hints_keep_arbitrary_public_names():
    class Custom(Block):
        inputs = (Port("origin", "data", visual_type="pose"),)
        outputs = (Port("window", "data", visual_type="region"),)

    child = Custom()
    graph = Graph()
    graph.add(child)
    block = ToolBlock(
        graph, {"renamed": Binding(child.id, "origin")}, {"other": Binding(child.id, "window")}
    )
    assert block.inputs[0].visual_type == "pose"
    assert port_type(block, block.outputs[0], True) == "region"


def test_node_badges_distinguish_invalid_input_from_exception_and_clear(app):
    class Plugin(Block):
        broken = True

        def process(self, inputs):
            if self.broken:
                raise ValueError("unexpected plugin exception")
            return Result({})

    registry = builtin_registry()
    graph = Graph()
    blur = graph.add(registry.create("cv.gaussian_blur"))
    plugin = graph.add(Plugin())
    graph.set_run_profile("lean")
    graph.run(injected={(blur.id, "image"): "not pixels"})
    scene = CanvasScene(graph)
    invalid, exception = scene.nodes[blur.id], scene.nodes[plugin.id]
    assert [item[0] for item in invalid.issue_badges()] == ["!"]
    assert [item[0] for item in exception.issue_badges()] == ["×"]
    invalid.refresh_issue_tooltip()
    exception.refresh_issue_tooltip()
    assert "Invalid data" in invalid.toolTip()
    assert "unexpected plugin exception" in exception.toolTip()
    display = registry.create("cv.gaussian_blur")
    display.copy_results_from(blur)
    assert display.error_kind == "invalid"

    plugin.broken = False
    graph.run(injected={(blur.id, "image"): np.zeros((16, 16), dtype=np.uint8)})
    assert blur.state == plugin.state == "pass"
    assert not invalid.issue_badges() and not exception.issue_badges()


def test_node_category_border_and_selection_preserve_result_header(app):
    graph = demo_graph(builtin_registry())
    graph.run()
    scene = CanvasScene(graph)
    node = next(iter(scene.nodes.values()))
    assert node.category == "Input"
    assert node.border_color.name() == BLOCK_COLORS["Input"]
    last = node.block.last_result
    state = node.block.state
    wire = scene.wires[0]
    original = wire.pen().color()
    wire.setSelected(True)
    node.setSelected(True)
    image = QImage(1000, 600, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    scene.render(painter)
    painter.end()
    assert wire.pen().color() == original
    assert node.block.last_result is last and node.block.state == state == "pass"
    assert not image.isNull()
    r = builtin_registry()
    assert block_type(r.create("region.shape")) == "Regions"
    assert block_type(r.create("pose.define")) == "Poses"


def test_color_legend_lists_type_and_status_purposes(app):
    window = MainWindow()
    seen = []

    def inspect():
        for widget in app.topLevelWidgets():
            if isinstance(widget, QDialog) and widget.windowTitle() == "Canvas color legend":
                seen.extend(label.text() for label in widget.findChildren(QLabel))
                widget.accept()

    QTimer.singleShot(0, inspect)
    window.show_color_legend()
    assert any("gold" in text for text in seen)
    assert any("run status" in text for text in seen)
    window.dirty = False
    window.close()
