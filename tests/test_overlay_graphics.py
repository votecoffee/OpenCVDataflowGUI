from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QTreeWidget

from opencv_dataflow.app import MainWindow
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.graphics_layers import (
    RESULT_GRAPHICS,
    GraphicLayer,
    GraphicsBundle,
    graphic_object,
)
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import ValuePacket, frame, moved_payload
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolblocks import encapsulate

EXAMPLE = Path("examples/overlay_result_graphics/overlay_result_graphics.json")


def test_graphics_ports_are_opt_in_and_compose_without_copying_the_backdrop():
    registry = builtin_registry()
    node = registry.create("inspect.contours")
    assert RESULT_GRAPHICS not in node.exposed_outputs
    port = next(port for port in node.output_ports if port.name == RESULT_GRAPHICS)
    assert port.kind == "graphics"

    graph = Graph.load(EXAMPLE, registry)
    graph.run()
    source = graph.blocks["source"].last_result.outputs["image"]
    result = graph.blocks["composite"].last_result
    assert "Overlaid 2 layers" in result.message
    assert np.any(result.outputs["image"] != source)
    np.testing.assert_array_equal(result.outputs["image"][0, 0], source[0, 0])
    tool = encapsulate(graph, ["region"], "Inspection region", registry)
    original = next(wire for wire in graph.connections if wire.source == tool.id)
    graph.connections.remove(original)
    tool.exposed_outputs.append(RESULT_GRAPHICS)
    graph.connect(tool.id, RESULT_GRAPHICS, original.target, original.input)
    graph = Graph.from_dict(graph.to_dict(), registry)
    graph.run()
    assert "Overlaid 2 layers" in graph.blocks["composite"].last_result.message
    graph.set_run_profile("lean")
    graph.run()
    result = graph.blocks["composite"].last_result
    assert "Overlaid 0 layers" in result.message
    np.testing.assert_array_equal(result.outputs["image"], source)


def test_composite_maps_drawing_objects_to_the_final_grid_and_skips_other_tickets():
    registry = builtin_registry()
    composite = registry.create("cv.overlay_result_graphics")
    canvas = ImageArray(np.full((5, 5, 4), [1, 2, 3, 41123], np.uint16), "rgba")
    color = (60, 100, 200)
    objects = (graphic_object("polyline", [(6, 5), (7, 5)], color, thickness=1),)
    clock = TicketClock()
    context = {"spatial": frame([10, 10], "source"), "ticket": clock.issue("source")}
    crop_context = moved_payload(context, np.float64([[1, 0, 4], [0, 1, 3], [0, 0, 1]]), [5, 5])
    layers = GraphicsBundle((
        GraphicLayer("outline", objects, context),
        GraphicLayer("old", objects, {**context, "ticket": clock.issue("source")}),
        GraphicLayer("unrelated", objects, {**context, "spatial": frame([10, 10], "other")}),
    ))
    result = composite.execute({"image": ValuePacket(canvas, crop_context), "graphics_1": layers})
    assert result is not None, composite.error
    assert "Overlaid 1 layers" in result.message and "skipped 2" in result.message
    output = result.outputs["image"]
    rgba = [200 * 257, 100 * 257, 60 * 257, 41123]
    assert output.dtype == np.uint16 and image_format(output) == "rgba"
    np.testing.assert_array_equal(output[2, 2:4], [rgba, rgba])
    np.testing.assert_array_equal(output[0, 0], canvas[0, 0])
    assert np.count_nonzero(output != canvas) == 6


def test_inspector_layer_checklist_persists_the_exclusion(app, monkeypatch):
    window = MainWindow()
    graph = Graph.load(EXAMPLE, window.registry)
    window.replace_graph(graph)
    graph.run()
    window.scene.nodes["composite"].setSelected(True)
    assert window.inspector.preview_choice.findText("Result / image") >= 0

    def choose(dialog):
        tree = dialog.findChild(QTreeWidget)
        assert tree.topLevelItemCount() == 2
        region = tree.findItems("region", Qt.MatchFlag.MatchExactly, 0)[0]
        assert region.checkState(0) == Qt.CheckState.Checked
        region.setCheckState(0, Qt.CheckState.Unchecked)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", choose)
    try:
        window.inspector.choose_result_graphics()
        assert window.inspector.apply()
        restored = Graph.from_dict(graph.to_dict(), window.registry)
        assert restored.blocks["composite"].parameters["excluded_graphics"] == {
            "graphics_1": ["region"]
        }
        restored.run()
        assert "Overlaid 1 layers" in restored.blocks["composite"].last_result.message
    finally:
        window.dirty = False
        window.close()
