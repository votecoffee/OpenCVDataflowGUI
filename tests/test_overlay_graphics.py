from pathlib import Path

import numpy as np
import pytest
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

EXAMPLE = Path("examples/assets/synthetic_graphs/overlay_result_graphics.json")


@pytest.mark.local_examples
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
    expected = graph.blocks["composite"].last_result.outputs["image"].copy()
    graph.set_graphics_mode("off")
    graph.run()
    assert all(not block.last_graphics for block in graph.blocks.values())
    np.testing.assert_array_equal(graph.blocks["composite"].last_result.outputs["image"], expected)
    graph.set_graphics_mode("inherit")
    graph.set_run_profile("lean")
    graph.run()
    result = graph.blocks["composite"].last_result
    assert "Overlaid 0 layers" in result.message
    np.testing.assert_array_equal(result.outputs["image"], source)
    graph.set_graphics_mode("on")
    graph.run()
    np.testing.assert_array_equal(graph.blocks["composite"].last_result.outputs["image"], expected)


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


def test_annotation_fill_reprojects_and_leaves_the_source_untouched():
    registry = builtin_registry()
    graph = Graph()
    annotation = graph.add(registry.create("cv.annotation_layer", parameters={
        "shape": "rectangle", "origin": [7, 5], "end": [10, 8],
        "color_rgb": [200, 90, 30], "filled": True, "layer_name": "fixture",
    }))
    crop = graph.add(registry.create(
        "cv.crop", parameters={"x": 6, "y": 4, "width": 7, "height": 7},
    ))
    overlay = graph.add(registry.create("cv.overlay_result_graphics"))
    graph.connect(annotation.id, RESULT_GRAPHICS, overlay.id, "graphics_1")
    graph.connect(crop.id, "image", overlay.id, "image")
    context = {"spatial": frame([20, 16], "source")}
    image = np.zeros((16, 20), np.uint16)
    packet = ValuePacket(image, context)
    graph.run(injected={(annotation.id, "image"): packet, (crop.id, "image"): packet})
    assert annotation.last_result is not None, annotation.error
    assert overlay.last_result is not None, overlay.error
    layers = annotation.last_result.outputs[RESULT_GRAPHICS].layers
    assert [layer.name for layer in layers] == ["fixture"]
    assert layers[0].objects[0].kind == "filled_polygon"
    expected = np.zeros((7, 7, 3), np.uint16)
    expected[1:5, 1:5] = [30 * 257, 90 * 257, 200 * 257]
    np.testing.assert_array_equal(overlay.last_result.outputs["image"], expected)
    assert not image.any()


def test_annotation_label_accepts_wired_text_and_rejects_foreign_coordinates():
    block = builtin_registry().create("cv.annotation_layer", parameters={"shape": "label"})
    context = {"spatial": frame([20, 16], "source")}
    image = ValuePacket(np.zeros((16, 20), np.uint8), context)
    result = block.execute({
        "image": image, "points": ValuePacket([[5.25, 7.5]], context), "text": "Pass 42",
    })
    assert result is not None, block.error
    item = result.outputs[RESULT_GRAPHICS].layers[0].objects[0]
    assert item.kind == "label" and item.points == ((5.25, 7.5),) and item.text == "Pass 42"
    foreign = ValuePacket([[5, 7]], {"spatial": frame([20, 16], "other")})
    assert block.execute({"image": image, "points": foreign}) is None
    assert "Transform annotation points" in block.error


@pytest.mark.local_examples
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
