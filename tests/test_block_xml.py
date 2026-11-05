import numpy as np
import pytest

from opencv_dataflow.block_xml import load_block_xml, save_block_xml
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.project_xml import export_block_xml, read_block_xml
from opencv_dataflow.toolblocks import Binding, ToolBlock


def test_block_xml_roundtrips_configuration_and_keeps_destination_layout_and_wires(tmp_path):
    registry = builtin_registry()
    graph = Graph()
    source = graph.add(registry.create("input.demo"))
    target = graph.add(registry.create("cv.morphology"))
    target.position = [120, 40]
    graph.connect(source.id, "image", target.id, "image")
    wires = list(graph.connections)

    preset = registry.create(
        "cv.morphology",
        parameters={"shape": "custom", "custom_kernel": [[0, 1], [1, 1]]},
    )
    preset.name = "Remove noise"
    preset.description = "Use after thresholding"
    preset.documentation = "<p><b>Custom kernel</b></p>"
    preset.position = [300, 200]
    preset.node_size = "large"
    preset.thumbnail = False
    preset.preview_output = "image"
    preset.input_levels = {"image": "1"}
    preset.input_failure_policies = {"image": "reject_failed"}
    preset.recipe_bindings = {"iterations": "settings.morph_iterations"}
    path = tmp_path / "remove-noise.cv.morphology.xml"
    save_block_xml(preset, path)
    assert read_block_xml(path) == preset.to_project_item()

    loaded = load_block_xml(graph, target.id, path, registry)
    assert loaded is graph.blocks[target.id]
    assert loaded.to_project_item() == {
        **preset.to_project_item(),
        "id": target.id,
        "position": [120, 40],
    }
    assert graph.connections == wires


def test_block_xml_rejects_wrong_type_and_preserves_block_if_wire_would_break(tmp_path):
    registry = builtin_registry()
    graph = Graph()
    source = graph.add(registry.create("input.demo"))
    target = graph.add(registry.create("cv.morphology"))
    downstream = graph.add(registry.create("cv.gray"))
    graph.connect(source.id, "image", target.id, "image")
    graph.connect(target.id, "image", downstream.id, "image")
    graph.run()
    incompatible, retained = graph.connections
    original = target.last_result
    path = tmp_path / "preset.cv.morphology.xml"
    export_block_xml(registry.create("cv.gray"), path)
    with pytest.raises(ValueError, match="Block type must match"):
        load_block_xml(graph, target.id, path, registry)

    preset = registry.create("cv.morphology")
    preset.exposed_inputs.remove("image")
    save_block_xml(preset, path)
    with pytest.raises(ValueError, match="approval to disconnect"):
        load_block_xml(graph, target.id, path, registry)
    assert load_block_xml(
        graph, target.id, path, registry, confirm_disconnect=lambda wires: False
    ) is None
    assert graph.blocks[target.id] is target and target.last_result is original
    assert graph.connections == [incompatible, retained]
    with pytest.raises(ValueError, match=r"must end in \.cv\.morphology\.xml"):
        save_block_xml(target, tmp_path / "wrong.xml")
    confirmed = []

    def confirm(wires):
        confirmed.extend(wires)
        return True

    loaded = load_block_xml(graph, target.id, path, registry, confirm_disconnect=confirm)
    assert loaded is graph.blocks[target.id]
    assert confirmed == [incompatible]
    assert graph.connections == [retained]


def test_block_xml_retains_nested_port_order_and_trained_image(tmp_path):
    registry = builtin_registry()
    inner = Graph()
    reference = inner.add(registry.create("sequence.image_memory"))
    reference.train_reference(np.arange(9, dtype=np.uint8).reshape(3, 3))
    outputs = {name: Binding(reference.id, "image") for name in ("second", "first")}
    tool = ToolBlock(inner, {}, outputs)
    path = tmp_path / "reference.core.toolblock.xml"
    save_block_xml(tool, path)
    graph = Graph()
    target = graph.add(ToolBlock(Graph(), {}, {}))
    loaded = load_block_xml(graph, target.id, path, registry)
    assert [port.name for port in loaded.outputs] == ["second", "first"]
    assert loaded.graph.blocks[reference.id].trained_image == reference.trained_image
