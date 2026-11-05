import copy

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry, demo_graph
from opencv_dataflow.core import Graph
from opencv_dataflow.toolblocks import Binding, ToolBlock, encapsulate


def inspection_toolblock():
    registry = builtin_registry()
    graph = demo_graph(registry)
    nodes = list(graph.blocks.values())
    block = encapsulate(graph, [b.id for b in nodes[1:]], "Inspect shapes", registry)
    return registry, graph, nodes, block


def test_encapsulation_preserves_pipeline_and_records():
    registry, graph, nodes, block = inspection_toolblock()
    assert len(graph.blocks) == 2
    assert sum(port.required for port in block.inputs) == 1
    assert len(block.inputs) == 7  # Image plus optional mask/region terminals.
    assert {p.kind for p in block.outputs} == {"image", "number", "data"}
    graph.run()
    assert block.state == "pass", block.error
    assert block.last_result.score == 1
    assert block.last_result.outputs["Find_contours_count"] == 2
    assert block.graph.blocks[nodes[-1].id].state == "pass"
    assert len(block.last_output_images) == 2
    assert len(block.last_graphics) == 1
    assert len(block.last_inputs) == 1
    # No runtime state is embedded in a definition.
    restored = Graph.from_dict(graph.to_dict(), registry)
    assert restored.to_dict() == graph.to_dict()
    assert restored.blocks[block.id].state == "idle"


def test_crossing_fanout_and_nested_graphs(tmp_path):
    registry = builtin_registry()
    graph = demo_graph(registry)
    nodes = list(graph.blocks.values())
    side = graph.add(registry.create("cv.threshold"))
    graph.connect(nodes[1].id, "image", side.id, "image")
    inner = encapsulate(graph, [nodes[1].id, nodes[2].id], "Preprocess", registry)
    outer = encapsulate(graph, [inner.id, nodes[3].id, nodes[4].id], "Inspect", registry)
    graph.run()
    assert all(b.state == "pass" for b in graph.blocks.values())
    assert outer.graph.blocks[inner.id].state == "pass"
    assert outer.graph.blocks[inner.id].graph.blocks[nodes[1].id].state == "pass"
    path = tmp_path / "nested.json"
    graph.save(path)
    loaded = Graph.load(path, registry)
    loaded.run()
    assert loaded.blocks[outer.id].last_result.score == 1
    np.testing.assert_array_equal(
        loaded.blocks[side.id].last_output_images["image"], side.last_output_images["image"]
    )


def test_toolblock_export_import_has_independent_instances(tmp_path):
    registry, graph, nodes, block = inspection_toolblock()
    path = tmp_path / "inspect.toolblock.json"
    block.description = "Count parts before picking."
    block.documentation = "<p><strong>Picking:</strong> verify the part count first.</p>"
    block.export(path)
    a, b = ToolBlock.read(path, registry), ToolBlock.read(path, registry)
    assert a.description == b.description == block.description
    assert a.documentation == b.documentation == block.documentation
    assert a.id != b.id
    a.graph.blocks[nodes[-1].id].parameters["expected_count"] = 3
    assert b.graph.blocks[nodes[-1].id].parameters["expected_count"] == 2
    graph.run()
    image = nodes[0].last_output_images["image"]
    terminal = a.inputs[0].name
    a.execute({terminal: image})
    b.execute({terminal: image})
    assert a.state == "fail"
    assert b.state == "pass"
    a.graph.blocks[nodes[-1].id].last_output_images["mask"][:] = 0
    assert b.graph.blocks[nodes[-1].id].last_output_images["mask"].any()


def test_inner_errors_propagate_and_clear_stale_records():
    _, graph, nodes, block = inspection_toolblock()
    graph.run()
    block.graph.blocks[nodes[2].id].parameters["kernel"] = 2
    graph.run()
    assert block.state == "error"
    assert "positive odd" in block.error
    assert block.last_result is None
    assert block.graph.blocks[nodes[-1].id].state == "skipped"
    assert not block.graph.blocks[nodes[-1].id].last_output_images


def test_connected_interface_rename_and_rejection_are_atomic():
    _, graph, _, block = inspection_toolblock()
    old_name = block.inputs[0].name
    binding = block.input_bindings[old_name]
    block.update_interface({"image": binding}, block.output_bindings, graph)
    assert graph.connections[0].input == "image"
    before = copy.deepcopy(graph.to_dict())
    with pytest.raises(ValueError, match="Disconnect"):
        block.update_interface({}, block.output_bindings, graph)
    assert graph.to_dict() == before
    graph.run()
    assert block.state == "pass"


def test_invalid_bindings_and_injected_input_conflicts():
    registry, graph, nodes, block = inspection_toolblock()
    definition = block.definition()
    definition["outputs"]["bad"] = {"block": "absent", "port": "image"}
    with pytest.raises(ValueError, match="missing internal port"):
        ToolBlock.from_definition(definition, registry)
    with pytest.raises(ValueError, match="internally connected"):
        graph.run(injected={(block.id, block.inputs[0].name): np.ones((2, 2), np.uint8)})
    with pytest.raises(ValueError, match="connected internally"):
        ToolBlock(block.graph, {"bad": Binding(nodes[2].id, "image")}, {})


def test_nonconvex_grouping_rejects_new_cycle_without_mutation():
    registry = builtin_registry()
    graph = demo_graph(registry)
    nodes = list(graph.blocks.values())
    before = copy.deepcopy(graph.to_dict())
    with pytest.raises(ValueError, match="Cycles"):
        encapsulate(graph, [nodes[1].id, nodes[3].id], "Bad grouping", registry)
    assert graph.to_dict() == before


def test_nested_cancellation_releases_resources(monkeypatch):
    _, graph, _, block = inspection_toolblock()
    seen = []
    children = list(block.graph.blocks.values())
    monkeypatch.setattr(children[0], "close", lambda: seen.append("closed"))
    block.execute({block.inputs[0].name: np.ones((10, 10, 3), np.uint8)}, lambda: True)
    assert all(b.state == "skipped" for b in children)
    assert block.state == "error"
    graph.close()
    assert seen == ["closed"]


def test_nesting_limit():
    registry = builtin_registry()
    graph = Graph()
    for _ in range(18):
        block = ToolBlock(graph, {}, {})
        outer = Graph()
        outer.add(block)
        graph = outer
    with pytest.raises(ValueError, match="16 levels"):
        Graph.from_dict(graph.to_dict(), registry)


def test_nested_grouping_and_renaming_preserve_owner_terminals():
    registry, graph, nodes, outer = inspection_toolblock()
    outer.update_interface(
        outer.input_bindings,
        {**outer.output_bindings, "gray": Binding(nodes[1].id, "image")},
        graph,
    )
    inner = encapsulate(
        outer.graph, [nodes[1].id, nodes[2].id], "Preprocess", registry, owner=outer
    )
    assert outer.output_bindings["gray"].block == inner.id
    inner.update_interface(
        {
            "image": next(iter(inner.input_bindings.values())),
            **dict(list(inner.input_bindings.items())[1:]),
        },
        inner.output_bindings,
        outer.graph,
        owner=outer,
    )
    assert next(iter(outer.input_bindings.values())) == Binding(inner.id, "image")
    graph.run()
    assert outer.state == "pass", outer.error
    assert outer.last_result.outputs["gray"].ndim == 2
    # Child run records are recursively copied into the UI display graph.
    display = Graph.from_dict(graph.to_dict(), registry)
    display.blocks[outer.id].copy_results_from(outer)
    displayed_inner = display.blocks[outer.id].graph.blocks[inner.id]
    assert displayed_inner.graph.blocks[nodes[1].id].state == "pass"


def test_shipped_inspection_pattern():
    registry = builtin_registry()
    block = ToolBlock.read("examples/inspect_shapes.toolblock.json", registry)
    source = registry.create("input.demo")
    source.execute({})
    block.execute({"image": source.last_output_images["image"]})
    assert block.state == "pass", block.error
    assert block.last_result.outputs["count"] == 2
    assert set(block.last_output_images) == {"overlay", "mask"}
