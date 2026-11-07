import copy
import threading

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Block, Graph, Port, Result
from opencv_dataflow.payloads import FlowGroup, ValuePacket
from opencv_dataflow.tickets import TicketClock


def entry(ticket, value=None, source="camera", sequence=None):
    clock = TicketClock("test")
    record = clock.issue(source, "external", ticket)
    if sequence is not None:
        record["sequence"] = sequence
    return ValuePacket(ticket if value is None else value, {"ticket": record})


def execute(block, inputs, state=None):
    block.execute(inputs)
    assert block.state == state if state else block.state in ("pass", "waiting")
    assert not block.error, block.error
    return block.last_result


@pytest.mark.parametrize("value", [42, 42.5, "part-A", {"timestamp": "2026-11-04T10:30:00-05:00"}])
def test_source_tickets_external_types_and_propagation(value):
    registry = builtin_registry()
    source = registry.create("input.demo")
    source.ticket_settings = {"mode": "external", "value": 1}
    result = execute(source, {"ticket": value})
    ticket = result.outputs["ticket"]
    converted = registry.create("gapi.gray")
    execute(converted, {"image": ValuePacket(result.outputs["image"], result.payloads["image"])})
    assert converted.last_result.payloads["image"]["ticket"] == ticket
    assert converted.last_output_images["image"].metadata["ticket"] == ticket
    if isinstance(value, dict):
        assert ticket["value"].startswith("2026-11-04T15:30") and ticket["kind"] == "timestamp"
    else:
        assert ticket["value"] == value


def test_auto_source_sequences_persist_across_runs_and_settings_roundtrip():
    r, graph = builtin_registry(), Graph()
    source = graph.add(r.create("input.demo"))
    graph.run()
    first = copy.deepcopy(source.last_result.outputs["ticket"])
    graph.run()
    second = source.last_result.outputs["ticket"]
    assert first["value"] + 1 == second["value"] and first["session"] == second["session"]
    source.ticket_settings = {"mode": "fixed", "value": "PLC-123"}
    clone = Graph.from_dict(graph.to_dict(), r)
    clone.run()
    assert clone.blocks[source.id].last_result.outputs["ticket"]["value"] == "PLC-123"


def test_auto_timestamp_tickets_are_unique_and_clone_keeps_clock():
    source = builtin_registry().create("input.demo")
    source.ticket_settings = {"mode": "auto_timestamp", "value": 1}
    first = execute(source, {}).outputs["ticket"]
    second = execute(source, {}).outputs["ticket"]
    assert first["kind"] == "timestamp" and first["value"] < second["value"]
    assert first["sequence"] + 1 == second["sequence"]
    clock = TicketClock()
    clock.issue("source", "auto_timestamp")
    clone = clock.clone()
    assert clone.last_timestamp == clock.last_timestamp and clone.sequence == clock.sequence


def test_manifest_fifo_distinguishes_cameras_with_shared_plc_part_ticket():
    fifo = builtin_registry().create("flow.reorder")
    left, right = entry("part-7", "left", source="cam1"), entry("part-7", "right", source="cam2")
    execute(fifo, {"expected": left})
    execute(fifo, {"expected": right})
    execute(fifo, {"value_1": right}, "waiting")
    output = execute(fifo, {"value_2": left})
    assert [p.value for p in output.emissions["value"]] == ["left", "right"]


def test_external_full_records_keep_part_identity_and_bind_acquisition_source():
    r = builtin_registry()
    shared = TicketClock().issue("plc", "external", "part-7")
    a, b = r.create("input.demo"), r.create("input.demo")
    for source in (a, b):
        source.ticket_settings = {"mode": "external", "value": 1}
        execute(source, {"ticket": shared})
        assert source.last_result.outputs["ticket"]["source"] == source.id
        assert source.last_result.outputs["ticket"]["session"] == shared["session"]
    join = r.create("flow.join")
    for name, source in (("branch_1", a), ("branch_2", b)):
        execute(
            join,
            {
                name: ValuePacket(
                    source.last_result.outputs["image"], source.last_result.payloads["image"]
                )
            },
        )
    assert join.state == "pass"


@pytest.mark.parametrize(
    "tickets",
    [
        [1, 2],
        ["one", "two"],
        [{"timestamp": "2026-11-04T12:00:00Z"}, {"timestamp": "2026-11-04T12:01:00Z"}],
    ],
)
def test_join_matches_ticket_identity_not_arrival_position(tickets):
    join = builtin_registry().create("flow.join")
    execute(join, {"branch_1": entry(tickets[0], "left-a")}, "waiting")
    execute(join, {"branch_2": entry(tickets[1], "right-b")}, "waiting")
    result = execute(join, {"branch_2": entry(tickets[0], "right-a")}, "pass")
    assert [p.value for p in result.outputs["group"].items] == ["left-a", "right-a"]
    result = execute(join, {"branch_1": entry(tickets[1], "left-b")})
    assert [p.value for p in result.outputs["group"].items] == ["left-b", "right-b"]
    join.execute({"branch_1": entry(tickets[1], "duplicate")})
    assert join.state == "error" and "Duplicate" in join.error and not join.runtime["pending"]


def test_batch_three_cameras_unbatch_and_select_preserve_each_ticket_and_frame():
    r = builtin_registry()
    batch = r.create("flow.batch", parameters={"distinct_sources": True})
    frames = [
        entry(f"cam-{i}", np.full((8, 10), i, np.uint8), source=f"cam{i}", sequence=i + 1)
        for i in range(3)
    ]
    for value in frames:
        result = execute(batch, {"item_1": value})
    group = result.outputs["group"]
    assert isinstance(group, FlowGroup)
    unbatch = r.create("flow.unbatch", parameters={"order": "sequence_descending"})
    output = execute(unbatch, {"group": group})
    assert [p.payload["ticket"]["value"] for p in output.emissions["value"]] == [
        "cam-2",
        "cam-1",
        "cam-0",
    ]
    selected = execute(r.create("flow.select_entry", parameters={"index": 1}), {"group": group})
    np.testing.assert_array_equal(selected.outputs["value"], frames[1].value)
    assert selected.payloads["value"]["ticket"] == frames[1].payload["ticket"]


def test_batch_state_survives_repeated_runs_and_clones_but_not_save_or_invalidation():
    r, graph = builtin_registry(), Graph()
    source = graph.add(r.create("input.demo"))
    batch = graph.add(r.create("flow.batch"))
    graph.connect(source.id, "image", batch.id, "item_1")
    graph.run()
    assert batch.state == "waiting"
    clone = Graph.from_dict(graph.to_dict(), r)
    assert not clone.blocks[batch.id].runtime["pending"]
    clone.copy_runtime_from(graph)
    clone.run()
    assert clone.blocks[batch.id].last_result.outputs["queued"] == 2
    clone.run()
    assert len(clone.blocks[batch.id].last_result.outputs["group"].items) == 3
    graph.invalidate()
    assert not batch.runtime["pending"]


def test_reorder_numeric_and_opaque_manifest_wait_for_missing_head():
    r = builtin_registry()
    numeric = r.create("flow.reorder", parameters={"mode": "number"})
    execute(numeric, {"value_1": entry(3)}, "waiting")
    execute(numeric, {"value_1": entry(2)}, "waiting")
    output = execute(numeric, {"value_1": entry(1)})
    assert [p.value for p in output.emissions["value"]] == [1, 2, 3]
    fifo = r.create("flow.reorder")
    for value in ["z-last-alphabet", "a-first-alphabet"]:
        execute(fifo, {"expected": entry(value)}, "waiting")
    execute(fifo, {"value_1": entry("a-first-alphabet")}, "waiting")
    output = execute(fifo, {"value_2": entry("z-last-alphabet")})
    assert [p.value for p in output.emissions["value"]] == ["z-last-alphabet", "a-first-alphabet"]


@pytest.mark.parametrize(
    "kind,inputs",
    [
        ("flow.batch", [{"item_1": entry(1)}, {"item_1": entry(2)}]),
        ("flow.join", [{"branch_1": entry(1)}, {"branch_1": entry(2)}]),
        ("flow.reorder", [{"value_1": entry(1)}, {"value_1": entry(2)}]),
    ],
)
def test_buffers_fail_closed_on_limit(kind, inputs):
    block = builtin_registry().create(kind, parameters={"max_pending": 1})
    execute(block, inputs[0])
    block.execute(inputs[1])
    assert block.state == "error" and "buffer limit" in block.error
    assert not block.runtime["pending"]


def test_case_select_executes_only_selected_branch_and_merge_fifo():
    r, graph = builtin_registry(), Graph()
    source = graph.add(r.create("input.demo"))
    route = graph.add(r.create("flow.case", parameters={"cases": [0, 1], "modulo": 2}))
    a, b = [graph.add(r.create("gapi.gray")) for _ in range(2)]
    fifo = graph.add(r.create("flow.reorder"))
    graph.connect(source.id, "image", route.id, "value")
    graph.connect(route.id, "case_1", a.id, "image")
    graph.connect(route.id, "case_2", b.id, "image")
    graph.connect(source.id, "image", fifo.id, "expected")
    graph.connect(a.id, "image", fifo.id, "value_1")
    graph.connect(b.id, "image", fifo.id, "value_2")
    for expected in [1, 2, 3]:
        graph.run()
        assert sorted([a.state, b.state]) == ["inactive", "pass"]
        assert fifo.state == "pass", fifo.error
        assert fifo.last_result.payloads["value"]["ticket"]["value"] == expected
        assert fifo.last_result.outputs["queued"] == 0


def test_parallel_split_uses_two_workers_and_isolates_sibling_mutation():
    barrier = threading.Barrier(2)
    threads = set()

    class ParallelTool(Block):
        inputs, outputs = (Port("image"),), (Port("image"),)

        def process(self, inputs):
            threads.add(threading.get_ident())
            barrier.wait(timeout=3)
            inputs["image"][:] = self.parameters.get("fill", 5)
            return Result({"image": inputs["image"]})

    r, graph = builtin_registry(), Graph()
    source = graph.add(r.create("input.demo"))
    split = graph.add(r.create("flow.split"))
    a, b = graph.add(ParallelTool()), graph.add(ParallelTool())
    a.parameters["fill"], b.parameters["fill"] = 11, 22
    join = graph.add(r.create("flow.join"))
    graph.connect(source.id, "image", split.id, "value")
    for index, tool in enumerate((a, b), 1):
        graph.connect(split.id, f"branch_{index}", tool.id, "image")
        graph.connect(tool.id, "image", join.id, f"branch_{index}")
    graph.run()
    assert all(b.state == "pass" for b in graph.blocks.values()), [
        (b.state, b.error) for b in graph.blocks.values()
    ]
    assert len(threads) == 2
    assert np.all(a.last_output_images["image"] == 11) and np.all(
        b.last_output_images["image"] == 22
    )
    assert not np.all(source.last_output_images["image"] == 11)


def test_nested_toolblock_unbatch_delivers_each_serial_entry_and_retains_buffer():
    from opencv_dataflow.toolblocks import encapsulate

    r, graph = builtin_registry(), Graph()
    source = graph.add(r.create("input.demo"))
    batch = graph.add(r.create("flow.batch"))
    unbatch = graph.add(r.create("flow.unbatch"))
    gray = graph.add(r.create("gapi.gray"))
    graph.connect(source.id, "image", batch.id, "item_1")
    graph.connect(batch.id, "group", unbatch.id, "group")
    graph.connect(unbatch.id, "value", gray.id, "image")
    nested = encapsulate(graph, [batch.id, unbatch.id], "Stream group", r)
    for _ in range(2):
        graph.run()
        assert nested.state == "waiting", nested.error
        assert gray.state == "inactive"
    graph.run()
    assert nested.state == gray.state == "pass", nested.error
    assert len(gray.last_result.emissions["image"]) == 3
    assert [p.payload["ticket"]["value"] for p in gray.last_result.emissions["image"]] == [1, 2, 3]


def test_gui_repeated_single_runs_preserve_runtime_and_reset_does_not_dirty(app, tmp_path):
    from test_pattern_editor import run

    from opencv_dataflow.app import MainWindow

    window = MainWindow()
    graph = Graph()
    source = graph.add(window.registry.create("input.demo"))
    batch = graph.add(window.registry.create("flow.batch"))
    graph.connect(source.id, "image", batch.id, "item_1")
    window.replace_graph(graph)
    window.show()
    app.processEvents()
    window.inspector.select(source)
    assert window.inspector.ticket_mode.currentText() == "auto_number"
    for count in (1, 2):
        run(window, app)
        assert batch.state == "waiting" and batch.last_result.outputs["queued"] == count
    run(window, app)
    assert batch.state == "pass" and len(batch.last_result.outputs["group"].items) == 3
    window.inspector.select(batch)
    assert "flow_group_v1" in window.inspector.details.toPlainText()
    window.inspector.select(source)
    window.inspector.ticket_mode.setCurrentText("fixed")
    window.inspector.ticket_value.setText('"PLC-job-7"')
    assert window.inspector.apply()
    project = tmp_path / "source_ticket.json"
    window.graph.save(project)
    loaded = Graph.load(project, window.registry)
    assert loaded.blocks[source.id].ticket_settings["value"] == "PLC-job-7"
    window.dirty = False
    window.reset_flow_state()
    assert batch.state == "idle" and not window.dirty and not batch.runtime["pending"]
    window.close()


def test_join_memory_limit_and_case_range_default_and_cancellation():
    r = builtin_registry()
    block = r.create("flow.join", parameters={"max_bytes_mb": 0.001})
    block.execute({"branch_1": entry(1, np.zeros((100, 100), np.uint8))})
    assert block.state == "error" and not block.runtime["pending"]
    route = r.create(
        "flow.case",
        parameters={"path": "value.area", "comparison": "range", "cases": [[1, 10], [11, 20]]},
    )
    result = execute(route, {"value": entry(1, {"area": 15})})
    assert result.outputs["case_2"] == {"area": 15}
    graph = Graph()
    source = graph.add(r.create("input.demo"))
    graph.run(cancelled=lambda: True)
    assert source.state == "skipped" and source.last_result is None


@pytest.mark.local_examples
def test_shipped_control_flow_example_repeats_drains_and_reports_each_event():
    import json
    from pathlib import Path

    from opencv_dataflow.examples import control_flow_graph
    from opencv_dataflow.runner import describe

    r = builtin_registry()
    graph = Graph.load(
        Path(__file__).parents[1] / "examples/flow tools/control_flow/control_flow.json", r
    )
    assert graph.to_dict() == control_flow_graph(r).to_dict()
    for _ in range(3):
        graph.run()
        assert all(b.state == "pass" for b in graph.blocks.values()), [
            (b.state, b.error) for b in graph.blocks.values()
        ]
        expected = graph.blocks["flow_expected"].last_result.emissions["value"]
        ordered = graph.blocks["flow_fifo"].last_result.emissions["value"]
        assert [p.payload["ticket"] for p in ordered] == [p.payload["ticket"] for p in expected]
        assert len(ordered) == 3 and graph.blocks["flow_fifo"].last_result.outputs["queued"] == 0
        assert len(graph.blocks["flow_histogram"].last_result.emissions["histogram"]) == 3
        receipt = json.dumps(describe(graph), default=lambda v: v.item())
        assert "flow_group_v1" in receipt and "image_shape" in receipt and len(receipt) < 300000
