import copy
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox

from opencv_dataflow.app import MainWindow
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Block, Graph, Port, Result
from opencv_dataflow.outcomes import TicketResults, failed_input
from opencv_dataflow.payloads import NO_VALUE, ValuePacket
from opencv_dataflow.runner import describe
from opencv_dataflow.tickets import TicketClock, ticket_key
from opencv_dataflow.toolblocks import Binding, ToolBlock


class ImageSource(Block):
    type_id = "input.outcome_fixture"
    outputs = (Port("image"),)
    defaults = {"passed": True, "pixel": 10}

    def process(self, inputs):
        return Result(
            {"image": np.full((12, 16, 3), self.parameters["pixel"], np.uint8)},
            passed=self.parameters["passed"],
            score=0.75,
            event_data={"measurement": 12.5, "arbitrary": ["part", 4]},
        )


def registry_with_source():
    registry = builtin_registry()
    registry.register(ImageSource)
    return registry


def ticketed(value, number, passed=True, clock=None):
    ticket = (clock or TicketClock("test")).issue("camera", "fixed", number)
    return ValuePacket(
        value,
        {
            "ticket": ticket,
            "outcome": {
                "model": "tool_outcome_v1",
                "passed": passed,
                "chain_passed": passed,
                "source": {"id": "upstream"},
                "score": 0.9,
                "data": {},
            },
        },
    )


def test_failures_propagate_without_erasure_and_final_judgments_are_explicit():
    registry = registry_with_source()
    graph = Graph()
    source = graph.add(registry.create(ImageSource.type_id, parameters={"passed": False}))
    gray = graph.add(registry.create("cv.gray"))
    judgment = graph.add(
        registry.create("inspect.final_judgment", parameters={"data": {"station": "A"}})
    )
    graph.connect(source.id, "image", gray.id, "image")
    graph.connect(gray.id, "image", judgment.id, "value")
    graph.run()
    payload = gray.last_result.payloads["image"]
    assert payload["outcome"]["passed"] is True
    assert payload["outcome"]["chain_passed"] is False
    assert payload["outcome"]["source"]["id"] == gray.id
    row = graph.ticket_results.find(payload["ticket"])
    assert row["status"] == "final_fail" and row["judgments"] == [False]
    assert [event["passed"] for event in row["events"]] == [False, True, False]
    assert row["events"][0]["data"]["measurement"] == 12.5
    assert row["events"][-1]["kind"] == "final_judgment"
    assert row["events"][-1]["data"] == {"station": "A"}
    judgment.parameters["mode"] = "pass"
    source.ticket_settings = {"mode": "fixed", "value": payload["ticket"]["value"]}
    # Same identity, different session: auto and external tickets do not collide.
    graph.run()
    row2 = graph.ticket_results.find(judgment.last_result.payloads["value"]["ticket"])
    assert row2["status"] == "final_pass" and row2["had_failure"]
    assert row2["ticket"]["session"] != row["ticket"]["session"]


def test_rejection_retains_last_processed_history_and_emits_nothing():
    registry = registry_with_source()
    graph = Graph()
    source = graph.add(registry.create(ImageSource.type_id))
    gate = graph.add(registry.create("cv.gray"))
    gate.input_failure_policies = {"image": "reject_failed"}
    sink = graph.add(registry.create("cv.gaussian_blur"))
    graph.connect(source.id, "image", gate.id, "image")
    graph.connect(gate.id, "image", sink.id, "image")
    graph.run()
    old_inputs, old_output = copy.deepcopy(gate.last_inputs), gate.last_result
    old_ticket = gate.input_payloads["image"]["ticket"]
    source.parameters.update(passed=False, pixel=200)
    graph.run()
    assert gate.state == "inactive" and gate.input_rejected
    np.testing.assert_array_equal(gate.last_inputs["image"], old_inputs["image"])
    assert gate.last_result is old_output
    assert gate.input_payloads["image"]["ticket"] == old_ticket
    assert gate.current_emissions == {}
    assert sink.state == "inactive" and sink.last_result is None
    assert describe(graph)[1]["emitted"] == {}
    new_ticket = source.last_result.outputs["ticket"]
    row = graph.ticket_results.find(new_ticket)
    assert [e["kind"] for e in row["events"]] == ["tool_result", "input_rejected"]
    assert row["status"] == "intermediate_fail"
    source.parameters.update(passed=True, pixel=90)
    graph.run()
    assert gate.last_inputs["image"].mean() == 90
    assert gate.current_emissions["image"] and sink.state == "pass"


class Burst(Block):
    type_id = "fixture.burst"
    outputs = (Port("image"),)
    outcome_passthrough = True
    per_packet_outcomes = True
    defaults = {"order": [1, 2, 3]}

    def process(self, inputs):
        packets = [
            ticketed(np.full((8, 10), n, np.uint8), n, n != 2) for n in self.parameters["order"]
        ]
        return Result({"image": packets[-1].value}, emissions={"image": packets})


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("order", [[1, 2, 3], [1, 3, 2]])
def test_bursts_and_batched_tickets_reject_individually(batched, order):
    registry = builtin_registry()
    registry.register(Burst)
    graph = Graph()
    source = graph.add(registry.create(Burst.type_id, parameters={"order": order}))
    previous, output = source, "image"
    if batched:
        batch = graph.add(registry.create("flow.batch", parameters={"size": 3}))
        unbatch = graph.add(registry.create("flow.unbatch"))
        graph.connect(source.id, "image", batch.id, "item_1")
        graph.connect(batch.id, "group", unbatch.id, "group")
        previous, output = unbatch, "value"
    gate = graph.add(registry.create("cv.gray"))
    gate.input_failure_policies = {"image": "reject_failed"}
    graph.connect(previous.id, output, gate.id, "image")
    graph.run()
    assert gate.state == "pass", gate.error
    assert [p.payload["ticket"]["value"] for p in gate.current_emissions["image"]] == [1, 3]
    assert gate.last_inputs["image"].mean() == 3
    rows = graph.ticket_results.snapshot()["tickets"]
    rejected = [e for row in rows for e in row["events"] if e["kind"] == "input_rejected"]
    assert len(rejected) == 1 and rejected[0]["ticket"]["value"] == 2


def test_nested_toolblock_does_not_reemit_a_retained_child_result():
    registry = registry_with_source()
    inner = Graph()
    gate = inner.add(registry.create("cv.gray", block_id="inner_gate"))
    gate.input_failure_policies = {"image": "reject_failed"}
    tool = ToolBlock(
        inner, {"image": Binding(gate.id, "image")}, {"image": Binding(gate.id, "image")}
    )
    outer = Graph()
    source = outer.add(registry.create(ImageSource.type_id))
    outer.add(tool)
    outer.connect(source.id, "image", tool.id, "image")
    outer.run()
    previous = gate.last_result
    previous.score = 0.4
    gate.last_graphics["retained diagnostic"] = gate.last_inputs["image"].copy()
    source.parameters["passed"] = False
    outer.run()
    assert gate.last_result is previous and gate.last_inputs["image"].mean() == 10
    assert gate.current_emissions == {}
    assert tool.current_emissions["image"] == []
    assert tool.last_result.outputs["image"] is NO_VALUE
    assert tool.last_result.score is None and tool.last_result.graphics == {}
    row = outer.ticket_results.find(source.last_result.outputs["ticket"])
    event = next(e for e in row["events"] if e["kind"] == "input_rejected")
    assert event["source"]["path"] == f"{tool.id}/{gate.id}"


def event(ticket, final=None, kind="tool_result", passed=True):
    return {
        "ticket": ticket,
        "source": {"id": "tool"},
        "kind": kind,
        "passed": passed,
        "chain_passed": passed,
        "score": 0.8,
        "message": "test",
        "data": {"anything": [1, "a"]},
        "final_judgment": final,
        "parent_ticket": None,
    }


def test_threadsafe_consumer_bounds_conflicts_and_identity_types():
    tracker = TicketResults(max_tickets=2, events_per_ticket=3)
    clock = TicketClock("shared")
    ticket = clock.issue("camera", "fixed", "part-1")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: tracker.consume(event(ticket)), range(20)))
    tracker.consume(event(ticket, final=True))
    tracker.consume(event(ticket, final=False, passed=False))
    row = tracker.find(ticket)
    assert row["event_count"] == 22 and row["discarded_events"] == 19
    assert len(row["events"]) == 3 and row["status"] == "conflicting_final"
    tracker.consume(event(ticket))
    assert tracker.find(ticket)["status"] == "conflicting_final"
    for value in (1, "1"):
        tracker.consume(event(clock.issue("camera", "fixed", value)))
    assert tracker.find(ticket) is None
    snapshot = tracker.snapshot()
    assert snapshot["evicted_tickets"] == 1
    assert {row["ticket"]["kind"] for row in snapshot["tickets"]} == {"number", "string"}
    restored = TicketResults()
    restored.restore(snapshot)
    assert restored.snapshot() == snapshot
    tiny = TicketResults(max_bytes=100)
    tiny.consume(event(ticket))
    assert tiny.snapshot()["tickets"] == [] and tiny.evicted_tickets == 1


@pytest.mark.parametrize(
    "mode, verdict", [("pass", True), ("fail", False), ("boolean_input", False)]
)
def test_explicit_final_modes(mode, verdict):
    block = builtin_registry().create("inspect.final_judgment", parameters={"mode": mode})
    block.execute(
        {"value": ticketed(np.zeros((8, 10), np.uint8), "part", False), "passed": verdict}
    )
    assert block.state == ("pass" if verdict else "fail"), block.error
    assert block.last_result.final_judgment is verdict
    assert failed_input(block.current_emissions["value"][0])


def test_errors_publish_for_the_consumed_ticket_and_do_not_finalize():
    registry = registry_with_source()
    graph = Graph()
    source = graph.add(registry.create(ImageSource.type_id))
    blur = graph.add(registry.create("cv.gaussian_blur", parameters={"kernel": 2}))
    graph.connect(source.id, "image", blur.id, "image")
    graph.run()
    row = graph.ticket_results.find(source.last_result.outputs["ticket"])
    assert row["status"] == "error" and row["judgments"] == []
    assert row["events"][-1]["kind"] == "error"
    assert row["events"][-1]["source"]["id"] == blur.id


def test_policy_roundtrip_rejects_bad_configuration_and_runtime_is_not_saved():
    registry = builtin_registry()
    graph = Graph()
    block = graph.add(registry.create("cv.gray"))
    block.input_failure_policies = {"image": "reject_failed"}
    restored = Graph.from_dict(graph.to_dict(), registry)
    assert restored.blocks[block.id].input_failure_policies == block.input_failure_policies
    assert "ticket_results" not in graph.to_dict()
    for policy in ({"missing": "continue"}, {"image": "maybe"}, []):
        data = copy.deepcopy(graph.to_dict())
        data["blocks"][0]["input_failure_policies"] = policy
        with pytest.raises(ValueError, match="failure policy"):
            Graph.from_dict(data, registry)


def pump(app, predicate):
    from PySide6.QtTest import QTest

    for _ in range(800):
        app.processEvents()
        if predicate():
            return
        QTest.qWait(10)
    raise AssertionError("Qt worker did not finish")


def test_gui_policy_lookup_and_retention_across_run_once(app):
    window = MainWindow()
    registry = window.registry
    registry.register(ImageSource)
    graph = Graph()
    source = graph.add(registry.create(ImageSource.type_id))
    gate = graph.add(registry.create("cv.gray"))
    graph.connect(source.id, "image", gate.id, "image")
    window.replace_graph(graph)
    window.inspector.select(gate)
    policy = window.inspector.failure_fields["image"]
    assert isinstance(policy, QComboBox)
    policy.setCurrentText("reject_failed")
    assert window.inspector.apply()
    window.start_run(False)
    pump(app, lambda: window.runner is None)
    source.parameters.update(passed=False, pixel=220)
    window.start_run(False)
    pump(app, lambda: window.runner is None)
    assert gate.last_inputs["image"].mean() == 10 and gate.input_rejected
    window.show_ticket_results()
    dialog = window.ticket_results_window
    dialog.refresh()
    assert dialog.table.topLevelItemCount() == 2
    dialog.search.setText("intermediate_fail")
    assert dialog.table.topLevelItemCount() == 1
    assert dialog.events.topLevelItemCount() == 2
    assert dialog.events.topLevelItem(1).text(2) == "Rejected"
    assert "input_rejected" in dialog.details.toPlainText()
    key = dialog.table.currentItem().data(0, Qt.ItemDataRole.UserRole)
    assert ticket_key(json.loads(dialog.details.toPlainText())["ticket"])
    dialog.refresh()
    assert dialog.table.currentItem().data(0, Qt.ItemDataRole.UserRole) == key
    dialog.close()
    window.dirty = False
    window.close()


@pytest.mark.parametrize("placement", ["direct", "leaf", "public"])
def test_queue_failure_gates_do_not_replace_history_or_emit_metrics(placement):
    registry = registry_with_source()
    graph = Graph()
    source = graph.add(registry.create(ImageSource.type_id))
    queue = registry.create("flow.image_queue")
    gate = queue
    if placement != "direct":
        inner = Graph()
        inner.add(queue)
        gate = ToolBlock(
            inner, {"image": Binding(queue.id, "image")}, {"image": Binding(queue.id, "image")}
        )
    (gate if placement == "public" else queue).input_failure_policies = {"image": "reject_failed"}
    graph.add(gate)
    graph.connect(source.id, "image", gate.id, "image")
    graph.run()
    previous = queue.last_inputs["image"].copy()
    source.parameters.update(passed=False, pixel=180)
    graph.run()
    assert gate.state in ("inactive", "waiting"), gate.error
    assert not any(gate.current_emissions.values())
    np.testing.assert_array_equal(queue.last_inputs["image"], previous)
    assert queue.stats()["enqueued"] == 1
    row = graph.ticket_results.find(source.last_result.outputs["ticket"])
    assert len([e for e in row["events"] if e["kind"] == "input_rejected"]) == 1


def test_background_capture_has_one_source_event_per_ticket_and_rejects_failed_frames():
    from opencv_dataflow.queued_execution import QueuedExecution

    class Alternating(ImageSource):
        type_id = "input.alternating"

        def process(self, inputs):
            self.counter = getattr(self, "counter", 0) + 1
            self.parameters.update(pixel=self.counter, passed=self.counter % 2 == 1)
            return super().process(inputs)

    registry = builtin_registry()
    registry.register(Alternating)
    graph = Graph()
    source = graph.add(registry.create(Alternating.type_id))
    queue = graph.add(
        registry.create(
            "flow.image_queue", parameters={"mode": "background_acquisition", "capacity": 2}
        )
    )
    queue.input_failure_policies = {"image": "reject_failed"}
    gray = graph.add(registry.create("cv.gray"))
    graph.connect(source.id, "image", queue.id, "image")
    graph.connect(queue.id, "image", gray.id, "image")
    tickets = []
    with QueuedExecution(graph, registry) as execution:
        for expected in (1, 3):
            assert execution.run_one()
            assert gray.last_inputs["image"].mean() == expected
            tickets.append(gray.input_payloads["image"]["ticket"])
    for ticket in tickets:
        row = graph.ticket_results.find(ticket)
        assert sum(e["source"]["id"] == source.id for e in row["events"]) == 1
        assert [e["kind"] for e in row["events"]] == ["tool_result"] * 3
    rejected = [
        row for row in graph.ticket_results.snapshot()["tickets"] if row["ticket"]["value"] == 2
    ][0]
    assert [e["kind"] for e in rejected["events"]] == ["tool_result", "input_rejected"]


def test_parallel_part_outcomes_remain_individual_in_the_common_consumer():
    from opencv_dataflow.parallel_inspection import ParallelInspect
    from opencv_dataflow.poses import local_offset, make_pose

    class PartCheck(Block):
        type_id = "fixture.part_check"
        inputs = (Port("image"), Port("pose", "data"))
        outputs = (Port("overlay"),)

        def process(self, inputs):
            index = self.input_payloads["image"]["inspection"]["index"]
            return Result({"overlay": inputs["image"]}, passed=index % 2 == 0, score=0.5)

    registry = builtin_registry()
    registry.register(PartCheck)
    inner = Graph()
    child = inner.add(registry.create(PartCheck.type_id))
    tool = ToolBlock(
        inner,
        {"image": Binding(child.id, "image"), "pose": Binding(child.id, "pose")},
        {"overlay": Binding(child.id, "overlay")},
    )
    inspect = ParallelInspect(inspection=tool, registry=registry)
    graph = Graph()
    graph.add(inspect)
    image = ticketed(np.zeros((12, 16, 3), np.uint8), "parent")
    poses = [make_pose(local_offset(4, 4, 0), [16, 12], {}) for _ in range(4)]
    graph.run(injected={(inspect.id, "image"): image, (inspect.id, "features"): poses})
    assert inspect.state == "fail", inspect.error
    for port in ("group", "images"):
        group = inspect.last_result.outputs[port]
        assert [failed_input(p) for p in group.items] == [False, True, False, True]
        for index, packet in enumerate(group.items):
            row = graph.ticket_results.find(packet.payload["ticket"])
            assert row["status"] == ("in_progress" if index % 2 == 0 else "intermediate_fail")
            assert any(e["parent_ticket"] == image.payload["ticket"] for e in row["events"])
            assert any(f"part-{index}" in e["source"]["path"] for e in row["events"])
    inspect.parameters.update(workers=1, stop_after_passes=1)
    graph.run(injected={(inspect.id, "image"): image, (inspect.id, "features"): poses})
    group_packet = inspect.current_emissions["group"][0]
    assert inspect.state == "fail" and len(group_packet.value.items) == 1
    assert failed_input(group_packet)  # Incomplete all-pass aggregate failed.
    assert not failed_input(group_packet.value.items[0])  # The inspected part passed.


def test_toolblock_public_policy_survives_export_and_terminal_rename(tmp_path):
    registry = builtin_registry()
    inner = Graph()
    child = inner.add(registry.create("cv.gray"))
    tool = ToolBlock(
        inner, {"input": Binding(child.id, "image")}, {"output": Binding(child.id, "image")}
    )
    tool.input_failure_policies = {"input": "reject_failed"}
    path = tmp_path / "tool.json"
    tool.export(path)
    restored = ToolBlock.read(path, registry)
    assert restored.input_failure_policies == tool.input_failure_policies
    parent = Graph()
    parent.add(restored)
    restored.update_interface(
        {"renamed": Binding(child.id, "image")}, restored.output_bindings, parent
    )
    assert restored.input_failure_policies == {"renamed": "reject_failed"}


def test_saved_example_final_status_and_headless_receipt(tmp_path, app):
    from opencv_dataflow.examples import ticket_results_graph
    from opencv_dataflow.runner import main

    registry = builtin_registry()
    graph = Graph.load("examples/ticket_results.json", registry)
    assert graph.to_dict() == ticket_results_graph(registry).to_dict()
    graph.run()
    gate = graph.blocks["ticket_accepted_preview"]
    previous = gate.last_result
    graph.blocks["ticket_inspect"].parameters["expected_count"] = 3
    graph.run()
    assert graph.ticket_results.snapshot()["tickets"][-1]["status"] == "final_fail"
    assert gate.last_result is previous and gate.input_rejected and gate.current_emissions == {}
    report = tmp_path / "report.json"
    assert main(["examples/ticket_results.json", "--repeat", "3", "--report", str(report)]) == 0
    receipt = json.loads(report.read_text(encoding="utf-8"))
    assert len(receipt["ticket_results"]["tickets"]) == 3
    assert all(row["status"] == "final_pass" for row in receipt["ticket_results"]["tickets"])
    window = MainWindow()
    window.load_ticket_results_example()
    assert "ticket_final" in window.graph.blocks
    window.dirty = False
    window.close()


def test_boolean_verdict_finalizes_the_value_ticket_only():
    registry = builtin_registry()
    graph = Graph()
    final = graph.add(
        registry.create("inspect.final_judgment", parameters={"mode": "boolean_input"})
    )
    value = ticketed(np.zeros((8, 10), np.uint8), "image")
    verdict = ticketed(True, "separate-controller-value")
    graph.run(injected={(final.id, "value"): value, (final.id, "passed"): verdict})
    assert graph.ticket_results.find(value.payload["ticket"])["status"] == "final_pass"
    assert graph.ticket_results.find(verdict.payload["ticket"]) is None
