import copy

import numpy as np
import pytest
from PySide6.QtWidgets import QComboBox

from opencv_dataflow.app import MainWindow
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Block, Graph, Port, Result
from opencv_dataflow.outcomes import failed_input
from opencv_dataflow.payloads import NO_VALUE, ValuePacket
from opencv_dataflow.tickets import TicketClock, ticket_key
from opencv_dataflow.toolblocks import Binding, ToolBlock


def event(number, source="camera", passed=True):
    ticket = TicketClock().issue(source, "fixed", number)
    image = np.full((8, 12), number, np.uint8)
    return ValuePacket(
        image,
        {
            "ticket": ticket,
            "custom": {"camera": source},
            "outcome": {
                "model": "tool_outcome_v1",
                "passed": passed,
                "chain_passed": passed,
                "source": {"id": source},
                "score": 0.8,
                "data": {},
            },
        },
    )


def coordinator(**parameters):
    block = builtin_registry().create("flow.coordinate", parameters=parameters)
    clock = [100.0]
    block.now = lambda: clock[0]
    return block, clock


def run(block, **values):
    block.execute(values)
    assert block.state in ("pass", "fail", "waiting"), block.error
    return block.last_result


def test_ticket_all_matches_across_sources_and_never_uses_the_next_ticket():
    block, clock = coordinator()
    assert run(block, input_1=event(1, "left")).outputs["group"] is NO_VALUE
    run(block, input_2=event(2, "right"))
    assert len(block.runtime["pending"]) == 2
    clock[0] += 0.1
    result = run(block, input_2=event(1, "right", passed=False))
    group = result.outputs["group"]
    assert group.labels == ("input_1", "input_2")
    assert [p.payload["ticket"]["value"] for p in group.items] == [1, 1]
    assert [p.payload["custom"]["camera"] for p in group.items] == ["left", "right"]
    assert [failed_input(p) for p in group.items] == [False, True]
    assert failed_input(result.emissions["group"][0])
    assert result.outputs["info"]["complete"] and not result.outputs["info"]["mixed_tickets"]
    assert result.outputs["info"]["age_ms"]["input_1"] == pytest.approx(100)
    assert len(block.runtime["pending"]) == 1


@pytest.mark.parametrize("policy", ["discard", "emit_partial", "error"])
def test_deadline_anchored_to_first_arrival_and_timeout_policies(policy):
    block, clock = coordinator(inputs=3, max_wait_ms=100, timeout=policy)
    run(block, input_1=event(1))
    clock[0] += 0.06
    run(block, input_2=event(1))
    clock[0] += 0.04
    block.execute({})
    assert not block.runtime["pending"]
    if policy == "error":
        assert block.state == "error" and "timeout" in block.error.lower()
        return
    assert block.runtime["timeouts"] == 1
    if policy == "emit_partial":
        assert block.state == "fail"
        info = block.last_result.outputs["info"]
        assert info["missing"] == ["input_3"] and info["reason"] == "timeout"
        assert failed_input(block.current_emissions["group"][0])
        assert not failed_input(block.last_result.outputs["group"].items[0])
    else:
        assert block.state == "waiting" and not any(block.current_emissions.values())
    run(block, input_3=event(1))
    assert block.runtime["late"] == 1 and not any(block.current_emissions.values())


def test_at_deadline_new_arrival_is_late_and_replacement_does_not_extend_wait():
    block, clock = coordinator(max_wait_ms=100, duplicates="replace")
    run(block, input_1=event(1))
    clock[0] += 0.06
    run(block, input_1=event(1))
    assert block.runtime["replaced"] == 1
    clock[0] += 0.04
    run(block, input_2=event(1))
    assert block.runtime["timeouts"] == 1 and block.runtime["late"] == 1
    assert not any(block.current_emissions.values())


def test_first_arrival_is_one_winner_per_ticket_with_bounded_history():
    block, _ = coordinator(mode="first_arrival", history_size=2)
    result = run(block, input_2=event(1, "right"))
    assert result.outputs["info"]["reason"] == "first_arrival"
    assert result.outputs["info"]["primary"] == "input_2"
    assert block.state == "pass" and not result.outputs["info"]["complete"]
    run(block, input_1=event(1, "left"))
    assert not any(block.current_emissions.values()) and block.runtime["late"] == 1
    for number in (2, 3):
        run(block, input_1=event(number))
    assert len(block.runtime["closed"]) == 2
    assert ticket_key(event(1).payload["ticket"]) not in block.runtime["closed"]


def test_latest_emits_every_new_event_after_warmup_and_marks_mixed_tickets():
    block, clock = coordinator(mode="latest")
    run(block, input_1=event(1, "left"))
    assert not any(block.current_emissions.values())
    run(block, input_2=event(2, "right"))
    first_group_ticket = block.last_result.payloads["group"]["ticket"]
    assert block.last_result.outputs["info"]["mixed_tickets"]
    assert block.last_result.outputs["value"].mean() == 1
    clock[0] += 0.1
    run(block, input_2=event(3, "right"))
    assert block.runtime["emitted"] == 2
    assert [p.payload["ticket"]["value"] for p in block.last_result.outputs["group"].items] == [
        1,
        3,
    ]
    assert block.last_result.payloads["group"]["ticket"] != first_group_ticket


def test_latest_freshness_and_explicit_partial_warmup():
    block, clock = coordinator(mode="latest", max_age_ms=100)
    run(block, input_1=event(1))
    clock[0] += 0.101
    run(block, input_2=event(2))
    assert not any(block.current_emissions.values())
    assert set(block.runtime["latest"]) == {"input_2"}
    graph = Graph()
    graph.add(block)
    graph.bind_ticket_results()
    block.parameters["warmup"] = "any"
    result = run(block, input_2=event(3))
    assert block.state == "fail" and result.outputs["info"]["missing"] == ["input_1"]
    assert failed_input(result.emissions["value"][0])
    assert graph.ticket_results.find(event(3).payload["ticket"])["had_failure"]


def test_zero_wait_and_age_disable_expiry_explicitly():
    block, clock = coordinator(max_wait_ms=0)
    run(block, input_1=event(1))
    clock[0] += 1000
    run(block)
    assert len(block.runtime["pending"]) == 1 and block.runtime["timeouts"] == 0
    block, clock = coordinator(mode="latest", max_age_ms=0)
    run(block, input_1=event(1))
    clock[0] += 1000
    result = run(block, input_2=event(2))
    assert result.outputs["info"]["complete"]


def test_trigger_latest_does_not_replay_a_trigger_when_supporting_data_arrives_late():
    block, _ = coordinator(mode="trigger_latest")
    run(block, input_1=event(1))
    assert block.runtime["not_ready"] == 1
    run(block, input_2=event(2))
    assert not any(block.current_emissions.values())
    run(block, input_1=event(3))
    assert block.last_result.outputs["info"]["trigger"] == "input_1"
    assert block.last_result.outputs["value"].mean() == 3
    run(block, input_2=event(4))
    assert block.runtime["emitted"] == 1 and not any(block.current_emissions.values())


@pytest.mark.parametrize(
    "parameters, arrivals, error",
    [
        ({}, ["input_1", "input_1"], "Duplicate"),
        ({"mode": "first_arrival", "late": "error"}, ["input_1", "input_2"], "Late"),
        ({"max_pending": 1}, ["input_1", "input_1"], "limit"),
    ],
)
def test_duplicate_late_and_capacity_error_reset_runtime(parameters, arrivals, error):
    block, _ = coordinator(**parameters)
    run(block, **{arrivals[0]: event(1)})
    second = 2 if "max_pending" in parameters else 1
    block.execute({arrivals[1]: event(second)})
    assert block.state == "error" and error in block.error
    assert block.runtime["pending"] == {} and block.runtime["latest"] == {}


def test_byte_bounds_missing_tickets_and_invalid_parameters_are_visible_errors():
    for parameters, values in [
        ({"max_bytes_mb": 0.00001}, {"input_1": event(1)}),
        ({}, {"input_1": np.zeros((8, 12), np.uint8)}),
        ({"primary": "input_4"}, {}),
        ({"max_wait_ms": -1}, {}),
        ({"mode": "unknown"}, {}),
    ]:
        block, _ = coordinator(**parameters)
        block.execute(values)
        assert block.state == "error" and block.error


class OptionalSource(Block):
    type_id = "fixture.optional_arrival"
    outputs = (Port("value", "any"),)
    defaults = {"active": True, "fail": False}

    def process(self, inputs):
        if self.parameters["fail"]:
            raise ValueError("Fixture source unavailable")
        value = event(1) if self.parameters["active"] else NO_VALUE
        return Result({"value": value})


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("source_error", [False, True])
def test_scheduler_polls_idle_deadlines_including_nested_and_error_sources(nested, source_error):
    registry = builtin_registry()
    registry.register(OptionalSource)
    graph = Graph()
    source = graph.add(registry.create(OptionalSource.type_id))
    block, clock = coordinator(max_wait_ms=100, timeout="emit_partial")
    if nested:
        inner = Graph()
        inner.add(block)
        target = graph.add(
            ToolBlock(
                inner,
                {"input": Binding(block.id, "input_1")},
                {"group": Binding(block.id, "group")},
            )
        )
        port = "input"
    else:
        target = graph.add(block)
        port = "input_1"
    graph.connect(source.id, "value", target.id, port)
    graph.run()
    source.parameters.update(active=False, fail=source_error)
    clock[0] += 0.101
    graph.run()
    assert block.runtime["timeouts"] == 1
    assert len(target.current_emissions["group"]) == 1
    row = graph.ticket_results.find(event(1).payload["ticket"])
    assert row["had_failure"] and any(e["kind"] == "coordination_timeout" for e in row["events"])
    assert not row["judgments"]


def test_runtime_copy_keeps_deadlines_but_project_and_reset_do_not_store_buffer():
    registry = builtin_registry()
    graph = Graph()
    block = graph.add(registry.create("flow.coordinate", parameters={"max_wait_ms": 100}))
    graph.bind_ticket_results()
    block.now = lambda: 10.0
    run(block, input_1=event(1))
    clone = Graph.from_dict(graph.to_dict(), registry)
    clone.copy_runtime_from(graph)
    copied = clone.blocks[block.id]
    copied.now = lambda: 10.101
    run(copied)
    assert copied.runtime["timeouts"] == 1
    assert len(block.runtime["pending"]) == 1
    assert "pending" not in graph.to_dict()["blocks"][0]
    assert not Graph.from_dict(graph.to_dict(), registry).blocks[block.id].runtime["pending"]
    graph.invalidate()
    assert not block.runtime["pending"] and not block.runtime["closed"]


def test_failure_policy_rejects_before_buffering_and_preserves_metadata():
    block, _ = coordinator()
    block.input_failure_policies = {"input_1": "reject_failed"}
    block.execute({"input_1": event(1, passed=False)})
    assert block.input_rejected and block.runtime["received"] == 0
    run(block, input_1=event(1))
    assert block.runtime["received"] == 1
    run(block, input_2=event(1))
    assert block.last_result.outputs["value"].metadata["custom"] == {"camera": "camera"}


def test_counter_is_exact_and_persists_through_runtime_copy():
    registry = builtin_registry()
    block = registry.create("input.counter", parameters={"start": 2**53 + 1, "step": 2})
    block.execute({})
    assert block.last_result.outputs["value"] == 2**53 + 1
    clone = registry.create("input.counter", parameters=block.parameters)
    clone.copy_runtime_from(block)
    clone.execute({})
    assert clone.last_result.outputs["value"] == 2**53 + 3
    clone.reset_runtime()
    clone.execute({})
    assert clone.last_result.outputs["value"] == 2**53 + 1


def test_inspector_enum_controls_and_policy_roundtrip(app):
    window = MainWindow()
    block = window.graph.add(window.registry.create("flow.coordinate"))
    window.inspector.select(block)
    for name in ("mode", "timeout", "warmup", "primary", "duplicates", "late"):
        assert isinstance(window.inspector.fields[name], QComboBox)
    window.inspector.fields["mode"].setCurrentText("first_arrival")
    assert window.inspector.apply()
    restored = Graph.from_dict(copy.deepcopy(window.graph.to_dict()), window.registry)
    assert restored.blocks[block.id].parameters["mode"] == "first_arrival"
    window.dirty = False
    window.close()


def test_saved_example_receipt_and_gui_repeated_runs(app, load_example, tmp_path):
    import json

    from PySide6.QtTest import QTest

    from opencv_dataflow.examples import input_coordination_graph
    from opencv_dataflow.runner import main

    registry = builtin_registry()
    graph = Graph.load("examples/flow tools/input_coordination/input_coordination.json", registry)
    assert graph.to_dict() == input_coordination_graph(registry).to_dict()
    report = tmp_path / "receipt.json"
    assert (
        main(
            [
                "examples/flow tools/input_coordination/input_coordination.json",
                "--repeat",
                "3",
                "--report",
                str(report),
            ]
        )
        == 0
    )
    receipt = json.loads(report.read_text(encoding="utf-8"))
    for row in receipt["ticket_results"]["tickets"]:
        if row["ticket"]["session"] == "external":
            assert row["status"] == "final_pass"
            assert len(row["sources"]) == 2
            assert row["events"][-1]["coordination"]["complete"]
    window = MainWindow()
    load_example(window, "input_coordination")
    for expected in (1, 2):
        window.start_run(False)
        for _ in range(800):
            app.processEvents()
            if window.runner is None:
                break
            QTest.qWait(10)
        assert window.runner is None
        block = window.graph.blocks["coord_inputs"]
        assert block.last_result.outputs["info"]["complete"]
        assert block.last_result.payloads["group"]["ticket"]["value"] == expected
        assert block.runtime["emitted"] == expected
    block.parameters["mode"] = "first_arrival"
    window.start_run(False)
    for _ in range(800):
        app.processEvents()
        if window.runner is None:
            break
        QTest.qWait(10)
    assert window.runner is None
    window.show_ticket_results()
    dialog = window.ticket_results_window
    dialog.search.setText("external")
    assert any(
        dialog.events.topLevelItem(i).text(2) == "Ignored late input"
        for i in range(dialog.events.topLevelItemCount())
    )
    dialog.close()
    window.dirty = False
    window.close()
