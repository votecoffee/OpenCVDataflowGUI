"""Timing boundaries, histories and end-to-end ticket propagation."""

import copy
import threading
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from opencv_dataflow import timing
from opencv_dataflow.app import MainWindow
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Block, Graph, Port, Result
from opencv_dataflow.examples import parallel_parts_graph, sequence_graph
from opencv_dataflow.outcomes import TicketResults
from opencv_dataflow.payloads import ValuePacket, validate_payload
from opencv_dataflow.queued_execution import QueuedExecution
from opencv_dataflow.runner import describe
from opencv_dataflow.ticket_window import TicketResultsWindow
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.timing import WorkTiming, advance_timing, merge_timing


@pytest.fixture
def clock(monkeypatch):
    clock = {"ns": 0}

    def mark():
        return {
            "ns": clock["ns"],
            "clock": "test-process",
            "utc": (
                datetime(2026, 1, 1, tzinfo=timezone.utc)
                + timedelta(microseconds=clock["ns"] / 1000)
            ).isoformat(),
        }

    monkeypatch.setattr(timing, "mark", mark)
    monkeypatch.setattr(timing.time, "perf_counter_ns", lambda: clock["ns"])
    clock["mark"] = mark
    return clock


class TimedImage(Block):
    type_id = "custom.timed_image"
    inputs = (Port("image"),)
    outputs = (Port("image"),)

    def process(self, inputs):
        self.clock["ns"] += self.delay_ns
        self.observed = self.timing.snapshot()
        if getattr(self, "raise_error", False):
            raise ValueError("Synthetic error")
        return Result({"image": inputs["image"]})


def tool(clock, delay=2_000_000):
    block = TimedImage()
    block.clock, block.delay_ns = clock, delay
    return block


def image():
    return np.zeros((12, 16), np.uint8)


def test_current_last_average_and_clear_results(clock):
    block = tool(clock)
    assert block.timing.snapshot()["last_ms"] is None
    block.execute({"image": image()})
    assert block.observed["running"] and block.observed["current_ms"] == 2
    assert block.timing.last_ms == 2 and block.elapsed_ms == 2
    block.delay_ns = 4_000_000
    block.execute({"image": image()})
    assert block.timing.snapshot() == {
        "running": False,
        "current_ms": None,
        "last_ms": 4,
        "average_ms": 3,
        "count": 2,
    }
    block.clear_result()
    assert block.timing.count == 2 and block.elapsed_ms == 4


def test_error_counts_but_rejection_does_not(clock):
    block = tool(clock)
    block.execute({"image": image()})
    block.raise_error = True
    block.execute({"image": image()})
    assert block.state == "error" and block.timing.count == 2
    assert block.timing.last_ms == 2
    block.input_failure_policies = {"image": "reject_failed"}
    failed = ValuePacket(
        image(),
        {
            "outcome": {
                "model": "tool_outcome_v1",
                "passed": False,
                "chain_passed": False,
                "source": {"id": "source"},
                "score": None,
                "data": {},
            }
        },
    )
    block.execute({"image": failed})
    assert block.input_rejected and block.timing.count == 2


def test_source_start_and_downstream_finish_include_waits(clock):
    source = builtin_registry().create("input.demo")
    source.execute({})
    original = source.current_emissions["image"][0]
    first = original.payload["processing_time"]
    assert first["started_ns"] == 0 and not first["inferred_start"]
    clock["ns"] += 10_000_000  # acquisition/queue/scheduling wait
    block = tool(clock)
    block.execute({"image": original})
    out = block.current_emissions["image"][0]
    measured = out.payload["processing_time"]
    assert measured["started_at"] == first["started_at"]
    assert measured["elapsed_ms"] == 12 and block.timing.last_ms == 2
    assert measured["finished_ns"] == 12_000_000
    assert out.value.metadata["processing_time"] == measured
    assert original.payload["processing_time"] == first  # no sibling mutation
    validate_payload(out.payload)


def test_error_event_has_current_work_and_total_ticket_time(clock):
    consumer = TicketResults()
    source = builtin_registry().create("input.demo")
    source._ticket_results = consumer
    source.execute({})
    packet = source.current_emissions["image"][0]
    clock["ns"] += 5_000_000
    block = tool(clock)
    block._ticket_results, block.raise_error = consumer, True
    block.execute({"image": packet})
    row = consumer.find(packet.payload["ticket"])
    event = row["events"][-1]
    assert event["kind"] == "error" and event["work_ms"] == 2
    assert row["processing_time"]["elapsed_ms"] == 7
    assert block.timing.count == 1


def test_missing_timing_in_legacy_ticket_gets_inferred_start(clock):
    block = tool(clock)
    ticket = TicketClock().issue("legacy")
    block.execute({"image": ValuePacket(image(), {"ticket": ticket})})
    stamp = block.last_result.payloads["image"]["processing_time"]
    assert stamp["inferred_start"] and stamp["elapsed_ms"] == 2


def test_statistics_survive_clone_reset_and_are_not_saved(tmp_path):
    registry, graph = builtin_registry(), Graph()
    source = graph.add(registry.create("input.demo"))
    graph.run()
    clone = Graph.from_dict(graph.to_dict(), registry)
    clone.copy_runtime_from(graph)
    clone.run()
    assert clone.blocks[source.id].timing.count == 2 and source.timing.count == 1
    graph.copy_runtime_from(clone)
    graph.invalidate()
    assert source.timing.count == 2 and source.elapsed_ms > 0
    path = tmp_path / "project.json"
    graph.save(path)
    assert Graph.load(path, registry).blocks[source.id].timing.count == 0
    assert describe(graph)[0]["timing"]["count"] == 2


def test_parallel_inspection_child_statistics_accumulate_all_parts():
    graph = parallel_parts_graph(builtin_registry())
    block = next(b for b in graph.blocks.values() if b.type_id == "flow.inspect_each")
    for count in (25, 50):
        graph.run()
        assert not block.error, block.error
        assert block.inspection.timing.count == count
        assert all(b.timing.count == count for b in block.graph.blocks.values())
    assert block.timing.count == 2
    assert block.inspection.timing.average_ms > 0


def test_acquisition_totals_keep_all_captures_and_survive_restart():
    registry, graph = builtin_registry(), Graph()
    source = graph.add(registry.create("input.demo"))
    queue = graph.add(
        registry.create(
            "flow.image_queue",
            parameters={
                "mode": "background_acquisition",
                "capacity": 3,
            },
        )
    )
    gray = graph.add(registry.create("cv.gray"))
    graph.connect(source.id, "image", queue.id, "image")
    graph.connect(queue.id, "image", gray.id, "image")
    previous = 0
    for _ in range(2):
        queued = QueuedExecution(graph, registry)
        try:
            queued.start()
            assert queued.run_one()
        finally:
            queued.close()
        count = queued.producer.blocks[source.id].timing.count
        assert source.timing.count == count and count > previous
        previous = count
    assert gray.timing.count == 2
    assert gray.last_result.payloads["image"]["processing_time"]["elapsed_ms"] >= 0
    graph.close()


def event(ticket, stamp, final=None):
    return {
        "ticket": ticket,
        "source": {"id": "tool"},
        "kind": "tool_result",
        "passed": True,
        "chain_passed": True,
        "score": None,
        "data": {},
        "message": "done",
        "final_judgment": final,
        "processing_time": stamp,
    }


def test_ticket_history_uses_latest_finish_even_when_events_arrive_out_of_order(clock):
    start = clock["mark"]()
    clock["ns"] = 20_000_000
    latest = advance_timing(None, start, clock["mark"](), source=True)
    clock["ns"] = 10_000_000
    earlier = advance_timing(None, start, clock["mark"](), source=True)
    consumer, ticket = TicketResults(events_per_ticket=1), TicketClock().issue("camera")
    consumer.consume(event(ticket, latest, final=True))
    consumer.consume(event(ticket, earlier))
    row = consumer.find(ticket)
    assert row["processing_time"]["elapsed_ms"] == 20
    assert row["finalized_at"] == latest["finished_at"]
    assert len(row["events"]) == 1
    restored = TicketResults()
    restored.restore(consumer.snapshot())
    assert restored.find(ticket)["processing_time"] == latest
    assert merge_timing(earlier, latest) == merge_timing(latest, earlier)


def test_foreign_clock_never_subtracts_unrelated_monotonic_readings(clock):
    start = clock["mark"]()
    seed = advance_timing(None, start, start, source=True)
    finish = {**start, "clock": "another-process", "ns": 50_000_000}
    result = advance_timing(seed, start, finish)
    assert result["elapsed_ms"] is None and result["started_at"] == seed["started_at"]
    validate_payload({"processing_time": result})
    result["elapsed_ms"] = 50
    with pytest.raises(ValueError, match="different processes"):
        validate_payload({"processing_time": result})


def test_utc_clock_adjustment_does_not_change_duration(clock):
    start = clock["mark"]()
    clock["ns"] = 10_000_000
    finish = {**clock["mark"](), "utc": "2025-01-01T00:00:00+00:00"}
    result = advance_timing(None, start, finish, source=True)
    assert result["elapsed_ms"] == 10
    validate_payload({"processing_time": result})


def test_editor_worker_handoff_and_ticket_window_columns(app):
    from test_editor import pump_until

    window = MainWindow()
    window.replace_graph(sequence_graph(window.registry))
    source = window.graph.blocks["sequence_source"]
    window.inspector.select(source)
    for count in (1, 2):
        window.start_run(False)
        pump_until(app, lambda: window.runner is None)
        assert source.timing.count == count
    assert "Average" in window.inspector.details.toPlainText()
    consumer = window.root_graph.ticket_results
    results = TicketResultsWindow(lambda: consumer)
    assert results.table.topLevelItem(0).text(7)
    assert results.table.topLevelItem(0).text(8)
    assert results.table.topLevelItem(0).text(9)
    assert results.events.topLevelItem(0).text(6)
    results.close()
    window.dirty = False
    window.close()


def test_weighted_parallel_statistics_merge():
    first = WorkTiming(count=2, last_ms=4, average_ms=3)
    second = WorkTiming(count=3, last_ms=6, average_ms=6)
    copied = copy.deepcopy(first)
    first.absorb(second)
    assert first.count == 5 and first.average_ms == pytest.approx(4.8)
    assert first.last_ms == 6 and copied.count == 2


@pytest.mark.parametrize(
    "field, value",
    [
        ("started_at", "2026-01-01T00:00:00"),
        ("elapsed_ms", -1),
        ("elapsed_ms", 100),
        ("started_ns", True),
    ],
)
def test_invalid_processing_timestamps_are_rejected(clock, field, value):
    point = clock["mark"]()
    record = advance_timing(None, point, point, source=True)
    record[field] = value
    with pytest.raises(ValueError):
        validate_payload({"processing_time": record})


def test_pyramid_levels_keep_timing_provenance(clock):
    registry = builtin_registry()
    source = registry.create("input.demo")
    source.execute({})
    pyramid = registry.create("cv.image_pyramid", parameters={"levels": 3})
    clock["ns"] = 10_000_000
    pyramid.execute({"image": source.current_emissions["image"][0]})
    assert not pyramid.error, pyramid.error
    for level in pyramid.last_result.outputs["images"].levels:
        stamp = level.metadata["processing_time"]
        assert stamp["started_ns"] == 0 and stamp["elapsed_ms"] == 10


def test_inspector_shows_current_time_from_live_worker(app):
    from test_editor import pump_until

    entered, release = threading.Event(), threading.Event()

    class SlowSource(Block):
        type_id = "input.timing_fixture"
        outputs = (Port("image"),)

        def process(self, inputs):
            entered.set()
            assert release.wait(5)
            return Result({"image": image()})

    window = MainWindow()
    window.registry.register(SlowSource)
    graph = Graph()
    source = graph.add(window.registry.create(SlowSource.type_id))
    window.replace_graph(graph)
    window.inspector.select(source)
    try:
        window.start_run(False)
        pump_until(app, entered.is_set)
        window.inspector.refresh_timing()
        label = window.inspector.timing_label.text()
        assert "Current: —" not in label and "Current:" in label
        assert source.timing.count == 0  # display history remains a private snapshot
    finally:
        release.set()
        pump_until(app, lambda: window.runner is None)
        window.dirty = False
        window.close()
    assert source.timing.count == 1
