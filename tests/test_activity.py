"""Block event correlation, contention, shared aggregation and offscreen timelines."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QFileDialog, QMessageBox

from opencv_dataflow.activity import ActivityLog, spans
from opencv_dataflow.activity_window import COLORS, ActivityWindow, row_key, ticket_id
from opencv_dataflow.app import MainWindow
from opencv_dataflow.blocks import builtin_registry, demo_graph
from opencv_dataflow.core import Block, Graph, Port, Result
from opencv_dataflow.examples import parallel_parts_graph
from opencv_dataflow.payloads import ValuePacket
from opencv_dataflow.queued_execution import QueuedExecution
from opencv_dataflow.tickets import TicketClock, ticket_key
from opencv_dataflow.timing import mark
from opencv_dataflow.toolblocks import Binding, ToolBlock


def packet(number=1):
    return ValuePacket(
        np.full((12, 16), number, np.uint8),
        {
            "ticket": TicketClock("test").issue("camera", "fixed", number),
        },
    )


def events(graph, kind):
    return [e for e in graph.activity.snapshot()["events"] if e["kind"] == kind]


def until(predicate):
    import time

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if predicate():
            return
        threading.Event().wait(0.005)
    raise AssertionError("Activity condition did not arrive")


def test_all_blocks_report_paired_start_finish_with_tickets():
    graph = demo_graph(builtin_registry())
    graph.run()
    snapshot = graph.activity.snapshot()
    assert len(events(graph, "start")) == len(graph.blocks)
    assert len(events(graph, "finish")) == len(graph.blocks)
    calls = spans(snapshot)
    assert len(calls) == len(graph.blocks)
    identities = {ticket_key(t) for s in calls for t in s["tickets"]}
    assert len(identities) == 1
    assert all(s["finish"]["ns"] >= s["start"]["ns"] for s in calls)
    assert all(s["judgment"] == "pass" for s in calls)
    assert all(s["event"]["duration_ms"] >= 0 for s in calls)
    assert events(graph, "ticket_assigned")
    json.dumps(snapshot, allow_nan=False)
    assert all(e["tickets"] for e in events(graph, "start"))


class Probe(Block):
    type_id = "custom.activity_probe"
    inputs, outputs = (Port("image"),), (Port("image"),)

    def process(self, inputs):
        self.report_event("measurement", {"score": 0.75})
        return Result({"image": inputs["image"]})


def test_custom_markers_inherit_ticket_and_are_private_json_snapshots():
    graph = Graph()
    block = graph.add(Probe())
    block.execute({"image": packet()})
    event = events(graph, "measurement")[0]
    assert event["tickets"][0]["value"] == 1
    assert event["invocation"] == events(graph, "start")[0]["invocation"]
    event["data"]["score"] = 99
    assert events(graph, "measurement")[0]["data"]["score"] == 0.75
    with pytest.raises(ValueError, match="8 KiB"):
        block.report_event("large", {"text": "x" * 9000})


def test_errors_always_have_finish_and_exception_markers():
    class Broken(Probe):
        def process(self, inputs):
            raise ValueError("Bad image")

    graph = Graph()
    block = graph.add(Broken())
    block.execute({"image": packet()})
    assert block.state == "error"
    assert "Bad image" in events(graph, "exception")[0]["data"]["message"]
    call = spans(graph.activity.snapshot())[0]
    assert call["judgment"] == "error" and call["tickets"][0]["value"] == 1


def test_rejected_input_is_a_flag_without_fake_work_bar_or_sample():
    graph = Graph()
    block = graph.add(Probe())
    block.execute({"image": packet()})
    block.input_failure_policies = {"image": "reject_failed"}
    incoming = packet(2)
    incoming.payload["outcome"] = {
        "model": "tool_outcome_v1",
        "passed": False,
        "chain_passed": False,
        "source": {"id": "upstream"},
        "score": None,
        "data": {},
    }
    block.execute({"image": incoming})
    assert block.timing.count == 1 and len(events(graph, "start")) == 1
    assert len(events(graph, "finish")) == 1
    assert events(graph, "rejected")[0]["tickets"][0]["value"] == 2


def test_overlapping_calls_are_flagged_and_serialized():
    entered, release = threading.Event(), threading.Event()

    class Slow(Probe):
        active, maximum = 0, 0

        def process(self, inputs):
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            if inputs["image"].mean() == 1:
                entered.set()
                assert release.wait(5)
            self.active -= 1
            return Result({"image": inputs["image"]})

    graph, block = Graph(), Slow()
    graph.add(block)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(block.execute, {"image": packet(1)})
        assert entered.wait(5)
        second = pool.submit(block.execute, {"image": packet(2)})
        try:
            until(lambda: events(graph, "blocked"))
            assert len(events(graph, "start")) == 1
        finally:
            release.set()
        first.result(), second.result()
    calls = sorted(spans(graph.activity.snapshot()), key=lambda s: s["start"]["ns"])
    assert block.maximum == 1 and block.timing.count == 2
    assert calls[0]["finish"]["ns"] <= calls[1]["start"]["ns"]
    assert [s["tickets"][0]["value"] for s in calls] == [1, 2]
    assert events(graph, "unblocked")


def test_blocked_call_can_cancel_without_changing_active_work():
    entered, release, cancel = threading.Event(), threading.Event(), threading.Event()

    class Slow(Probe):
        def process(self, inputs):
            entered.set()
            assert release.wait(5)
            return Result({"image": inputs["image"]})

    graph = Graph()
    block = graph.add(Slow())
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(block.execute, {"image": packet(1)})
        assert entered.wait(5)
        second = pool.submit(block.execute, {"image": packet(2)}, cancel.is_set)
        try:
            until(lambda: events(graph, "blocked"))
            cancel.set()
            assert second.result(timeout=5) is None
            assert block.timing.started is not None
        finally:
            release.set()
        first.result()
    assert block.timing.count == 1 and events(graph, "cancelled_wait")


def test_distinct_blocks_can_execute_in_parallel_in_same_log():
    barrier = threading.Barrier(2)

    class Concurrent(Probe):
        def process(self, inputs):
            barrier.wait(timeout=5)
            return Result({"image": inputs["image"]})

    graph = Graph()
    blocks = [graph.add(Concurrent()) for _ in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda b: b.execute({"image": packet()}), blocks))
    calls = spans(graph.activity.snapshot())
    assert max(s["start"]["ns"] for s in calls) <= min(s["finish"]["ns"] for s in calls)
    assert len({s["event"]["thread"] for s in calls}) == 2


def test_retention_reconstructs_finish_even_when_start_evicted():
    graph = Graph()
    graph.bind_activity(ActivityLog(max_events=1))
    block = graph.add(Probe())
    block.execute({"image": packet()})
    snapshot = graph.activity.snapshot()
    assert len(snapshot["events"]) == 1 and snapshot["evicted_events"] > 0
    assert len(spans(snapshot)) == 1
    assert spans(snapshot)[0]["tickets"][0]["value"] == 1
    tiny = ActivityLog(max_bytes=1)
    tiny.emit(snapshot["events"][0])
    assert tiny.snapshot()["events"] == []


def test_nested_and_parallel_jobs_share_log_and_parent_tickets():
    graph = parallel_parts_graph(builtin_registry())
    graph.run()
    block = next(b for b in graph.blocks.values() if b.type_id == "flow.inspect_each")
    calls = [s for s in spans(graph.activity.snapshot()) if "/part-" in s["source"]["path"]]
    assert calls and all(s["parent_tickets"] for s in calls)
    children = [s for s in calls if s["source"]["id"] in block.graph.blocks]
    assert len(children) == 25 * len(block.graph.blocks)
    assert len({row_key(s["source"]) for s in children}) == len(block.graph.blocks)


def test_ordinary_toolblock_paths_and_collector_are_shared():
    registry, inner, outer = builtin_registry(), Graph(), Graph()
    child = inner.add(registry.create("cv.gray"))
    tool = outer.add(
        ToolBlock(
            inner, {"image": Binding(child.id, "image")}, {"image": Binding(child.id, "image")}
        )
    )
    tool.execute({"image": packet()})
    assert tool.graph.activity is outer.activity
    assert {s["source"]["path"] for s in spans(outer.activity.snapshot())} == {
        tool.id,
        f"{tool.id}/{child.id}",
    }


def test_queue_reports_busy_arrivals_drops_and_capacity_wait():
    graph, registry = Graph(), builtin_registry()
    queue = graph.add(
        registry.create(
            "flow.image_queue",
            parameters={
                "capacity": 1,
                "overflow": "drop_newest",
            },
        )
    )
    queue.enqueue(packet(1))
    queue.take()
    queue.enqueue(packet(2))
    assert not queue.enqueue(packet(3))
    assert events(graph, "queue_busy") and events(graph, "dropped")
    assert events(graph, "dropped")[0]["tickets"][0]["value"] == 3
    queue.parameters["overflow"] = "block"
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(queue.wait_for_capture_space, lambda: False)
        try:
            until(lambda: events(graph, "queue_wait"))
        finally:
            queue.complete()
            queue.take()
        assert pending.result(timeout=5)
    assert events(graph, "queue_resume")
    queue.close()


def test_background_acquisition_reports_once_into_shared_job_log():
    graph, registry = Graph(), builtin_registry()
    source = graph.add(registry.create("input.demo"))
    queue = graph.add(
        registry.create(
            "flow.image_queue",
            parameters={
                "mode": "background_acquisition",
                "capacity": 2,
            },
        )
    )
    gray = graph.add(registry.create("cv.gray"))
    graph.connect(source.id, "image", queue.id, "image")
    graph.connect(queue.id, "image", gray.id, "image")
    queued = QueuedExecution(graph, registry)
    try:
        queued.start()
        assert queued.run_one()
    finally:
        queued.close()
    finishes = events(graph, "finish")
    assert sum(e["source"]["id"] == source.id for e in finishes) == source.timing.count
    assert sum(e["source"]["id"] == gray.id for e in finishes) == 1
    assert queued.producer.activity is graph.activity
    graph.close()


def emit_span(log, source, start, finish, ticket, judgment="pass", invocation="call"):
    point = mark()
    first, last = {**point, "ns": start}, {**point, "ns": finish}
    common = {
        "source": source,
        "tickets": [ticket],
        "parent_tickets": [],
        "thread": 1,
        "data": {},
        "invocation": invocation,
    }
    log.emit({**common, "kind": "start", "time": first})
    log.emit(
        {
            **common,
            "kind": "finish",
            "time": last,
            "started": first,
            "judgment": judgment,
            "duration_ms": (finish - start) / 1e6,
        }
    )


def test_ticket_gantt_filter_selection_and_offscreen_render(app):
    log = ActivityLog()
    now = mark()["ns"]
    source = {"id": "blur", "path": "blur", "name": "Gaussian blur", "type": "cv.gaussian_blur"}
    tickets = [packet(i).payload["ticket"] for i in (1, 2)]
    for i, ticket in enumerate(tickets):
        emit_span(log, source, now - 20_000_000, now - 10_000_000, ticket, invocation=str(i))
    window = ActivityWindow(lambda: log)
    window.ticket.setCurrentIndex(window.ticket.findData(ticket_id(tickets[0])))
    assert window.bar_count == 1
    bars = [
        i for i in window.scene.items() if isinstance(i.data(0), dict) and "judgment" in i.data(0)
    ]
    bars[0].setSelected(True)
    assert "Gaussian blur" in window.details.toPlainText()
    assert bars[0].brush().color() == QColor(COLORS["pass"])
    image = QImage(1250, 760, QImage.Format.Format_ARGB32)
    image.fill(QColor("black"))
    painter = QPainter(image)
    window.scene.render(painter)
    painter.end()
    assert not image.isNull()
    window.search.setText("missing")
    assert window.bar_count == 0
    window.close()


def test_rolling_window_excludes_old_bars_and_pause_freezes_data(app):
    log, now = ActivityLog(), mark()["ns"]
    source = {"id": "inspect", "path": "inspect", "name": "Inspect", "type": "custom"}
    ticket = packet().payload["ticket"]
    emit_span(log, source, now - 30_000_000_000, now - 29_000_000_000, ticket, invocation="old")
    emit_span(log, source, now - 20_000_000, now - 10_000_000, ticket, invocation="new")
    window = ActivityWindow(lambda: log, "concurrency")
    assert window.bar_count == 1
    window.pause.setChecked(True)
    emit_span(log, source, now - 8_000_000, now - 5_000_000, ticket, invocation="added")
    window.refresh()
    assert window.bar_count == 1
    window.pause.setChecked(False)
    assert window.bar_count == 2
    window.close()


def test_exception_flag_remains_visible_after_finish(app):
    graph = Graph()
    block = graph.add(Probe())
    block.execute({"image": packet()})
    block.report_event("exception", {"message": "diagnostic exception"}, {"image": packet()})
    window = ActivityWindow(lambda: graph.activity)
    assert window.flag_count == 2  # custom measurement and exception
    window.close()


def test_live_active_span_and_verdict_colors(app):
    log = ActivityLog()
    point, ticket = mark(), packet().payload["ticket"]
    source = {"id": "active", "name": "Active block", "path": "active", "type": "custom"}
    common = {
        "source": source,
        "tickets": [ticket],
        "parent_tickets": [],
        "thread": 1,
        "data": {},
        "invocation": "active",
    }
    log.emit({**common, "kind": "start", "time": point})
    window = ActivityWindow(lambda: log, "concurrency")
    assert window.bar_count == 1
    bar = next(i for i in window.scene.items() if i.data(0))
    assert bar.brush().color() == QColor(COLORS["running"])
    log.emit(
        {
            **common,
            "kind": "finish",
            "time": mark(),
            "started": point,
            "judgment": "fail",
            "duration_ms": 1,
        }
    )
    window.refresh()
    bar = next(i for i in window.scene.items() if i.data(0))
    assert bar.brush().color() == QColor(COLORS["fail"])
    window.close()


def test_gui_worker_log_survives_f5_and_menu_views(app):
    from test_editor import pump_until

    window = MainWindow()
    for count in (1, 2):
        window.start_run(False)
        pump_until(app, lambda: window.runner is None)
        assert len(events(window.root_graph, "finish")) == count * len(window.graph.blocks)
    window.show_activity("ticket")
    window.show_activity("concurrency")
    assert window.activity_ticket_window.bar_count == len(window.graph.blocks)
    assert window.activity_concurrency_window.bar_count == 2 * len(window.graph.blocks)
    window.activity_ticket_window.close()
    window.activity_concurrency_window.close()
    window.dirty = False
    window.close()


def test_source_reserves_ticket_before_work_and_reports_failed_acquisition():
    class BrokenSource(Block):
        type_id = "input.activity_failure"
        outputs = (Port("image"),)

        def process(self, inputs):
            assert events(graph, "start")[-1]["tickets"] == [self._source_ticket]
            raise ValueError("Acquisition failed")

    graph = Graph()
    source = graph.add(BrokenSource())
    graph.bind_ticket_results()
    source.execute({})
    ticket = events(graph, "start")[0]["tickets"][0]
    assert events(graph, "exception")[0]["tickets"] == [ticket]
    assert events(graph, "finish")[0]["tickets"] == [ticket]
    row = graph.ticket_results.find(ticket)
    assert row["status"] == "error" and not row["processing_time"]["inferred_start"]


@pytest.mark.parametrize(
    "field, value",
    [
        ("kind", ""),
        ("source", {}),
        ("time", {}),
        ("tickets", ["bad"]),
        ("data", {"bad": float("nan")}),
    ],
)
def test_invalid_events_are_rejected_without_advancing_log(field, value):
    graph = Graph()
    graph.add(Probe()).execute({"image": packet()})
    original = graph.activity.snapshot()["events"][-1]
    original[field] = value
    log = ActivityLog()
    with pytest.raises((ValueError, TypeError, KeyError)):
        log.emit(original)
    assert log.snapshot()["sequence"] == 0


def test_parallel_ticket_view_defaults_to_parent_ticket(app):
    graph = parallel_parts_graph(builtin_registry())
    graph.run()
    window = ActivityWindow(lambda: graph.activity)
    parent = next(e["parent_tickets"][0] for e in events(graph, "finish") if e["parent_tickets"])
    assert window.ticket.currentData() == ticket_id(parent)
    assert window.bar_count > 25
    window.close()


def test_activity_history_is_not_saved_in_project(tmp_path):
    registry = builtin_registry()
    graph = demo_graph(registry)
    graph.run()
    path = tmp_path / "project.json"
    graph.save(path)
    loaded = Graph.load(path, registry)
    assert not events(loaded, "finish")
    assert events(graph, "finish")


def test_export_preserves_existing_files_and_includes_event_schema(app, tmp_path, monkeypatch):
    graph = demo_graph(builtin_registry())
    graph.run()
    window = ActivityWindow(lambda: graph.activity)
    path = tmp_path / "activity.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), ""))
    window.export_events()
    before = path.read_bytes()
    receipt = json.loads(before)
    assert receipt["model"] == "block_activity_v1"
    assert all(e["model"] == "block_event_v1" for e in receipt["events"])
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: warnings.append(a))
    window.export_events()
    assert warnings and path.read_bytes() == before
    window.close()


def test_headless_receipt_contains_paired_activity_events(tmp_path):
    from opencv_dataflow.runner import main

    path = tmp_path / "receipt.json"
    assert (
        main(
            [
                "examples/sequence_memory/sequence_memory.json",
                "--repeat",
                "2",
                "--report",
                str(path),
            ]
        )
        == 0
    )
    receipt = json.loads(path.read_text("utf-8"))
    calls = spans(receipt["activity"])
    assert len(calls) == 10 and all(s["finish"] is not None for s in calls)


def test_recursive_call_is_an_exception_instead_of_deadlock():
    class Recursive(Probe):
        def process(self, inputs):
            return self.execute(inputs)

    graph = Graph()
    block = graph.add(Recursive())
    block.execute({"image": packet()})
    assert block.state == "error" and "recursively execute" in block.error
    assert len(events(graph, "finish")) == 1 and events(graph, "blocked")
