import copy
import threading
import time

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Block, Graph, Port, Result
from opencv_dataflow.image_queue import ImageQueue
from opencv_dataflow.payloads import NO_VALUE, ValuePacket
from opencv_dataflow.queued_execution import QueuedExecution, background_queue
from opencv_dataflow.tickets import TicketClock


def frame(number):
    return ValuePacket(
        np.full((8, 10), number, np.uint8),
        {"ticket": TicketClock().issue("camera", "external", number), "custom": {"number": number}},
    )


def release(queue):
    queue.execute({})
    assert queue.state in ("pass", "waiting"), queue.error
    value = queue.last_result.outputs["image"]
    queue.complete()
    queue.update_metrics()
    return value


def test_fifo_private_snapshots_one_inflight_and_metadata():
    queue = ImageQueue()
    original = frame(1)
    queue.enqueue(original)
    queue.enqueue(frame(2))
    original.value[:] = 9
    queue.execute({})
    assert np.all(queue.last_result.outputs["image"] == 1)
    assert queue.last_result.payloads["image"]["custom"]["number"] == 1
    assert queue.stats()["busy"] == 1
    with pytest.raises(ValueError, match="in flight"):
        queue.take()
    queue.complete()
    assert np.all(release(queue) == 2)
    assert release(queue) is NO_VALUE and queue.state == "waiting"
    assert queue.stats()["completed"] == 2


@pytest.mark.parametrize("policy,expected", [("drop_oldest", [2, 3]), ("drop_newest", [1, 2])])
def test_explicit_overflow_policies_count_drops(policy, expected):
    queue = ImageQueue(parameters={"capacity": 2, "overflow": policy})
    for value in (1, 2, 3):
        queue.enqueue(frame(value))
    assert [int(release(queue)[0, 0]) for _ in range(2)] == expected
    assert queue.stats()["dropped"] == 1 and queue.stats()["high_water"] == 2


def test_full_error_preserves_accepted_frames_and_byte_limit():
    queue = ImageQueue(parameters={"capacity": 1, "overflow": "error"})
    queue.enqueue(frame(1))
    with pytest.raises(ValueError, match="queue full"):
        queue.enqueue(frame(2))
    assert np.all(release(queue) == 1) and queue.stats()["rejected"] == 1
    tiny = ImageQueue(parameters={"max_bytes_mb": 0.00001})
    with pytest.raises(ValueError, match="byte limit"):
        tiny.enqueue(frame(1))
    assert tiny.stats()["queued"] == 0


def test_blocking_backpressure_waits_for_consumer_and_cancels_cleanly():
    queue = ImageQueue(parameters={"capacity": 1})
    queue.enqueue(frame(1))
    cancel = threading.Event()
    started, finished = threading.Event(), threading.Event()

    def produce():
        started.set()
        queue.enqueue(frame(2), cancel.is_set, background=True)
        finished.set()

    thread = threading.Thread(target=produce)
    thread.start()
    assert started.wait(1)
    assert not finished.wait(0.03)
    assert np.all(release(queue) == 1)
    assert finished.wait(1)
    thread.join()
    assert np.all(release(queue) == 2)
    queue.enqueue(frame(3))
    thread = threading.Thread(
        target=lambda: queue.enqueue(frame(4), cancel.is_set, background=True)
    )
    thread.start()
    cancel.set()
    thread.join(timeout=1)
    assert not thread.is_alive() and queue.stats()["queued"] == 1


def test_burst_is_released_once_per_graph_run_and_drains_without_new_input():
    class Burst(Block):
        outputs = (Port("image"),)
        sent = False

        def process(self, inputs):
            if self.sent:
                return Result({"image": NO_VALUE})
            self.sent = True
            entries = [frame(i) for i in (1, 2, 3)]
            return Result({"image": entries[-1].value}, emissions={"image": entries})

    r, graph = builtin_registry(), Graph()
    source = graph.add(Burst())
    queue = graph.add(r.create("flow.image_queue", parameters={"capacity": 2}))
    gray = graph.add(r.create("cv.convert"))
    graph.connect(source.id, "image", queue.id, "image")
    graph.connect(queue.id, "image", gray.id, "image")
    for number in (1, 2, 3):
        graph.run()
        assert gray.state == "pass", gray.error
        assert len(gray.last_result.emissions["image"]) == 1
        assert np.all(gray.last_result.outputs["image"] == number)
        assert queue.stats()["queued"] == 3 - number and queue.stats()["busy"] == 0
    graph.run()
    assert queue.state == "waiting" and gray.state == "inactive"


def setup_background(capacity=3, fail_after=0):
    registry = builtin_registry()

    class Source(Block):
        type_id = "input.queue_test"
        outputs = (Port("image"), Port("info", "data"))
        acquired, closed, threads = [], [], set()
        enough = threading.Event()

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.counter = 0

        def process(self, inputs):
            self.counter += 1
            if fail_after and self.counter > fail_after:
                raise ValueError("camera offline")
            self.threads.add(threading.get_ident())
            self.acquired.append(self.counter)
            if len(self.acquired) >= 3:
                self.enough.set()
            return Result(
                {
                    "image": np.full((10, 12), self.counter % 200, np.uint8),
                    "info": {"capture": self.counter},
                }
            )

        def close(self):
            if self.counter:
                self.closed.append(threading.get_ident())

    class Inspect(Block):
        type_id = "custom.queue_inspect"
        inputs = (Port("image"), Port("info", "data"))
        outputs = (Port("data", "data"),)
        active, max_active = 0, 0
        seen = []
        require_background = True
        lock = threading.Lock()

        def process(self, inputs):
            with self.lock:
                type(self).active += 1
                type(self).max_active = max(self.max_active, self.active)
            try:
                if self.require_background:
                    assert Source.enough.wait(2), (
                        "Acquisition did not continue while inspection was running"
                    )
                time.sleep(0.005)
                value = inputs["info"]["capture"]
                assert np.all(inputs["image"] == value % 200), "Image/ancillary snapshot mismatch"
                self.seen.append((value, copy.deepcopy(self.input_payloads["image"]["ticket"])))
                return Result({"data": {"capture": value}})
            finally:
                with self.lock:
                    type(self).active -= 1

    registry.register(Source)
    registry.register(Inspect)
    graph = Graph()
    source = graph.add(registry.create(Source.type_id))
    queue = graph.add(
        registry.create(
            "flow.image_queue", parameters={"mode": "background_acquisition", "capacity": capacity}
        )
    )
    inspect = graph.add(registry.create(Inspect.type_id))
    graph.connect(source.id, "image", queue.id, "image")
    graph.connect(queue.id, "image", inspect.id, "image")
    graph.connect(source.id, "info", inspect.id, "info")
    return registry, graph, source, queue, inspect, Source, Inspect


def test_background_capture_runs_during_inspection_coherently_and_never_overlaps_frames():
    registry, graph, source, queue, inspect, Source, Inspect = setup_background()
    with QueuedExecution(graph, registry) as execution:
        for _ in range(5):
            assert execution.run_one()
            assert inspect.state == "pass", inspect.error
            assert queue.stats()["busy"] == 0 and queue.stats()["queued"] <= 3
    assert [number for number, _ in Inspect.seen] == [1, 2, 3, 4, 5]
    assert Inspect.max_active == 1 and Inspect.active == 0
    assert Source.threads == set(Source.closed) and threading.get_ident() not in Source.threads
    assert queue.stats()["completed"] == 5
    assert source.last_result.outputs["info"]["capture"] == 5


def test_producer_error_drains_accepted_frames_then_reports_error():
    registry, graph, _, queue, inspect, Source, Inspect = setup_background(fail_after=2)
    Inspect.require_background = False
    with QueuedExecution(graph, registry) as execution:
        assert execution.run_one() and execution.run_one()
        with pytest.raises(RuntimeError, match="camera offline"):
            execution.run_one()
    assert [number for number, _ in Inspect.seen] == [1, 2]
    assert queue.stats()["queued"] == 0


def test_pending_frames_clone_and_resume_before_new_captures_but_not_project_data(tmp_path):
    registry, graph, _, queue, _, Source, Inspect = setup_background()
    with QueuedExecution(graph, registry) as execution:
        execution.run_one()
    assert queue.stats()["queued"] > 0
    expected = int(queue.runtime["pending"][0][0].frame.value[0, 0])
    clone = Graph.from_dict(graph.to_dict(), registry)
    clone.copy_runtime_from(graph)
    clone_queue = clone.blocks[queue.id]
    with QueuedExecution(clone, registry) as execution:
        execution.run_one()
        assert int(clone_queue.last_result.outputs["image"][0, 0]) == expected
    path = tmp_path / "queued.json"
    graph.save(path)
    fresh = Graph.load(path, registry)
    assert fresh.blocks[queue.id].stats()["queued"] == 0
    graph.invalidate()
    assert queue.stats()["queued"] == 0


def test_two_background_queues_require_explicit_configuration():
    registry, graph, *_ = setup_background()
    graph.add(registry.create("flow.image_queue", parameters={"mode": "background_acquisition"}))
    with pytest.raises(ValueError, match="one background"):
        background_queue(graph)


def test_gui_background_stop_retains_queue_and_run_once_drains_before_acquiring(app):
    from test_editor import pump_until
    from test_pattern_editor import run as run_window

    from opencv_dataflow.app import MainWindow

    registry, graph, source, queue, inspect, Source, Inspect = setup_background()
    window = MainWindow()
    window.registry = registry
    window.replace_graph(graph)
    window.show()
    app.processEvents()
    window.start_run(True)
    pump_until(app, lambda: queue.last_result is not None and queue.stats()["completed"] >= 1)
    window.stop_run()
    pump_until(app, lambda: window.runner is None)
    assert queue.stats()["queued"] > 0 and Inspect.active == 0
    expected = queue.runtime["pending"][0][0]
    expected_number = expected.snapshots[source.id]["last_result"].outputs["info"]["capture"]
    acquisitions = len(Source.acquired)
    run_window(window, app)
    assert inspect.state == "pass", inspect.error
    assert inspect.last_result.outputs["data"]["capture"] == expected_number
    assert len(Source.acquired) == acquisitions
    window.inspector.select(queue)
    assert window.inspector.fields["overflow"].currentText() == "block"
    assert "queued" in window.inspector.details.toPlainText()
    window.reset_flow_state()
    assert queue.stats()["queued"] == 0
    window.dirty = False
    window.close()


def test_nested_queue_drains_on_inactive_input():
    from opencv_dataflow.toolblocks import encapsulate

    class Burst(Block):
        outputs = (Port("image"),)
        sent = False

        def process(self, inputs):
            if self.sent:
                return Result({"image": NO_VALUE})
            self.sent = True
            packets = [frame(i) for i in (1, 2, 3)]
            return Result({"image": packets[-1].value}, emissions={"image": packets})

    r, graph = builtin_registry(), Graph()
    r.register(Burst)
    source = graph.add(Burst())
    queue = graph.add(r.create("flow.image_queue"))
    gray = graph.add(r.create("cv.convert"))
    graph.connect(source.id, "image", queue.id, "image")
    graph.connect(queue.id, "image", gray.id, "image")
    nested = encapsulate(graph, [queue.id, gray.id], "Queued inspection", r)
    for _ in range(3):
        graph.run()
        assert nested.state == "pass", nested.error
    assert nested.graph.blocks[queue.id].stats()["completed"] == 3


def test_sync_burst_overflow_rejects_before_mutating_accepted_queue():
    queue = ImageQueue(parameters={"capacity": 1, "overflow": "error"})
    queue.enqueue(frame(1))
    queue.pending_events = [frame(2), frame(3)]
    queue.execute({})
    assert queue.state == "error" and queue.stats()["queued"] == 1
    assert queue.stats()["rejected"] == 2 and queue.lease is None
    assert np.all(release(queue) == 1)


def test_failed_downstream_marks_current_frame_and_does_not_retry():
    registry, graph, _, queue, inspect, _, Inspect = setup_background()
    Inspect.require_background = False
    queue.parameters["mode"] = "buffer_only"
    # A broken public input makes the downstream step fail after dequeueing.
    inspect.inputs = (*inspect.inputs, Port("missing", "data"))
    graph.run()
    assert inspect.state == "error"
    assert queue.stats()["completed"] == 1 and queue.stats()["failed"] == 1
    assert queue.stats()["busy"] == 0


def test_background_nested_acquisition_preserves_internal_preview_snapshots():
    from opencv_dataflow.toolblocks import encapsulate

    registry, graph, source, queue, inspect, Source, Inspect = setup_background()
    convert = graph.add(registry.create("cv.convert"))
    graph.connections = [
        c for c in graph.connections if not (c.source == source.id and c.target == queue.id)
    ]
    graph.connect(source.id, "image", convert.id, "image")
    graph.connect(convert.id, "image", queue.id, "image")
    acquisition = encapsulate(graph, [source.id, convert.id], "Acquisition", registry)
    with QueuedExecution(graph, registry) as execution:
        for expected in (1, 2, 3):
            execution.run_one()
            assert inspect.state == "pass", inspect.error
            captured = acquisition.graph.blocks[source.id].last_result.outputs["info"]["capture"]
            assert captured == expected
            assert np.all(
                acquisition.graph.blocks[convert.id].last_result.outputs["image"] == expected
            )


def test_saved_queued_example_headless_receipt_and_gui_load(app, tmp_path):
    import json
    from pathlib import Path

    from opencv_dataflow.app import MainWindow
    from opencv_dataflow.examples import queued_input_graph
    from opencv_dataflow.runner import main

    registry = builtin_registry()
    project = Path(__file__).parents[1] / "examples/queued_input/queued_input.json"
    graph = Graph.load(project, registry)
    assert graph.to_dict() == queued_input_graph(registry).to_dict()
    report = tmp_path / "receipt.json"
    assert main([str(project), "--repeat", "3", "--report", str(report)]) == 0
    receipt = json.loads(report.read_text(encoding="utf-8"))
    for row in receipt["runs"]:
        queue = next(b for b in row["blocks"] if b["id"] == "input_queue")
        inspect = next(b for b in row["blocks"] if b["id"] == "parts_inspect")
        assert queue["emitted"]["image"] == 1 and queue["outputs"]["busy"] == 0
        assert inspect["outputs"]["results"]["pass_count"] == 25
    window = MainWindow()
    window.load_queued_input_example()
    assert window.graph.blocks["input_queue"].parameters["mode"] == "background_acquisition"
    window.dirty = False
    window.close()


def test_queue_selects_pyramid_level_and_keeps_coordinate_provenance():
    registry = builtin_registry()
    source = registry.create("input.demo")
    source.execute({})
    pyramid = registry.create("cv.image_pyramid", parameters={"levels": 3})
    pyramid.execute({"image": source.last_result.outputs["image"]})
    queue = registry.create("flow.image_queue")
    queue.input_levels = {"image": "2"}
    queue.enqueue(pyramid.last_result.outputs["images"])
    output = release(queue)
    assert output.shape[:2] == (90, 120)
    assert output.metadata["pyramid"]["level"] == 2
    assert output.metadata["ticket"] == source.last_result.outputs["ticket"]
