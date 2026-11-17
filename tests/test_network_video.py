"""Native loopback transport plus deterministic cancellation, failure and persistence checks."""

import copy
import importlib.util
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.network_capture import CaptureFailure, CaptureWorker
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.project_xml import export_project_xml, read_project_xml

TYPE = "input.network_video"
ROOT = Path(__file__).resolve().parents[1]


def source(url="http://127.0.0.1:18764/stream", **settings):
    return builtin_registry().create(TYPE, parameters={"url": url, **settings})


@pytest.fixture
def stream():
    spec = importlib.util.spec_from_file_location(
        "network_fixture", ROOT / "examples/opencv tools/network_video/serve_stream.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    images = []
    for index in range(8):
        image = np.full((64, 96, 3), 20, np.uint8)
        image[16:48, 12 + 4 * index : 36 + 4 * index] = [160, 220, 250]
        ok, data = cv2.imencode(
            ".jpg",
            image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95,
                cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
                cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
            ],
        )
        assert ok
        images.append(data.tobytes())
    server = module.make_server(images=images)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", images
    finally:
        server.stopped.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("path,runs", [("/stream", 5), ("/drop", 10)])
def test_real_ffmpeg_frames_reconnect_tickets_and_new_coordinate_segments(stream, path, runs):
    url, images = stream
    block = source(url + path, open_timeout_ms=2000, read_timeout_ms=1000, reconnect_delay_ms=0)
    records, tickets, grids = [], [], []
    try:
        for index in range(runs):
            result = block.execute({})
            assert result is not None, block.error
            info = result.outputs["info"]
            records.append(info)
            tickets.append(result.outputs["ticket"])
            grids.append(result.payloads["image"]["spatial"])
            local = info["connection_frame_index"] % len(images)
            expected = cv2.imdecode(np.frombuffer(images[local], np.uint8), cv2.IMREAD_COLOR)
            assert np.max(np.abs(result.outputs["image"].astype(int) - expected)) <= 3
            assert info["frame_index"] == index and info["backend"] == "FFMPEG"
            assert info["received_unix_ns"] > 0
            assert result.outputs["stream_start"] is (info["connection_frame_index"] == 0)
            assert result.payloads["image"]["ticket"] == result.outputs["ticket"]
        assert len({t["sequence"] for t in tickets}) == runs
        if path == "/drop":
            assert [i["connection"] for i in records] == [1] * 4 + [2] * 4 + [3] * 2
            assert grids[0] != grids[4] and grids[0] == grids[1]
        else:
            assert all(i["connection"] == 1 for i in records)
    finally:
        block.close()
        assert block._worker is None


@pytest.mark.parametrize("path", ["/open-stall", "/stall"])
def test_native_stalls_hit_bounded_open_or_read_deadline_and_can_recover(stream, path):
    url, _ = stream
    block = source(url + path, open_timeout_ms=500, read_timeout_ms=150, reconnect_attempts=0)
    try:
        started = time.monotonic()
        for _ in range(8):
            result = block.execute({})
            if result is None:
                break
        else:
            pytest.fail("Stream did not stall")
        assert time.monotonic() - started < 5
        assert "timeout" in block.error or "failed" in block.error
        assert block._worker is None and not any(block.current_emissions.values())
        block.parameters["url"] = url + "/stream"
        assert block.execute({}) is not None, block.error
    finally:
        block.close()


def test_stop_interrupts_native_open_before_its_long_timeout_without_ticket(stream):
    url, _ = stream
    block = source(url + "/open-stall", open_timeout_ms=30000)
    cancelled = threading.Event()
    outcome = []
    thread = threading.Thread(target=lambda: outcome.append(block.execute({}, cancelled.is_set)))
    thread.start()
    deadline = time.monotonic() + 5
    while (block._worker is None or block._worker.process is None) and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.5)
    started = time.monotonic()
    cancelled.set()
    thread.join(timeout=1.5)
    assert not thread.is_alive() and time.monotonic() - started < 1.5
    assert outcome[0].outputs["image"] is NO_VALUE
    assert outcome[0].outputs["ticket"] is NO_VALUE and block._stream["next"] == 0
    assert block._worker is None
    block.close()


@pytest.mark.parametrize(
    "parameters",
    [
        {"url": "C:/local/file.avi"},
        {"url": "file:///secret.avi"},
        {"url": "ftp://host/test"},
        {"url": "http://"},
        {"url": "http://host:99999/test"},
        {"url": "http://host/a\nb"},
        {"url": "http://host/test#fragment"},
        {"open_timeout_ms": True},
        {"reconnect_attempts": 1.0},
        {"maximum_frame_bytes": 0},
        {"source_name": ""},
        {"pixel_depth": "uint16", "color_format": "hsv"},
        {"backend": "auto"},
    ],
)
def test_invalid_configuration_never_constructs_worker(monkeypatch, parameters):
    from opencv_dataflow import network_video

    def forbidden():
        pytest.fail("Invalid configuration opened a decoder")

    monkeypatch.setattr(network_video, "CaptureWorker", forbidden)
    block = source(**parameters)
    assert block.execute({}) is None and block.error_kind == "invalid"
    assert not block.last_result and not any(block.current_emissions.values())


def test_native_frame_budget_failure_is_not_retried_or_ticketed(stream):
    url, _ = stream
    block = source(url + "/stream", maximum_frame_pixels=10, reconnect_attempts=5)
    try:
        assert block.execute({}) is None and "frame_limit_or_format" in block.error
        assert block._stream["connection"] == 0 and block._stream["next"] == 0
        assert block._worker is None and not any(block.current_emissions.values())
    finally:
        block.close()


def test_public_diagnostics_and_outputs_do_not_retain_url_secrets(stream):
    url, _ = stream
    secret = url.replace("http://", "http://user:secret@") + "/missing?token=private"
    block = source(secret, reconnect_attempts=0)
    assert block.execute({}) is None
    assert not any(token in block.error for token in ("secret", "private", "user", "token"))
    assert block._worker is None


def test_copy_close_resume_xml_and_external_ticket_are_handle_free(stream, tmp_path):
    url, _ = stream
    block = source(url + "/stream", color_format="rgb", pixel_depth="uint16")
    block.ticket_settings = {**block.ticket_settings, "mode": "external"}
    try:
        first = block.execute({"ticket": "PLC-42"})
        assert first is not None, block.error
        assert first.outputs["image"].dtype == np.uint16
        assert first.outputs["image"].color_format == "rgb"
        assert first.outputs["ticket"]["value"] == "PLC-42"
        other = source(url + "/stream", color_format="rgb", pixel_depth="uint16")
        other.copy_runtime_from(block)
        assert other._worker is None and other._stream == block._stream
        other._stream["next"] = 999
        assert block._stream["next"] == 1
        state = copy.deepcopy(block._stream)
        block.close()
        assert block._stream == state
        again = block.execute({"ticket": "PLC-43"})
        assert again.outputs["info"]["frame_index"] == 1
        assert again.outputs["stream_start"] and again.outputs["info"]["connection"] == 2
        graph = Graph()
        graph.add(block)
        path = tmp_path / "network.xml"
        export_project_xml(graph.to_dict(), path)
        assert read_project_xml(path) == graph.to_dict()
        restored = Graph.from_dict(read_project_xml(path), builtin_registry())
        assert restored.blocks[block.id]._worker is None
    finally:
        block.close()


def test_worker_deadline_is_independent_of_a_response_or_native_timeout():
    class NeverResponds:
        def poll(self):
            return None

    worker = CaptureWorker()
    worker.process = NeverResponds()
    started = time.monotonic()
    with pytest.raises(CaptureFailure, match="read_timeout"):
        worker._wait(60, lambda: False, "read")
    assert time.monotonic() - started < 0.3


def test_shared_slot_and_owned_process_are_released_after_native_read(stream):
    from multiprocessing.shared_memory import SharedMemory

    url, _ = stream
    block = source(url + "/stream")
    assert block.execute({}) is not None, block.error
    worker = block._worker
    name, process, reader = worker.memory.name, worker.process, worker.reader
    block.close()
    assert process.poll() is not None and not reader.is_alive()
    assert worker.process is None and worker.memory is None
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=name, track=False)
    block.close()


def test_unexpected_decoder_exit_recovers_without_duplicate_ticket_or_frame_index(stream):
    url, _ = stream
    block = source(url + "/stream", reconnect_delay_ms=0)
    try:
        first = block.execute({})
        assert first is not None, block.error
        block._worker.process.kill()
        block._worker.process.wait(timeout=2)
        second = block.execute({})
        assert second is not None, block.error
        assert second.outputs["info"]["connection"] == 2
        assert second.outputs["info"]["frame_index"] == 1
        assert second.outputs["stream_start"] is True
        assert first.outputs["ticket"] != second.outputs["ticket"]
    finally:
        block.close()


def test_retry_budget_exhaustion_is_bounded_and_all_partial_workers_close(monkeypatch):
    from opencv_dataflow import network_video

    workers = []

    class Unavailable:
        def __init__(self):
            self.closed = False
            workers.append(self)

        def open(self, settings, cancelled):
            raise CaptureFailure("open_failed")

        def close(self):
            self.closed = True

    monkeypatch.setattr(network_video, "CaptureWorker", Unavailable)
    block = source(reconnect_attempts=2, reconnect_delay_ms=0)
    assert block.execute({}) is None and "open_failed" in block.error
    assert len(workers) == 3 and all(w.closed for w in workers)
    assert block._stream["next"] == 0 and not any(block.current_emissions.values())


def test_stop_cancels_reconnect_delay_without_another_attempt(monkeypatch):
    from opencv_dataflow import network_video

    cancelled = threading.Event()
    attempts = []

    class Unavailable:
        def open(self, settings, check):
            attempts.append(1)
            threading.Timer(0.04, cancelled.set).start()
            raise CaptureFailure("open_failed")

        def close(self):
            pass

    monkeypatch.setattr(network_video, "CaptureWorker", Unavailable)
    block = source(reconnect_delay_ms=30000)
    started = time.monotonic()
    result = block.execute({}, cancelled.is_set)
    assert time.monotonic() - started < 0.5 and len(attempts) == 1
    assert result.outputs["ticket"] is NO_VALUE and result.outputs["image"] is NO_VALUE


def test_qt_runner_stop_cancels_stalled_native_read_and_resume_opens_new_segment(app, stream):
    from PySide6.QtCore import Qt

    from opencv_dataflow.app import Runner

    url, _ = stream
    graph = Graph()
    block = graph.add(source(url + "/stall", read_timeout_ms=30000))
    runner = Runner(graph, "continuous")
    seen = threading.Event()
    count = []

    def completed():
        count.append(1)
        runner.ack.set()
        if len(count) >= 6:
            seen.set()

    runner.frame.connect(completed, Qt.ConnectionType.DirectConnection)
    try:
        runner.start()
        assert seen.wait(6), block.error
        time.sleep(0.1)
        started = time.monotonic()
        runner.stop()
        assert runner.wait(1500)
        assert time.monotonic() - started < 1.5 and not runner.ended_with_error
        assert block._worker is None and block._stream["next"] == 6
        result = block.execute({})
        assert result is not None, block.error
        assert result.outputs["info"]["frame_index"] == 6
        assert result.outputs["info"]["connection"] == 2
        assert result.outputs["stream_start"]
    finally:
        runner.detached = True
        runner.stop()
        runner.ack.set()
        runner.wait(2000)
        graph.close()


def test_required_url_validation_and_editing_do_not_acquire(app):
    from PySide6.QtWidgets import QComboBox

    from opencv_dataflow.app import MainWindow

    block = builtin_registry().create(TYPE)
    with pytest.raises(ValidationError) as exc:
        block.validate_parameters()
    assert exc.value.fields == ("url",)
    window = MainWindow()
    try:
        graph = Graph()
        graph.add(block)
        window.replace_graph(graph)
        window.inspector.select(block)
        combo = window.inspector.fields["rtsp_transport"]
        assert isinstance(combo, QComboBox)
        assert [combo.itemText(i) for i in range(combo.count())] == ["tcp", "udp"]
        assert block._worker is None
    finally:
        window.dirty = False
        window.close()
