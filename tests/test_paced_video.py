"""Deterministic media-clock scheduling and real decoder/GUI/queue integration."""

import copy
import threading
import time

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.paced_video import MediaPacer
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import NO_VALUE

TYPE = "input.paced_video_file"


class Clock:
    def __init__(self):
        self.value = 100.0
        self.sleeps = []

    def now(self):
        return self.value

    def sleep(self, delay):
        self.sleeps.append(delay)
        self.value += delay

    def advance(self, delay):
        self.value += delay


def settings(**values):
    return {**builtin_registry().create(TYPE).parameters, **values}


def info(timestamp, fps=25):
    return {
        "decoder_timestamp_ms": timestamp,
        "fps": fps,
        "frame_index": 0,
        "timestamp_ms": timestamp,
        "timestamp_source": "decoder",
    }


@pytest.mark.parametrize("speed", [0.5, 1, 2])
def test_variable_media_intervals_and_processing_time_do_not_accumulate_drift(speed):
    clock = Clock()
    pacer = MediaPacer(clock.now, clock.sleep)
    state = {}
    for i, stamp in enumerate((1000, 1040, 1130, 1310)):
        if i:
            clock.advance(0.003)
        result = pacer.release(state, i, info(stamp), settings(speed=speed), lambda: False)
        assert result is not None
        assert clock.now() == pytest.approx(100 + (stamp - 1000) / 1000 / speed)
        assert result["timestamp_ms"] == stamp
    assert max(clock.sleeps) <= 0.01000001


def test_stride_gaps_and_loop_boundary_keep_cadence_with_one_cycle_start():
    clock = Clock()
    pacer = MediaPacer(clock.now, clock.sleep)
    state = {}
    p = settings(stride=2)
    first = pacer.release(state, 2, info(200, 10), p, lambda: False)
    second = pacer.release(state, 4, info(400, 10), p, lambda: False)
    state["loop"] = True
    third = pacer.release(state, 2, info(200, 10), p, lambda: False)
    assert clock.now() == pytest.approx(100.4)
    assert [v["playback"]["cycle_start"] for v in (first, second, third)] == [True, False, True]
    assert third["playback"]["cycle"] == 1
    assert third["playback"]["interval_source"] == "loop_frame_rate_estimate"


@pytest.mark.parametrize("policy", ["decoder_or_fps", "configured_fps"])
def test_repeated_missing_or_backwards_decoder_times_use_explicit_rate_provenance(policy):
    clock = Clock()
    pacer = MediaPacer(clock.now, clock.sleep)
    state = {}
    p = settings(timing_policy=policy, fallback_fps=20 if policy == "configured_fps" else 0)
    output = [
        pacer.release(state, i, info(stamp, 10), p, lambda: False)
        for i, stamp in enumerate((0, None, 0, 10))
    ]
    step = 50 if policy == "configured_fps" else 100
    assert [v["timestamp_ms"] for v in output] == [i * step for i in range(4)]
    assert output[1]["timestamp_source"] == (
        "configured_rate_estimate" if policy == "configured_fps" else "frame_rate_estimate"
    )


def test_strict_timing_and_gap_limit_reject_before_any_wait():
    clock = Clock()
    pacer = MediaPacer(clock.now, clock.sleep)
    state = {}
    pacer.release(state, 0, info(0), settings(), lambda: False)
    with pytest.raises(ValidationError, match="increasing decoder"):
        pacer.release(state, 1, info(0), settings(timing_policy="decoder_only"), lambda: False)
    with pytest.raises(ValidationError, match="Maximum gap"):
        pacer.release(state, 1, info(20000), settings(), lambda: False)
    assert not clock.sleeps


@pytest.mark.parametrize("policy", ["rebase", "catch_up"])
def test_slow_processing_lateness_and_idle_resume_do_not_drop_entries(policy):
    clock = Clock()
    pacer = MediaPacer(clock.now, clock.sleep)
    state = {}
    p = settings(late_policy=policy)
    pacer.release(state, 0, info(0), p, lambda: False)
    clock.advance(1)
    late = pacer.release(state, 1, info(40), p, lambda: False)
    assert late["playback"]["action"] == ("late_rebase" if policy == "rebase" else "late_catch_up")
    assert late["playback"]["lateness_ms"] == pytest.approx(960)
    last = pacer.release(state, 2, info(80), p, lambda: False)
    assert clock.now() == pytest.approx(101.04 if policy == "rebase" else 101)
    state["resume"] = True
    clock.advance(10)
    resumed = pacer.release(state, 3, info(120), p, lambda: False)
    assert resumed["playback"]["action"] == "resumed" and resumed["playback"]["waited_ms"] == 0
    assert not last["playback"]["cycle_start"] and not resumed["playback"]["cycle_start"]


def test_cancelled_wait_does_not_commit_clock_and_resume_rebases():
    clock = Clock()
    pacer = MediaPacer(clock.now, clock.sleep)
    state = {}
    pacer.release(state, 0, info(0), settings(), lambda: False)
    before = copy.deepcopy(state)
    result = pacer.release(state, 1, info(5000), settings(), lambda: clock.now() >= 100.025)
    assert result is None and state == before and clock.now() < 100.04
    state["resume"] = True
    result = pacer.release(state, 1, info(5000), settings(), lambda: False)
    assert result["playback"]["waited_ms"] == 0 and state["index"] == 1


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "paced.avi"
    writer = builtin_registry().create(
        "output.video",
        parameters={"path": str(path), "backend": "ffmpeg", "codec": "FFV1", "fps": 10},
    )
    try:
        for index in range(6):
            assert (
                writer.execute({"image": np.full((40, 60, 3), 20 + index * 30, np.uint8)})
                is not None
            ), writer.error
    finally:
        writer.close()
    return path


def test_native_file_stride_matches_media_time_pixels_tickets_and_reopen(clip):
    clock = Clock()
    reader = builtin_registry().create(
        TYPE,
        parameters={"path": str(clip), "backend": "ffmpeg", "stride": 2, "color_format": "gray"},
    )
    reader._pacer = MediaPacer(clock.now, clock.sleep)
    try:
        results = [reader.execute({}) for _ in range(3)]
        assert all(r is not None for r in results), reader.error
        assert [r.outputs["info"]["frame_index"] for r in results] == [0, 2, 4]
        np.testing.assert_allclose(
            [r.outputs["image"].mean() for r in results], [20, 80, 140], atol=0
        )
        assert [r.outputs["info"]["timestamp_ms"] for r in results] == pytest.approx([0, 200, 400])
        assert clock.now() == pytest.approx(100.4)
        assert reader.exhausted and reader.execute({}).outputs["image"] is NO_VALUE
        assert all(r.payloads["image"]["video_frame"] == r.outputs["info"] for r in results)
        assert len({r.outputs["ticket"]["session"] for r in results}) == 3
    finally:
        reader.close()


def test_stop_during_native_source_clock_preserves_pending_frame_and_worker_copy(clip):
    clock = Clock()
    registry = builtin_registry()
    params = {"path": str(clip), "backend": "ffmpeg"}
    reader = registry.create(TYPE, parameters=params)
    reader._pacer = MediaPacer(clock.now, clock.sleep)
    first = reader.execute({})
    assert first is not None, reader.error
    assert reader.execute({}, cancelled=lambda: clock.now() >= 100.025).outputs["image"] is NO_VALUE
    assert reader._source_ticket is None and reader._sequence["next"] == 1
    assert not any(reader.current_emissions.values())
    clone = registry.create(TYPE, parameters=params)
    clone._pacer = MediaPacer(clock.now, clock.sleep)
    clone.copy_runtime_from(reader)
    try:
        second = clone.execute({})
        assert second is not None, clone.error
        assert second.outputs["info"]["frame_index"] == 1
        assert second.outputs["info"]["playback"]["action"] == "resumed"
        assert second.outputs["image"].mean() == 50
        assert reader._sequence["next"] == 1 and clone._sequence["next"] == 2
    finally:
        reader.close()
        clone.close()


def test_qt_runner_stop_interrupts_wait_and_resume_emits_next_frame(app, clip):
    from PySide6.QtCore import Qt

    from opencv_dataflow.app import Runner

    registry, graph = builtin_registry(), Graph()
    source = graph.add(
        registry.create(TYPE, parameters={"path": str(clip), "backend": "ffmpeg", "speed": 0.1})
    )
    runner = Runner(graph, "continuous", registry)
    seen = threading.Event()

    def completed():
        runner.ack.set()
        seen.set()

    runner.frame.connect(completed, Qt.ConnectionType.DirectConnection)
    try:
        runner.start()
        assert seen.wait(2)
        start = time.monotonic()
        runner.stop()
        assert runner.wait(1000)
        assert time.monotonic() - start < 0.75 and not runner.ended_with_error
        assert source._sequence["next"] == 1
        result = source.execute({})
        assert result is not None, source.error
        assert result.outputs["info"]["frame_index"] == 1
    finally:
        runner.detached = True
        runner.stop()
        runner.ack.set()
        runner.wait(2000)
        graph.close()


def test_native_mjpeg_clock_preserves_relative_spacing(clip, tmp_path):
    path = tmp_path / "mjpeg.avi"
    writer = builtin_registry().create(
        "output.video", parameters={"path": str(path), "backend": "opencv_mjpeg", "fps": 10}
    )
    try:
        for index in range(4):
            assert (
                writer.execute({"image": np.full((40, 60, 3), 20 + index * 30, np.uint8)})
                is not None
            )
    finally:
        writer.close()
    clock = Clock()
    reader = builtin_registry().create(
        TYPE, parameters={"path": str(path), "backend": "opencv_mjpeg"}
    )
    reader._pacer = MediaPacer(clock.now, clock.sleep)
    try:
        rows = [reader.execute({}) for _ in range(4)]
        assert all(r is not None for r in rows), reader.error
        times = [r.outputs["info"]["timestamp_ms"] for r in rows]
        np.testing.assert_allclose(np.diff(times), 100)
        assert clock.now() == pytest.approx(100.3)
        assert all(r.outputs["info"]["backend"] == "CV_MJPEG" for r in rows)
    finally:
        reader.close()


def test_unknown_native_timing_requires_explicit_rate_and_keeps_pixels(monkeypatch, clip):
    from opencv_dataflow import video_files

    native = cv2.VideoCapture

    class MissingTiming:
        def __init__(self, *args):
            self.real = native(*args)

        def __getattr__(self, name):
            return getattr(self.real, name)

        def get(self, property):
            if property in (cv2.CAP_PROP_POS_MSEC, cv2.CAP_PROP_FPS):
                return float("nan")
            return self.real.get(property)

    monkeypatch.setattr(video_files.cv2, "VideoCapture", MissingTiming)
    clock = Clock()
    registry = builtin_registry()
    missing = registry.create(TYPE, parameters={"path": str(clip), "backend": "ffmpeg"})
    missing._pacer = MediaPacer(clock.now, clock.sleep)
    assert missing.execute({}) is None and missing.error_kind == "invalid"
    assert not any(missing.current_emissions.values()) and not clock.sleeps
    reader = registry.create(
        TYPE, parameters={"path": str(clip), "backend": "ffmpeg", "fallback_fps": 20}
    )
    reader._pacer = MediaPacer(clock.now, clock.sleep)
    try:
        first, second = reader.execute({}), reader.execute({})
        assert first is not None and second is not None, reader.error
        assert second.outputs["info"]["timestamp_source"] == "configured_rate_estimate"
        assert second.outputs["info"]["decoder_timestamp_ms"] is None
        assert second.outputs["image"].mean() == 50 and clock.now() == pytest.approx(100.05)
    finally:
        missing.close()
        reader.close()


def test_background_queue_retains_paced_order_and_media_metadata(clip):
    from opencv_dataflow.queued_execution import QueuedExecution

    registry, graph = builtin_registry(), Graph()
    source = graph.add(
        registry.create(
            TYPE,
            block_id="video",
            parameters={"path": str(clip), "backend": "ffmpeg", "speed": 100},
        )
    )
    queue = graph.add(
        registry.create(
            "flow.image_queue", block_id="queue", parameters={"mode": "background_acquisition"}
        )
    )
    graph.connect(source.id, "image", queue.id, "image")
    queued = QueuedExecution(graph, registry)
    queued.start()
    try:
        rows = []
        for _ in range(6):
            assert queued.run_one()
            result = graph.blocks["queue"].last_result
            rows.append(result.payloads["image"]["video_frame"])
        assert [r["frame_index"] for r in rows] == list(range(6))
        assert [r["timestamp_ms"] for r in rows] == pytest.approx([i * 100 for i in range(6)])
        assert not queued.run_one()
    finally:
        queued.close()
        graph.close()


def test_clock_errors_and_close_reopen_leave_no_decoder_or_stale_emission(clip, monkeypatch):
    registry = builtin_registry()
    reader = registry.create(TYPE, parameters={"path": str(clip), "backend": "ffmpeg"})

    def broken(*args, **kwargs):
        raise RuntimeError("Controlled clock failure")

    monkeypatch.setattr(reader._pacer, "release", broken)
    assert reader.execute({}) is None and reader.error_kind == "exception"
    assert reader._capture is None and not any(reader.current_emissions.values())
    reader.reset_runtime()
    reader._pacer = MediaPacer(Clock().now, lambda delay: None)
    # First frame has no wait; an independent restored runtime does not reuse a native handle.
    assert reader.execute({}).outputs["info"]["frame_index"] == 0
    reader.close()
    assert reader._capture is None


@pytest.mark.parametrize(
    "parameters",
    [
        {"speed": 0},
        {"maximum_gap_ms": 0},
        {"timing_policy": "configured_fps"},
        {"end_frame": 0},
        {"stride": 1.5},
        {"late_policy": "skip_frames"},
        {"frame_ticket": "false"},
    ],
)
def test_invalid_playback_parameters_do_not_open_decoder(parameters, monkeypatch):
    reader = builtin_registry().create(TYPE, parameters=parameters)
    monkeypatch.setattr(
        cv2, "VideoCapture", lambda *a: pytest.fail("Invalid settings opened decoder")
    )
    assert reader.execute({}) is None and reader.error_kind == "invalid"


def test_paced_controls_xml_toolbox_and_editor_are_pure(app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QComboBox

    from opencv_dataflow.app import MainWindow
    from opencv_dataflow.project_xml import export_project_xml, read_project_xml
    from opencv_dataflow.toolbox import block_entry

    registry, graph = builtin_registry(), Graph()
    source = graph.add(registry.create(TYPE))
    assert block_entry(source).path[0] == "OpenCV" and "videoio" in block_entry(source).path[1]
    assert "cycle_start" not in source.exposed_outputs
    path = tmp_path / "paced.xml"
    snapshot = graph.to_dict()
    export_project_xml(snapshot, path)
    opened = Graph.from_dict(read_project_xml(path), registry)
    assert opened.to_dict() == snapshot
    opened.close()
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a: pytest.fail("Editing opened video"))
    window = MainWindow()
    try:
        window.replace_graph(graph)
        window.inspector.select(source)
        assert all(
            isinstance(window.inspector.fields[n], QComboBox)
            for n in ("timing_policy", "late_policy", "backend")
        )
        assert source._capture is None and source.last_result is None
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
        graph.close()


def test_equal_native_and_fallback_rates_still_report_the_rate_actually_used():
    clock = Clock()
    pacer = MediaPacer(clock.now, clock.sleep)
    state = {}
    p = settings(fallback_fps=10)
    pacer.release(state, 0, info(0, 10), p, lambda: False)
    result = pacer.release(state, 1, info(None, 10), p, lambda: False)
    assert result["timestamp_source"] == "frame_rate_estimate"


def test_loop_without_fps_uses_declared_last_interval_estimate_or_rejects_unknown_duration():
    clock = Clock()
    pacer = MediaPacer(clock.now, clock.sleep)
    state = {}
    p = settings(timing_policy="decoder_only")
    pacer.release(state, 0, info(0, None), p, lambda: False)
    state["loop"] = True
    with pytest.raises(ValidationError, match="one-frame loop"):
        pacer.release(state, 0, info(0, None), p, lambda: False)
    state["loop"] = False
    pacer.release(state, 1, info(75, None), p, lambda: False)
    state["loop"] = True
    result = pacer.release(state, 0, info(0, None), p, lambda: False)
    assert result["playback"]["interval_source"] == "loop_last_interval_estimate"
    assert clock.now() == pytest.approx(100.15)
