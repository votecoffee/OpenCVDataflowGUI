"""Check continuous smoothing, bounded state, source completion and delayed tickets."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.buffered_stream import BufferedStreamBlock
from opencv_dataflow.core import Block, Graph, Port, Result
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.outcomes import TicketResults
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket, frame
from opencv_dataflow.tickets import DIRECT_CLOCK
from opencv_dataflow.toolblocks import Binding, ToolBlock

TYPE = "sequence.stabilize_stream"


def _texture(seed=314):
    image = np.random.default_rng(seed).integers(0, 256, (112, 160), np.uint8)
    return cv2.GaussianBlur(image, (5, 5), 0.9)


def _shift(image, x):
    return cv2.warpAffine(
        image,
        np.array([[1.0, 0.0, x], [0.0, 1.0, 0.0]]),
        image.shape[1::-1],
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _packet(image, index, *, source="camera"):
    ticket = DIRECT_CLOCK.issue(source)
    return ValuePacket(
        image,
        {
            "ticket": ticket,
            "spatial": frame(image.shape[1::-1], "raw"),
            "video_frame": {
                "filename": "original.avi",
                "frame_index": index,
                "timestamp_ms": index * 1000 / 30,
            },
        },
    )


def _block(**parameters):
    return builtin_registry().create(TYPE, parameters={"motion_model": "translation", **parameters})


def _run(block, images):
    outputs = {name: [] for name in ("image", "coverage", "transform", "report")}
    for index, image in enumerate(images):
        result = block.execute({"image": _packet(image, index)})
        assert result is not None and result.passed, block.error
        for name in outputs:
            outputs[name].extend(result.emissions[name])
    result = block.execute({"flush": True})
    assert result is not None and result.passed, block.error
    for name in outputs:
        outputs[name].extend(result.emissions[name])
    return outputs


@pytest.mark.parametrize("kind", ["uint8", "rgb16", "float32", "float64"])
@pytest.mark.parametrize("model", ["translation", "similarity"])
def test_stream_matches_batch_smoothing_and_preserves_each_frame_context(kind, model):
    base = _texture()
    images = [_shift(base, offset) for offset in (0, 0, 6, 0, -3, 1, 0, 4, 0)]
    if kind == "rgb16":
        images = [
            ImageArray(np.repeat((im.astype(np.uint16) * 257)[..., None], 3, 2), "rgb")
            for im in images
        ]
    elif kind.startswith("float"):
        images = [im.astype(kind) / 255 for im in images]
    parameters = {"radius_frames": 2, "sigma_frames": 1.5, "motion_model": model}
    packets = [_packet(image, index) for index, image in enumerate(images)]
    batch = builtin_registry().create("sequence.stabilize_batch", parameters=parameters)
    expected = batch.execute(
        {"frames": FlowGroup(tuple(packets), tuple(map(str, range(len(images)))))}
    )
    assert expected is not None and expected.passed, batch.error
    block = _block(**parameters)
    emitted = {name: [] for name in ("image", "coverage", "transform", "report")}
    for index, packet in enumerate(packets):
        result = block.execute({"image": packet})
        assert result is not None and result.passed, block.error
        assert result.outputs["buffered"] <= parameters["radius_frames"]
        assert result.outputs["status"]["stored_frames"] <= 2 * parameters["radius_frames"] + 1
        if index < parameters["radius_frames"]:
            assert result.outputs["image"] is NO_VALUE
            assert not result.outputs["ready"]
        for name in emitted:
            emitted[name].extend(result.emissions[name])
    tail = block.execute({"flush": True})
    assert tail is not None and tail.passed, block.error
    for name in emitted:
        emitted[name].extend(tail.emissions[name])
    assert not block.pending_output and block.runtime["entries"] == []
    assert len(emitted["image"]) == len(images)
    for i, (image, mask, transform, report) in enumerate(zip(*emitted.values(), strict=True)):
        np.testing.assert_allclose(image.value, expected.outputs["corrected"].items[i].value)
        np.testing.assert_array_equal(mask.value, expected.outputs["coverage"].items[i].value)
        assert image.payload["ticket"] == packets[i].payload["ticket"]
        assert image.payload["video_frame"] == packets[i].payload["video_frame"]
        assert image.value.dtype == images[i].dtype
        assert image_format(image.value) == image_format(images[i])
        assert mask.payload["spatial"] == image.payload["spatial"]
        assert report.value["window"] == expected.outputs["report"]["frames"][i]["window"]
        assert report.value["look_ahead_frames"] <= parameters["radius_frames"]
        np.testing.assert_allclose(
            transform.value["source_to_target"],
            expected.outputs["transforms"].items[i].value["source_to_target"],
            atol=1e-10,
        )


def test_long_stream_has_fixed_storage_and_flush_does_not_duplicate_frames():
    image = _texture()
    block = _block(radius_frames=3)
    count, peak = 0, 0
    for index in range(180):
        result = block.execute({"image": _packet(_shift(image, 3 * np.sin(index / 3)), index)})
        assert result is not None and result.passed, block.error
        status = result.outputs["status"]
        assert status["stored_frames"] <= 7 and status["buffered"] <= 3
        peak = max(peak, status["stored_bytes"])
        count += len(result.emissions["image"])
    tail = block.execute({"flush": True})
    assert tail is not None
    count += len(tail.emissions["image"])
    assert count == 180 and peak < 200_000
    again = block.execute({"flush": True})
    assert again is not None and not again.emissions["image"]
    assert not block.pending_output


def test_scene_cuts_flush_the_old_tail_without_mixing_coordinate_segments():
    a, b = _texture(), _texture(315)
    images = [a, _shift(a, 2), np.zeros_like(a), b, _shift(b, -2)]
    outputs = _run(_block(radius_frames=2), images)
    reports = [p.value for p in outputs["report"]]
    assert [r["index"] for r in reports] == list(range(5))
    assert [r["segment"] for r in reports] == [0, 0, 1, 2, 2]
    assert [r["window"] for r in reports] == [[0, 2], [0, 2], [2, 3], [3, 5], [3, 5]]
    roots = [p.payload["spatial"]["root_id"] for p in outputs["image"]]
    assert roots[0] == roots[1] and roots[3] == roots[4] and len(set(roots)) == 3


def test_explicit_reset_discards_pending_and_copy_runtime_resumes_the_same_frames():
    image = _texture()
    original = _block(radius_frames=2)
    for i in range(2):
        assert original.execute({"image": _packet(image, i)}) is not None
    clone = _block(radius_frames=2)
    clone.copy_runtime_from(original)
    assert clone.snapshot_runtime()["epoch"] == original.snapshot_runtime()["epoch"]
    tail = clone.execute({"flush": True})
    assert [p.value["index"] for p in tail.emissions["report"]] == [0, 1]
    assert original.pending_output
    reset = original.execute({"reset": True, "image": _packet(image, 2)})
    assert reset is not None and reset.outputs["status"]["reset_discarded"] == 2
    assert original.runtime["received"] == 1
    original.reset_runtime()
    assert not original.pending_output


def test_gap_and_recording_loop_start_new_segments():
    image = _texture()
    block = _block(radius_frames=2, max_gap_ms=50)
    ticks = iter([0.0, 0.01, 1.0])
    block.now = lambda: next(ticks)
    for i in range(3):
        assert block.execute({"image": _packet(image, i)}) is not None
    assert block.runtime["segment"] == 1
    block.now = lambda: 1.01
    loop = block.execute({"image": _packet(image, 0)})
    assert loop is not None and block.runtime["segment"] == 2


def test_budget_and_malformed_controls_fail_safely_and_clear_buffer():
    block = _block(radius_frames=63, max_bytes_mb=0.01)
    assert block.execute({"image": _texture()}) is None
    assert block.error_kind == "invalid" and "memory budget" in block.error
    assert not block.pending_output
    for name in ("flush", "reset"):
        assert _block().execute({name: "yes"}) is None


def test_failed_trigger_does_not_change_a_delayed_good_frames_outcome():
    image = _texture()
    block = _block(radius_frames=1)
    good, failed = _packet(image, 0), _packet(image, 1)
    failed.payload["outcome"] = {
        "model": "tool_outcome_v1",
        "passed": False,
        "chain_passed": False,
        "source": {"id": "upstream"},
        "fail_reasons": [{"code": "upstream.reject"}],
    }
    assert block.execute({"image": good}) is not None
    emitted = block.execute({"image": failed})
    first = emitted.emissions["image"][0]
    assert first.payload["ticket"] == good.payload["ticket"]
    assert first.payload["outcome"]["chain_passed"]
    tail = block.execute({"flush": True})
    assert tail.emissions["image"][0].payload["outcome"]["chain_passed"] is False


class FiniteImages(Block):
    type_id = "input.test_frames"
    outputs = (Port("image"),)
    file_source = True
    finite_source = True

    def __init__(self, images, **kwargs):
        self.images = images
        self.index = 0
        super().__init__(**kwargs)

    @property
    def exhausted(self):
        return self.index == len(self.images)

    def process(self, inputs):
        if self.exhausted:
            return Result({"image": NO_VALUE})
        image = self.images[self.index]
        payload = _packet(image, self.index).payload
        payload.pop("ticket")
        self.index += 1
        return Result({"image": image}, payloads={"image": payload})


def test_graph_flushes_chained_streams_and_nested_toolblock_only_on_last_call():
    images = [_shift(_texture(), offset) for offset in (0, 0, 4, 0, -2, 0, 1, 0, 0)]
    graph = Graph()
    source = graph.add(FiniteImages(images, block_id="source"))
    first = graph.add(_block(radius_frames=2))
    child = Graph()
    inner = child.add(_block(radius_frames=2))
    second = graph.add(
        ToolBlock(
            child,
            {"image": Binding(inner.id, "image")},
            {"image": Binding(inner.id, "image")},
            block_id="nested",
        )
    )
    graph.connect(source.id, "image", first.id, "image")
    graph.connect(first.id, "image", second.id, "image")
    indices = []
    for _ in range(20):
        graph.run()
        assert first.state in ("pass", "waiting", "inactive"), first.error
        assert second.state in ("pass", "waiting", "inactive"), second.error
        indices.extend(
            p.payload["video_frame"]["frame_index"]
            for p in second.current_emissions.get("image", [])
        )
        if graph.sources_exhausted():
            break
    assert graph.sources_exhausted() and indices == list(range(len(images)))
    assert not graph.pending_output
    assert first.runtime["segment"] == inner.runtime["segment"] == 0


def test_production_completion_waits_for_pending_frame_tickets():
    graph = Graph()
    source = graph.add(FiniteImages([_texture()] * 5))
    smooth = graph.add(_block(radius_frames=1))
    graph.connect(source.id, "image", smooth.id, "image")
    for index in range(5):
        graph.run()
        assert smooth.state in ("pass", "waiting"), smooth.error
        expected = 5 if index == 4 else index
        assert graph.ticket_results.production.summary()["total"] == expected
        pending = smooth.pending_tickets
        rows = graph.ticket_results.snapshot()["tickets"]
        for ticket in pending:
            row = next(row for row in rows if row["ticket"] == ticket)
            assert "completion" not in row
    assert graph.sources_exhausted()


def test_output_consumer_attributes_coverage_failure_to_the_emitted_ticket():
    image = _texture()
    block = _block(radius_frames=1, min_coverage_fraction=1.0)
    consumer = TicketResults()
    block._ticket_results = consumer
    first, second = _packet(image, 0), _packet(_shift(image, 6), 1)
    assert block.execute({"image": first}) is not None
    result = block.execute({"image": second})
    assert result is not None and not result.passed, block.error
    assert block.state == "fail"
    report = result.emissions["report"][0]
    assert report.payload["ticket"] == first.payload["ticket"]
    assert report.payload["outcome"]["passed"] is False
    assert result.payloads["status"]["ticket"] == second.payload["ticket"]
    assert result.payloads["status"]["outcome"]["passed"] is True
    rows = consumer.snapshot()["tickets"]
    events = {row["ticket"]["sequence"]: row["events"][-1] for row in rows}
    assert events[first.payload["ticket"]["sequence"]]["passed"] is False
    current = events[second.payload["ticket"]["sequence"]]
    assert current["passed"] is True and current["fail_reasons"] == []


def test_finite_video_flushes_all_frames_with_bounded_lookahead(tmp_path):
    path = tmp_path / "long.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 30, (160, 112))
    assert writer.isOpened()
    base = _texture()
    for i in range(90):
        writer.write(cv2.cvtColor(_shift(base, 3 * np.sin(i / 3)), cv2.COLOR_GRAY2BGR))
    writer.release()
    graph = Graph()
    source = graph.add(
        builtin_registry().create(
            "input.video_file",
            parameters={"path": str(path), "color_format": "gray", "repeat": "stop"},
        )
    )
    smooth = graph.add(_block(radius_frames=3))
    graph.connect(source.id, "image", smooth.id, "image")
    observed, count = [], 0
    try:
        while not graph.sources_exhausted() and count < 100:
            graph.run()
            assert smooth.state in ("pass", "waiting"), smooth.error
            observed.extend(
                p.payload["video_frame"]["frame_index"]
                for p in smooth.current_emissions.get("image", [])
            )
            assert smooth.last_result.outputs["status"]["stored_frames"] <= 7
            count += 1
        assert graph.sources_exhausted() and observed == list(range(90))
        assert not smooth.pending_output
    finally:
        graph.close()


class DrainOne(BufferedStreamBlock):
    """A plugin-style buffer that needs extra scheduler cycles to finish."""

    type_id = "test.drain_one"
    inputs = (
        Port("image", required=False, required_when_connected=True),
        Port("level", "number", required=False),
    )
    outputs = (Port("image"),)

    def __init__(self, **kwargs):
        self.pending = []
        super().__init__(**kwargs)

    @property
    def pending_output(self):
        return bool(self.pending)

    @property
    def pending_tickets(self):
        return tuple(packet.payload["ticket"] for packet in self.pending)

    def process(self, inputs):
        if "image" in inputs:
            self.pending.append(ValuePacket(inputs["image"], self.input_payloads["image"]))
        elif self.end_of_input and self.pending:
            packet = self.pending.pop(0)
            return Result({"image": packet.value}, payloads={"image": packet.payload})
        return Result({"image": NO_VALUE})


class ScalarControl(Block):
    type_id = "input.test_scalar"
    outputs = (Port("level", "number"),)

    def process(self, inputs):
        return Result({"level": 1})


def test_generic_buffer_drains_after_source_exhaustion_with_scalar_controls():
    graph = Graph()
    source = graph.add(FiniteImages([_texture()] * 3))
    buffer = graph.add(DrainOne())
    control = graph.add(ScalarControl())
    graph.connect(source.id, "image", buffer.id, "image")
    graph.connect(control.id, "level", buffer.id, "level")
    emitted, runs = [], 0
    while not graph.sources_exhausted() and runs < 10:
        graph.run()
        emitted.extend(buffer.current_emissions.get("image", []))
        runs += 1
    assert runs == 6 and len(emitted) == 3
    assert graph.sources_exhausted() and not graph.pending_output


@pytest.mark.local_examples
@pytest.mark.parametrize("hold_until_end", [False, True])
def test_background_acquisition_keeps_the_complete_stabilized_tail(hold_until_end):
    from opencv_dataflow.queued_execution import QueuedExecution

    registry = builtin_registry()
    registry.register(DrainOne)
    graph = Graph()
    path = (
        Path(__file__).parents[1]
        / "examples/opencv tools/continuous_camera_stabilization"
        / "assets/inspection-jitter.avi"
    )
    source = graph.add(
        registry.create(
            "input.video_file",
            parameters={
                "path": str(path),
                "color_format": "gray",
                "repeat": "stop",
                "end_frame": 25,
            },
        )
    )
    queue = graph.add(
        registry.create(
            "flow.image_queue",
            parameters={
                "mode": "background_acquisition",
                "capacity": 3,
            },
        )
    )
    smooth = graph.add(DrainOne() if hold_until_end else _block(radius_frames=3))
    graph.connect(source.id, "image", queue.id, "image")
    graph.connect(queue.id, "image", smooth.id, "image")
    indices = []
    try:
        with QueuedExecution(graph, registry) as execution:
            for _ in range(60):
                if not execution.run_one():
                    break
                assert smooth.state in ("pass", "waiting"), smooth.error
                indices.extend(
                    p.payload["video_frame"]["frame_index"]
                    for p in smooth.current_emissions.get("image", [])
                )
        assert indices == list(range(25)) and not graph.pending_output
    finally:
        graph.close()


@pytest.mark.local_examples
def test_saved_streaming_project_runs_all_frames_and_smooths_the_path():
    folder = Path(__file__).parents[1] / "examples/opencv tools/continuous_camera_stabilization"
    graph = Graph.load(folder / "continuous_camera_stabilization.json", builtin_registry())
    graph.set_run_profile("lean")
    raw, smoothed, indices = [], [], []
    try:
        for _ in range(150):
            graph.run()
            block = graph.blocks["smooth"]
            assert block.state in ("pass", "waiting"), block.error
            for packet in block.current_emissions.get("report", []):
                raw.append(packet.value["observed_pose"])
                smoothed.append(packet.value["smoothed_pose"])
                indices.append(packet.payload["video_frame"]["frame_index"])
            if graph.sources_exhausted():
                break
        assert graph.sources_exhausted() and indices == list(range(144))
        jitter = [
            float(np.sqrt(np.mean(np.sum(np.diff(path, n=2, axis=0)[:, :2] ** 2, axis=1))))
            for path in (np.asarray(raw), np.asarray(smoothed))
        ]
        assert jitter[1] < 0.2 * jitter[0]
        assert not block.pending_output
    finally:
        graph.close()
