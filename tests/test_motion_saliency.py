"""Verify source-grid motion, readiness and native sequence restart behavior."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.run_profiles import profile
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

PROJECT = Path(__file__).parents[1] / "examples/opencv tools/motion_saliency/motion_saliency.json"


@pytest.fixture(scope="module")
def registry():
    return builtin_registry()


def test_saved_clip_tracks_motion_with_video_times(registry):
    graph = Graph.load(PROJECT, registry)
    try:
        previous = np.zeros((96, 128), bool)
        for index in range(100):
            graph.run()
            expected = np.zeros_like(previous)
            if 72 <= index < 87:
                left = 20 + 4 * (index - 72)
                expected[35:57, left : left + 22] = True
            for name in ("wang", "retina"):
                block = graph.blocks[name]
                result = block.last_result
                assert result and result.passed, block.error
                status = result.outputs["status"]
                assert status["frame_index"] == index
                assert status["timestamp_ms"] == pytest.approx(index * 50)
                assert result.outputs["ready"] is (index >= 59)
                if index < 59:
                    assert result.outputs["mask"] is NO_VALUE
                    continue
                observed = np.asarray(result.outputs["mask"]) != 0
                assert result.payloads["mask"]["spatial"]["root_id"] == "video:image"
                assert result.payloads["mask"]["video_frame"]["frame_index"] == index
                if name == "wang":
                    np.testing.assert_array_equal(observed, expected)
                elif index == 77:
                    transient = expected | previous
                    assert np.count_nonzero(observed & transient) / np.count_nonzero(
                        observed | transient
                    ) == pytest.approx(0.415, abs=0.01)
                elif index in (59, 71, 88, 99):
                    assert not observed.any()
                if observed.any():
                    assert result.drawings["motion saliency"].objects
            previous = expected
        assert block_entry(graph.blocks["wang"]).path[:2] == ("OpenCV", "Saliency (saliency)")
    finally:
        graph.close()


def test_bin_wang_odd_grid_selection_stream_and_reset(registry):
    block = registry.create("sequence.motion_saliency", parameters={"warmup": 1})
    block.run_profile = profile("lean")
    base = np.full((49, 65), 60, np.uint8)
    motion = base.copy()
    motion[20:36, 20:40] = 220
    allowed = np.zeros_like(base)
    allowed[24:32, 24:36] = 255
    spatial = frame([65, 49], "camera")
    clock = TicketClock("motion")

    def run(image, index, **options):
        ticket = clock.issue("camera", "fixed", f"frame-{index}")
        payload = {
            "spatial": spatial,
            "ticket": ticket,
            "video_frame": {"frame_index": index, "timestamp_ms": index * 40},
        }
        result = block.execute({"image": ValuePacket(image, payload), **options})
        assert result and result.passed, block.error
        return result, ticket

    first, _ = run(base, 0)
    assert not first.outputs["ready"]
    for i in range(1, 60):
        ready, _ = run(base, i)
    assert ready.outputs["ready"] and not ready.outputs["mask"].any()
    result, ticket = run(motion, 60, mask=ValuePacket(allowed, {"spatial": spatial}))
    np.testing.assert_array_equal(result.outputs["mask"], allowed)
    assert result.payloads["mask"]["ticket"] == ticket
    assert not result.drawings
    new_stream, _ = run(motion, 60, stream="other")
    assert new_stream.outputs["mask"] is NO_VALUE
    rewind, _ = run(base, 0)
    assert rewind.outputs["mask"] is NO_VALUE
    assert rewind.outputs["status"]["learning_activations"] == 1
    assert rewind.outputs["status"]["model_restart"] == "video position restarted"
    explicit, _ = run(base, 1, reset=True)
    assert explicit.outputs["status"]["generation"] != rewind.outputs["status"]["generation"]
    assert explicit.outputs["mask"] is NO_VALUE


def test_bin_wang_counts_persistent_motion_before_selection(registry, monkeypatch):
    monkeypatch.setattr("opencv_dataflow.motion_saliency.PROMOTION_GUARD", 6)
    block = registry.create("sequence.motion_saliency", parameters={"warmup": 1})
    block.run_profile = profile("lean")
    base = np.full((24, 24), 61, np.uint8)
    bright = np.full_like(base, 220)
    exclude_all = np.zeros_like(base)
    for index in range(7):
        image = base if index % 2 == 0 else bright
        result = block.execute({"image": image, "mask": exclude_all})
        assert result and result.passed, block.error
        status = result.outputs["status"]
        assert status["motion_pixels"] == 0
        if status["model_restart"] is not None:
            assert index == 6
            assert status["model_restart"] == "foreground persistence limit"
            assert status["learning_activations"] == 1
            assert result.outputs["mask"] is NO_VALUE
            break
    else:
        pytest.fail("The native model did not return to warmup")


def test_retina_threshold_and_exposure_response(registry):
    base = np.full((64, 96), 60, np.uint8)
    brighter = np.full_like(base, 210)
    blocks = [
        registry.create(
            "sequence.motion_saliency",
            parameters={
                "method": "retina transient",
                "transient_threshold": threshold,
                "warmup": 30,
            },
        )
        for threshold in (100, 100000)
    ]
    for block in blocks:
        block.run_profile = profile("lean")
        for _ in range(40):
            result = block.execute({"image": base})
            assert result and result.passed, block.error
        assert result.outputs["ready"] and not result.outputs["mask"].any()
    responsive, suppressed = [block.execute({"image": brighter}) for block in blocks]
    assert responsive and suppressed
    assert responsive.outputs["status"]["motion_pixels"] > (base.size // 3)
    assert suppressed.outputs["status"]["motion_pixels"] == 0


def test_native_memory_limit_is_checked_before_creation(registry):
    block = registry.create(
        "sequence.motion_saliency",
        parameters={"method": "retina transient", "max_bytes_mb": 0.1},
    )
    assert block.execute({"image": np.zeros((72, 96), np.uint8)}) is None
    assert "sequence memory byte limit" in block.error
