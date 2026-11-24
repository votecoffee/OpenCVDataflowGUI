"""Verify timed decay, source-grid direction and native motion region outputs."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.run_profiles import profile
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

PROJECT = Path(__file__).parents[1] / "examples/opencv tools/motion_history/motion_history.json"


@pytest.fixture(scope="module")
def registry():
    return builtin_registry()


@pytest.mark.parametrize("initial_time, duration", [(0.0, 1.000014), (1_800_000_000_000.0, 250.0)])
def test_inclusive_expiry_at_zero_and_large_epochs(registry, initial_time, duration):
    block = registry.create(
        "sequence.motion_history",
        parameters={"duration_ms": duration, "segment_regions": False},
    )
    mask = np.zeros((24, 32), np.uint8)
    mask[8, 9] = 255
    blank = np.zeros_like(mask)
    first = block.execute({"image": mask, "timestamp_ms": initial_time})
    assert first and first.passed, block.error
    assert first.outputs["recency"][8, 9] == 1
    assert first.outputs["labels"] is NO_VALUE
    assert first.outputs["direction_deg"] is NO_VALUE
    edge = block.execute({"image": blank, "timestamp_ms": initial_time + duration})
    assert edge and edge.passed, block.error
    assert edge.outputs["active"][8, 9] == 255
    assert edge.outputs["age_ms"][8, 9] == pytest.approx(duration)
    assert edge.outputs["recency"][8, 9] == 0
    after = block.execute({"image": blank, "timestamp_ms": initial_time + duration + 0.125})
    assert after and after.passed, block.error
    assert not after.outputs["active"].any()
    assert np.all(after.outputs["age_ms"] == -1)


def test_mask_roi_stream_and_clock_reset(registry):
    block = registry.create("sequence.motion_history")
    block.run_profile = profile("lean")
    grid = frame([32, 24], "camera")
    clock = TicketClock("history")
    current = np.zeros((24, 32), np.uint8)
    current[4:14, 4:14] = 255
    select = np.zeros_like(current)
    select[6:16, 5:11] = 255
    region = {
        "model": "image_region_v1",
        "shape": "rectangle",
        "image_size": [32, 24],
        "rectangle": [7, 8, 8, 9],
    }

    def step(image, time, call, **kwargs):
        ticket = clock.issue("camera", "fixed", f"frame-{call}")
        payload = {"spatial": grid, "ticket": ticket, "video_frame": {"timestamp_ms": time}}
        result = block.execute({"image": ValuePacket(image, payload), **kwargs})
        assert result and result.passed, block.error
        return result, ticket

    start, ticket = step(
        current, 100, 1, mask=ValuePacket(select, {"spatial": grid}), region=region
    )
    expected = np.zeros_like(current)
    expected[8:14, 7:11] = 255
    np.testing.assert_array_equal(start.outputs["active"], expected)
    assert start.payloads["age_ms"]["ticket"] == ticket
    assert start.payloads["age_ms"]["spatial"] == grid
    assert not start.drawings
    empty = np.zeros_like(current)
    isolated, _ = step(empty, 150, 2, stream="other")
    assert not isolated.outputs["active"].any()
    original, _ = step(empty, 150, 3)
    assert original.outputs["age_ms"][9, 8] == 50
    rewound, _ = step(empty, 10, 4)
    assert rewound.outputs["status"]["clock_restarted"]
    assert rewound.outputs["status"]["learning_activations"] == 1
    assert not rewound.outputs["active"].any()
    step(current, 20, 5, timestamp_ms=20)
    reset, _ = step(empty, 30, 6, timestamp_ms=30, reset=True)
    assert not reset.outputs["active"].any()
    assert reset.outputs["status"]["previous_ticket"] is None


def test_segments_label_all_regions_and_limit_records(registry):
    block = registry.create(
        "sequence.motion_history",
        parameters={"duration_ms": 600000, "maximum_regions": 1},
    )
    earlier = np.zeros((24, 48), np.uint8)
    earlier[15:17, 3:6] = 255
    epoch = 1_800_000_000_000.0
    first = block.execute({"image": earlier, "timestamp_ms": epoch})
    assert first and first.passed
    current = np.zeros_like(earlier)
    current[3:7, 2:6] = 255
    current[9:15, 31:42] = 255
    result = block.execute({"image": current, "timestamp_ms": epoch + 0.03125})
    assert result and result.passed, block.error
    labels = result.outputs["labels"]
    assert labels.dtype == np.int32
    assert set(np.unique(labels)) == {0, 1, 2}
    assert labels[15, 3] == 0 and result.outputs["active"][15, 3] == 255
    assert result.outputs["segments"] == [{"label": 2, "rectangle": [31, 9, 11, 6], "pixels": 66}]


def test_invalid_time_and_gradient_limits(registry):
    block = registry.create("sequence.motion_history")
    with pytest.raises(ValidationError, match="history duration"):
        block.validate_parameters({**block.parameters, "maximum_delta_ms": 251})
    assert block.execute({"image": np.zeros((24, 32), np.uint8)}) is None
    assert "capture timestamp" in block.error
    assert block.execute({"image": np.zeros((4, 32), np.uint8), "timestamp_ms": 0}) is None
    assert "uint8 silhouette" in block.error


def test_saved_clip_directions_and_independent_age_reference(registry):
    graph = Graph.load(PROJECT, registry)
    try:
        times = np.full((96, 128), -np.inf, float)
        for index in range(44):
            graph.run()
            block = graph.blocks["history"]
            result = block.last_result
            assert result and result.passed, (index, block.error)
            status = result.outputs["status"]
            now = status["timestamp_ms"]
            assert now == pytest.approx(index * 50)
            if 2 <= index <= 12:
                left = 18 + 3 * (index - 2)
                times[18:34, left : left + 15] = now
            if 22 <= index <= 32:
                top = 18 + 3 * (index - 22)
                times[top : top + 16, 74:91] = now
            age = now - times
            valid = age <= 250
            np.testing.assert_array_equal(result.outputs["active"] != 0, valid)
            np.testing.assert_allclose(result.outputs["age_ms"], np.where(valid, age, -1))
            if index in (6, 12, 25, 32):
                expect = 0 if index < 22 else 90
                assert result.outputs["direction_deg"] == pytest.approx(expect, abs=0.1)
                assert result.drawings["motion history"].from_output
            if index in (18, 38):
                assert not valid.any()
                assert result.outputs["direction_deg"] is NO_VALUE
        assert block_entry(graph.blocks["history"]).path[:2] == (
            "OpenCV",
            "Motion Templates (motempl)",
        )
    finally:
        graph.close()
