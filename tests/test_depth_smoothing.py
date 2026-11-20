"""Qualify masked depth smoothing and the saved cloud reconstruction workflow."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import DepthArray
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples/opencv tools/depth_smoothing/depth_smoothing.json"
)


def test_depth_smoothing_matches_independent_confidence_weighted_neighborhood():
    values = np.full((7, 7), 1200, np.float32)
    values[4, 3], values[3, 2], values[2, 3], values[3, 4] = 1260, 1170, np.nan, 1500
    confidence = np.ones(values.shape, np.float32)
    confidence[4, 3], confidence[3, 4] = 0.4, 0.1
    coord = frame([7, 7], "depth-camera")
    packet = {"spatial": coord, "ticket": TicketClock().issue("camera")}
    registry = builtin_registry()
    block = registry.create("3d.smooth_depth", parameters={"radius": 1, "sigma_space_px": 1.2})
    result = block.execute(
        {
            "depth": ValuePacket(DepthArray(values, "mm", 800, 1700), packet),
            "confidence": ValuePacket(confidence, packet),
        }
    )
    assert result and result.passed, block.error
    z = np.array([1200, 1170, 1260])
    distance_px = np.array([0, 1, 1])
    weight = [1, 1, 0.4] * np.exp(-0.5 * ((distance_px / 1.2) ** 2 + ((z - 1200) / 80) ** 2))
    depth = result.outputs["depth"]
    assert depth[3, 3] == pytest.approx(np.average(z, weights=weight), abs=0.001)
    assert (depth.units, depth.display_min, depth.display_max) == ("mm", 800, 1700)
    assert np.isnan(depth[2, 3]) and np.isnan(depth[3, 4])
    assert result.outputs["valid_mask"][2, 3] == result.outputs["valid_mask"][3, 4] == 0
    assert result.outputs["rejected_mask"][2, 3] == 0
    assert result.outputs["rejected_mask"][3, 4] == 255
    for name in ("depth", "valid_mask", "rejected_mask"):
        assert result.payloads[name]["ticket"] == packet["ticket"]
        assert result.payloads[name]["spatial"] == coord
    meters = block.execute({"depth": DepthArray(values / 1000), "confidence": confidence})
    np.testing.assert_allclose(meters.outputs["depth"], depth / 1000, atol=2e-7)
    assert values[3, 4] == 1500 and confidence[3, 4] == np.float32(0.1)


def test_selection_limits_writes_but_still_uses_outside_neighbors():
    depth = np.full((5, 5), 1000, np.uint16)
    depth[2, 2] = 1080
    confidence = np.full((5, 5), 255, np.uint8)
    confidence[2, 3:] = 0
    mask = np.zeros_like(confidence)
    mask[2, :] = 255
    region = {
        "model": "image_region_v1",
        "image_size": [5, 5],
        "shape": "rectangle",
        "rectangle": [2, 1, 2, 3],
    }
    block = builtin_registry().create("3d.smooth_depth", parameters={"minimum_confidence": 0.0})
    result = block.execute(
        {"depth": depth, "confidence": confidence, "mask": mask, "region": region}
    )
    assert result and result.passed, block.error
    out = result.outputs["depth"]
    expected = depth.copy()
    expected[2, 2:4] = (out[2, 2], 0)
    np.testing.assert_array_equal(out, expected)
    assert out.dtype == np.uint16
    assert 1000 < out[2, 2] < 1080
    assert result.outputs["valid_mask"][2, 4] == 255  # Outside selection.
    assert result.outputs["report"]["selected_samples"] == 2
    assert result.outputs["report"]["smoothed_samples"] == 1


def test_no_valid_centers_returns_failure_and_preserves_invalid_sentinels():
    values = np.array([[0, np.nan], [-1, np.inf]], np.float32)
    block = builtin_registry().create("3d.smooth_depth")
    result = block.execute({"depth": values})
    assert result is not None and not result.passed
    assert result.fail_reasons[0]["code"] == "no_trusted_depth_in_selection"
    assert not result.outputs["valid_mask"].any()
    np.testing.assert_array_equal(result.outputs["depth"], values)
    result = block.execute(
        {
            "depth": np.full((2, 2), 1000, np.uint16),
            "confidence": np.zeros((2, 2), np.float32),
        }
    )
    assert result is not None and not result.passed
    assert result.outputs["report"]["confidence_rejected_samples"] == 4
    assert not result.outputs["depth"].any()


@pytest.mark.parametrize(
    ("confidence", "message"),
    [
        (np.ones((5, 6), np.float32), "same dimensions"),
        (np.full((5, 5), 1.1, np.float32), "finite float"),
        (np.full((5, 5), np.nan, np.float32), "finite float"),
        (
            ValuePacket(np.ones((5, 5), np.float32), {"spatial": frame([5, 5], "other")}),
            "different coordinate frames",
        ),
    ],
)
def test_bad_confidence_is_rejected(confidence, message):
    block = builtin_registry().create("3d.smooth_depth")
    depth = ValuePacket(np.ones((5, 5), np.float32), {"spatial": frame([5, 5], "depth")})
    assert block.execute({"depth": depth, "confidence": confidence}) is None
    assert message in block.error


def test_conflicting_ticket_is_rejected():
    block = builtin_registry().create("3d.smooth_depth")
    image = np.ones((5, 5), np.float32)
    clock = TicketClock()
    depth = ValuePacket(image, {"ticket": clock.issue("depth")})
    confidence = ValuePacket(image, {"ticket": clock.issue("confidence")})
    assert block.execute({"depth": depth, "confidence": confidence}) is None
    assert "same ticket" in block.error


def test_native_guide_resolution_checks_trusted_samples_only():
    block = builtin_registry().create("3d.smooth_depth", parameters={"sigma_depth_m": 0.001})
    values = np.ones((5, 5), np.float64)
    values[2, 2] = 12
    assert block.execute({"depth": values}) is None
    assert "4096 depth sigmas" in block.error
    confidence = np.ones((5, 5), np.float64)
    confidence[2, 2] = 0
    result = block.execute({"depth": values, "confidence": confidence})
    assert result and result.passed, block.error
    assert result.outputs["depth"].dtype == np.float64
    assert np.isnan(result.outputs["depth"][2, 2])
    np.testing.assert_array_equal(result.outputs["depth"][confidence > 0], 1)


def test_saved_pipeline_smooths_depth_without_crossing_the_step():
    graph = Graph.load(EXAMPLE, builtin_registry())
    try:
        graph.run()
        assert all(
            block.state not in ("error", "failed", "skipped") for block in graph.processing_blocks()
        ), {block.id: block.error for block in graph.processing_blocks()}
        inputs = graph.blocks["split"].last_result.outputs
        result = graph.blocks["smooth"].last_result
        depth, truth = result.outputs["depth"], inputs["channel_2"]
        include = result.outputs["valid_mask"] != 0
        cloud = graph.blocks["cloud"].last_result.outputs["cloud"]
        assert result.outputs["report"]["invalid_input_samples"] == 114
        assert result.outputs["report"]["confidence_rejected_samples"] == 50
        assert result.outputs["report"]["smoothed_samples"] == include.sum() == len(cloud.points)
        assert not depth[~include].any()
        raw_error = inputs["channel_0"].astype(float) - truth
        smooth_error = depth.astype(float) - truth
        raw_rms = np.sqrt(np.mean(raw_error[include] ** 2))
        smoothed_rms = np.sqrt(np.mean(smooth_error[include] ** 2))
        assert smoothed_rms < 0.4 * raw_rms
        edge = include[:, 59:61]
        assert np.abs(smooth_error[:, 59:61][edge]).max() <= 35
        assert cloud.units == "m"
        x, y = cloud.pixels.T
        np.testing.assert_allclose(cloud.points[:, 2], depth[y, x] * 0.001, atol=3e-7)
        assert cloud.metadata["ticket"] == result.payloads["depth"]["ticket"]
    finally:
        graph.close()
