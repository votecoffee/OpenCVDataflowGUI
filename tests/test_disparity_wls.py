"""Qualify guided filtering, fixed-point conventions and rejected match support."""

from pathlib import Path
from runpy import run_path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.disparity_wls import FilterExternalDisparity
from opencv_dataflow.images import DepthArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/external_disparity_wls"


def run_filter(disparity, right=None, *, guide=None, options=None, **ports):
    tool = FilterExternalDisparity(parameters=options or {})
    if guide is None:
        guide = np.full(disparity.shape, 96, np.uint8)
    if right is not None:
        ports["right_disparity"] = right
    result = tool.execute({"disparity": disparity, "guide": guide, **ports})
    return tool, result


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_pipeline_masks_holes_and_filters_each_side_of_depth_edge(profile):
    fixture = run_path(str(EXAMPLE / "build_example.py"))
    graph = Graph.load(EXAMPLE / "external_disparity_wls.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.set_branch_workers(2)
        graph.run()
        left, truth = (graph.blocks[key].last_result.outputs["image"] for key in ("left", "truth"))
        raw_valid = left != fixture["INVALID"]
        raw_error = np.abs(left[raw_valid] / fixture["SCALE"] - truth[raw_valid])
        assert raw_error.mean() > 0.34
        guided, paired = (graph.blocks[key].last_result for key in ("guided", "paired"))
        assert guided and guided.passed and paired and paired.passed
        g, p = guided.outputs, paired.outputs
        np.testing.assert_array_equal(g["valid_mask"] != 0, raw_valid)
        assert np.isnan(g["disparity"][fixture["HOLE"]]).all()
        assert g["confidence"] is NO_VALUE and g["right_support"] is NO_VALUE
        np.testing.assert_allclose(
            g["disparity"][:, fixture["EDGE"] - 1 : fixture["EDGE"] + 1],
            truth[:, fixture["EDGE"] - 1 : fixture["EDGE"] + 1],
            atol=0.04,
        )
        occluded = (slice(10, 28), slice(91, 93))
        assert p["right_support"][occluded].all()
        assert not p["confidence"][occluded].any()
        assert not p["valid_mask"][occluded].any()
        hole_projection = (slice(35, 40), slice(128, 132))
        assert not p["right_support"][hole_projection].any()
        assert not p["confidence"][hole_projection].any()
        assert np.isnan(p["disparity"][hole_projection]).all()
        assert np.min(p["confidence"][54:59, 135:145]) > 0.6
        for key in ("check_guided", "check_paired"):
            checked = graph.blocks[key].last_result
            assert checked and checked.passed
            assert checked.outputs["report"]["bad_pixels"] == 0
            assert checked.outputs["report"]["mae_px"] < 0.03
        assert ("Filtered disparity" in guided.graphics) == (profile == "debug")
    finally:
        graph.close()


@pytest.mark.parametrize("disparity", [3.25, 0, -2.5])
def test_dense_native_path_uses_signed_subpixel_matches_and_explicit_scales(disparity):
    left = np.full((8, 32), disparity * 64, np.int16)
    positive_convention = np.full(left.shape, disparity * 8, np.int16)
    tool, result = run_filter(
        left,
        positive_convention,
        options={
            "samples_per_pixel": 64,
            "right_samples_per_pixel": 8,
            "right_sign": "left_minus_right",
            "depth_discontinuity_radius": 0,
            "left_right_threshold_px": 0,
        },
    )
    assert result and result.passed, tool.error
    outputs = result.outputs
    expected_x = np.arange(left.shape[1]) - int(np.floor(disparity))
    expected = np.broadcast_to((expected_x >= 0) & (expected_x < left.shape[1]), left.shape)
    np.testing.assert_array_equal(outputs["right_support"] != 0, expected)
    np.testing.assert_array_equal(outputs["confidence"], expected)
    np.testing.assert_array_equal(outputs["valid_mask"] != 0, expected)
    np.testing.assert_allclose(outputs["disparity"][expected], disparity, atol=2e-5)
    assert np.isnan(outputs["disparity"][~expected]).all()


def test_weighted_left_only_preserves_float_holes_mask_and_arbitrary_sentinel():
    src = np.full((12, 26), -6.75, np.float64)
    src[2:6, 4:9] = -100000
    src[7:9, 15:20] = np.inf
    src[3:5, 21:24] = np.nan
    src[9, 5] = 1700
    mask = np.full(src.shape, 255, np.uint8)
    mask[9, 5] = 0
    before = src.copy()
    tool, result = run_filter(
        src, valid_mask=mask, options={"invalid_mode": "also_use_code", "invalid_code": -100000}
    )
    assert result and result.passed, tool.error
    valid = np.isfinite(src) & (src != -100000) & (mask != 0)
    np.testing.assert_array_equal(result.outputs["valid_mask"] != 0, valid)
    np.testing.assert_allclose(result.outputs["disparity"][valid], -6.75, atol=2e-5)
    assert np.isnan(result.outputs["disparity"][~valid]).all()
    np.testing.assert_array_equal(src, before)


def test_missing_right_support_emits_zero_confidence_and_a_judgement():
    left = np.ones((10, 20), np.float32)
    right = np.full(left.shape, np.nan, np.float32)
    tool, result = run_filter(left, right)
    assert result and not result.passed, tool.error
    assert result.fail_reasons[0]["code"] == "external_disparity_wls_no_valid_pixels"
    assert not result.outputs["confidence"].any()
    assert not result.outputs["right_support"].any()
    assert np.isnan(result.outputs["disparity"]).all()


def test_selected_outputs_preserve_left_frame_and_ticket():
    left = np.ones((8, 14), np.float32)
    spatial = frame(left.shape[::-1], "rectified-left")
    ticket = TicketClock("wls-test").issue("stereo", "fixed", "sample-1")
    packet = ValuePacket(left, {"spatial": spatial, "ticket": ticket})
    guide = ValuePacket(np.uint8(left * 30), {"spatial": spatial})
    selection = {
        "model": "image_region_v1",
        "image_size": [14, 8],
        "shape": "rectangle",
        "rectangle": [2, 2, 8, 4],
    }
    tool, result = run_filter(
        packet, guide=guide, region=selection, options={"alignment": "same_frame"}
    )
    assert result and result.passed, tool.error
    for name in ("disparity", "valid_mask"):
        assert result.payloads[name]["spatial"] == spatial
        assert result.payloads[name]["ticket"] == ticket
    expected = np.zeros(left.shape, np.uint8)
    expected[2:6, 2:10] = 255
    np.testing.assert_array_equal(result.outputs["valid_mask"], expected)

    other = ValuePacket(guide.value, {"spatial": frame(left.shape[::-1], "rectified-right")})
    tool, result = run_filter(packet, guide=other, options={"alignment": "same_frame"})
    assert result is None and "share a pixel frame" in tool.error


@pytest.mark.parametrize(
    ("disparity", "options", "error"),
    [
        (DepthArray(np.ones((2, 4), np.float32)), {}, "monochrome"),
        (np.full((2, 4), 3000.0, np.float32), {}, "16-bit range"),
        (
            np.zeros((2, 4), np.uint16),
            {"invalid_mode": "also_use_code", "invalid_code": -1},
            "representable",
        ),
        (np.zeros((2, 4), np.uint16), {"minimum_confidence": 0.5}, "Connect right disparity"),
    ],
)
def test_incompatible_inputs_fail_before_native_filter(disparity, options, error):
    tool, result = run_filter(disparity, options=options)
    assert result is None
    assert error in tool.error
