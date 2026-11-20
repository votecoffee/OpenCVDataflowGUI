"""Qualify stereo disparity range derivation, support masks and legacy execution."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.calibration_revisions import digest
from opencv_dataflow.core import Graph
from opencv_dataflow.disparity_search import plan_disparity_search
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame

TYPE = "3d.disparity"
EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples/opencv tools/stereo_disparity_search/stereo_disparity_search.json"
)


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_example_limits_search_and_recovers_known_depth(profile):
    graph = Graph.load(EXAMPLE, builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        for key, count, expected_range in (("manual", 34865, 64), ("depth", 44269, 16)):
            block = graph.blocks[key]
            result = block.last_result
            assert result and result.passed, block.error
            outputs = result.outputs
            report = outputs["search_report"]
            assert report["num_disparities"] == expected_range
            assert np.count_nonzero(outputs["support_mask"]) == count
            support = outputs["support_mask"] != 0
            valid = outputs["valid_mask"] != 0
            assert np.all(support[valid])
            assert np.isnan(outputs["disparity"][~support]).all()
            assert np.nanmedian(outputs["disparity"]) == pytest.approx(8, abs=0.06)
            assert report["support_fraction"] == pytest.approx(support.mean())
        depth = graph.blocks["depth"].last_result.outputs["search_report"]
        assert depth["min_disparity"] == 5 and depth["max_disparity"] == 20
        assert depth["expected_disparity_px"] == pytest.approx([24 / 4.4, 24 / 2.4])
        assert depth["depth_units"] == "m"
        assert np.nanmedian(graph.blocks["cloud"].last_result.outputs["depth"]) == pytest.approx(
            3.0, abs=0.03
        )
    finally:
        graph.close()


def shifted_pair(disparity, width=112, height=54):
    rng = np.random.default_rng(11)
    left = cv2.GaussianBlur(rng.integers(0, 256, (height, width), np.uint8), (3, 3), 0)
    right = np.zeros_like(left)
    if disparity > 0:
        right[:, :-disparity] = left[:, disparity:]
    else:
        right[:, -disparity:] = left[:, :disparity]
    return left, right


def rectification_record(w, h, roi1=None, roi2=None, sign=1):
    return {
        "model": "stereo_rectification_v1",
        "image_size": [w, h],
        "roi_left": roi1,
        "roi_right": roi2,
        "units": "mm",
        "P2": [[200, 0, 0, -sign * 24000], [0, 200, 0, 0], [0, 0, 1, 0]],
        "Q": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 200], [0, 0, sign / 120, -sign * 3 / 120]],
    }


def test_signed_range_and_mask_footprints_match_independent_search_windows():
    minimum, count, block_size = -18, 16, 5
    a, b = shifted_pair(-8)
    h, w = a.shape
    left_roi, right_roi = [2, 1, w - 4, h - 2], [3, 2, w - 8, h - 4]
    lm, rm = np.full_like(a, 255), np.full_like(b, 255)
    lm[10:13, 17:23] = 0
    rm[16:20, 43:49] = 0
    block = builtin_registry().create(
        TYPE, parameters={"min_disparity": minimum, "num_disparities": count}
    )
    result = block.execute(
        {
            "left": a,
            "right": b,
            "left_mask": lm,
            "right_mask": rm,
            "rectification": rectification_record(w, h, left_roi, right_roi),
        }
    )
    assert result and result.passed, block.error
    outputs = result.outputs
    # Enumerate the complete windows independently, then intersect the native ROI.
    active = np.zeros((2, h, w), bool)
    for grid, roi, mask in zip(active, (left_roi, right_roi), (lm, rm), strict=True):
        x, y, width, height = roi
        grid[y : y + height, x : x + width] = mask[y : y + height, x : x + width] != 0
    x, y, width, height = cv2.getValidDisparityROI(left_roi, right_roi, minimum, count, block_size)
    expected = np.zeros((h, w), bool)
    radius = block_size // 2
    for yy in range(y, y + height):
        ys = slice(yy - radius, yy + radius + 1)
        for xx in range(x, x + width):
            expected[yy, xx] = active[0, ys, xx - radius : xx + radius + 1].all() and all(
                radius <= xx - d < w - radius
                and active[1, ys, xx - d - radius : xx - d + radius + 1].all()
                for d in range(minimum, minimum + count)
            )
    np.testing.assert_array_equal(outputs["support_mask"] != 0, expected)
    assert outputs["search_report"]["valid_roi"] != outputs["search_report"]["opencv_roi"]
    assert np.isnan(outputs["disparity"][~expected]).all()
    assert np.nanmedian(outputs["disparity"]) == pytest.approx(-8, abs=0.1)


@pytest.mark.parametrize(("baseline", "bounds", "minimum"), [(1, (8, 13), 8), (-1, (-7, -2), -7)])
def test_depth_from_q_keeps_units_sign_and_principal_point_offset(baseline, bounds, minimum):
    registry = builtin_registry()
    block = registry.create(
        TYPE, parameters={"search_mode": "depth", "minimum_depth": 2400, "maximum_depth": 4800}
    )
    a, b = shifted_pair(8 if baseline == 1 else -4)
    rec = rectification_record(*a.shape[::-1], sign=baseline)
    plan = plan_disparity_search(
        {"left": a, "right": b, "rectification": rec}, block.parameters, a.shape, 5
    )
    assert plan.minimum == minimum and plan.count == 16
    assert plan.report["expected_disparity_px"] == pytest.approx(bounds)
    assert plan.report["depth_units"] == "mm"


@pytest.mark.parametrize("algorithm", ["bm", "sgbm"])
def test_matching_with_support_and_wls_masks_all_diagnostic_outputs(algorithm):
    a, b = shifted_pair(8, width=160, height=75)
    rm = np.full_like(b, 255)
    rm[25:43, 67:75] = 0
    block = builtin_registry().create(
        TYPE, parameters={"algorithm": algorithm, "num_disparities": 16, "wls": True}
    )
    result = block.execute({"left": a, "right": b, "right_mask": rm})
    assert result and result.passed, block.error
    out = result.outputs
    excluded = out["support_mask"] == 0
    assert excluded.any()
    assert np.all(out["valid_mask"][excluded] == 0)
    assert np.all(out["confidence"][excluded] == 0)
    assert np.all(out["preview"][excluded] == 0)
    assert np.isnan(out["disparity"][excluded]).all()
    assert np.nanmedian(out["disparity"]) == pytest.approx(8, abs=0.15)


def test_empty_support_fails_before_constructing_a_matcher(monkeypatch):
    def unexpected(**kwargs):
        pytest.fail("No matching work is needed with empty support")

    monkeypatch.setattr(cv2, "StereoSGBM_create", unexpected)
    a, b = shifted_pair(8)
    block = builtin_registry().create(TYPE, parameters={"num_disparities": 16})
    result = block.execute({"left": a, "right": b, "right_mask": np.zeros_like(b)})
    assert result and not result.passed
    assert not result.outputs["support_mask"].any()
    assert np.isnan(result.outputs["disparity"]).all()
    assert result.fail_reasons[0]["code"] == "stereo_search_no_support"


def test_rejects_wrong_mask_frame_and_mismatched_sealed_rectification():
    a, b = shifted_pair(8)
    size = a.shape[::-1]
    rec = rectification_record(*size)
    packet = {}
    for value, side in ((a, "left"), (b, "right")):
        spatial = frame(size, side)
        stamp = {"sha256": digest(rec), "side": side, "spatial": spatial}
        packet[side] = ValuePacket(value, {"spatial": spatial, "stereo_calibration_grid": stamp})
    block = builtin_registry().create(TYPE, parameters={"num_disparities": 16})
    wrong = ValuePacket(np.full_like(a, 255), {"spatial": frame(size, "right")})
    assert block.execute({**packet, "rectification": rec, "left_mask": wrong}) is None
    assert "coordinate frame differs" in block.error
    alternate = {**rec, "roi_left": [1, 1, size[0] - 2, size[1] - 2]}
    assert block.execute({**packet, "rectification": alternate}) is None
    assert "differs from the sealed stereo images" in block.error


def test_legacy_manual_path_keeps_disparity_and_optional_support():
    a, b = shifted_pair(8)
    block = builtin_registry().create(TYPE, parameters={"num_disparities": 16})
    result = block.execute({"left": a, "right": b})
    assert result and result.passed
    assert result.outputs["support_mask"] is NO_VALUE
    assert result.outputs["search_report"]["valid_roi"] is None
    assert np.nanmedian(result.outputs["disparity"]) == pytest.approx(8, abs=0.1)
    block.parameters["search_mode"] = "depth"
    assert block.execute({"left": a, "right": b}) is None
    assert "requires calibrated stereo rectification" in block.error


def test_depth_search_padding_preserves_valid_interval_near_fixed_point_ceiling():
    w, h = 4096, 12
    rectification = rectification_record(w, h)
    block = builtin_registry().create(
        TYPE,
        parameters={
            "search_mode": "depth",
            "minimum_depth": 24000 / (2046.8 - 3),
            "maximum_depth": 24000 / (2045.1 - 3),
        },
    )
    image = np.zeros((h, w), np.uint8)
    search = plan_disparity_search(
        {"rectification": rectification, "left": image, "right": image},
        block.parameters,
        image.shape,
        5,
    )
    assert search.minimum == 2032 and search.count == 16
    assert search.report["expected_disparity_px"] == pytest.approx([2045.1, 2046.8])
    assert search.report["max_disparity"] == 2047
