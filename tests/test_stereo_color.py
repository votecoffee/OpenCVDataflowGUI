"""Qualify stereo photometric alignment and its effect on held-out disparity."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow import stereo_color
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.calibration_revisions import digest
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.stereo3d import STEREO_GRID
from opencv_dataflow.stereo_color import MatchStereoColor
from opencv_dataflow.tickets import TicketClock

PROJECT = (
    Path(__file__).resolve().parents[1] / "examples/opencv tools/stereo_color/stereo_color.json"
)


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_calibrated_comparison_improves_held_out_native_bm(profile):
    graph = Graph.load(PROJECT, builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        for name in ("rectify", "guide", "correct", "raw_bm_check", "corrected_bm_check"):
            block = graph.blocks[name]
            assert block.last_result and block.last_result.passed, (name, block.error)
        rectified = graph.blocks["rectify"].last_result
        corrected = graph.blocks["correct"].last_result
        exposure = corrected.outputs["report"]
        assert exposure["usable_pairs"] == exposure["inlier_pairs"] == 9752
        assert exposure["mae_after"] < 0.1 * exposure["mae_before"]
        assert corrected.payloads["right"]["spatial"] == rectified.payloads["right"]["spatial"]
        assert corrected.payloads["right"]["ticket"] == rectified.payloads["right"]["ticket"]
        x0, y0, width, height = graph.blocks["check_region"].parameters["rectangle"]
        iy, ix = slice(y0, y0 + height), slice(x0, x0 + width)
        assert not np.any(corrected.outputs["fit_mask"][iy, ix])
        reference = np.asarray(rectified.outputs["left"][iy, ix], float)
        ix_right = slice(x0 - 8, x0 + width - 8)
        before = np.asarray(rectified.outputs["right"][iy, ix_right], float)
        after = np.asarray(corrected.outputs["right"][iy, ix_right], float)
        assert np.mean(np.abs(reference - after)) < 0.1 * np.mean(np.abs(reference - before))
        np.testing.assert_array_equal(
            graph.blocks["raw_bm"].last_result.outputs["support_mask"],
            graph.blocks["corrected_bm"].last_result.outputs["support_mask"],
        )
        raw = graph.blocks["raw_bm_check"].last_result.outputs["report"]
        repaired = graph.blocks["corrected_bm_check"].last_result.outputs["report"]
        assert raw["reference_valid_pixels"] == repaired["reference_valid_pixels"] == 9540
        assert raw["evaluated_pixels"] == 9086 and repaired["evaluated_pixels"] == 9360
        assert repaired["mae_px"] < 0.45 * raw["mae_px"]
        assert raw["bad_pixels"] == repaired["bad_pixels"] == 0
        assert bool(corrected.graphics) == (profile == "debug")
    finally:
        graph.close()


def _rectification(size):
    return {
        "model": "stereo_rectification_v1",
        "image_size": list(size),
        "P2": [[30, 0, 12, -15], [0, 30, 7, 0], [0, 0, 1, 0]],
    }


def test_fractional_negative_disparity_requires_both_taps_and_keeps_rgb_order_and_alpha(
    monkeypatch,
):
    monkeypatch.setattr(stereo_color, "_BAND_PIXELS", 43)
    rng = np.random.default_rng(731)
    h, w = 7, 19
    right = rng.integers(8192, 36000, (h, w, 4), np.uint16)
    right_before = right.copy()
    signal = right[..., 2::-1].astype(float) / 65535
    gain, bias = np.array([1.10, 0.70, 1.30]), np.array([0.05, 0.03, 0.02])
    reference = np.full((h, w, 3), 0.4, np.float32)
    reference[:, :-1] = (signal[:, :-1] + signal[:, 1:]) * 0.5 * gain + bias
    guide = np.full((h, w), -0.5, np.float32)
    right_mask = np.full((h, w), 255, np.uint8)
    right_mask[:, 7] = 0
    left_mask = np.full((h, w), 255, np.uint8)
    left_mask[0, :] = 0
    rec = _rectification((w, h))
    rec["roi_right"] = [0, 0, w - 1, h]
    block = MatchStereoColor(parameters={"minimum_samples": 8, "residual_tolerance": 0.00002})
    result = block.execute(
        {
            "left": ImageArray(reference, "rgb"),
            "right": ImageArray(right, "bgra"),
            "disparity": guide,
            "rectification": rec,
            "left_mask": left_mask,
            "right_mask": right_mask,
        }
    )
    assert result and result.passed, block.error
    fit = result.outputs["report"]
    np.testing.assert_allclose(fit["right_to_left_gain"], gain, atol=1e-7)
    np.testing.assert_allclose(fit["right_to_left_bias"], bias, atol=1e-7)
    found = result.outputs["fit_mask"] != 0
    expected_support = np.ones((h, w), bool)
    expected_support[0, :] = False
    expected_support[:, [6, 7, w - 2, w - 1]] = False
    np.testing.assert_array_equal(found, expected_support)
    output = result.outputs["right"]
    assert output.dtype == np.uint16 and output.color_format == "bgra"
    np.testing.assert_array_equal(output[..., 3], right_before[..., 3])
    np.testing.assert_array_equal(output[:, [7, w - 1], :], right_before[:, [7, w - 1], :])
    expected = np.rint((signal * gain + bias) * 65535).astype(np.uint16)
    write_mask = right_mask != 0
    write_mask[:, -1] = False
    np.testing.assert_allclose(output[..., 2::-1][write_mask], expected[write_mask], atol=1)
    np.testing.assert_array_equal(right, right_before)


def test_gain_only_rejects_bad_matches_and_saturated_samples():
    rng = np.random.default_rng(670)
    source = rng.uniform(0.2, 0.6, (20, 30)).astype(np.float32)
    expected_gain = 1.35
    reference = source * expected_gain
    corrupted = (slice(3, 8), slice(8, 20))
    reference[corrupted] = 0.93
    source[0, :7] = 0
    block = MatchStereoColor(parameters={"model": "gain_only", "residual_tolerance": 0.01})
    result = block.execute({"left": reference, "right": source, "disparity": np.zeros_like(source)})
    assert result and result.passed, block.error
    report = result.outputs["report"]
    assert report["sampled_pairs"] == 600 and report["saturated_pairs"] == 7
    assert report["inlier_pairs"] == 533
    assert report["right_to_left_gain"] == pytest.approx([expected_gain], abs=1e-7)
    assert report["right_to_left_bias"] == [0.0]
    assert not np.any(result.outputs["fit_mask"][corrupted])


@pytest.mark.parametrize(
    ("fault", "reason"),
    [
        ("no_support", "stereo_color_insufficient_samples"),
        ("flat", "stereo_color_insufficient_signal"),
        ("inverted", "stereo_color_implausible_model"),
    ],
)
def test_unreliable_fit_reports_reason_without_emitting_corrected_image(fault, reason):
    right = np.linspace(0.2, 0.8, 240, dtype=np.float32).reshape(12, 20)
    disparity = np.zeros_like(right)
    left = right.copy()
    if fault == "no_support":
        disparity.fill(np.nan)
    elif fault == "flat":
        right.fill(0.5)
    else:
        left = 1 - right
    block = MatchStereoColor()
    result = block.execute({"left": left, "right": right, "disparity": disparity})
    assert result and not result.passed and not block.error
    assert result.outputs["right"] is NO_VALUE and not block.current_emissions.get("right")
    assert [record["code"] for record in result.fail_reasons] == [reason]


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("tickets", "same ticket"),
        ("left_grid", "different left grid"),
        ("seal", "same sealed left rectification"),
        ("vertical", "horizontal rectified baseline"),
        ("count", "Maximum samples must cover minimum samples"),
    ],
)
def test_invalid_stereo_contracts_fail_explicitly(fault, message):
    w, h = 25, 12
    array = np.full((h, w), 128, np.uint8)
    guide = np.zeros((h, w), np.float32)
    data = {"left": array, "right": array, "disparity": guide}
    settings = {}
    if fault == "tickets":
        clock = TicketClock()
        for name in ("left", "right"):
            data[name] = ValuePacket(array, {"ticket": clock.issue(name)})
    elif fault == "left_grid":
        data["left"] = ValuePacket(array, {"spatial": frame((w, h), "left")})
        data["disparity"] = ValuePacket(guide, {"spatial": frame((w, h), "other")})
    elif fault == "seal":
        data["rectification"] = rec = _rectification((w, h))
        for name in ("left", "right"):
            grid = frame((w, h), name)
            stamp = {"sha256": digest(rec), "side": name, "spatial": grid}
            data[name] = ValuePacket(array, {"spatial": grid, STEREO_GRID: stamp})
    elif fault == "vertical":
        data["rectification"] = rec = _rectification((w, h))
        rec["P2"][1][3] = -15
        rec["P2"][0][3] = 0
    else:
        settings["maximum_samples"] = 100
    block = MatchStereoColor(parameters=settings)
    assert block.execute(data) is None and block.state == "error"
    assert message in block.error
