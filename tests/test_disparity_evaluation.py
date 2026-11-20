"""Check region, coverage, masks, confidence and native stereo interoperability."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.disparity_evaluation import EvaluateDisparity
from opencv_dataflow.images import DepthArray
from opencv_dataflow.payloads import ValuePacket, frame

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples/opencv tools/disparity_reference/disparity_reference.json"
)


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_example_reports_exact_entire_and_regional_results(profile):
    graph = Graph.load(EXAMPLE, builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        for key in ("whole", "selected"):
            assert graph.blocks[key].last_result.passed, graph.blocks[key].error
        whole = graph.blocks["whole"].last_result.outputs
        report = whole["report"]
        assert report["reference_valid_pixels"] == 11520
        assert report["evaluated_pixels"] == 9216
        assert report["coverage"] == 0.8
        assert report["bad_pixels"] == 4608
        assert report["bad_fraction"] == 0.5
        assert report["bad_or_missing_fraction"] == 0.6
        assert report["bias_px"] == 0.5
        assert report["mae_px"] == 1.5
        assert report["mse_px2"] == 3.5
        assert report["confidence_slices"]["below_cutoff"]["bad_fraction"] == 1
        assert report["confidence_slices"]["at_or_above_cutoff"]["bad_fraction"] == 1 / 3
        assert report["confidence_slices"]["unknown"]["bad_fraction"] is None
        for name, row in (
            ("evaluated_mask", [0, 255, 255, 255, 0, 255]),
            ("bad_mask", [0, 0, 0, 255, 0, 255]),
            ("missing_mask", [0, 0, 0, 0, 255, 0]),
            ("absolute_error", [np.nan, 0, 1, 2, np.nan, 3]),
        ):
            expected = np.broadcast_to(np.repeat(row, 24), (96, 144))
            np.testing.assert_array_equal(whole[name], expected)
        selected = graph.blocks["selected"].last_result.outputs["report"]
        assert selected["selected_pixels"] == selected["evaluated_pixels"] == 4608
        assert selected["coverage"] == 1 and selected["bad_fraction"] == 0
        assert selected["mae_px"] == 0.5 and selected["max_error_px"] == 1
        result = graph.blocks["whole"].last_result
        estimate = graph.blocks["estimate"].last_result
        for field in ("spatial", "ticket"):
            assert result.payloads["absolute_error"][field] == estimate.payloads["image"][field]
        if profile == "debug":
            np.testing.assert_array_equal(
                result.graphics["disparity_error"][0, [0, 96]], [[0, 0, 0], [255, 0, 255]]
            )
        else:
            assert not result.graphics
    finally:
        graph.close()


def test_matches_native_bad_pixel_and_mse_on_common_fixed_point_domain():
    # Use OpenCV's float representation of 16x raw values for nonsaturating squaring.
    gt = np.array([[-32, 16, 24], [32, 48, -16]], dtype=np.int16)
    estimate = gt + np.array([[0, 1, -16], [17, -32, 49]], dtype=np.int16)
    result = EvaluateDisparity().execute({"estimate": estimate, "reference": gt})
    assert result and result.passed
    reference_raw, estimate_raw = (x.astype(np.float32) for x in (gt, estimate))
    roi = (0, 0, 3, 2)
    report = result.outputs["report"]
    assert report["mse_px2"] == pytest.approx(
        cv2.ximgproc.computeMSE(reference_raw, estimate_raw, roi)
    )
    assert 100 * report["bad_fraction"] == pytest.approx(
        cv2.ximgproc.computeBadPixelPercent(reference_raw, estimate_raw, roi, 17)
    )


def test_reference_coverage_and_bad_gate_report_separate_failures_and_unknown_confidence():
    prediction = np.array([[-2, 0, np.nan, 5, 9, 5, 100]], dtype=np.float64)
    truth = np.array([[0, 0, -2, 4, 4, 4, np.inf]], dtype=np.float64)
    confidence = np.array([[np.nan, 0.5, 0, 1, 0.25, 0, 99]], dtype=np.float64)
    block = EvaluateDisparity(parameters={"minimum_coverage": 0.9, "maximum_bad_fraction": 0.2})
    result = block.execute(
        {
            "estimate": prediction,
            "reference": truth,
            "estimate_valid": np.array([[255, 255, 255, 255, 0, 255, 255]], dtype=np.uint8),
            "reference_valid": np.array([[255, 255, 255, 255, 255, 0, 255]], dtype=np.uint8),
            "confidence": confidence,
        }
    )
    assert result is not None and not result.passed and not block.error
    assert {reason["code"] for reason in result.fail_reasons} == {
        "disparity_coverage_below_minimum",
        "disparity_bad_fraction_above_maximum",
    }
    report = result.outputs["report"]
    assert report["reference_valid_pixels"] == 5
    assert report["coverage"] == 3 / 5
    assert report["bad_fraction"] == 1 / 3
    assert result.score == 2 / 5
    slices = report["confidence_slices"]
    assert slices["unknown"]["evaluated_pixels"] == slices["unknown"]["bad_pixels"] == 1
    assert slices["at_or_above_cutoff"]["evaluated_pixels"] == 2
    assert slices["below_cutoff"]["evaluated_pixels"] == 0


@pytest.mark.parametrize(
    ("prediction", "reference", "code"),
    [
        ([1, 2], [np.nan, np.nan], "disparity_no_reference"),
        ([np.nan, np.nan], [1, 2], "disparity_no_overlap"),
    ],
)
def test_absent_reference_and_absent_estimates_keep_distinct_reports(prediction, reference, code):
    block = EvaluateDisparity()
    result = block.execute(
        {"estimate": np.array([prediction], float), "reference": np.array([reference], float)}
    )
    assert result and not result.passed
    assert result.fail_reasons[0]["code"] == code
    outputs = result.outputs
    assert outputs["report"]["mae_px"] is None
    assert outputs["report"]["coverage"] == (None if code == "disparity_no_reference" else 0)
    assert np.isnan(outputs["absolute_error"]).all()
    assert not outputs["bad_mask"].any()


def test_region_mask_intersection_and_known_grid_mismatch():
    grid = frame((4, 4), "left")
    shifted = frame((4, 4), "left")
    shifted["to_root"][0][2] = 1
    data = np.full((4, 4), 4.0, np.float32)
    region = {
        "model": "image_region_v1",
        "shape": "circle",
        "center": [1, 1],
        "radius": 1,
        "image_size": [4, 4],
        "coordinate_context": grid,
    }
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[:2] = 255
    inputs = {
        "estimate": ValuePacket(data, {"spatial": grid}),
        "reference": ValuePacket(data, {"spatial": shifted}),
        "mask": ValuePacket(mask, {"spatial": grid}),
        "region": region,
    }
    block = EvaluateDisparity()
    assert block.execute(inputs) is None and "same pixel frame" in block.error
    block = EvaluateDisparity(parameters={"alignment": "pixel_grid"})
    result = block.execute(inputs)
    assert result and result.passed
    assert result.outputs["report"]["selected_pixels"] == 4
    wrong_region = {**region, "coordinate_context": shifted}
    assert block.execute({**inputs, "region": wrong_region}) is None
    assert "coordinate frame" in block.error


@pytest.mark.parametrize(
    ("overrides", "parameters", "message"),
    [
        ({"reference": np.ones((1, 2), np.float32)}, {}, "same pixel dimensions"),
        ({"reference": DepthArray(np.ones((2, 2), np.float32))}, {}, "monochrome"),
        ({"estimate_valid": np.ones((2, 2), np.uint16)}, {}, "uint8 pixels"),
        ({"confidence": np.full((2, 2), 1.1, np.float32)}, {}, "between 0 and 1"),
        (
            {},
            {"reference_invalid_mode": "also_use_code", "reference_invalid_code": 0.5},
            "representable",
        ),
    ],
)
def test_reports_invalid_rasters_and_settings(overrides, parameters, message):
    raw = np.zeros((2, 2), np.uint16)
    block = EvaluateDisparity(parameters=parameters)
    assert block.execute({"estimate": raw, "reference": raw, **overrides}) is None
    assert message in block.error


def test_native_sgbm_wls_outputs_integrate_on_known_eight_pixel_pair():
    registry = builtin_registry()
    pair = registry.create("input.stereo_demo", block_id="pair").execute({})
    matcher = registry.create("3d.disparity", parameters={"num_disparities": 16, "wls": True})
    stereo = matcher.execute(
        {key: ValuePacket(pair.outputs[key], pair.payloads[key]) for key in ("left", "right")}
    )
    assert stereo and stereo.passed, matcher.error
    shape = stereo.outputs["disparity"].shape
    evaluator = EvaluateDisparity(
        parameters={"minimum_coverage": 0.8, "maximum_bad_fraction": 0.01}
    )
    result = evaluator.execute(
        {
            "estimate": ValuePacket(stereo.outputs["disparity"], stereo.payloads["disparity"]),
            "estimate_valid": ValuePacket(
                stereo.outputs["valid_mask"], stereo.payloads["valid_mask"]
            ),
            "reference": np.full(shape, 128, np.int16),
            "confidence": ValuePacket(stereo.outputs["confidence"], stereo.payloads["confidence"]),
        }
    )
    assert result and result.passed, evaluator.error
    report = result.outputs["report"]
    expected = stereo.outputs["valid_mask"] != 0
    assert report["coverage"] == expected.mean()
    assert report["mae_px"] == pytest.approx(
        np.abs(stereo.outputs["disparity"][expected] - 8).astype(float).mean()
    )
    assert report["mae_px"] < 0.02
    assert (
        sum(s["evaluated_pixels"] for s in report["confidence_slices"].values()) == expected.sum()
    )
