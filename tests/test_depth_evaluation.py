"""Check metric normalization, mask accounting and depth-step residuals."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.depth_evaluation import EvaluateDepth
from opencv_dataflow.examples3d import stereo_cloud_graph
from opencv_dataflow.images import DepthArray
from opencv_dataflow.payloads import ValuePacket, frame

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples/opencv tools/depth_reference/depth_reference.json"
)


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_graph_resolves_mixed_units_and_preserves_the_known_depth_step(profile):
    graph = Graph.load(EXAMPLE, builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        shape = (80, 128)
        expected = np.zeros(shape)
        expected[:, 64] = -500
        expected[20:30, 5:15] = 15.625
        expected[42:52, 88:98] = -31.25
        missing = np.zeros(shape, bool)
        missing[30:38, 100:110] = True
        estimate_only = np.zeros(shape, bool)
        estimate_only[12:20, 10:20] = True
        overlap = ~(missing | estimate_only)
        expected[~overlap] = np.nan
        estimate = graph.blocks["estimate"].last_result
        for name in ("whole", "step"):
            block = graph.blocks[name]
            result = block.last_result
            assert result and result.passed, block.error
            outputs = result.outputs
            report = outputs["report"]
            scoped = overlap.copy()
            if name == "step":
                scoped[:, :62] = False
                scoped[:, 66:] = False
            assert (report["estimate_units"], report["reference_units"]) == ("m", "mm")
            assert report["units"] == "mm"
            assert report["evaluated_pixels"] == int(scoped.sum())
            np.testing.assert_array_equal(outputs["residual"], np.where(scoped, expected, np.nan))
            np.testing.assert_array_equal(outputs["overlap_mask"], scoped * 255)
            np.testing.assert_array_equal(
                outputs["bad_mask"], (scoped & (abs(expected) > 20)) * 255
            )
            assert report["mae_mm"] == pytest.approx(abs(expected[scoped]).mean())
            assert report["bias_mm"] == pytest.approx(expected[scoped].mean())
            assert report["rmse_mm"] == pytest.approx(np.sqrt(np.mean(expected[scoped] ** 2)))
            for field in ("spatial", "ticket"):
                assert result.payloads["residual"][field] == estimate.payloads["depth"][field]
        whole = graph.blocks["whole"].last_result
        report = whole.outputs["report"]
        assert report["coverage"] == 10080 / 10160
        assert report["validity_iou"] == 10080 / 10240
        assert report["bad_pixels"] == 180
        np.testing.assert_array_equal(whole.outputs["missing_mask"], missing * 255)
        np.testing.assert_array_equal(whole.outputs["estimate_only_mask"], estimate_only * 255)
        step = graph.blocks["step"].last_result.outputs["report"]
        assert (step["coverage"], step["bad_fraction"], step["mae_mm"]) == (1, 0.25, 125)
        if profile == "debug":
            graphic = whole.graphics["depth_error"]
            np.testing.assert_array_equal(graphic[30, 100], [255, 0, 255])
            np.testing.assert_array_equal(graphic[12, 10], [128, 128, 128])
        else:
            assert not whole.graphics
    finally:
        graph.close()


def test_raw_uint16_and_typed_units_compare_at_inclusive_tolerance():
    estimate = np.array([[500, 503, 497]], dtype=np.uint16)
    truth = DepthArray([[1000, 1000, 1000]], "mm")
    block = EvaluateDepth(
        parameters={
            "estimate_units": "mm",
            "estimate_units_per_sample": 2,
            "reference_units": "m",
            "reference_units_per_sample": 20,
            "error_tolerance_mm": 6,
            "minimum_coverage": 1,
            "maximum_bad_fraction": 0,
        }
    )
    result = block.execute({"estimate": estimate, "reference": truth})
    assert result and result.passed, block.error
    report = result.outputs["report"]
    assert report["reference_mm_per_sample"] == 1
    assert report["estimate_mm_per_sample"] == 2
    assert report["bad_pixels"] == 0
    assert report["mae_mm"] == 4
    assert report["mse_mm2"] == 24
    np.testing.assert_array_equal(result.outputs["residual"], [[0, 6, -6]])
    np.testing.assert_array_equal(estimate, [[500, 503, 497]])


def test_mask_selection_validity_and_quality_gates_have_separate_denominators():
    estimate = DepthArray([[1100, 1000, 1000, 900, -1, 1000, 1000]], "mm")
    reference = DepthArray([[1000, np.inf, 1000, 1000, 1000, 1000, 1000]], "mm")
    result = EvaluateDepth(
        parameters={"minimum_coverage": 0.8, "maximum_bad_fraction": 0.1}
    ).execute(
        {
            "estimate": estimate,
            "reference": reference,
            "mask": np.array([[1, 1, 1, 1, 1, 0, 1]], np.uint8),
            "estimate_valid": np.array([[1, 1, 0, 1, 1, 1, 1]], np.uint8),
            "reference_valid": np.array([[1, 1, 1, 0, 1, 1, 1]], np.uint8),
        }
    )
    assert result and not result.passed
    assert {item["code"] for item in result.fail_reasons} == {
        "depth_coverage_below_minimum",
        "depth_bad_fraction_above_maximum",
    }
    outputs = result.outputs
    report = outputs["report"]
    assert report["selected_pixels"] == 6
    assert report["coverage"] == 0.5 and report["validity_iou"] == 1 / 3
    assert report["bad_fraction"] == 0.5 and result.score == 0.25
    np.testing.assert_array_equal(outputs["missing_mask"], [[0, 0, 255, 0, 255, 0, 0]])
    np.testing.assert_array_equal(outputs["estimate_only_mask"], [[0, 255, 0, 255, 0, 0, 0]])


@pytest.mark.parametrize(
    ("estimate", "truth", "coverage", "iou", "code"),
    [
        ([0, np.nan], [10, 20], 0, 0, "depth_no_overlap"),
        ([10, 0], [np.nan, -1], None, 0, "depth_no_reference"),
        ([np.inf, 0], [0, -1], None, None, "depth_no_reference"),
    ],
)
def test_no_comparison_samples_keeps_masks_and_undefined_statistics(
    estimate, truth, coverage, iou, code
):
    result = EvaluateDepth().execute(
        {"estimate": DepthArray([estimate], "mm"), "reference": DepthArray([truth], "mm")}
    )
    assert result and not result.passed
    assert result.fail_reasons[0]["code"] == code
    assert result.outputs["report"]["coverage"] == coverage
    assert result.outputs["report"]["validity_iou"] == iou
    assert result.outputs["report"]["mae_mm"] is None
    assert not result.outputs["overlap_mask"].any()
    assert np.isnan(result.outputs["residual"]).all()


def test_same_frame_rejects_shift_and_pixel_grid_still_rejects_foreign_region():
    depth = DepthArray(np.ones((2, 2)), "m")
    estimate_grid = frame((2, 2), "depth")
    other_grid = frame((2, 2), "depth")
    other_grid["to_root"][1][2] = 1
    inputs = {
        "estimate": ValuePacket(depth, {"spatial": estimate_grid}),
        "reference": ValuePacket(depth, {"spatial": other_grid}),
    }
    strict = EvaluateDepth()
    assert strict.execute(inputs) is None and "same pixel frame" in strict.error
    pixel_grid = EvaluateDepth(parameters={"alignment": "pixel_grid"})
    assert pixel_grid.execute(inputs).passed
    region = {
        "model": "image_region_v1",
        "shape": "rectangle",
        "rectangle": [0, 0, 1, 1],
        "image_size": [2, 2],
        "coordinate_context": other_grid,
    }
    assert pixel_grid.execute({**inputs, "region": region}) is None
    assert "coordinate frame" in pixel_grid.error


@pytest.mark.parametrize(
    ("reference", "extras", "message"),
    [
        (DepthArray([[1, 1]], "m"), {}, "same pixel dimensions"),
        (DepthArray(np.ones((2, 2)), "unitless"), {}, "meters or millimeters"),
        (
            DepthArray(np.ones((2, 2)), "m"),
            {"reference_valid": np.ones((2, 2), np.uint16)},
            "uint8 pixels",
        ),
        (np.full((2, 2), 1e250), {}, "numeric range"),
    ],
)
def test_rejects_incompatible_depths_and_numeric_overflow(reference, extras, message):
    block = EvaluateDepth()
    estimate = DepthArray(np.ones((2, 2)), "m")
    assert block.execute({"estimate": estimate, "reference": reference, **extras}) is None
    assert message in block.error


def test_accepts_native_stereo_reprojection_against_known_three_meter_plane():
    registry = builtin_registry()
    graph = stereo_cloud_graph(registry)
    try:
        block = graph.add(
            registry.create(
                "3d.evaluate_depth",
                block_id="evaluate",
                parameters={
                    "alignment": "pixel_grid",
                    "minimum_coverage": 0.85,
                    "error_tolerance_mm": 60,
                    "maximum_bad_fraction": 0.01,
                },
            )
        )
        # This synthetic rig is fronto-parallel and the truth is constant across the left grid.
        graph.connect("pair", "depth", block.id, "reference")
        graph.connect("cloud", "depth", block.id, "estimate")
        graph.run()
        result = block.last_result
        assert result and result.passed, [b.error for b in graph.blocks.values()]
        truth = graph.blocks["pair"].last_result.outputs["depth"]
        estimate = graph.blocks["cloud"].last_result.outputs["depth"]
        valid = np.isfinite(truth) & np.isfinite(estimate)
        expected = (np.asarray(estimate[valid], dtype=float) - 3) * 1000
        report = result.outputs["report"]
        assert report["coverage"] == np.count_nonzero(valid) / np.count_nonzero(np.isfinite(truth))
        assert report["mae_mm"] == pytest.approx(np.mean(np.abs(expected)))
        assert report["mae_mm"] < 4
    finally:
        graph.close()
