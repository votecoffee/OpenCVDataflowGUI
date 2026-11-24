"""Qualify exact and native percentile fitting with known and selected samples."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.percentile_white_balance import PercentileWhiteBalance
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/percentile_white_balance"
TYPE = "cv.percentile_white_balance"


@pytest.mark.parametrize(("profile", "workers"), [("debug", 1), ("lean", 3)])
def test_saved_chart_neutralizes_steps_and_clips_outliers(profile, workers):
    graph = Graph.load(EXAMPLE / "percentile_white_balance.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.set_branch_workers(workers)
        graph.run()
        assert block_entry(graph.blocks["trimmed"]).path[:2] == (
            "OpenCV", "Extended Computational Photography (xphoto)",
        )
        for name, expected in (
            ("untrimmed", [[36, 66, 48], [73, 97, 84], [110, 128, 120], [147, 159, 156]]),
            ("trimmed", [[16] * 3, [89] * 3, [162] * 3, [235] * 3]),
            ("native", [[16] * 3, [88] * 3, [161] * 3, [233] * 3]),
        ):
            block = graph.blocks[name]
            result = block.last_result
            assert result and result.passed, block.error
            image = result.outputs["image"]
            np.testing.assert_array_equal(image[50, [15, 47, 79, 111]], expected)
            np.testing.assert_array_equal(image[0], np.full_like(image[0], 16))
            np.testing.assert_array_equal(image[-1], np.full_like(image[-1], 235))
            assert result.payloads["image"]["ticket"]["value"] == "cast_chart.png"
            assert result.payloads["image"]["spatial"]["root_id"] == "source:image"
        assert graph.blocks["filmstrip"].last_result.passed
    finally:
        graph.close()


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_native_matches_direct_opencv_with_input_and_output_clipping(dtype):
    source = np.random.default_rng(548).integers(0, 256, (37, 43, 3), dtype=np.uint8).astype(dtype)
    factor = 1 if dtype == np.uint8 else 255
    if dtype == np.float32:
        source /= factor
    low, high = 20 / factor, 220 / factor
    out_low, out_high = 35 / factor, 195 / factor
    p = {
        "method": "opencv_histogram", "tail_percent": 5.0, "range_mode": "explicit",
        "input_min": low, "input_max": high, "output_min": out_low, "output_max": out_high,
    }
    native = cv2.xphoto.createSimpleWB()
    native.setInputMin(low)
    native.setInputMax(high)
    native.setOutputMin(out_low)
    native.setOutputMax(out_high)
    native.setP(p["tail_percent"])
    expected = native.balanceWhite(np.clip(source, low, high).astype(dtype))
    expected = np.clip(expected, out_low, out_high).astype(dtype)
    block = PercentileWhiteBalance(parameters=p)
    result = block.execute({"image": source})
    assert result and result.passed, block.error
    np.testing.assert_array_equal(result.outputs["image"], expected)
    assert result.outputs["report"]["fitted_bgr"] is None
    assert result.outputs["image"].min() == dtype(out_low)
    bin_width = (out_high - out_low) / (4095 if dtype == np.float32 else 255)
    assert 0 <= float(dtype(out_high) - result.outputs["image"].max()) <= 2 * bin_width


def test_uint16_rgba_region_and_mask_fit_applies_to_entire_frame():
    # The selection admits three neutral steps with different offsets/slopes.
    # Both unselected columns fall beyond the fitted anchors.
    bgr = np.array([
        [10, 20, 30], [100, 200, 300], [200, 400, 600],
        [300, 600, 900], [700, 1200, 1500],
    ], np.uint16)
    rgba = np.column_stack((bgr[:, ::-1], np.arange(1000, 1005, dtype=np.uint16)))
    source = ImageArray(np.repeat(rgba[None, ...], 3, axis=0), "rgba")
    context = frame([5, 3], "inspection")
    region = {
        "shape": "rectangle", "rectangle": [1, 0, 4, 3], "image_size": [5, 3],
        "model": "image_region_v1", "coordinate_context": context,
    }
    mask = np.full((3, 5), 255, np.uint8)
    mask[:, 4] = 0
    initial = source.copy()
    block = PercentileWhiteBalance(parameters={"tail_percent": 0})
    result = block.execute({
        "image": ValuePacket(source, {"spatial": context}),
        "region": region,
        "mask": ValuePacket(mask, {"spatial": context}),
    })
    assert result and result.passed, block.error
    assert result.outputs["report"]["sample_count"] == 9
    assert result.outputs["report"]["fitted_bgr"] == {
        "B": [100, 300], "G": [200, 600], "R": [300, 900],
    }
    output = result.outputs["image"]
    assert output.dtype == np.uint16 and output.color_format == "rgba"
    np.testing.assert_array_equal(output[0, :, :3], np.repeat(
        np.uint16([0, 0, 32768, 65535, 65535])[:, None], 3, axis=1,
    ))
    np.testing.assert_array_equal(output[..., 3], initial[..., 3])
    np.testing.assert_array_equal(source, initial)
    assert result.payloads["image"]["spatial"] == context


def test_flat_exact_channels_and_empty_fit():
    image = np.array([[[12, 80, 10], [46, 80, 90], [80, 80, 170]]], np.uint8)
    block = PercentileWhiteBalance(parameters={"range_mode": "explicit", "output_max": 50})
    result = block.execute({"image": image})
    assert result and result.passed, block.error
    assert result.outputs["report"]["fitted_bgr"]["G"] == [80, 80]
    np.testing.assert_array_equal(result.outputs["image"], [[[0, 50, 0], [25, 50, 25], [50, 50, 50]]])
    failed = block.execute({"image": image, "mask": np.zeros(image.shape[:2], dtype=np.uint8)})
    assert failed and not failed.passed
    assert failed.outputs["image"] is NO_VALUE
    assert failed.fail_reasons[0]["code"] == "percentile_white_balance_empty_selection"


@pytest.mark.parametrize(("parameters", "dtype", "error"), [
    ({"tail_percent": 50}, np.uint8, "between 0 and 49.99"),
    ({"range_mode": "explicit", "input_min": 256}, np.uint8, "Minimum must be below maximum"),
    ({"range_mode": "explicit", "output_min": 0.25}, np.uint8, "whole limits"),
    ({"method": "opencv_histogram"}, np.uint16, "exact percentiles for uint16"),
    ({"method": "opencv_histogram", "range_mode": "explicit",
      "input_min": 1e6, "input_max": 1e6 + 1}, np.float32, "too narrow"),
])
def test_parameter_validation(parameters, dtype, error):
    block = builtin_registry().create(TYPE, parameters=parameters)
    assert block.execute({"image": np.zeros((3, 4, 3), dtype=dtype)}) is None
    assert error in block.error


def test_native_rejects_a_selection_instead_of_fitting_the_wrong_pixels():
    block = PercentileWhiteBalance(parameters={"method": "opencv_histogram"})
    assert block.execute({
        "image": np.ones((4, 6, 3), np.uint8),
        "mask": np.ones((4, 6), np.uint8),
    }) is None
    assert "Use exact percentiles" in block.error
