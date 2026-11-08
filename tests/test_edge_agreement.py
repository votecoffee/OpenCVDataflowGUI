"""Validate tolerant edge scoring and masked native thresholds through the registry."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.result_graphics import ACCEPTED, REJECTED


def test_edge_comparison_handles_shifted_missing_and_unexpected_edges():
    reference = np.zeros((32, 40), np.uint8)
    reference[6:16, 8] = 255
    reference[19:23, 25] = 255
    sample = np.zeros_like(reference)
    sample[6:16, 9] = 255
    sample[20:22, 5] = 255
    block = builtin_registry().create(
        "inspect.compare_edge_maps", parameters={"minimum_precision": 0.9, "minimum_recall": 0.8}
    )
    result = block.execute({"image": sample, "reference": reference})
    assert result is not None and not result.passed, block.error
    assert result.outputs["precision"] == 10 / 12
    assert result.outputs["recall"] == 10 / 14
    assert result.outputs["f1"] == pytest.approx(20 / 26)
    assert result.outputs["measurements"]["unexpected_count"] == 2
    assert result.outputs["measurements"]["missed_count"] == 4
    assert [reason["code"] for reason in result.fail_reasons] == [
        "edge_precision_below_minimum",
        "edge_recall_below_minimum",
    ]
    assert tuple(result.outputs["map"][10, 9]) == ACCEPTED
    assert tuple(result.outputs["map"][20, 5]) == REJECTED

    block.parameters["tolerance_pixels"] = 0
    exact = block.execute({"image": sample, "reference": reference})
    assert exact is not None and exact.outputs["precision"] == 0


def test_edge_comparison_selection_and_frame_contract():
    image = np.zeros((22, 30), np.uint8)
    image[5:12, 5] = 255
    reference = image.copy()
    reference[6:10, 20] = 255
    context = {"spatial": frame([30, 22], "sample")}
    block = builtin_registry().create("inspect.compare_edge_maps", parameters={"draw_map": False})
    inputs = {
        "image": ValuePacket(image, context),
        "reference": ValuePacket(reference, {"spatial": frame([30, 22], "reference")}),
        "region": {
            "model": "image_region_v1",
            "image_size": [30, 22],
            "shape": "rectangle",
            "rectangle": [0, 0, 15, 22],
        },
    }
    assert block.execute(inputs) is None
    assert "share a coordinate frame" in block.error
    block.parameters["alignment"] = "pixel_grid"
    result = block.execute(inputs)
    assert result is not None and result.passed, block.error
    assert result.outputs["f1"] == 1
    assert result.payloads["f1"]["spatial"] == context["spatial"]
    assert result.outputs["map"] is NO_VALUE


@pytest.mark.parametrize(
    ("sample_has_edge", "reference_has_edge", "expected"),
    [(False, False, 1.0), (False, True, 0.0), (True, False, 0.0)],
)
def test_edge_comparison_empty_edge_sets(sample_has_edge, reference_has_edge, expected):
    sample = np.zeros((9, 12), np.uint8)
    reference = sample.copy()
    sample[4, 5] = 255 if sample_has_edge else 0
    reference[4, 5] = 255 if reference_has_edge else 0
    block = builtin_registry().create("inspect.compare_edge_maps")
    result = block.execute({"image": sample, "reference": reference})
    assert result is not None, block.error
    assert result.outputs["precision"] == result.outputs["recall"] == expected


@pytest.mark.parametrize(
    ("method", "dtype", "mode"),
    [
        ("otsu", np.uint8, cv2.THRESH_OTSU),
        ("otsu", np.uint16, cv2.THRESH_OTSU),
        ("triangle", np.uint8, cv2.THRESH_TRIANGLE),
    ],
)
@pytest.mark.parametrize("invert", [False, True])
def test_native_automatic_threshold_matches_selected_samples(method, dtype, mode, invert):
    rng = np.random.default_rng(51)
    maximum = np.iinfo(dtype).max
    image = rng.integers(0, maximum + 1, (79, 121), dtype=dtype)
    original = image.copy()
    mask = (rng.random(image.shape) > 0.7).astype(np.uint8)
    basic = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    level, _ = cv2.threshold(image[mask != 0].reshape(-1, 1), 0, maximum, mode | basic)
    _, expected = cv2.threshold(image, level, maximum, basic)
    expected[mask == 0] = 0
    block = builtin_registry().create(
        "cv.auto_threshold", parameters={"method": method, "invert": invert}
    )
    result = block.execute({"image": image, "mask": mask})
    assert result is not None, block.error
    assert result.outputs["threshold"] == level
    np.testing.assert_array_equal(result.outputs["image"], expected)
    np.testing.assert_array_equal(image, original)
