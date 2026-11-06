"""Verify adaptive background estimates, selected writes and safe division."""

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import ValuePacket, frame


def test_closing_background_preserves_dark_defect_and_unselected_pixels():
    source = np.tile(np.arange(64, dtype=np.uint8) + 80, (40, 1))
    source[17:22, 27:31] -= 50
    mask = np.zeros_like(source)
    mask[:, 8:56] = 255
    spatial = frame([64, 40], "camera")
    block = builtin_registry().create(
        "cv.adaptive_background",
        parameters={
            "estimator": "closing",
            "shape": "rect",
            "radius": 6,
            "target_level": 128 / 255,
        },
    )
    result = block.execute({"image": ValuePacket(source, {"spatial": spatial}), "mask": mask})
    assert result is not None, block.error
    estimate = cv2.morphologyEx(
        source.astype(np.float64),
        cv2.MORPH_CLOSE,
        np.ones((13, 13), np.uint8),
        borderType=cv2.BORDER_REFLECT_101,
    )
    np.testing.assert_allclose(result.outputs["background"], estimate)
    output = result.outputs["image"]
    np.testing.assert_array_equal(output[mask == 0], source[mask == 0])
    assert output[15, 25] == output[15, 45] == 128
    assert output[18, 29] < 85
    assert result.payloads["background"]["spatial"] == spatial


def test_division_preserves_invalid_rgb_samples_and_alpha():
    source = np.full((12, 20, 4), [25, 50, 75, 240], dtype=np.uint8)
    source[:, :6, 1] = 0
    block = builtin_registry().create(
        "cv.adaptive_background",
        parameters={
            "estimator": "opening",
            "shape": "rect",
            "radius": 2,
            "operation": "divide",
            "output_depth": "float32",
            "target_level": 100 / 255,
        },
    )
    result = block.execute({"image": ImageArray(source, "rgba")})
    assert result is not None, block.error
    output = result.outputs["image"]
    assert image_format(output) == "rgba" and output.dtype == np.float32
    assert image_format(result.outputs["background"]) == "rgb"
    np.testing.assert_array_equal(output[:, :6], source[:, :6])
    np.testing.assert_allclose(output[:, 8:, :3], 100)
    np.testing.assert_array_equal(output[..., 3], source[..., 3])
    assert np.all(result.outputs["invalid_mask"][:, :6] == 255)
    assert not np.any(result.outputs["invalid_mask"][:, 8:])

    result = block.execute({"image": np.zeros((12, 20), np.uint8)})
    assert result is not None and not result.passed, block.error
    assert result.fail_reasons[0]["code"] == "adaptive_background_no_valid_pixels"


def test_gaussian_subtraction_can_emit_signed_float_samples():
    source = np.full((7, 9), 10, np.uint16)
    source[3, 4] = 0
    block = builtin_registry().create(
        "cv.adaptive_background",
        parameters={
            "estimator": "gaussian",
            "radius": 1,
            "sigma": 1.0,
            "target_level": 0,
            "output_depth": "float64",
        },
    )
    result = block.execute({"image": source})
    assert result is not None, block.error
    expected = source - cv2.GaussianBlur(source.astype(float), (3, 3), 1.0)
    np.testing.assert_allclose(result.outputs["image"], expected)
    assert result.outputs["image"][3, 4] < 0
