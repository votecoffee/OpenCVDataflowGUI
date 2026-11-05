"""Flat-field correction preserves source encoding, selection, and ticket frame."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ValuePacket, frame


def test_flat_field_removes_gradient_and_preserves_source_metadata():
    block = builtin_registry().create("cv.flat_field")
    dark = np.full((3, 4), 10, np.uint8)
    flat = np.tile([50, 90, 130, 170], (3, 1)).astype(np.uint8)
    sample = dark + (flat - dark) // 2
    spatial = frame([4, 3], "production_camera")
    result = block.execute(
        {
            "image": ValuePacket(sample, {"spatial": spatial}),
            "flat": ValuePacket(flat, {"spatial": frame([4, 3], "calibration_capture")}),
            "dark": dark,
        }
    )
    assert result is not None, block.error
    assert result.passed and result.outputs["corrected_pixels"] == 12
    np.testing.assert_array_equal(result.outputs["image"], np.full_like(sample, 50))
    assert not result.outputs["invalid_mask"].any()
    assert result.payloads["image"]["spatial"] == spatial
    assert result.payloads["invalid_mask"]["spatial"] == spatial


def test_flat_field_preserves_alpha_and_flags_invalid_selected_pixels():
    block = builtin_registry().create(
        "cv.flat_field", parameters={"target_mode": "fixed", "target_level": 0.5}
    )
    sample = ImageArray(np.full((2, 3, 4), 1000, np.uint16), "rgba")
    flat = sample.copy()
    flat[..., :3] = [10000, 20000, 40000]
    flat[0, 1, 0] = 0
    selection = np.array([[255, 255, 0], [0, 0, 0]], np.uint8)
    result = block.execute({"image": sample, "flat": flat, "mask": selection})
    assert result is not None, block.error
    out = result.outputs["image"]
    assert out.color_format == "rgba" and out.dtype == np.uint16
    assert out[0, 0].tolist() == [3277, 1638, 819, 1000]
    np.testing.assert_array_equal(out[0, 1:], sample[0, 1:])
    np.testing.assert_array_equal(out[1], sample[1])
    assert result.outputs["invalid_mask"].tolist() == [[0, 255, 0], [0, 0, 0]]
    assert result.outputs["corrected_pixels"] == 1


@pytest.mark.parametrize("response", [0.0, 0.001])
def test_unusable_float_reference_returns_original_with_failure(response):
    block = builtin_registry().create("cv.flat_field")
    sample = np.full((2, 2), 0.25, np.float32)
    result = block.execute({"image": sample, "flat": np.full_like(sample, response)})
    assert result is not None, block.error
    assert not result.passed
    assert result.fail_reasons[0]["code"] == "flat_field_no_valid_pixels"
    np.testing.assert_array_equal(result.outputs["image"], sample)
    assert (result.outputs["invalid_mask"] == 255).all()


def test_flat_field_rejects_mismatched_camera_grid():
    block = builtin_registry().create("cv.flat_field")
    sample = np.full((2, 2), 10, np.uint8)
    assert block.execute({"image": sample, "flat": sample[:, :1]}) is None
    assert block.error_kind == "invalid"
    assert "Flat must match image dimensions" in block.error
