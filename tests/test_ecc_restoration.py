"""Verify native alignment, registered inspection, and selected image restoration."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.toolblocks import ToolBlock


@pytest.mark.parametrize("levels", [1, 3])
def test_ecc_alignment_and_transform_direction(levels):
    reference = np.zeros((96, 96), np.uint8)
    cv2.circle(reference, (27, 35), 11, 127, -1)
    cv2.rectangle(reference, (51, 56), (74, 67), 213, -1)
    shift = np.float32([[1, 0, 4], [0, 1, -3]])
    sample = cv2.warpAffine(reference, shift, reference.shape[::-1])
    mask = np.zeros_like(reference)
    mask[10:85, 10:85] = 255
    block = builtin_registry().create(
        "inspect.align_ecc",
        parameters={
            "motion": "translation",
            "pyramid_levels": levels,
            "iterations_per_level": [50, 35, 20] if levels > 1 else [],
        },
    )
    result = block.execute(
        {"image": sample, "reference": reference, "mask": mask, "reference_mask": mask}
    )
    assert result is not None, block.error
    assert result.outputs["coefficient"] > 0.99
    np.testing.assert_allclose(
        result.outputs["transform"]["source_to_target"],
        [[1, 0, -4], [0, 1, 3], [0, 0, 1]],
        atol=0.1,
    )
    np.testing.assert_array_equal(result.outputs["image"], reference)


@pytest.mark.parametrize("depth,norm", [(np.uint8, "l2"), (np.uint16, "l1")])
def test_nlm_preserves_unselected_pixels(depth, norm):
    noise = np.random.default_rng(15).integers(-12, 13, (32, 32))
    image = (130 + noise).astype(depth)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[6:26, 6:26] = 255
    block = builtin_registry().create(
        "cv.non_local_means", parameters={"norm": norm, "strength": 12}
    )
    result = block.execute({"image": image, "mask": mask})
    assert result is not None, block.error
    output = result.outputs["image"]
    assert output.dtype == image.dtype
    np.testing.assert_array_equal(output[mask == 0], image[mask == 0])
    assert output[mask != 0].std() < image[mask != 0].std()


@pytest.mark.parametrize("method", ["telea", "navier_stokes"])
def test_inpaint_uses_mask(method):
    image = np.full((32, 32), 120, np.uint8)
    mask = np.zeros_like(image)
    image[12:19, 12:19] = 0
    mask[12:19, 12:19] = 255
    block = builtin_registry().create("cv.inpaint", parameters={"method": method})
    result = block.execute({"image": image, "mask": mask})
    assert result is not None, block.error
    output = result.outputs["image"]
    np.testing.assert_array_equal(output[mask == 0], image[mask == 0])
    assert np.abs(output[mask != 0].astype(float) - 120).max() <= 5


def test_inpaint_preserves_high_depth_color_and_alpha():
    pixels = np.full((24, 32, 4), [8000, 16000, 24000, 35000], np.uint16)
    pixels[10:14, 13:19, :3] = 0
    mask = np.zeros(pixels.shape[:2], np.uint8)
    mask[10:14, 13:19] = 255
    block = builtin_registry().create("cv.inpaint")
    result = block.execute({"image": ImageArray(pixels, "rgba"), "mask": mask})
    assert result is not None, block.error
    output = result.outputs["image"]
    assert output.dtype == np.uint16 and image_format(output) == "rgba"
    np.testing.assert_array_equal(output[..., 3], pixels[..., 3])
    np.testing.assert_array_equal(output[mask == 0], pixels[mask == 0])
    for channel in range(3):
        expected = cv2.inpaint(pixels[..., channel], mask, 3, cv2.INPAINT_TELEA)
        np.testing.assert_array_equal(output[..., channel], expected)


@pytest.mark.parametrize("with_defect", [False, True])
def test_registered_defect_toolblock_judges_translated_parts(with_defect):
    example = (
        Path(__file__).resolve().parents[1]
        / "examples/registered_defect_inspection/registered_defect_inspection.toolblock.json"
    )
    tool = ToolBlock.from_definition(
        json.loads(example.read_text(encoding="utf-8")), builtin_registry()
    )
    try:
        reference = np.zeros((96, 96), np.uint8)
        cv2.circle(reference, (27, 35), 11, 127, -1)
        cv2.rectangle(reference, (51, 56), (74, 67), 213, -1)
        sample = reference.copy()
        if with_defect:
            cv2.circle(sample, (27, 35), 3, 0, -1)
        sample = cv2.warpAffine(sample, np.float32([[1, 0, 4], [0, 1, -3]]), (96, 96))
        result = tool.execute({"sample": sample, "reference": reference})
        assert result is not None, tool.error
        assert result.passed is (not with_defect)
        assert result.outputs["defect_count"] == int(with_defect)
        assert result.outputs["alignment_coefficient"] > 0.98
    finally:
        tool.close()


@pytest.mark.parametrize("output_depth", ["unchanged", "float32"])
def test_guided_filter_keeps_edge_alpha_and_unselected_pixels(output_depth):
    noise = np.random.default_rng(15).integers(-900, 901, (40, 48))
    guide = np.full((40, 48), 8000, np.uint16)
    guide[:, 24:] = 48000
    noisy = (guide.astype(int) + noise).astype(np.uint16)
    image = np.stack([noisy, noisy, noisy, np.full_like(noisy, 29000)], axis=2)
    mask = np.zeros(guide.shape, np.uint8)
    mask[5:-5, 5:-5] = 255
    block = builtin_registry().create(
        "cv.guided_filter", parameters={"radius": 5, "scale": 0.5, "output_depth": output_depth}
    )
    result = block.execute({"image": ImageArray(image, "rgba"), "guide": guide, "mask": mask})
    assert result is not None, block.error
    output = result.outputs["image"]
    multiplier = 65535 if output_depth == "float32" else 1
    original = image.astype(float) / multiplier
    assert output.dtype == (np.float32 if output_depth == "float32" else np.uint16)
    assert image_format(output) == "rgba"
    np.testing.assert_allclose(output[mask == 0], original[mask == 0], rtol=1e-7)
    np.testing.assert_allclose(output[..., 3], original[..., 3], rtol=1e-7)
    assert output[10:30, 8:18, 0].std() < original[10:30, 8:18, 0].std()
    assert float(output[20, 27, 0]) * multiplier > 44000
    assert float(output[20, 20, 0]) * multiplier < 12000


def test_guided_filter_rejects_different_frames_and_tiny_scale():
    block = builtin_registry().create("cv.guided_filter")
    image = np.full((5, 5), 100, np.uint8)
    result = block.execute(
        {
            "image": ValuePacket(image, {"spatial": frame([5, 5], "a")}),
            "guide": ValuePacket(image, {"spatial": frame([5, 5], "b")}),
        }
    )
    assert result is None and block.error_kind == "invalid"
    assert "share the source pixel frame" in block.error
    block.parameters["scale"] = 0.1
    assert block.execute({"image": image}) is None and block.error_kind == "invalid"
    assert "too small" in block.error
