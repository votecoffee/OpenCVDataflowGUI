"""Verify native alignment direction and selected image restoration."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry


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
