"""Verify centered convolution PSFs, image format, and selected deblurring."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import ValuePacket, frame


def test_wiener_inverts_an_asymmetric_circular_blur_without_a_pixel_shift():
    rng = np.random.default_rng(17)
    original = rng.normal(80, 20, (23, 26))
    blurred = (
        0.1 * np.roll(original, -1, axis=1) + 0.75 * original + 0.15 * np.roll(original, 1, axis=1)
    )
    block = builtin_registry().create(
        "cv.wiener_deblur",
        parameters={
            "boundary": "circular",
            "noise_to_signal": 1e-12,
            "output_depth": "float64",
        },
    )
    context = {"spatial": frame([26, 23], "source")}
    result = block.execute(
        {
            "image": ValuePacket(blurred, context),
            "psf": [[1, 7.5, 1.5]],
        }
    )
    assert result is not None, block.error
    np.testing.assert_allclose(result.outputs["image"], original, atol=1e-8)
    np.testing.assert_allclose(result.outputs["psf"], [[0.1, 0.75, 0.15]])
    assert result.payloads["image"]["spatial"] == context["spatial"]
    assert "spatial" not in result.payloads["psf"]


def test_wiener_gaussian_retains_rgba_alpha_and_unselected_pixels():
    rng = np.random.default_rng(20)
    image = rng.integers(40, 200, (16, 22, 4), dtype=np.uint8)
    source = ImageArray(image, "rgba")
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[4:12, 5:17] = 255
    block = builtin_registry().create(
        "cv.wiener_deblur",
        parameters={
            "kernel_width": 5,
            "kernel_height": 3,
            "sigma_x": 1.1,
            "sigma_y": 0.8,
        },
    )
    result = block.execute({"image": source, "mask": mask})
    assert result is not None, block.error
    output = result.outputs["image"]
    assert image_format(output) == "rgba" and output.dtype == np.uint8
    np.testing.assert_array_equal(output[..., 3], image[..., 3])
    np.testing.assert_array_equal(output[mask == 0], image[mask == 0])
    assert np.any(output[mask != 0, :3] != image[mask != 0, :3])
    x, y = cv2.getGaussianKernel(5, 1.1), cv2.getGaussianKernel(3, 0.8)
    np.testing.assert_allclose(result.outputs["psf"], y @ x.T)


@pytest.mark.parametrize("kernel", [[], [[0, 0, 0]], [[1, -1, 1]], [[1, 1]], [[1], [2, 3]]])
def test_wiener_rejects_invalid_point_spread_functions(kernel):
    block = builtin_registry().create("cv.wiener_deblur")
    assert block.execute({"image": np.zeros((5, 5), np.uint8), "psf": kernel}) is None
    assert block.error_kind == "invalid"
    assert "PSF must be" in block.error
