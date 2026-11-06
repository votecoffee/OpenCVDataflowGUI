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


@pytest.mark.parametrize("angle", [0, 90])
def test_wiener_motion_model_inverts_a_known_linear_blur(angle):
    rng = np.random.default_rng(19)
    original = rng.normal(110, 18, (29, 31))
    weights = np.array([0.5, 1, 1, 1, 1, 1, 1, 1, 0.5]) / 8
    axis = 1 if angle == 0 else 0
    blurred = sum(
        weight * np.roll(original, offset, axis=axis)
        for offset, weight in enumerate(weights, start=-4)
    )
    block = builtin_registry().create(
        "cv.wiener_deblur",
        parameters={
            "model": "motion",
            "motion_length": 8,
            "motion_angle": angle,
            "boundary": "circular",
            "noise_to_signal": 1e-12,
        },
    )
    result = block.execute({"image": blurred})
    assert result is not None, block.error
    psf = np.asarray(result.outputs["psf"])
    assert psf.shape == ((1, 9) if angle == 0 else (9, 1))
    np.testing.assert_allclose(psf.ravel(), weights)
    np.testing.assert_allclose(result.outputs["image"], original, atol=1e-5)


def test_wiener_defocus_model_has_centered_disk_and_supports_zero_radius():
    registry = builtin_registry()
    block = registry.create(
        "cv.wiener_deblur",
        parameters={"model": "defocus", "defocus_radius": 2.5, "noise_to_signal": 1e-12},
    )
    image = np.arange(81, dtype=np.uint8).reshape(9, 9)
    result = block.execute({"image": image})
    assert result is not None, block.error
    kernel = np.asarray(result.outputs["psf"])
    assert kernel.shape == (7, 7)
    assert kernel.sum() == pytest.approx(1)
    assert kernel[0, 0] == 0 and 0 < kernel[3, 5] < kernel[3, 3]
    np.testing.assert_allclose(kernel, kernel.T)
    np.testing.assert_allclose(kernel, kernel[::-1, ::-1])

    block.parameters["defocus_radius"] = 0
    result = block.execute({"image": image})
    assert result is not None, block.error
    assert result.outputs["psf"] == [[1]]
    np.testing.assert_array_equal(result.outputs["image"], image)


def test_motion_model_positive_angle_follows_image_coordinates():
    block = builtin_registry().create(
        "cv.wiener_deblur", parameters={"model": "motion", "motion_length": 6, "motion_angle": 45}
    )
    result = block.execute({"image": np.ones((16, 16), np.float64)})
    assert result is not None, block.error
    kernel = np.asarray(result.outputs["psf"])
    np.testing.assert_allclose(kernel, kernel[::-1, ::-1], atol=1e-15)
    assert kernel[2, 2] > 0 and kernel[2, 4] == 0
