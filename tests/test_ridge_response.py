"""Verify native ridge filtering, polarity, and response selection."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock


@pytest.mark.parametrize(("polarity", "precision", "output", "aperture", "dx", "dy"), [
    ("dark", "float32", "uint16", 3, 1, 1),
    ("bright", "float64", "float64", 5, 2, 1),
    ("both", "float64", "uint8", 1, 2, 1),
])
def test_ridge_matches_native_responses_and_clips_to_selected_pixels(
    polarity, precision, output, aperture, dx, dy
):
    rng = np.random.default_rng(27)
    bgr = rng.integers(0, 4096, (20, 26, 3), dtype=np.uint16)
    rgb = ImageArray(bgr[:, :, ::-1].copy(), "rgb")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    spatial = frame([26, 20], "part")
    ticket = TicketClock().issue("camera")
    mask = np.zeros(gray.shape, np.uint8)
    mask[3:17, 4:20] = 255
    region = {
        "model": "image_region_v1", "image_size": [26, 20],
        "shape": "rectangle", "rectangle": [2, 5, 14, 9],
    }
    selected = np.zeros(gray.shape, dtype=bool)
    selected[5:14, 4:16] = True
    depths = {"float32": cv2.CV_32F, "float64": cv2.CV_64F,
              "uint8": cv2.CV_8U, "uint16": cv2.CV_16U}
    native = cv2.ximgproc.RidgeDetectionFilter_create(
        depths[precision], dx, dy, aperture, depths[output], 0.35, 0.7, cv2.BORDER_REPLICATE
    )
    if polarity == "dark":
        expected = native.getRidgeFilteredImage(gray)
    else:
        expected = native.getRidgeFilteredImage(-gray.astype(precision))
        if polarity == "both":
            expected = np.maximum(expected, native.getRidgeFilteredImage(gray))
    expected[~selected] = 0
    block = builtin_registry().create("cv.ridge_response", parameters={
        "polarity": polarity, "derivative_depth": precision, "output_depth": output,
        "kernel_size": str(aperture), "dx": dx, "dy": dy, "scale": 0.35, "delta": 0.7,
        "border": "replicate",
    })
    result = block.execute({
        "image": ValuePacket(rgb, {"ticket": ticket, "spatial": spatial}),
        "mask": mask, "region": region,
    })
    assert result is not None, block.error
    np.testing.assert_array_equal(result.outputs["image"], expected)
    assert result.outputs["image"].dtype == expected.dtype
    assert result.payloads["image"]["ticket"] == ticket
    assert result.payloads["image"]["spatial"] == spatial


def test_ridge_polarity_peaks_on_the_corresponding_line_centers():
    image = np.full((45, 60), 120, np.uint8)
    image[:, 14] = 40
    image[:, 43] = 200
    block = builtin_registry().create("cv.ridge_response")
    dark = block.execute({"image": image}).outputs["image"]
    block.parameters["polarity"] = "bright"
    bright = block.execute({"image": image}).outputs["image"]
    assert dark[22, 14] > 0 and dark[22, 43] <= 0
    assert bright[22, 43] > 0 and bright[22, 14] <= 0


def test_ridge_validates_derivative_order_against_aperture():
    block = builtin_registry().create(
        "cv.ridge_response", parameters={"kernel_size": "3", "dy": 3}
    )
    assert block.execute({"image": np.ones((16, 20), np.uint8)}) is None
    assert "smaller than the Sobel axis" in block.error
