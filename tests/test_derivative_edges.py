"""Test paired derivatives and the native derivative overload of Canny."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ValuePacket, frame


@pytest.mark.parametrize(
    ("border", "mode"),
    [("reflect_101", cv2.BORDER_REFLECT_101), ("replicate", cv2.BORDER_REPLICATE)],
)
def test_paired_sobel_matches_independent_derivatives_and_masks_output(border, mode):
    y, x = np.indices((20, 24))
    gray = (90 + 3 * x - 4 * y).astype(np.uint8)
    block = builtin_registry().create("cv.spatial_gradient", parameters={"border": border})
    payload = {"spatial": frame([24, 20], "ramp")}
    image = ValuePacket(np.repeat(gray[..., None], 3, axis=2), payload)
    mask = np.zeros_like(gray)
    mask[:, 2:] = 255
    result = block.execute({"image": image, "mask": mask})
    assert result is not None, block.error
    for name, x_order, y_order in (("dx", 1, 0), ("dy", 0, 1)):
        expected = cv2.Sobel(gray, cv2.CV_16S, x_order, y_order, borderType=mode)
        expected[:, :2] = 0
        np.testing.assert_array_equal(result.outputs[name], expected)
        assert result.outputs[name].dtype == np.int16
        assert result.payloads[name]["spatial"] == payload["spatial"]


@pytest.mark.parametrize("l2", [False, True])
def test_canny_from_wired_derivatives_matches_canny_before_output_selection(l2):
    registry = builtin_registry()
    source = np.zeros((32, 40), np.uint8)
    source[5:26, 8:30] = 170
    source[12:19, 15:23] = 20
    parameters = {"low": 50, "high": 100, "l2_gradient": l2}
    native = registry.create("cv.canny", parameters=parameters).execute({"image": source})
    derivatives = registry.create("cv.spatial_gradient", parameters={"border": "replicate"})
    pair = derivatives.execute({"image": source})
    assert pair is not None, derivatives.error
    detector = registry.create("cv.canny_derivatives", parameters=parameters)
    selected = np.ones_like(source)
    selected[:, :20] = 0
    result = detector.execute({**pair.outputs, "mask": selected})
    assert result is not None, detector.error
    expected = np.where(selected, native.outputs["image"], 0)
    np.testing.assert_array_equal(result.outputs["image"], expected)
    assert result.outputs["image"].dtype == np.uint8


def test_derivative_canny_validates_formats_and_coordinate_frames():
    block = builtin_registry().create("cv.canny_derivatives")
    base = np.zeros((5, 7, 3), np.int16)
    a = ImageArray(base, "bgr")
    b = ImageArray(base, "rgb")
    assert block.execute({"dx": a, "dy": a}) is not None
    assert block.execute({"dx": a, "dy": b}) is None
    assert "matching channel order" in block.error
    assert block.execute({"dx": base, "dy": base.astype(np.float32)}) is None
    assert "Unsupported image depth" in block.error

    gray = base[..., 0]
    first = ValuePacket(gray, {"spatial": frame([7, 5], "first")})
    second = ValuePacket(gray, {"spatial": frame([7, 5], "second")})
    assert block.execute({"dx": first, "dy": second}) is None
    assert "same coordinate frame" in block.error
    block.parameters["high"] = 0
    assert block.execute({"dx": gray, "dy": gray}) is None
    assert "low <= high" in block.error
