"""Validate the native vector-field conversions through their graph port contract."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import ValuePacket, frame


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("units", ["degrees", "radians"])
def test_polar_round_trip_uses_tagged_units_and_preserves_signed_fields(dtype, units):
    registry = builtin_registry()
    forward = registry.create("cv.cart_to_polar", parameters={"angle_units": units})
    inverse = registry.create("cv.polar_to_cart")
    x = np.array([[2, 0, -2, 0, 3]], dtype=dtype)
    y = np.array([[0, 2, 0, -2, -4]], dtype=dtype)
    context = {"spatial": frame([5, 1], "camera")}
    polar = forward.execute({"x": x, "y": ValuePacket(y, context)})
    assert polar is not None, forward.error
    assert polar.payloads["angle"]["angle_units"] == units
    assert "angle_units" not in polar.payloads["magnitude"]
    result = inverse.execute(polar.outputs)
    assert result is not None, inverse.error
    assert result.outputs["x"].dtype == result.outputs["y"].dtype == dtype
    np.testing.assert_allclose(result.outputs["x"], x, atol=0.004)
    np.testing.assert_allclose(result.outputs["y"], y, atol=0.004)
    assert result.payloads["x"]["spatial"] == context["spatial"]
    assert "angle_units" not in result.payloads["x"]


def test_polar_to_cartesian_handles_units_omitted_magnitude_and_invalid_metadata():
    block = builtin_registry().create("cv.polar_to_cart")
    angle = np.array([[0, np.pi / 2, -np.pi / 2]])
    result = block.execute({"angle": angle})
    assert result is not None, block.error
    np.testing.assert_allclose(result.outputs["x"], [[1, 0, 0]], atol=1e-6)
    np.testing.assert_allclose(result.outputs["y"], [[0, 1, -1]], atol=1e-6)
    assert block.execute({"angle": ValuePacket(angle, {"angle_units": "turns"})}) is None
    assert "radians or degrees" in block.error
    block.parameters["angle_units"] = "degrees"
    result = block.execute({"angle": np.array([[360e12 + 90]])})
    assert result is not None, block.error
    np.testing.assert_allclose(result.outputs["y"], [[1]], atol=1e-6)


def test_polar_to_cartesian_checks_magnitude_and_frames():
    registry = builtin_registry()
    block = registry.create("cv.polar_to_cart")
    angle = np.zeros((3, 4), np.float32)
    magnitude = np.ones_like(angle)
    assert block.execute({"angle": angle, "magnitude": -magnitude}) is None
    assert "nonnegative" in block.error
    assert block.execute({"angle": angle, "magnitude": magnitude.astype(np.float64)}) is None
    assert "matching monochrome floating arrays" in block.error
    assert block.execute({"angle": angle, "magnitude": magnitude[:2]}) is None
    a = ValuePacket(angle, {"spatial": frame([4, 3], "a")})
    b = ValuePacket(magnitude, {"spatial": frame([4, 3], "b")})
    assert block.execute({"angle": a, "magnitude": b}) is None
    assert "same coordinate frame" in block.error
    forward = registry.create("cv.cart_to_polar")
    assert forward.execute({"x": a, "y": b}) is None
    assert "same coordinate frame" in forward.error
