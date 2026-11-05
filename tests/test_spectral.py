"""Check native spectral transforms through the graph block contracts."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import ValuePacket, frame, packet_for
from opencv_dataflow.tickets import TicketClock


@pytest.mark.parametrize(
    "mode,shape,padded,dtype",
    [
        ("2d", (11, 17), (12, 18), np.uint8),
        ("rows", (11, 17), (11, 18), np.uint16),
        ("2d", (9, 1), (10, 1), np.float64),
        ("rows", (9, 1), (9, 1), np.float32),
    ],
)
def test_dct_round_trip_preserves_source_size_depth_and_frame(mode, shape, padded, dtype):
    image = np.random.default_rng(26).integers(0, 256, size=shape).astype(dtype)
    context = frame(list(shape[::-1]), "fixture")
    registry = builtin_registry()
    forward = registry.create("cv.dct", parameters={"mode": mode})
    ticket = TicketClock().issue("fixture", "fixed", 38)
    result = forward.execute({"image": ValuePacket(image, {"spatial": context, "ticket": ticket})})
    assert result is not None, forward.error
    packet = result.outputs["coefficients"]
    assert packet.values.shape == padded
    assert result.graphics["magnitude"].shape == padded
    inverse = registry.create("cv.idct")
    restored = inverse.execute({"coefficients": packet_for(result, "coefficients")})
    assert restored is not None, inverse.error
    output = restored.outputs["image"]
    assert output.dtype == dtype and output.shape == shape
    assert restored.payloads["image"]["spatial"] == context
    assert restored.payloads["image"]["ticket"] == ticket
    if image.dtype.kind == "u":
        np.testing.assert_array_equal(output, image)
    else:
        np.testing.assert_allclose(output, image, atol=1e-4, rtol=1e-6)


def test_dct_matches_native_values_and_validates_odd_policy_and_packets():
    image = np.arange(24, dtype=np.float32).reshape(3, 8)
    registry = builtin_registry()
    forward = registry.create("cv.dct", parameters={"mode": "rows", "odd_size": "reject"})
    result = forward.execute({"image": image})
    assert result is not None, forward.error
    packet = result.outputs["coefficients"]
    np.testing.assert_allclose(packet.values, cv2.dct(image, flags=cv2.DCT_ROWS))
    assert forward.execute({"image": image[:, :-1]}) is None
    assert forward.error_kind == "invalid" and "even transformed dimensions" in forward.error
    inverse = registry.create("cv.idct")
    assert inverse.execute({"coefficients": replace(packet, image_shape=(4, 8))}) is None
    assert inverse.error_kind == "invalid"


@pytest.mark.parametrize("mode,padded", [("2d", (12, 18, 2)), ("rows", (11, 18, 2))])
def test_dft_round_trip_preserves_shape_and_depth_with_optimal_padding(mode, padded):
    image = np.random.default_rng(40).integers(0, 65536, (11, 17), dtype=np.uint16)
    registry = builtin_registry()
    forward = registry.create("cv.dft", parameters={"mode": mode, "size": "optimal"})
    result = forward.execute({"image": image})
    assert result is not None, forward.error
    assert result.outputs["spectrum"].values.shape == padded
    assert result.graphics["magnitude"].shape == padded[:2]
    inverse = registry.create("cv.idft")
    restored = inverse.execute({"spectrum": packet_for(result, "spectrum")})
    assert restored is not None, inverse.error
    output = restored.outputs["image"]
    assert output.dtype == np.uint16
    np.testing.assert_array_equal(output, image)


@pytest.mark.parametrize(
    "mode,cutoff,upper_cutoff,expected",
    [
        ("low_pass", 0.1, 0.3, "slow"),
        ("high_pass", 0.1, 0.3, "stripes"),
        ("band_pass", 0.2, 0.3, "stripes"),
        ("band_stop", 0.2, 0.3, "slow"),
    ],
)
def test_fourier_filter_selects_known_frequencies_and_preserves_frame(
    mode, cutoff, upper_cutoff, expected
):
    height, width = 48, 64
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    slow = 0.5 + 0.12 * np.cos(2 * np.pi * y / 24)
    stripes = 0.18 * np.cos(2 * np.pi * x / 4)
    image = (slow + stripes).astype(np.float32)
    context = frame([width, height], "stripe-fixture")
    registry = builtin_registry()
    forward = registry.create("cv.dft")
    transformed = forward.execute({"image": ValuePacket(image, {"spatial": context})})
    assert transformed is not None, forward.error
    band = registry.create(
        "cv.filter_fourier_spectrum",
        parameters={"mode": mode, "cutoff": cutoff, "upper_cutoff": upper_cutoff},
    )
    filtered = band.execute({"spectrum": packet_for(transformed, "spectrum")})
    assert filtered is not None, band.error
    inverse = registry.create("cv.idft")
    result = inverse.execute({"spectrum": packet_for(filtered, "spectrum")})
    assert result is not None, inverse.error
    assert result.payloads["image"]["spatial"] == context
    assert result.outputs["image"].dtype == np.float32
    np.testing.assert_allclose(
        result.outputs["image"], slow if expected == "slow" else stripes, atol=2e-7
    )
