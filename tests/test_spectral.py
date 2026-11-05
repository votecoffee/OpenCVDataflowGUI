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
