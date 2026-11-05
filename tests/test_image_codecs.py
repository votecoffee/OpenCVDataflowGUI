"""Verify in-memory image packets and compatible file encoder settings."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import ValuePacket, describe_value, frame, packet_for
from opencv_dataflow.port_types import compatible
from opencv_dataflow.tickets import TicketClock


def test_lossless_png_retains_16_bit_rgba_frame_and_ticket():
    registry = builtin_registry()
    raw = np.random.default_rng(20).integers(0, 65536, (19, 23, 4), dtype=np.uint16)
    image = ImageArray(raw, "rgba")
    context = frame([23, 19], "source")
    ticket = TicketClock().issue("source")
    encoder = registry.create(
        "cv.encode_image", parameters={"png_compression": 9, "png_strategy": "rle"}
    )
    result = encoder.execute({"image": ValuePacket(image, {"spatial": context, "ticket": ticket})})
    assert result is not None, encoder.error
    encoded = result.outputs["encoded"]
    assert encoded.buffer.startswith(b"\x89PNG")
    assert result.outputs["byte_count"] == len(encoded.buffer)
    assert describe_value(encoded) == {
        "model": "encoded_image_v1",
        "codec": "png",
        "image_size": [23, 19],
        "color_format": "rgba",
        "byte_count": len(encoded.buffer),
    }
    assert not compatible("encoded_image", "image")
    assert not compatible("image", "encoded_image")
    decoder = registry.create("cv.decode_image")
    output = decoder.execute({"encoded": packet_for(result, "encoded")})
    assert output is not None, decoder.error
    assert image_format(output.outputs["image"]) == "rgba"
    np.testing.assert_array_equal(output.outputs["image"], raw)
    assert output.payloads["image"]["spatial"] == context
    assert output.payloads["image"]["ticket"] == ticket

    with pytest.raises(ValidationError, match="declared image codec"):
        replace(encoded, codec="jpeg")
    assert decoder.execute({"encoded": replace(encoded, image_size=(22, 19))}) is None
    assert decoder.error_kind == "invalid" and "size differs" in decoder.error


def test_memory_jpeg_and_save_image_share_options(tmp_path):
    registry = builtin_registry()
    pixels = np.tile(np.array([[[140, 90, 30]]], np.uint8), (24, 32, 1))
    image = ImageArray(pixels, "rgb")
    parameters = {"jpeg_quality": 91, "jpeg_progressive": True, "jpeg_optimize": True}
    encoder = registry.create("cv.encode_image", parameters={"codec": "jpeg", **parameters})
    encoded = encoder.execute({"image": image})
    assert encoded is not None, encoder.error
    path = tmp_path / "camera.jpg"
    sink = registry.create("output.image", parameters={"path": str(path), **parameters})
    assert sink.execute({"image": image}) is not None, sink.error
    assert path.read_bytes() == encoded.outputs["encoded"].buffer
    decoder = registry.create(
        "cv.decode_image", parameters={"color_format": "gray", "pixel_depth": "uint16"}
    )
    output = decoder.execute({"encoded": encoded.outputs["encoded"]})
    assert output is not None, decoder.error
    native = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    expected = cv2.cvtColor(native.astype(np.uint16) * 257, cv2.COLOR_BGR2GRAY)
    np.testing.assert_array_equal(output.outputs["image"], expected)
    assert output.outputs["image"].dtype == np.uint16


def test_in_memory_codecs_reject_lossy_depth_or_alpha_and_validate_binary_png():
    registry = builtin_registry()
    encoder = registry.create("cv.encode_image", parameters={"codec": "jpeg"})
    assert encoder.execute({"image": np.zeros((8, 8), np.uint16)}) is None
    assert "JPEG requires uint8" in encoder.error
    assert encoder.execute({"image": np.zeros((8, 8, 4), np.uint8)}) is None
    assert encoder.error_kind == "invalid"
    encoder.parameters.update(codec="png", png_bilevel=True)
    assert encoder.execute({"image": np.full((8, 8), 100, np.uint8)}) is None
    assert "Bilevel PNG" in encoder.error
    mask = np.tile(np.array([[0, 255]], np.uint8), (8, 4))
    result = encoder.execute({"image": mask})
    assert result is not None, encoder.error
    decoder = registry.create("cv.decode_image")
    decoded = decoder.execute({"encoded": result.outputs["encoded"]})
    assert decoded is not None, decoder.error
    np.testing.assert_array_equal(decoded.outputs["image"], mask)
