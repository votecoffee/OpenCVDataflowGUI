"""Verify opt-in metadata and explicit EXIF orientation."""

import struct

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.image_metadata import orient_image, summarize_exif


def exif_fixture(orientation, order="<"):
    endian = b"II" if order == "<" else b"MM"
    root_offset = 8
    capture_ifd_offset = root_offset + 2 + 4 * 12 + 4
    text_offset = capture_ifd_offset + 2 + 12 + 4
    make = b"Dataflow Synthetic\0"
    capture = b"2026:01:02 03:04:05\0"
    entries = (
        struct.pack(f"{order}HHII", 0x010F, 2, len(make), text_offset),
        struct.pack(f"{order}HHI4s", 0x0110, 2, 4, b"Lab\0"),
        struct.pack(f"{order}HHIH", 0x0112, 3, 1, orientation) + b"\0\0",
        struct.pack(f"{order}HHII", 0x8769, 4, 1, capture_ifd_offset),
    )
    root = struct.pack(f"{order}H", len(entries)) + b"".join(entries) + b"\0" * 4
    sub = struct.pack(f"{order}HHHII", 1, 0x9003, 2, len(capture), text_offset + len(make))
    return (
        endian
        + struct.pack(f"{order}HI", 42, root_offset)
        + root
        + sub
        + b"\0" * 4
        + make
        + capture
    )


def jpeg_with_exif(image, orientation):
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    payload = b"Exif\0\0" + exif_fixture(orientation)
    raw = encoded.tobytes()
    return raw[:2] + b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload + raw[2:]


def test_image_file_metadata_is_opt_in_and_orientation_keeps_ticket(tmp_path):
    image = np.arange(72, dtype=np.uint8).reshape(6, 12)
    path = tmp_path / "orientaciÃ³n.jpg"
    path.write_bytes(jpeg_with_exif(image, 6))
    source = builtin_registry().create(
        "input.file", parameters={"path": str(path), "color_format": "unchanged"}
    )
    default = source.execute({})
    assert default is not None, source.error
    pixels = default.outputs["image"]
    assert pixels.shape == (6, 12)
    assert "metadata" not in default.outputs["info"]

    source.parameters["metadata_mode"] = "read"
    observed = source.execute({})
    assert observed is not None, source.error
    np.testing.assert_array_equal(observed.outputs["image"], pixels)
    metadata = observed.outputs["info"]["metadata"]
    assert metadata["decoded_size"] == [12, 6]
    assert not metadata["orientation_applied"]
    assert metadata["exif"] == {
        "orientation": 6,
        "make": "Dataflow Synthetic",
        "model": "Lab",
        "date_time_original": "2026:01:02 03:04:05",
    }
    assert metadata["chunks"][0]["type"] == "exif"

    source.parameters["metadata_mode"] = "apply_orientation"
    rotated = source.execute({})
    assert rotated is not None, source.error
    np.testing.assert_array_equal(rotated.outputs["image"], np.rot90(pixels, -1))
    assert rotated.outputs["info"]["metadata"]["orientation_applied"]
    assert rotated.payloads["image"]["spatial"]["image_size"] == [6, 12]
    assert rotated.outputs["ticket"]["value"] == path.name


@pytest.mark.parametrize("order", ["<", ">"])
def test_exif_endianness_and_truncated_ifds(order):
    data = exif_fixture(8, order)
    assert summarize_exif(data)["orientation"] == 8
    assert summarize_exif(b"Exif\0\0" + data)["model"] == "Lab"
    assert summarize_exif(data[:17]) == {}
    assert "orientation" not in summarize_exif(exif_fixture(9, order))


@pytest.mark.parametrize(
    "orientation,expected",
    [
        (1, [[1, 2, 3], [4, 5, 6]]),
        (2, [[3, 2, 1], [6, 5, 4]]),
        (3, [[6, 5, 4], [3, 2, 1]]),
        (4, [[4, 5, 6], [1, 2, 3]]),
        (5, [[1, 4], [2, 5], [3, 6]]),
        (6, [[4, 1], [5, 2], [6, 3]]),
        (7, [[6, 3], [5, 2], [4, 1]]),
        (8, [[3, 6], [2, 5], [1, 4]]),
    ],
)
def test_exif_orientation_mappings(orientation, expected):
    image = np.arange(1, 7, dtype=np.uint8).reshape(2, 3)
    np.testing.assert_array_equal(orient_image(image, orientation), expected)


def test_png_orientation_preserves_16bit_alpha(tmp_path):
    pixels = np.arange(60, dtype=np.uint16).reshape(3, 5, 4) * 1000
    ok, encoded = cv2.imencodeWithMetadata(
        ".png",
        pixels,
        [cv2.IMAGE_METADATA_EXIF],
        [np.frombuffer(exif_fixture(8), np.uint8)],
    )
    assert ok
    path = tmp_path / "alpha.png"
    path.write_bytes(encoded.tobytes())
    block = builtin_registry().create(
        "input.file",
        parameters={
            "path": str(path),
            "metadata_mode": "apply_orientation",
            "color_format": "unchanged",
            "pixel_depth": "unchanged",
        },
    )
    result = block.execute({})
    assert result is not None, block.error
    image = result.outputs["image"]
    assert image.dtype == np.uint16
    np.testing.assert_array_equal(image, np.rot90(pixels))
