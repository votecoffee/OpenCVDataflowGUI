"""Verify packed YUV layouts, raw-file input and decoded provenance."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import image_format
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock


def packed_red(layout):
    y, u, v = 82, 90, 240
    if layout in ("NV12", "NV21", "I420", "YV12"):
        chroma = {
            "NV12": [u, v] * 4,
            "NV21": [v, u] * 4,
            "I420": [u] * 4 + [v] * 4,
            "YV12": [v] * 4 + [u] * 4,
        }[layout]
        return np.array([y] * 16 + chroma, dtype=np.uint8).reshape(6, 4)
    pair = {"YUY2": [y, u, y, v], "UYVY": [u, y, v, y], "YVYU": [y, v, y, u]}[layout]
    return np.tile(np.array([pair], dtype=np.uint8), (4, 2))


@pytest.mark.parametrize("layout", ["NV12", "NV21", "I420", "YV12", "YUY2", "UYVY", "YVYU"])
def test_decode_packed_yuv_recovers_color_and_unscaled_luma(layout):
    buffer = packed_red(layout)
    ticket = TicketClock().issue("packed")
    source = ValuePacket(
        buffer, {"spatial": frame(list(buffer.shape[::-1]), "storage"), "ticket": ticket}
    )
    block = builtin_registry().create(
        "cv.decode_yuv", parameters={"layout": layout, "color_format": "rgba"}
    )
    result = block.execute({"image": source})
    assert result is not None, block.error
    image = result.outputs["image"]
    assert image_format(image) == "rgba" and image.shape == (4, 4, 4)
    np.testing.assert_allclose(image[1, 1], [255, 0, 0, 255], atol=2)
    assert result.payloads["image"]["ticket"] == ticket
    assert result.payloads["image"]["spatial"]["image_size"] == [4, 4]
    assert result.payloads["image"]["spatial"]["root_id"] != "storage"
    block.parameters["color_format"] = "gray"
    result = block.execute({"image": source})
    assert result is not None, block.error
    np.testing.assert_array_equal(result.outputs["image"], np.full((4, 4), 82, np.uint8))


def test_raw_yuv_file_checks_byte_count_and_supports_project_relative_unicode_paths(tmp_path):
    raw = tmp_path / "cámara.yuv"
    data = packed_red("NV21").tobytes()
    raw.write_bytes(data)
    block = builtin_registry().create(
        "input.raw_yuv",
        parameters={
            "path": raw.name,
            "path_base": "project",
            "layout": "NV21",
            "width": 4,
            "height": 4,
            "color_format": "rgb",
        },
    )
    block.project_dir = tmp_path
    result = block.execute({})
    assert result is not None, block.error
    assert result.outputs["info"]["byte_count"] == 24
    assert result.outputs["ticket"]["value"] == raw.name
    np.testing.assert_allclose(result.outputs["image"][0, 0], [255, 0, 0], atol=2)
    for wrong in (data[:-1], data + b"0"):
        raw.write_bytes(wrong)
        assert block.execute({}) is None
        assert block.error_kind == "invalid" and "exactly 24 bytes" in block.error
    block.parameters["height"] = 3
    assert block.execute({}) is None
    assert "even height" in block.error


def test_yuv_decoder_validates_incompatible_matrix_layouts():
    block = builtin_registry().create("cv.decode_yuv")
    assert block.execute({"image": np.zeros((5, 4), np.uint8)}) is None
    assert "4:2:0" in block.error
    block.parameters["layout"] = "UYVY"
    assert block.execute({"image": np.zeros((4, 6), np.uint8)}) is None
    assert "four bytes" in block.error
    assert block.execute({"image": np.zeros((4, 8), np.uint16)}) is None
    assert "2D uint8" in block.error
