"""High-bit-depth codec qualification on real PNG/OpenJPEG CPU backends."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ValuePacket, frame, packet_for
from opencv_dataflow.tickets import TicketClock


def execute(kind, inputs=None, **parameters):
    block = builtin_registry().create(kind, parameters=parameters)
    result = block.execute(inputs or {})
    assert result is not None, block.error
    return result


@pytest.mark.parametrize("codec", ["png", "jpeg2000"])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("encoding,channels", [("gray", 1), ("rgb", 3), ("rgba", 4)])
def test_exact_memory_and_unicode_file_roundtrips_preserve_every_sample(
    tmp_path, codec, dtype, encoding, channels
):
    maximum = np.iinfo(dtype).max
    shape = (65, 97) if channels == 1 else (65, 97, channels)
    values = np.random.default_rng(126).integers(0, int(maximum) + 1, shape, dtype=dtype)
    values[0, 0], values[-1, -1] = 0, maximum
    source = ImageArray(values, encoding)
    ticket = TicketClock().issue("camera")
    spatial = frame([97, 65], "fixture")
    payload = {"ticket": ticket, "spatial": spatial, "inspection": {"calibration_revision": "r1"}}
    settings = {"png_compression": 6, "png_filter": "paeth", "png_zlib_buffer_size": 131072}
    encoded = execute(
        "cv.encode_image", {"image": ValuePacket(source, payload)}, codec=codec, **settings
    )
    packet = encoded.outputs["encoded"]
    output = execute("cv.decode_image", {"encoded": packet_for(encoded, "encoded")})
    np.testing.assert_array_equal(output.outputs["image"], source)
    assert output.outputs["image"].dtype == dtype
    assert output.outputs["image"].color_format == encoding
    assert output.payloads["image"]["ticket"] == ticket
    assert output.payloads["image"]["spatial"] == spatial
    assert output.payloads["image"]["inspection"] == payload["inspection"]
    if codec == "png":
        assert packet.buffer[24] == np.dtype(dtype).itemsize * 8
    else:
        assert packet.buffer.startswith(b"\x00\x00\x00\x0cjP  \r\n\x87\n")
    extension = ".png" if codec == "png" else ".jp2"
    path = tmp_path / f"{np.dtype(dtype).name}-{encoding}-图像{extension}"
    execute("output.image", {"image": source}, path=str(path), **settings)
    assert path.read_bytes() == packet.buffer
    loaded = execute("input.file", path=str(path), color_format=encoding, pixel_depth="unchanged")
    np.testing.assert_array_equal(loaded.outputs["image"], source)
    assert loaded.outputs["image"].dtype == dtype


@pytest.mark.parametrize(
    "filter_name,buffer_size",
    [
        ("none", 6),
        ("sub", 8192),
        ("up", 131072),
        ("avg", 262144),
        ("paeth", 1048576),
        ("fast", 0),
        ("all", 0),
    ],
)
def test_png_filters_and_buffer_endpoints_preserve_16bit_samples(filter_name, buffer_size):
    values = np.random.default_rng(19).integers(0, 65536, (13, 19, 4), dtype=np.uint16)
    encoded = execute(
        "cv.encode_image",
        {"image": ImageArray(values, "rgba")},
        png_compression=9,
        png_strategy="filtered",
        png_filter=filter_name,
        png_zlib_buffer_size=buffer_size,
    )
    output = execute("cv.decode_image", {"encoded": encoded.outputs["encoded"]}).outputs["image"]
    np.testing.assert_array_equal(output, values)
    assert output.dtype == np.uint16


def test_lossy_jpeg2000_requires_explicit_mode_and_retains_storage_depth():
    source = np.random.default_rng(8).integers(0, 65536, (65, 97), dtype=np.uint16)
    encoded = execute(
        "cv.encode_image",
        {"image": source},
        codec="jpeg2000",
        jpeg2000_mode="lossy",
        jpeg2000_rate_x1000=100,
    )
    output = execute("cv.decode_image", {"encoded": encoded.outputs["encoded"]}).outputs["image"]
    assert output.dtype == np.uint16 and output.shape == source.shape
    assert np.max(np.abs(output.astype(float) - source)) > 0


def test_jpeg2000_rejects_unsupported_depth_and_small_geometry_before_publication(tmp_path):
    path = tmp_path / "not-created.jp2"
    saver = builtin_registry().create("output.image", parameters={"path": str(path)})
    assert saver.execute({"image": np.ones((65, 97), np.float32)}) is None
    assert "requires uint8 or uint16" in saver.error and not path.exists()
    assert saver.execute({"image": np.ones((31, 97), np.uint16)}) is None
    assert "at least 32 pixels" in saver.error and not path.exists()
    encoder = builtin_registry().create("cv.encode_image", parameters={"codec": "jpeg2000"})
    for value in (0, 1000):
        encoder.parameters["jpeg2000_rate_x1000"] = value
        assert encoder.execute({"image": np.ones((65, 97), np.uint16)}) is None
        assert "Jpeg2000 rate x1000" in encoder.error
    encoder.parameters.update(codec="png", jpeg2000_rate_x1000=250, png_zlib_buffer_size=5)
    assert encoder.execute({"image": np.ones((8, 9), np.uint16)}) is None
    assert "Png zlib buffer size" in encoder.error


def test_missing_jpeg2000_backend_is_explicit_and_preserves_existing_files(tmp_path, monkeypatch):
    native = cv2.haveImageWriter
    monkeypatch.setattr(
        cv2,
        "haveImageWriter",
        lambda extension: False if extension == ".jp2" else native(extension),
    )
    source = np.ones((65, 97), np.uint16)
    encoder = builtin_registry().create("cv.encode_image", parameters={"codec": "jpeg2000"})
    assert encoder.execute({"image": source}) is None
    assert encoder.error_kind == "invalid" and ".jp2 is unavailable" in encoder.error
    path = tmp_path / "preserved.jp2"
    path.write_bytes(b"original")
    saver = builtin_registry().create("output.image", parameters={"path": str(path)})
    assert saver.execute({"image": source}) is None and path.read_bytes() == b"original"
    saver.parameters["path"] = str(tmp_path / "not-created.jp2")
    assert saver.execute({"image": source}) is None
    assert not (tmp_path / "not-created.jp2").exists()


@pytest.mark.parametrize("project", ["project.json", "grayscale.json"])
def test_public_lossless_16bit_examples(project):
    path = (
        Path(__file__).resolve().parents[1]
        / "examples/opencv tools/high_bit_depth_codecs"
        / project
    )
    graph = Graph.load(path, builtin_registry())
    try:
        graph.run()
        assert all(
            block.state == "pass" for block in graph.blocks.values() if not block.layout_only
        )
        for codec in ("png", "jpeg2000"):
            assert graph.blocks[f"verify_{codec}"].last_result.outputs["value"] == 0
            assert graph.blocks[f"decode_{codec}"].last_result.outputs["image"].dtype == np.uint16
    finally:
        graph.close()
