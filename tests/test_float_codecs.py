"""Exact floating TIFF samples, bounded RGBE loss and explicit optional EXR support."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray, format_image
from opencv_dataflow.payloads import FlowGroup, ValuePacket, frame, packet_for
from opencv_dataflow.tickets import TicketClock


def execute(kind, inputs=None, **parameters):
    block = builtin_registry().create(kind, parameters=parameters)
    result = block.execute(inputs or {})
    assert result is not None, block.error
    return result


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "shape,encoding", [((17, 23), "gray"), ((17, 23, 3), "rgb"), ((17, 23, 4), "rgba")]
)
def test_tiff_memory_and_unicode_file_roundtrip_preserve_samples_order_and_payload(
    tmp_path, dtype, shape, encoding
):
    source = ImageArray(np.random.default_rng(125).uniform(-2, 20, shape).astype(dtype), encoding)
    spatial = frame([23, 17], "float-camera")
    ticket = TicketClock().issue("inspection")
    encoded = execute(
        "cv.encode_image",
        {"image": ValuePacket(source, {"spatial": spatial, "ticket": ticket})},
        codec="tiff",
    )
    decoded = execute("cv.decode_image", {"encoded": packet_for(encoded, "encoded")})
    output = decoded.outputs["image"]
    np.testing.assert_array_equal(output, source)
    assert output.dtype == dtype and output.color_format == encoding
    assert output.min() < 0 and output.max() > 1
    assert decoded.payloads["image"]["spatial"] == spatial
    assert decoded.payloads["image"]["ticket"] == ticket
    path = tmp_path / f"float-{np.dtype(dtype).name}-{encoding}-图像.tiff"
    execute("output.image", {"image": source}, path=str(path))
    assert path.read_bytes() == encoded.outputs["encoded"].buffer
    loaded = execute("input.file", path=str(path), color_format=encoding, pixel_depth="unchanged")
    np.testing.assert_array_equal(loaded.outputs["image"], source)
    assert loaded.outputs["info"]["source_pixel_depth"] == np.dtype(dtype).name


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_deflate_floating_predictor_preserves_color_samples(dtype):
    source = ImageArray(np.random.default_rng(99).normal(0, 2, (17, 23, 3)).astype(dtype), "rgb")
    packet = execute(
        "cv.encode_image",
        {"image": source},
        codec="tiff",
        tiff_compression="deflate",
        tiff_predictor="floating",
        tiff_rows_per_strip=5,
    )
    decoded = execute("cv.decode_image", {"encoded": packet.outputs["encoded"]})
    np.testing.assert_array_equal(decoded.outputs["image"], source)


@pytest.mark.parametrize("compression", ["rle", "none"])
def test_radiance_hdr_has_bounded_shared_exponent_loss_and_preserves_range(tmp_path, compression):
    source = np.random.default_rng(7).uniform(0.01, 40, (17, 23, 3)).astype(np.float32)
    source[0, :3] = [[0, 0, 0], [1e4, 1e-5, 1], [1e-34, 1e-34, 1e-34]]
    image = ImageArray(source, "rgb")
    packet = execute("cv.encode_image", {"image": image}, codec="hdr", hdr_compression=compression)
    output = execute("cv.decode_image", {"encoded": packet.outputs["encoded"]}).outputs["image"]
    assert output.dtype == np.float32 and output.color_format == "rgb" and output.max() > 1
    maximum = source.max(axis=2)
    _, exponent = np.frexp(maximum)
    step = np.exp2(exponent.astype(float) - 8)
    bound = np.where(maximum < 1e-32, maximum, step)
    difference = np.abs(source.astype(float) - output)
    assert (difference <= bound[..., None] + 1e-6).all()
    assert difference.max() > 0
    path = tmp_path / f"radiance-{compression}.hdr"
    execute("output.image", {"image": image}, path=str(path), hdr_compression=compression)
    assert path.read_bytes() == packet.outputs["encoded"].buffer
    loaded = execute("input.file", path=str(path), color_format="rgb", pixel_depth="unchanged")
    np.testing.assert_array_equal(loaded.outputs["image"], output)


@pytest.mark.parametrize(
    "source,target",
    [
        ("bgr", "rgb"),
        ("bgra", "rgba"),
        ("bgra", "bgr"),
        ("gray", "bgra"),
        ("bgr", "gray"),
        ("bgr", "rgba"),
    ],
)
def test_float64_color_conversion_keeps_precision_and_alpha(source, target):
    base = np.array([[0.125, 1.23456789012345, -0.5]], dtype=np.float64)
    bgr = np.dstack((base, base + 2.123456789012345, base * 7))
    image = base if source == "gray" else np.dstack((bgr, base / 2)) if source == "bgra" else bgr
    output = format_image(image, target, "unchanged")
    assert output.dtype == np.float64 and output.color_format == target
    color = np.repeat(base[..., None], 3, axis=2) if source == "gray" else bgr
    expected = (
        color[..., 0] * 0.114 + color[..., 1] * 0.587 + color[..., 2] * 0.299
        if target == "gray"
        else color[..., ::-1]
        if target in ("rgb", "rgba")
        else color
    )
    if target in ("bgra", "rgba"):
        expected = np.dstack((expected, base / 2 if source == "bgra" else np.ones(base.shape)))
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_multipage_float_tiff_roundtrip(tmp_path, dtype):
    images = [
        np.random.default_rng(0).normal(0, 2, (17, 23, 3)).astype(dtype),
        np.random.default_rng(1).normal(0, 2, (17, 23)).astype(dtype),
    ]
    group = FlowGroup(tuple(ValuePacket(image, {}) for image in images), ("one", "two"))
    path = tmp_path / "float-pages.tiff"
    execute(
        "output.tiff_file",
        {"pages": group},
        path=str(path),
        compression="deflate",
        predictor="floating",
    )
    reader = builtin_registry().create("input.tiff_file", parameters={"path": str(path)})
    try:
        for expected in images:
            result = reader.execute({})
            assert result is not None, reader.error
            np.testing.assert_array_equal(result.outputs["image"], expected)
    finally:
        reader.close()


def test_invalid_depth_range_predictor_and_output_existence_are_explicit(tmp_path):
    with pytest.raises(ValueError, match="HSV and Lab"):
        format_image(np.ones((8, 9, 3), np.float64), "hsv", "unchanged")
    encoder = builtin_registry().create("cv.encode_image", parameters={"codec": "hdr"})
    assert encoder.execute({"image": np.ones((8, 9), np.float32)}) is None
    assert "float32 RGB/BGR" in encoder.error
    assert encoder.execute({"image": np.full((8, 9, 3), -1, np.float32)}) is None
    assert "nonnegative" in encoder.error
    assert encoder.execute({"image": np.full((8, 9, 3), 2**127, np.float32)}) is None
    assert "below 2**127" in encoder.error
    encoder.parameters.update(codec="tiff", tiff_compression="deflate", tiff_predictor="horizontal")
    assert encoder.execute({"image": np.ones((8, 9), np.float32)}) is None
    assert "predictor does not match" in encoder.error
    encoder.parameters.update(codec="exr", exr_storage="float16")
    assert encoder.execute({"image": np.full((8, 9), 70000, np.float32)}) is None
    assert "65504" in encoder.error
    path = tmp_path / "preserve.png"
    path.write_bytes(b"original")
    saver = builtin_registry().create("output.image", parameters={"path": str(path)})
    assert saver.execute({"image": np.ones((8, 9), np.float32)}) is None
    assert path.read_bytes() == b"original"
    path = tmp_path / "no-silent-downcast.png"
    saver.parameters["path"] = str(path)
    assert saver.execute({"image": np.ones((8, 9), np.float32)}) is None
    assert "use TIFF" in saver.error and not path.exists()


def test_file_saver_keeps_existing_signed_tiff_support(tmp_path):
    image = np.arange(72, dtype=np.int16).reshape(8, 9) - 36
    path = tmp_path / "signed.tiff"
    execute("output.image", {"image": image}, path=str(path))
    decoded = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(decoded, image)


def test_unavailable_exr_fails_without_creating_a_file(tmp_path, monkeypatch):
    native = cv2.haveImageWriter
    monkeypatch.setattr(
        cv2,
        "haveImageWriter",
        lambda extension: False if extension == ".exr" else native(extension),
    )
    image = np.ones((8, 9, 3), np.float32)
    encoder = builtin_registry().create("cv.encode_image", parameters={"codec": "exr"})
    assert encoder.execute({"image": image}) is None
    assert encoder.error_kind == "invalid" and ".exr is unavailable" in encoder.error
    path = tmp_path / "not-created.exr"
    saver = builtin_registry().create("output.image", parameters={"path": str(path)})
    assert saver.execute({"image": image}) is None
    assert not path.exists()


@pytest.mark.parametrize("storage", ["float32", "float16"])
def test_optional_native_exr_roundtrip(storage):
    try:
        available = cv2.haveImageWriter(".exr")
    except cv2.error:
        available = False
    if not available:
        pytest.skip("The installed OpenCV build has no enabled OpenEXR codec")
    source = ImageArray(
        np.random.default_rng(5).uniform(-2, 8, (17, 23, 4)).astype(np.float32), "rgba"
    )
    encoded = execute("cv.encode_image", {"image": source}, codec="exr", exr_storage=storage)
    decoded = execute("cv.decode_image", {"encoded": encoded.outputs["encoded"]}).outputs["image"]
    expected = (
        np.asarray(source).astype(np.float16).astype(np.float32) if storage == "float16" else source
    )
    np.testing.assert_array_equal(decoded, expected)


@pytest.mark.parametrize("project", ["project.json", "float64.json"])
def test_public_float_codec_examples(project):
    path = (
        Path(__file__).resolve().parents[1] / "examples/opencv tools/float_image_codecs" / project
    )
    graph = Graph.load(path, builtin_registry())
    try:
        graph.run()
        assert all(
            block.state == "pass" for block in graph.blocks.values() if not block.layout_only
        )
        assert graph.blocks["verify_tiff"].last_result.outputs["value"] == 0
        if "verify_hdr" in graph.blocks:
            error = graph.blocks["verify_hdr"].last_result.outputs["value"]
            assert 0 < error <= 0.0625
    finally:
        graph.close()
