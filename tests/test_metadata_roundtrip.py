"""Native metadata round trips, explicit orientation and publish-only-after-verification."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import ImageCms

from opencv_dataflow.block_xml import save_block_xml
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.codec_metadata import ImageMetadata, make_exif
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import ValuePacket, frame, packet_for
from opencv_dataflow.project_xml import read_block_xml
from opencv_dataflow.tickets import TicketClock, ticket_key

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/metadata_roundtrip"


def chunks(orientation=1, *, color=True):
    values = [(0, make_exif(orientation, "Dataflow test", "Original synthetic fixture"))]
    if color:
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        values += [
            (1, b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><value>Original</value></x:xmpmeta>'),
            (2, profile),
            (3, bytes([1, 13, 0, 1])),
        ]
    return ImageMetadata(tuple(values))


def execute(block, **inputs):
    result = block.execute(inputs)
    assert result is not None, block.error
    return result


@pytest.mark.parametrize("encoding", ["gray", "rgb", "rgba"])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_png_supported_chunks_pixels_ticket_frame_and_repeated_xmp(encoding, dtype):
    raw = np.arange(
        24 * 32 * (1 if encoding == "gray" else 4 if encoding == "rgba" else 3), dtype=dtype
    )
    raw = raw.reshape((24, 32) if encoding == "gray" else (24, 32, 4 if encoding == "rgba" else 3))
    image = ImageArray(raw, encoding)
    metadata = chunks()
    if encoding == "gray":
        metadata = ImageMetadata(tuple(chunk for chunk in metadata.chunks if chunk[0] != 2))
    expected_metadata = dict(metadata.chunks)
    context = {
        "ticket": TicketClock().issue("camera"),
        "spatial": frame([32, 24], "sensor"),
        "lot": "fixture",
    }
    registry = builtin_registry()
    encoder = registry.create(
        "cv.encode_image", parameters={"png_compression": 9, "png_strategy": "rle"}
    )
    decoder = registry.create("cv.decode_image", parameters={"metadata_mode": "read"})
    for _ in range(3):
        encoded = execute(encoder, image=ValuePacket(image, context), metadata=metadata)
        decoded = execute(decoder, encoded=packet_for(encoded, "encoded"))
        image, metadata = decoded.outputs["image"], decoded.outputs["metadata"]
        assert dict(metadata.chunks) == expected_metadata
        assert not dict(metadata.chunks)[1].endswith(b"\0")
        np.testing.assert_array_equal(image, raw)
        assert image.dtype == dtype and image.color_format == encoding
        assert decoded.payloads["image"]["ticket"] == context["ticket"]
        assert decoded.payloads["image"]["spatial"] == context["spatial"]
        assert decoded.payloads["image"]["lot"] == "fixture"


@pytest.mark.parametrize("codec", ["png", "jpeg"])
def test_native_file_writer_and_memory_writer_share_metadata_and_options(codec, tmp_path):
    registry = builtin_registry()
    image = ImageArray(
        np.random.default_rng(9).integers(0, 256, (24, 32, 3), dtype=np.uint8), "rgb"
    )
    meta = chunks(color=codec == "png")
    if codec == "jpeg":
        meta = ImageMetadata(tuple(chunk for chunk in chunks().chunks if chunk[0] in (0, 1, 2)))
    extension = ".png" if codec == "png" else ".jpg"
    parameters = {"png_compression": 7, "jpeg_quality": 93, "jpeg_progressive": True}
    encoder = registry.create("cv.encode_image", parameters={"codec": codec, **parameters})
    memory = execute(encoder, image=image, metadata=meta).outputs["encoded"].buffer
    path = tmp_path / ("\u6d4b\u8bd5-\u00e9" + extension)
    sink = registry.create("output.image", parameters={"path": str(path), **parameters})
    execute(sink, image=image, metadata=meta)
    assert path.read_bytes() == memory
    pixels, types, raw = cv2.imdecodeWithMetadata(
        np.frombuffer(memory, np.uint8), cv2.IMREAD_UNCHANGED
    )
    actual = ImageMetadata(
        tuple((int(k), item.tobytes()) for k, item in zip(types, raw, strict=True))
    )
    assert dict(actual.chunks) == dict(meta.chunks)
    assert pixels.shape == image.shape
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sink.execute({"image": image, "metadata": meta}) is None
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert not list(tmp_path.glob(".metadata-*"))


@pytest.mark.parametrize("orientation", range(1, 9))
def test_decode_orientation_uses_known_pixels_and_correct_coordinate_transform(orientation):
    registry = builtin_registry()
    pixels = np.arange(6, dtype=np.uint16).reshape(2, 3)
    context = {"spatial": frame([3, 2], "sensor"), "ticket": TicketClock().issue("sensor")}
    encoder = registry.create("cv.encode_image")
    source = execute(
        encoder, image=ValuePacket(pixels, context), metadata=chunks(orientation, color=False)
    )
    decoder = registry.create("cv.decode_image", parameters={"metadata_mode": "apply_orientation"})
    decoded = execute(decoder, encoded=packet_for(source, "encoded"))
    expected = {
        1: pixels,
        2: pixels[:, ::-1],
        3: pixels[::-1, ::-1],
        4: pixels[::-1],
        5: pixels.T,
        6: np.rot90(pixels, -1),
        7: pixels.T[::-1, ::-1],
        8: np.rot90(pixels),
    }[orientation]
    image = decoded.outputs["image"]
    np.testing.assert_array_equal(image, expected)
    output = decoded.payloads["image"]["spatial"]
    assert output["image_size"] == [image.shape[1], image.shape[0]]
    assert ticket_key(decoded.payloads["image"]["ticket"]) == ticket_key(context["ticket"])
    y, x = np.indices(image.shape)
    native = np.asarray(output["to_root"]) @ np.stack([x.ravel(), y.ravel(), np.ones(x.size)])
    sampled = pixels[np.rint(native[1]).astype(int), np.rint(native[0]).astype(int)].reshape(
        image.shape
    )
    np.testing.assert_array_equal(sampled, image)
    if orientation != 1:
        assert encoder.execute({"image": image, "metadata": decoded.outputs["metadata"]}) is None
        assert encoder.error_kind == "invalid" and "fresh metadata" in encoder.error
    fresh = execute(registry.create("cv.image_metadata"), **{}).outputs["metadata"]
    normalized = execute(encoder, image=image, metadata=fresh)
    twice = execute(decoder, encoded=packet_for(normalized, "encoded"))
    np.testing.assert_array_equal(twice.outputs["image"], expected)
    assert not twice.outputs["metadata"].pixels_oriented


def test_exposed_file_metadata_follows_filename_ticket_and_orientation_state(tmp_path):
    registry = builtin_registry()
    path = tmp_path / "source.png"
    sink = registry.create("output.image", parameters={"path": str(path)})
    execute(
        sink, image=np.arange(12, dtype=np.uint8).reshape(3, 4), metadata=chunks(6, color=False)
    )
    source = registry.create("input.file", parameters={"path": str(path), "metadata_mode": "read"})
    assert "metadata" not in source.exposed_outputs
    source.exposed_outputs.append("metadata")
    raw = execute(source)
    assert raw.outputs["metadata"].image_size == (4, 3)
    assert raw.payloads["metadata"]["spatial"] == raw.payloads["image"]["spatial"]
    assert raw.outputs["ticket"]["value"] == "source.png"
    source.parameters["metadata_mode"] = "apply_orientation"
    oriented = execute(source)
    assert oriented.outputs["metadata"].pixels_oriented


@pytest.mark.parametrize("backend", ["buffer", "file"])
def test_silent_native_metadata_loss_fails_before_file_publication(backend, tmp_path, monkeypatch):
    registry = builtin_registry()
    image = np.zeros((12, 16, 3), np.uint8)
    if backend == "buffer":
        monkeypatch.setattr(
            cv2,
            "imencodeWithMetadata",
            lambda ext, image, types, data, options: cv2.imencode(ext, image, options),
        )
        block = registry.create("cv.encode_image")
    else:
        monkeypatch.setattr(
            cv2,
            "imwriteWithMetadata",
            lambda path, image, types, data, options: cv2.imwrite(path, image, options),
        )
        block = registry.create(
            "output.image", parameters={"path": str(tmp_path / "unpublished.png")}
        )
    assert block.execute({"image": image, "metadata": chunks(color=False)}) is None
    assert block.error_kind == "invalid" and "did not preserve" in block.error
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "case", ["duplicate", "unknown", "oversized", "exif", "xmp", "icc", "cicp", "shape", "oriented"]
)
def test_invalid_packets_and_native_limits_are_explicit(case):
    data = chunks(color=False)
    if case == "duplicate":
        with pytest.raises(ValidationError):
            ImageMetadata(data.chunks * 2)
        return
    if case == "unknown":
        with pytest.raises(ValidationError):
            ImageMetadata(((7, b"unknown"),))
        return
    if case in ("oversized", "exif", "xmp", "icc", "cicp"):
        kind, raw = {
            "oversized": (1, b"x" * 1_048_577),
            "exif": (0, b"II*"),
            "xmp": (1, b"<broken>"),
            "icc": (2, b"not a profile"),
            "cicp": (3, b"abc"),
        }[case]
        with pytest.raises(ValidationError):
            ImageMetadata(((kind, raw),))
        return
    data = ImageMetadata(data.chunks, (20, 20) if case == "shape" else None, case == "oriented")
    encoder = builtin_registry().create("cv.encode_image")
    assert encoder.execute({"image": np.zeros((12, 16), np.uint8), "metadata": data}) is None
    assert encoder.error_kind == "invalid"


@pytest.mark.parametrize("codec", ["jpeg", "tiff"])
def test_unsupported_codec_chunks_emit_nothing(codec, tmp_path):
    meta = chunks()
    block = builtin_registry().create("cv.encode_image", parameters={"codec": codec})
    assert block.execute({"image": np.zeros((16, 24, 3), np.uint8), "metadata": meta}) is None
    assert block.error_kind == "invalid" and not block.current_emissions


def test_metadata_grid_must_match_processed_image_and_creation_fields_are_validated():
    registry = builtin_registry()
    meta = chunks(color=False)
    block = registry.create("cv.encode_image")
    image = ImageArray(np.zeros((12, 16), np.uint8), "gray", {"spatial": frame([16, 12], "A")})
    assert (
        block.execute(
            {"image": image, "metadata": ValuePacket(meta, {"spatial": frame([16, 12], "B")})}
        )
        is None
    )
    assert block.error_kind == "invalid" and "frames differ" in block.error
    maker = registry.create("cv.image_metadata", parameters={"software": "\u00e9"})
    assert maker.execute({}) is None and maker.error_kind == "invalid"
    maker.parameters.update(software="test", cicp=[1, 2])
    assert maker.execute({}) is None and maker.error_kind == "invalid"
    maker.parameters.update(cicp=[], include_exif=False)
    assert execute(maker).outputs["metadata"].chunks == ()


def test_jpeg_xmp_roundtrips_repeatedly_without_accumulating_namespace_headers():
    registry = builtin_registry()
    metadata = ImageMetadata(tuple(item for item in chunks().chunks if item[0] != 3))
    expected = dict(metadata.chunks)
    image = np.zeros((16, 24, 3), np.uint8)
    for _ in range(3):
        encoder = registry.create("cv.encode_image", parameters={"codec": "jpeg"})
        encoded = execute(encoder, image=image, metadata=metadata)
        decoder = registry.create("cv.decode_image", parameters={"metadata_mode": "read"})
        decoded = execute(decoder, encoded=packet_for(encoded, "encoded"))
        image, metadata = decoded.outputs["image"], decoded.outputs["metadata"]
        assert dict(metadata.chunks) == expected


def test_incompatible_rgb_profile_is_rejected_for_grayscale_before_native_write():
    encoder = builtin_registry().create("cv.encode_image")
    assert encoder.execute({"image": np.zeros((16, 24), np.uint8), "metadata": chunks()}) is None
    assert encoder.error_kind == "invalid" and "ICC color space" in encoder.error


def test_explicit_empty_or_summary_metadata_is_not_silently_treated_as_disconnected():
    encoder = builtin_registry().create("cv.encode_image")
    for value in (None, {"chunks": []}):
        assert encoder.execute({"image": np.zeros((16, 24), np.uint8), "metadata": value}) is None
        assert encoder.error_kind == "invalid" and not encoder.current_emissions


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize(
    "name", ["png_roundtrip", "jpeg_roundtrip", "oriented_once", "unsupported_jpeg_cicp"]
)
def test_runnable_examples_exact_png_orientation_metadata_and_presets(profile, name, tmp_path):
    registry = builtin_registry()
    graph = Graph.load(EXAMPLE / (name + ".json"), registry)
    truth = json.loads((EXAMPLE / "truth.json").read_text())
    try:
        graph.set_run_profile(profile)
        graph.run()
        encoder = graph.blocks["encode"]
        if name == "unsupported_jpeg_cicp":
            assert encoder.error_kind == "invalid" and "cicp" in encoder.error
            assert not encoder.current_emissions
            return
        assert all(not b.error for b in graph.blocks.values())
        final = graph.blocks["final" if "final" in graph.blocks else "decode"].last_result
        source = graph.blocks["source"].last_result
        result = final.outputs["image"]
        metadata = final.outputs["metadata"].describe()
        assert [item["type"] for item in metadata["chunks"]] == truth[
            "jpeg_chunk_types" if name == "jpeg_roundtrip" else "png_chunk_types"
        ]
        if name == "oriented_once":
            reference = cv2.imread(str(EXAMPLE / "oriented_reference.png"), cv2.IMREAD_UNCHANGED)[
                ..., [2, 1, 0, 3]
            ]
            np.testing.assert_array_equal(result, reference)
            assert metadata["exif"]["orientation"] == 1 and not metadata["pixels_oriented"]
            np.testing.assert_array_equal(
                final.payloads["image"]["spatial"]["to_root"], [[0, 1, 0], [-1, 0, 31], [0, 0, 1]]
            )
        elif name == "png_roundtrip":
            np.testing.assert_array_equal(result, source.outputs["image"])
            assert result.dtype == np.uint16 and result.color_format == "rgba"
            assert metadata["exif"]["orientation"] == 6
            assert dict(final.outputs["metadata"].chunks) == dict(source.outputs["metadata"].chunks)
        else:
            assert result.dtype == np.uint8 and result.color_format == "rgb"
            assert np.abs(result.astype(float) - source.outputs["image"]).mean() < 8
        assert ticket_key(final.payloads["image"]["ticket"]) == ticket_key(
            source.payloads["image"]["ticket"]
        )
        preset = tmp_path / "encoder.cv.encode_image.xml"
        save_block_xml(encoder, preset)
        assert read_block_xml(preset) == encoder.to_project_item()
        if "fresh" in graph.blocks:
            maker = graph.blocks["fresh"]
            preset = tmp_path / "metadata.cv.image_metadata.xml"
            save_block_xml(maker, preset)
            assert read_block_xml(preset) == maker.to_project_item()
    finally:
        graph.close()
