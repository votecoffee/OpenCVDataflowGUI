"""Native/independent page samples, provenance, bounds and editor persistence."""

import copy
import io
import json
import struct

import cv2
import numpy as np
import pytest
from PIL import Image

from opencv_dataflow.block_xml import save_block_xml
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import DepthArray, ImageArray, format_image, image_format, opencv_image
from opencv_dataflow.multipage_blocks import MULTIPAGE_BLOCKS, DecodePages, EncodePages
from opencv_dataflow.multipage_codec import EncodedPages
from opencv_dataflow.payloads import FlowGroup, ValuePacket, frame, packet_for
from opencv_dataflow.project_xml import read_block_xml
from opencv_dataflow.tickets import TicketClock, ticket_key
from opencv_dataflow.tiff_directory import directory


def successful(block, inputs):
    result = block.execute(inputs)
    assert result is not None, block.error
    assert block.state == "pass", block.error
    return result


def group_of(*images):
    clock = TicketClock()
    packets = []
    for i, image in enumerate(images):
        payload = {
            "ticket": clock.issue(f"camera-{i}"),
            "spatial": frame([image.shape[1], image.shape[0]], f"grid-{i}"),
            "acquisition": {"encoder": 1200 + i},
            "inspection": {"region": "part"},
        }
        packets.append(ValuePacket(image, payload))
    return FlowGroup(tuple(packets), tuple(f"page-{i}" for i in range(len(images))))


@pytest.fixture
def group():
    y, x = np.indices((17, 23))
    rgb = ImageArray(np.dstack((x * 2000, y * 3000, x * 500 + y)).astype(np.uint16), "rgb")
    gray = ((x[:11, :15] + y[:11, :15]) * 7).astype(np.uint8)
    floating = np.linspace(-20, 200, 7 * 13 * 4).reshape(7, 13, 4).astype(np.float64)
    floating[..., 3] = 0.75
    return group_of(gray, rgb, ImageArray(floating, "rgba"))


@pytest.mark.parametrize("compression", ["none", "lzw", "deflate", "packbits"])
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_mixed_native_roundtrip_preserves_order_types_and_page_context(group, compression, profile):
    original = copy.deepcopy(group)
    graph = Graph()
    encoder = graph.add(
        EncodePages(
            parameters={"compression": compression, "rows_per_strip": 3, "x_dpi": 300, "y_dpi": 300}
        )
    )
    graph.set_run_profile(profile)
    encoded = successful(encoder, {"pages": group})
    decoder = DecodePages()
    decoded = successful(decoder, {"encoded": packet_for(encoded, "encoded")})
    restored = decoded.outputs["pages"]
    assert restored.labels == group.labels
    assert decoded.outputs["info"]["restored_page_context"]
    assert encoded.outputs["info"]["exact_samples"]
    assert encoded.outputs["info"]["requested_compression"] == compression
    actual_pages = directory(encoded.outputs["buffer"])
    assert [p["compression_code"] for p in encoded.outputs["info"]["pages"]] == [
        p.compression for p in actual_pages
    ]
    assert decoded.outputs["info"]["selected_indexes"] == [0, 1, 2]
    for source, output, saved in zip(group.items, restored.items, original.items, strict=True):
        np.testing.assert_array_equal(source.value, saved.value)
        assert source.value.tobytes() == output.value.tobytes()
        assert image_format(source.value) == image_format(output.value)
        assert source.value.dtype == output.value.dtype
        assert source.payload["spatial"] == output.payload["spatial"]
        assert source.payload["acquisition"] == output.payload["acquisition"]
        assert ticket_key(source.payload["ticket"]) == ticket_key(output.payload["ticket"])
    json.dumps(encoded.outputs["encoded"].describe(), allow_nan=False)
    graph.close()


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
@pytest.mark.parametrize("color", ["gray", "bgr", "rgb", "bgra", "rgba"])
def test_all_supported_sample_depths_and_logical_colors(dtype, color):
    channels = 1 if color == "gray" else 4 if "a" in color else 3
    shape = (11, 9) if channels == 1 else (11, 9, channels)
    values = (np.arange(np.prod(shape)).reshape(shape) % 251).astype(dtype)
    if values.dtype.kind == "f":
        values = values / 13 - 6
        values.flat[0] = -0.0
    image = ImageArray(values, color)
    packet = successful(EncodePages(), {"pages": group_of(image)}).outputs["encoded"]
    restored = successful(DecodePages(), {"encoded": packet}).outputs["image"]
    assert restored.dtype == image.dtype and restored.tobytes() == image.tobytes()
    assert image_format(restored) == color


def test_independent_integer_tiff_reader_and_resolution_tags():
    rgb = ImageArray(np.full((9, 13, 3), [17, 103, 245], np.uint8), "rgb")
    gray = np.arange(63, dtype=np.uint16).reshape(7, 9) * 613
    result = successful(
        EncodePages(parameters={"compression": "deflate", "x_dpi": 240, "y_dpi": 300}),
        {"pages": group_of(rgb, gray)},
    )
    with Image.open(io.BytesIO(result.outputs["buffer"])) as independent:
        assert independent.n_frames == 2
        np.testing.assert_array_equal(np.asarray(independent), rgb)
        assert tuple(independent.info["dpi"]) == (240, 300)
        independent.seek(1)
        np.testing.assert_array_equal(np.asarray(independent), gray)


def test_stride_subset_preview_conversion_and_native_call_ranges(group, monkeypatch):
    encoded = successful(EncodePages(), {"pages": group}).outputs["encoded"]
    calls, native = [], cv2.imdecodemulti

    def observed(*args, **kwargs):
        calls.append(kwargs["range"])
        return native(*args, **kwargs)

    monkeypatch.setattr(cv2, "imdecodemulti", observed)
    decoder = DecodePages(
        parameters={
            "stride": 2,
            "page_count": 2,
            "preview_page": 1,
            "color_format": "gray",
            "pixel_depth": "float32",
        }
    )
    result = successful(decoder, {"encoded": encoded})
    assert calls == [(0, 1), (2, 3)]
    assert result.outputs["pages"].labels == ("page-0", "page-2")
    assert result.outputs["info"]["selected_indexes"] == [0, 2]
    np.testing.assert_array_equal(
        result.outputs["image"], format_image(opencv_image(group.items[2].value), "gray", "float32")
    )
    assert result.payloads["image"]["ticket"] == group.items[2].payload["ticket"]


@pytest.mark.parametrize("vector", ["bytes", "flat", "row", "column"])
def test_raw_buffers_use_canonical_order_and_new_grids(group, vector):
    raw = successful(EncodePages(), {"pages": group}).outputs["buffer"]
    value = raw if vector == "bytes" else np.frombuffer(raw, np.uint8)
    if vector in ("row", "column"):
        value = value.reshape(1, -1) if vector == "row" else value.reshape(-1, 1)
    input_size = (
        list(value.shape[1::-1]) if isinstance(value, np.ndarray) and value.ndim == 2 else [1, 1]
    )
    context = {
        "ticket": group.items[0].payload["ticket"],
        "spatial": frame(input_size, "unrelated-input"),
        "arbitrary": {"station": 4},
    }
    result = successful(DecodePages(), {"encoded": ValuePacket(value, context)})
    assert not result.outputs["info"]["restored_page_context"]
    for i, packet in enumerate(result.outputs["pages"].items):
        np.testing.assert_array_equal(packet.value, opencv_image(group.items[i].value))
        assert packet.payload["ticket"] == context["ticket"]
        assert packet.payload["arbitrary"] == context["arbitrary"]
        assert packet.payload["spatial"]["image_size"] == list(packet.value.shape[1::-1])
        assert packet.payload["spatial"]["root_id"] != "unrelated-input"
    assert len({p.payload["spatial"]["root_id"] for p in result.outputs["pages"].items}) == 3


def test_immutable_packet_context_is_validated_and_not_aliased(group):
    packet = successful(EncodePages(), {"pages": group}).outputs["encoded"]
    contexts = packet.contexts
    contexts[0]["payload"]["acquisition"]["encoder"] = -1
    assert packet.contexts[0]["payload"]["acquisition"]["encoder"] == 1200
    for mutate in (
        lambda p: p[0].update(size=[2, 3]),
        lambda p: p[0].update(dtype="uint16"),
        lambda p: p[0].update(color_format="rgba"),
        lambda p: p.pop(),
        lambda p: p[0].update(label="x" * 257),
        lambda p: p[0]["payload"].update(spatial=frame([1, 1])),
    ):
        entries = packet.contexts
        mutate(entries)
        with pytest.raises(ValueError):
            EncodedPages(packet.buffer, json.dumps({"version": 1, "pages": entries}).encode())


@pytest.mark.parametrize(
    "settings",
    [
        {"maximum_pixels": 1},
        {"maximum_decoded_bytes": 1},
        {"start_page": 5},
        {"preview_page": 3},
        {"color_format": "hsv"},
    ],
)
def test_selection_and_conversion_budget_rejected_before_pixel_decoding(
    group, settings, monkeypatch
):
    data = successful(EncodePages(), {"pages": group}).outputs["encoded"]
    calls = []
    monkeypatch.setattr(cv2, "imdecodemulti", lambda *a, **k: calls.append(1))
    block = DecodePages(parameters=settings)
    assert block.execute({"encoded": data}) is None and block.error_kind == "invalid"
    assert not calls and not block.current_emissions


def alter_tag(data, tag, *, value=None, count=None):
    data = bytearray(data)
    offset = int.from_bytes(data[4:8], "little")
    fields = int.from_bytes(data[offset : offset + 2], "little")
    for i in range(fields):
        entry = offset + 2 + 12 * i
        if int.from_bytes(data[entry : entry + 2], "little") == tag:
            if value is not None:
                struct.pack_into("I", data, entry + 8, value)
            if count is not None:
                struct.pack_into("I", data, entry + 4, count)
            return bytes(data)
    raise AssertionError(f"Missing test tag {tag}")


@pytest.mark.parametrize(
    "mutation", ["huge_width", "huge_samples", "offset", "cycle", "truncated", "orientation"]
)
def test_malformed_or_excessive_declarations_never_reach_native_decoder(
    group, mutation, monkeypatch
):
    raw = successful(EncodePages(), {"pages": group}).outputs["buffer"]
    if mutation == "huge_width":
        raw = alter_tag(raw, 256, value=50000)
    elif mutation == "huge_samples":
        raw = alter_tag(raw, 258, count=2**31)
    elif mutation == "offset":
        raw = alter_tag(raw, 273, value=2**31)
    elif mutation == "orientation":
        raw = alter_tag(raw, 284, value=2)
    elif mutation == "truncated":
        raw = raw[:31]
    else:
        raw = bytearray(raw)
        offset = int.from_bytes(raw[4:8], "little")
        fields = int.from_bytes(raw[offset : offset + 2], "little")
        struct.pack_into("I", raw, offset + 2 + 12 * fields, offset)
        raw = bytes(raw)
    calls = []
    monkeypatch.setattr(cv2, "imdecodemulti", lambda *a, **k: calls.append(1))
    block = DecodePages()
    assert block.execute({"encoded": raw}) is None, mutation
    assert block.error_kind == "invalid" and not calls and not block.current_emissions


@pytest.mark.parametrize(
    "settings",
    [
        {"compression": "deflate", "predictor": "horizontal"},
        {"compression": "lzw", "predictor": "floating"},
    ],
)
def test_predictor_requires_homogeneous_sample_kind(group, settings):
    block = EncodePages(parameters=settings)
    assert block.execute({"pages": group}) is None and "sample type" in block.error


@pytest.mark.parametrize(
    "settings",
    [
        {"compression": "none", "predictor": "horizontal"},
        {"x_dpi": 300},
        {"rows_per_strip": -1},
        {"compression": "jpeg"},
    ],
)
def test_invalid_controls_are_contained_without_output(group, settings):
    block = EncodePages(parameters=settings)
    assert block.execute({"pages": group}) is None and block.error_kind == "invalid"
    assert not block.current_emissions


def test_native_sample_loss_and_exceptions_cannot_publish_success(group, monkeypatch):
    native = cv2.imdecodemulti

    def lossy(*args, **kwargs):
        ok, pages = native(*args, **kwargs)
        pages[0].flat[0] += 1
        return ok, pages

    monkeypatch.setattr(cv2, "imdecodemulti", lossy)
    block = EncodePages()
    assert block.execute({"pages": group}) is None and "changed samples" in block.error
    assert not block.current_emissions

    def error(*args, **kwargs):
        raise cv2.error("controlled decoder failure")

    monkeypatch.setattr(cv2, "imdecodemulti", error)
    assert block.execute({"pages": group}) is None and "controlled decoder" in block.error


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((2, 2), np.int16),
        np.full((2, 2), np.nan, np.float32),
        DepthArray(np.ones((2, 2))),
        "not pixels",
    ],
)
def test_unsupported_page_data_is_contained(value):
    group = FlowGroup((ValuePacket(value, {}),), ("page",))
    block = EncodePages()
    assert block.execute({"pages": group}) is None and block.error_kind == "invalid"


def test_saved_graph_and_block_xml_preserve_all_controls(tmp_path):
    registry, graph = builtin_registry(), Graph()
    for cls in MULTIPAGE_BLOCKS:
        block = graph.add(registry.create(cls.type_id))
        path = tmp_path / ("settings." + cls.type_id + ".xml")
        save_block_xml(block, path)
        assert read_block_xml(path)["parameters"] == block.parameters
    path = tmp_path / "pages.json"
    graph.save(path)
    restored = Graph.load(path, registry)
    assert restored.to_dict() == graph.to_dict()
    graph.close()
    restored.close()


def test_pyramid_and_image_attached_metadata_survive(group):
    page = group.items[1]
    page.payload["pyramid"] = {
        "level": 1,
        "largest_size": [46, 34],
        "to_largest": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
    }
    page.value.metadata = {"medical": {"spacing": [0.4, 0.4], "source": "reference"}}
    packet = successful(EncodePages(), {"pages": group}).outputs["encoded"]
    result = successful(
        DecodePages(parameters={"start_page": 1, "page_count": 1}), {"encoded": packet}
    )
    payload = result.outputs["pages"].items[0].payload
    assert payload["pyramid"] == page.payload["pyramid"]
    assert payload["medical"] == page.value.metadata["medical"]
    assert payload["spatial"] == page.payload["spatial"]


def test_large_page_inventory_requires_bounded_explicit_subset(monkeypatch):
    images = [np.full((2, 3), i, np.uint8) for i in range(65)]
    ok, raw = cv2.imencodemulti(".tiff", images, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
    assert ok and len(directory(raw.tobytes())) == 65
    calls, native = [], cv2.imdecodemulti

    def observed(*args, **kwargs):
        calls.append(kwargs["range"])
        return native(*args, **kwargs)

    monkeypatch.setattr(cv2, "imdecodemulti", observed)
    block = DecodePages()
    assert block.execute({"encoded": raw}) is None and "64 pages" in block.error
    assert not calls
    block.parameters.update(start_page=60, page_count=3, stride=2)
    result = successful(block, {"encoded": raw})
    assert calls == [(60, 61), (62, 63), (64, 65)]
    assert [int(p.value[0, 0]) for p in result.outputs["pages"].items] == [60, 62, 64]


@pytest.mark.parametrize("kind,dtype", [("horizontal", np.uint16), ("floating", np.float32)])
def test_explicit_predictors_roundtrip_when_supported(kind, dtype):
    images = [np.linspace(0, 100, 99).reshape(9, 11).astype(dtype) for _ in range(2)]
    result = successful(
        EncodePages(parameters={"compression": "deflate", "predictor": kind}),
        {"pages": group_of(*images)},
    )
    restored = successful(DecodePages(), {"encoded": result.outputs["encoded"]})
    for packet in restored.outputs["pages"].items:
        assert packet.value.tobytes() == images[0].tobytes()


def test_toolbox_controls_and_nondefault_xml_roundtrip(app, tmp_path):
    from PySide6.QtWidgets import QComboBox

    from opencv_dataflow.app import MainWindow
    from opencv_dataflow.toolbox import block_entry

    graph = Graph()
    encoder = graph.add(
        EncodePages(
            parameters={
                "compression": "lzw",
                "predictor": "horizontal",
                "rows_per_strip": 5,
                "x_dpi": 150,
                "y_dpi": 300,
            }
        )
    )
    decoder = graph.add(
        DecodePages(
            parameters={
                "start_page": 1,
                "page_count": 2,
                "stride": 3,
                "preview_page": 1,
                "color_format": "hsv",
                "pixel_depth": "float32",
                "maximum_pixels": 1234,
                "maximum_decoded_bytes": 2**20,
            }
        )
    )
    window = MainWindow()
    try:
        window.replace_graph(graph)
        for block in (encoder, decoder):
            assert block_entry(type(block)).path == ("OpenCV", "Image Codecs (imgcodecs)", "Codecs")
            window.inspector.select(block)
            for field in block.parameter_choices:
                assert isinstance(window.inspector.fields[field], QComboBox)
        path = tmp_path / "multipage.xml"
        graph.save(path)
        restored = Graph.load(path, builtin_registry())
        try:
            assert restored.to_dict() == graph.to_dict()
        finally:
            restored.close()
    finally:
        window.dirty = False
        window.close()
        app.processEvents()


@pytest.mark.parametrize("endian,big", [("<", False), (">", False), ("<", True), (">", True)])
def test_directory_byte_orders_and_classic_bigtiff_layouts(endian, big):
    # Independent one-pixel, one-strip TIFF builder; no native writer in this reference.
    count_format, offset_format = ("Q", "Q") if big else ("H", "I")
    entry_format, inline = ("HHQQ", 8) if big else ("HHII", 4)
    start = 16 if big else 8
    tags = [(256, 1), (257, 1), (258, 8), (259, 1), (262, 1), (273, 0), (279, 1)]
    pixel = (
        start
        + struct.calcsize(count_format)
        + len(tags) * struct.calcsize(endian + entry_format)
        + inline
    )
    header = b"II" if endian == "<" else b"MM"
    header += struct.pack(
        endian + ("HHHQ" if big else "HI"), *((43, 8, 0, start) if big else (42, start))
    )
    raw = header + struct.pack(endian + count_format, len(tags))
    for tag, value in tags:
        value = pixel if tag == 273 else value
        # A one-element SHORT is in the first two inline bytes in either byte order.
        raw += struct.pack(endian + "HH" + offset_format, tag, 3, 1)
        raw += struct.pack(endian + "H", value).ljust(inline, b"\0")
    raw += struct.pack(endian + offset_format, 0) + b"\x79"
    pages = directory(raw)
    assert pages[0].size == (1, 1) and pages[0].nbytes == 1
    assert successful(DecodePages(), {"encoded": raw}).outputs["image"][0, 0] == 121


@pytest.mark.parametrize("fault", [None, "tile_allocation", "outside_buffer"])
def test_independent_tiled_tiff_and_predecode_tile_guards(fault, monkeypatch):
    tile = 16384 if fault == "tile_allocation" else 32
    tags = [
        (256, 1),
        (257, 1),
        (258, 8),
        (259, 1),
        (262, 1),
        (322, tile),
        (323, tile),
        (324, 0),
        (325, 1024),
    ]
    pixel = 8 + 2 + len(tags) * 12 + 4
    raw = b"II" + struct.pack("<HIH", 42, 8, len(tags))
    for tag, value in tags:
        if tag == 324:
            value = 65500 if fault == "outside_buffer" else pixel
        raw += struct.pack("<HHI", tag, 3, 1) + struct.pack("<H", value).ljust(4, b"\0")
    raw += struct.pack("<I", 0) + b"\x79" * 1024
    calls, native = [], cv2.imdecodemulti

    def observed(*args, **kwargs):
        calls.append(True)
        return native(*args, **kwargs)

    monkeypatch.setattr(cv2, "imdecodemulti", observed)
    decoder = DecodePages()
    result = decoder.execute({"encoded": raw})
    if fault:
        assert result is None and decoder.error_kind == "invalid"
        assert not calls
    else:
        assert result is not None, decoder.error
        assert result.outputs["image"][0, 0] == 121 and calls
