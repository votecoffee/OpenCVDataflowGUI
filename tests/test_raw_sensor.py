"""Known packing bytes, stride bounds and Bayer phase through image geometry."""

import json
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Block, Graph, Port, Result
from opencv_dataflow.payloads import ValuePacket, frame, packet_for
from opencv_dataflow.runner import describe
from opencv_dataflow.tickets import TicketClock


def execute(kind, inputs=None, **parameters):
    block = builtin_registry().create(kind, parameters=parameters)
    result = block.execute(inputs or {})
    assert result is not None, block.error
    return block, result


def pack_reference(image, bits, header=b"", padding=0):
    """Scalar encoder independent of the decoder's vectorized row extraction."""
    pixels, low_bits = (4, 2) if bits == 10 else (2, 4)
    result = bytearray(header)
    for row in image:
        for start in range(0, len(row), pixels):
            samples = [int(value) for value in row[start : start + pixels]]
            result.extend(value >> low_bits for value in samples)
            result.append(
                sum(
                    (value % (1 << low_bits)) * (1 << (index * low_bits))
                    for index, value in enumerate(samples)
                )
            )
        result.extend([0xEE] * padding)
    return bytes(result)


@pytest.mark.parametrize(
    "packing,encoded,expected",
    [
        ("mipi_raw10", "00 00 00 00 e4 ff ff ff ff e4", [[0, 1, 2, 3, 1020, 1021, 1022, 1023]]),
        ("mipi_raw10", "55 aa ff 00 39", [[341, 682, 1023, 0]]),
        ("mipi_raw12", "00 00 10 ff ff fe 12 ab c3", [[0, 1, 4094, 4095, 0x123, 0xABC]]),
    ],
)
def test_published_layout_golden_bytes(packing, encoded, expected):
    data = bytes.fromhex(encoded)
    for buffer in (data, bytearray(data), np.frombuffer(data, np.uint8)):
        _, result = execute(
            "cv.decode_raw_sensor",
            {"buffer": buffer},
            packing=packing,
            width=len(expected[0]),
            height=1,
        )
        np.testing.assert_array_equal(result.outputs["image"], np.asarray(expected, np.uint16))


@pytest.mark.parametrize("bits", [10, 12])
@pytest.mark.parametrize("alignment", ["right", "left"])
def test_stride_offset_alignment_and_wire_payload(bits, alignment):
    source = np.random.default_rng(44).integers(0, 1 << bits, (7, 12), dtype=np.uint16)
    data = pack_reference(source, bits, b"header", 7)
    buffer = np.frombuffer(data, np.uint8).reshape(1, -1).copy()
    saved = buffer.copy()
    ticket = TicketClock().issue("camera")
    pixel_grid = frame([12, 7], "sensor")
    pixel_grid.update(root_size=[640, 480], to_root=[[1, 0, 21], [0, 1, 17], [0, 0, 1]])
    block, result = execute(
        "cv.decode_raw_sensor",
        {
            "buffer": ValuePacket(
                buffer,
                {
                    "spatial": frame([len(data), 1], "transport"),
                    "ticket": ticket,
                    "plc": {"encoder": 123},
                },
            ),
            "pixel_frame": pixel_grid,
        },
        packing=f"mipi_raw{bits}",
        width=12,
        height=7,
        byte_offset=6,
        row_stride=12 * bits // 8 + 7,
        alignment=alignment,
        bayer_pattern="BGGR",
        column_offset="odd",
    )
    np.testing.assert_array_equal(
        result.outputs["image"], source if alignment == "right" else source << (16 - bits)
    )
    np.testing.assert_array_equal(buffer, saved)
    payload = result.payloads["image"]
    assert payload["spatial"] == pixel_grid and payload["ticket"] == ticket
    assert payload["plc"]["encoder"] == 123
    assert payload["raw_sensor"]["bit_depth"] == bits
    assert payload["raw_sensor"]["column_offset"] == "odd"
    assert result.outputs["info"]["row_stride"] == 12 * bits // 8 + 7
    assert result.graphics["sensor counts preview"].dtype == np.uint8
    block.set_graphics_mode("off")
    lean = block.execute({"buffer": data, "pixel_frame": pixel_grid})
    assert lean is not None and not lean.graphics and not lean.drawings


@pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
@pytest.mark.parametrize(
    "row,column", [("even", "even"), ("even", "odd"), ("odd", "even"), ("odd", "odd")]
)
def test_every_bayer_sensor_phase_recovers_expected_color(pattern, row, column):
    colors = {"R": 3000, "G": 1500, "B": 600}
    oy, ox = int(row == "odd"), int(column == "odd")
    mosaic = np.array(
        [
            [colors[pattern[2 * ((y + oy) % 2) + (x + ox) % 2]] for x in range(16)]
            for y in range(12)
        ],
        np.uint16,
    )
    _, decoded = execute(
        "cv.decode_raw_sensor",
        {"buffer": pack_reference(mosaic, 12)},
        packing="mipi_raw12",
        width=16,
        height=12,
        bayer_pattern=pattern,
        row_offset=row,
        column_offset=column,
    )
    _, result = execute(
        "cv.demosaic_bayer",
        {"image": packet_for(decoded, "image")},
        use_sensor_metadata=True,
        color_format="rgb",
    )
    np.testing.assert_array_equal(
        result.outputs["image"][2:-2, 2:-2], np.tile([3000, 1500, 600], (8, 12, 1))
    )
    assert result.payloads["image"]["raw_sensor"]["mosaic"] is False
    assert result.outputs["image"].dtype == np.uint16
    assert "sensor color preview" in result.graphics


def test_phase_follows_crop_flip_and_rejects_resampling():
    colors, pattern = {"R": 900, "G": 500, "B": 100}, "RGGB"
    mosaic = np.array(
        [[colors[pattern[2 * (y % 2) + x % 2]] for x in range(24)] for y in range(20)], np.uint16
    )
    _, decoded = execute(
        "cv.decode_raw_sensor", {"buffer": pack_reference(mosaic, 10)}, width=24, height=20
    )
    _, crop = execute(
        "cv.crop", {"image": packet_for(decoded, "image")}, x=1, y=3, width=20, height=14
    )
    _, flipped = execute("cv.flip", {"image": packet_for(crop, "image")}, direction="horizontal")
    _, result = execute(
        "cv.demosaic_bayer",
        {"image": packet_for(flipped, "image")},
        use_sensor_metadata=True,
        color_format="rgb",
    )
    np.testing.assert_array_equal(
        result.outputs["image"][2:-2, 2:-2], np.tile([900, 500, 100], (10, 16, 1))
    )
    for rotation in ("clockwise", "counterclockwise", "180"):
        _, rotated = execute(
            "cv.rotate90", {"image": packet_for(decoded, "image")}, rotation=rotation
        )
        _, color = execute(
            "cv.demosaic_bayer",
            {"image": packet_for(rotated, "image")},
            use_sensor_metadata=True,
            color_format="rgb",
        )
        interior = color.outputs["image"][2:-2, 2:-2]
        np.testing.assert_array_equal(interior, np.broadcast_to([900, 500, 100], interior.shape))
    _, resized = execute("cv.resize", {"image": packet_for(decoded, "image")}, width=48, height=40)
    block = builtin_registry().create("cv.demosaic_bayer", parameters={"use_sensor_metadata": True})
    assert block.execute({"image": packet_for(resized, "image")}) is None
    assert "unresampled pixel lattice" in block.error


def test_buffer_length_stride_group_bounds_and_monochrome_validation():
    block = builtin_registry().create("cv.decode_raw_sensor", parameters={"width": 4, "height": 1})
    for wrong in (b"1234", b"123456"):
        assert block.execute({"buffer": wrong}) is None
        assert "exactly 5 bytes" in block.error
    block.parameters["size_policy"] = "at_least"
    result = block.execute({"buffer": bytes.fromhex("55 aa ff 00 39 ee")})
    assert result is not None and result.outputs["image"][0, 0] == 341
    block.parameters["width"] = 5
    assert block.execute({"buffer": b"123456"}) is None and "multiple of 4" in block.error
    block.parameters.update(width=4, row_stride=4)
    assert block.execute({"buffer": b"123456"}) is None and "stride is smaller" in block.error
    block.parameters["row_stride"] = 0
    assert block.execute({"buffer": np.ones((1, 5), np.uint16)}) is None
    assert "uint8 vector" in block.error
    block.parameters.update(width=16, height=8, bayer_pattern="monochrome", size_policy="exact")
    decoded = block.execute({"buffer": pack_reference(np.ones((8, 16), np.uint16), 10)})
    demosaic = builtin_registry().create(
        "cv.demosaic_bayer", parameters={"use_sensor_metadata": True}
    )
    assert demosaic.execute({"image": packet_for(decoded, "image")}) is None
    assert "unconsumed Bayer sensor metadata" in demosaic.error
    block.parameters.update(width=4, height=1)
    assert block.execute({"buffer": b"12345", "pixel_frame": frame([3, 1])}) is None
    assert "dimensions must match" in block.error
    block.parameters["byte_offset"] = 128 * 1024 * 1024
    assert block.execute({"buffer": b"12345"}) is None and "128 MiB" in block.error


def test_file_source_tickets_unicode_path_and_exact_size(tmp_path):
    path = tmp_path / "caméra.raw12"
    image = np.arange(48, dtype=np.uint16).reshape(6, 8)
    data = pack_reference(image, 12, b"HEAD", 3)
    path.write_bytes(data)
    block = builtin_registry().create(
        "input.raw_sensor",
        parameters={
            "path": path.name,
            "path_base": "project",
            "width": 8,
            "height": 6,
            "packing": "mipi_raw12",
            "byte_offset": 4,
            "row_stride": 15,
        },
    )
    block.project_dir = tmp_path
    result = block.execute({})
    assert result is not None, block.error
    np.testing.assert_array_equal(result.outputs["image"], image)
    assert result.outputs["ticket"]["value"] == path.name
    assert result.outputs["info"]["filename"] == path.name
    for wrong in (data[:-1], data + b"tail"):
        path.write_bytes(wrong)
        assert block.execute({}) is None and "exactly" in block.error
    block.parameters["size_policy"] = "at_least"
    assert block.execute({}) is not None


def test_graph_byte_buffer_receipts_are_json_safe():
    class BufferSource(Block):
        type_id = "test.raw_buffer"
        outputs = (Port("buffer", "any"),)

        def process(self, inputs):
            return Result({"buffer": bytes.fromhex("55 aa ff 00 39")})

    graph = Graph()
    source = graph.add(BufferSource(block_id="source"))
    decoder = graph.add(
        builtin_registry().create(
            "cv.decode_raw_sensor", block_id="decoder", parameters={"width": 4, "height": 1}
        )
    )
    graph.connect(source.id, "buffer", decoder.id, "buffer")
    try:
        graph.run()
        assert decoder.state == "pass", decoder.error
        assert json.loads(json.dumps(describe(graph), allow_nan=False))
    finally:
        graph.close()


@pytest.mark.parametrize("project", ["raw10.json", "raw12.json"])
def test_public_raw_examples_match_reference_counts(project):
    path = Path(__file__).resolve().parents[1] / "examples/opencv tools/packed_raw_sensor" / project
    graph = Graph.load(path, builtin_registry())
    try:
        graph.run()
        assert all(
            block.state == "pass" for block in graph.blocks.values() if not block.layout_only
        )
        assert graph.blocks["verify"].last_result.outputs["value"] == 0
        assert graph.blocks["demosaic"].last_result.outputs["image"].dtype == np.uint16
    finally:
        graph.close()
