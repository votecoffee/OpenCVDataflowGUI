"""Native QR header references, strict assembly, resource/payload/editor contracts."""

import hashlib
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest

from opencv_dataflow.block_xml import save_block_xml
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket, frame
from opencv_dataflow.project_xml import read_block_xml
from opencv_dataflow.qr_append import ReadQRAppend, assemble
from opencv_dataflow.qr_append_header import structured_header
from opencv_dataflow.qr_encoding import CORRECTION, EncodeQRCode
from opencv_dataflow.tickets import TicketClock

TEXT = "PART-0042/LOT-A-INSPECTION-1234567890"


def symbols(text=TEXT, count=3):
    block = EncodeQRCode(
        parameters={
            "mode": "structured_append",
            "text": text,
            "structure_number": count,
            "pixels_per_module": 6,
        }
    )
    result = block.execute({})
    assert result is not None, block.error
    return result.outputs["symbols"]


@pytest.mark.parametrize("version", [1, 2, 3, 4, 7, 10, 20, 32, 40])
@pytest.mark.parametrize("level", list(CORRECTION))
def test_rectified_header_layout_matches_native_encoder_across_rs_blocks(version, level):
    p = cv2.QRCodeEncoder_Params()
    p.mode, p.structure_number, p.version, p.correction_level = (
        cv2.QRCodeEncoder_MODE_STRUCTURED_APPEND,
        3,
        version,
        CORRECTION[level],
    )
    payload = "ABCDEFGHI"
    parity = 0
    for value in payload.encode():
        parity ^= value
    for index, native in enumerate(cv2.QRCodeEncoder_create(p).encodeStructuredAppend(payload)):
        header = structured_header(native[2:-2, 2:-2])
        assert (header["index"], header["count"], header["parity"]) == (index, 3, parity)
        assert (header["version"], header["correction_level"]) == (version, level)
        assert 0 <= header["mask"] <= 7


@pytest.mark.parametrize("mutation", ["format", "data", "nonbinary", "shape", "ordinary"])
def test_header_parser_rejects_untrusted_or_ordinary_modules(mutation):
    p = cv2.QRCodeEncoder_Params()
    p.mode, p.structure_number = cv2.QRCodeEncoder_MODE_STRUCTURED_APPEND, 3
    grid = cv2.QRCodeEncoder_create(p).encodeStructuredAppend("ABCDEFGHI")[0][2:-2, 2:-2].copy()
    if mutation == "format":
        grid[0, 8] = 255 - grid[0, 8]
    elif mutation == "data":
        grid[-1, -1] = 255 - grid[-1, -1]
    elif mutation == "nonbinary":
        grid[10, 10] = 127
    elif mutation == "shape":
        grid = grid[:-1]
    else:
        grid = cv2.QRCodeEncoder_create().encode("ordinary")[2:-2, 2:-2]
    with pytest.raises(ValidationError):
        structured_header(grid)


@pytest.mark.parametrize("backend", ["standard", "aruco"])
@pytest.mark.parametrize("order", [(2, 0, 1), (1, 2, 0), (0, 1, 2)])
def test_out_of_order_image_groups_ignore_generator_payload_order_and_keep_member_tickets(
    backend, order
):
    entries, clock = symbols().items, TicketClock()
    packets = []
    for i in order:
        image = entries[i].value
        payload = {
            "ticket": clock.issue("scanner", "external", f"physical-{i}"),
            "spatial": frame(image.shape[::-1], f"capture-{i}"),
            "qr_label": {"position": 999, "count": 1, "group_sha256": "untrusted"},
        }
        packets.append(ValuePacket(image, payload))
    group = FlowGroup(tuple(packets), ("same",) * 3)
    originals = [p.value.copy() for p in packets]
    block = ReadQRAppend(
        parameters={
            "backend": backend,
            "expected_parts": 3,
            "expected_sha256": hashlib.sha256(TEXT.encode()).hexdigest(),
        }
    )
    result = block.execute({"images": group})
    assert result is not None, block.error
    assert result.passed and result.outputs["text"] == TEXT
    assert bytes.fromhex(result.outputs["bytes_hex"]) == TEXT.encode()
    assert result.outputs["complete"]
    records = result.outputs["fragments"]
    assert [r["header"]["index"] for r in records] == list(order)
    assert [r["ticket"]["value"] for r in records] == [f"physical-{i}" for i in order]
    assert [r["spatial"]["root_id"] for r in records] == [f"capture-{i}" for i in order]
    assert result.outputs["overlay"] is NO_VALUE and not result.drawings
    for packet, original in zip(packets, originals, strict=True):
        np.testing.assert_array_equal(packet.value, original)


def test_identical_payload_fragments_are_distinct_when_header_indices_are_distinct():
    text = "A" * 48
    block = ReadQRAppend()
    result = block.execute({"images": symbols(text)})
    assert result is not None, block.error
    assert result.passed and result.outputs["text"] == text
    assert len({r["bytes_hex"] for r in result.outputs["fragments"]}) == 1
    assert not result.outputs["status"]["duplicate_indices"]


@pytest.mark.parametrize("encoding", ["ascii", "utf8", "hex"])
def test_text_decoding_is_explicit_and_hex_preserves_bytes(encoding):
    block = ReadQRAppend(parameters={"text_encoding": encoding})
    result = block.execute({"images": symbols()})
    assert result is not None, block.error
    assert result.outputs["text"] == (TEXT.encode().hex() if encoding == "hex" else TEXT)


@pytest.mark.parametrize("count", [2, 16])
def test_minimum_and_maximum_part_counts_keep_repeated_fragment_payloads_distinct(count):
    text = "A" * 48
    block = ReadQRAppend(parameters={"expected_parts": count})
    result = block.execute({"images": symbols(text, count)})
    assert result is not None, block.error
    assert result.passed and result.outputs["text"] == text
    assert sorted(f["header"]["index"] for f in result.outputs["fragments"]) == list(range(count))


def test_utf8_is_decoded_after_reassembling_split_multibyte_sequences():
    text = "A\u6f22\u5b57XYZ"
    p = cv2.QRCodeEncoder_Params()
    p.mode, p.structure_number = cv2.QRCodeEncoder_MODE_STRUCTURED_APPEND, 3
    values = cv2.QRCodeEncoder_create(p).encodeStructuredAppend(text)
    group = FlowGroup(
        tuple(
            ValuePacket(np.repeat(np.repeat(np.pad(value, 2, constant_values=255), 6, 0), 6, 1), {})
            for value in values
        ),
        ("first", "second", "third"),
    )
    block = ReadQRAppend(parameters={"text_encoding": "utf8"})
    result = block.execute({"images": group})
    assert result is not None, block.error
    assert result.passed and result.outputs["text"] == text
    assert bytes.fromhex(result.outputs["bytes_hex"]) == text.encode()
    with pytest.raises(UnicodeError):
        bytes.fromhex(result.outputs["fragments"][0]["bytes_hex"]).decode("utf8")
    block.parameters["text_encoding"] = "ascii"
    invalid = block.execute({"images": group})
    assert not invalid.passed and invalid.outputs["text"] is NO_VALUE
    assert invalid.outputs["bytes_hex"] is NO_VALUE
    assert invalid.outputs["status"]["reasons"] == ["qr_payload_text_encoding_mismatch"]
    assert not block.current_emissions.get("text")
    block.parameters["text_encoding"] = "hex"
    assert block.execute({"images": group}).outputs["text"] == text.encode().hex()


def test_raw_parity_is_checked_independently_of_headers_and_expected_digest():
    fragments = [
        {"header": {"index": i, "count": 2, "parity": 0}, "bytes_hex": raw.hex()}
        for i, raw in enumerate((b"A", b"B"))
    ]
    raw, status = assemble(fragments)
    assert raw is None and status["reasons"] == ["qr_payload_parity_mismatch"]


@pytest.mark.parametrize(
    "parameters,field",
    [
        ({"expected_parts": 1}, "expected_parts"),
        ({"expected_parts": 4, "maximum_symbols": 3}, "maximum_symbols"),
        ({"expected_sha256": "x" * 64}, "expected_sha256"),
        ({"text_encoding": "latin1"}, "text_encoding"),
        ({"mode": "single"}, "mode"),
        ({"maximum_images": 0}, "maximum_images"),
        ({"maximum_total_pixels": float("nan")}, "maximum_total_pixels"),
    ],
)
def test_invalid_parameters_identify_fields_without_native_work(parameters, field, monkeypatch):
    import opencv_dataflow.qr_append as module

    monkeypatch.setattr(module, "make_detector", lambda _: pytest.fail("Unexpected native call"))
    block = ReadQRAppend(parameters=parameters)
    assert block.execute({"images": symbols()}) is None
    assert block.error_kind == "invalid"
    with pytest.raises(ValidationError) as error:
        block.validate_parameters()
    assert field in error.value.fields


@pytest.mark.parametrize(
    "parameters,inputs",
    [
        ({"maximum_pixels": 441}, "group"),
        ({"maximum_total_pixels": 441}, "group"),
        ({"maximum_images": 2}, "group"),
        ({}, "both"),
        ({}, "neither"),
        ({}, "wrong_group"),
        ({}, "group_selection"),
    ],
)
def test_input_resource_and_shape_preflight_precedes_native_work(parameters, inputs, monkeypatch):
    import opencv_dataflow.qr_append as module

    group = symbols()
    values = {"images": group}
    if inputs == "both":
        values["image"] = group.items[0].value
    elif inputs == "neither":
        values = {}
    elif inputs == "wrong_group":
        values = {"images": [1, 2, 3]}
    elif inputs == "group_selection":
        values["mask"] = np.zeros_like(group.items[0].value)
    monkeypatch.setattr(module, "make_detector", lambda _: pytest.fail("Native preflight bypassed"))
    block = ReadQRAppend(parameters=parameters)
    assert block.execute(values) is None
    assert block.error_kind == "invalid" and not any(block.current_emissions.values())


def test_native_exception_has_no_outputs_and_next_invocation_recovers(monkeypatch):
    import opencv_dataflow.qr_append as module

    original = module.make_detector

    class Broken:
        def detectMulti(self, image):
            raise cv2.error("Controlled structured QR failure")

    monkeypatch.setattr(module, "make_detector", lambda _: Broken())
    block = ReadQRAppend()
    assert block.execute({"images": symbols()}) is None
    assert block.error_kind == "exception" and not any(block.current_emissions.values())
    monkeypatch.setattr(module, "make_detector", original)
    assert block.execute({"images": symbols()}).passed


def test_candidate_limit_precedes_fragment_decoding_and_payload_limit_emits_nothing(monkeypatch):
    import opencv_dataflow.qr_append as module

    native = module.make_detector

    class Guard:
        def __init__(self, p):
            self.detector = native(p)

        def detectMulti(self, image):
            return self.detector.detectMulti(image)

        def decodeBytes(self, *args):
            pytest.fail("Candidate cap must precede decoding")

    group = symbols()
    monkeypatch.setattr(module, "make_detector", Guard)
    block = ReadQRAppend(parameters={"maximum_symbols": 2})
    assert block.execute({"images": group}) is None
    assert block.error_kind == "invalid" and "symbol limit" in block.error
    monkeypatch.setattr(module, "make_detector", native)
    block.parameters.update(maximum_symbols=16, maximum_payload_bytes=5)
    assert block.execute({"images": group}) is None
    assert block.error_kind == "invalid" and not any(block.current_emissions.values())


def test_parallel_instances_and_graphics_off_are_independent():
    groups = [symbols("BRANCH-A-0123456789"), symbols("BRANCH-B-0123456789")]

    def run(group):
        block = ReadQRAppend(parameters={"draw_overlay": False})
        block.set_graphics_mode("off")
        result = block.execute({"images": group})
        assert result is not None, block.error
        assert not result.drawings and result.outputs["overlay"] is NO_VALUE
        return result.outputs["text"]

    with ThreadPoolExecutor(max_workers=2) as workers:
        assert list(workers.map(run, groups)) == ["BRANCH-A-0123456789", "BRANCH-B-0123456789"]


def test_xml_project_and_inspector_dropdown_roundtrip(app, tmp_path):
    from PySide6.QtWidgets import QComboBox

    from opencv_dataflow.app import MainWindow

    registry, graph = builtin_registry(), Graph()
    reader = graph.add(registry.create("inspect.read_qr_append", parameters={"expected_parts": 3}))
    xml = tmp_path / "read.inspect.read_qr_append.xml"
    save_block_xml(reader, xml)
    assert read_block_xml(xml)["parameters"] == reader.parameters
    graph.save(tmp_path / "program.json")
    reopened = Graph.load(tmp_path / "program.json", registry)
    assert reopened.blocks[reader.id].parameters == reader.parameters
    reopened.close()
    window = MainWindow()
    try:
        window.replace_graph(graph)
        window.inspector.select(reader)
        for name in reader.parameter_choices:
            assert isinstance(window.inspector.fields[name], QComboBox)
        assert window.inspector.fields["expected_sha256"].toolTip()
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
        graph.close()
