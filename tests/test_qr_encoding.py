"""Native QR controls, independent render geometry, payload recovery and safe output limits."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.block_xml import save_block_xml
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame, packet_for
from opencv_dataflow.project_xml import read_block_xml
from opencv_dataflow.qr_encoding import CORRECTION, MODES, EncodeQRCode
from opencv_dataflow.tickets import TicketClock, ticket_key

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/qr_encoding"


def encode(parameters=None, **inputs):
    block = EncodeQRCode(parameters=parameters or {})
    result = block.execute(inputs)
    assert result is not None, block.error
    return block, result


@pytest.mark.parametrize(
    "mode,text",
    [
        ("auto", "PART-0042/LOT-A"),
        ("numeric", "001234567890"),
        ("alphanumeric", "PART 0042/LOT A"),
        ("byte", "Part-42 / mixed"),
        ("eci_utf8", "Caf\u00e9 \u6f22\u5b57"),
        ("kanji", "\u6f22\u5b57"),
    ],
)
def test_native_mode_parity_and_recovery(mode, text):
    parameters = {"text": text, "mode": mode, "pixels_per_module": 5, "quiet_zone": 4}
    block = EncodeQRCode(parameters=parameters)
    block.exposed_outputs.append("modules")
    result = block.execute({})
    assert result is not None, block.error
    p = cv2.QRCodeEncoder_Params()
    p.mode, p.correction_level = MODES[mode], cv2.QRCodeEncoder_CORRECT_LEVEL_M
    raw = cv2.QRCodeEncoder_create(p).encode(text.encode("shift_jis") if mode == "kanji" else text)
    # Current native output has two border modules. The public module output has none.
    native_core = raw[2:-2, 2:-2]
    np.testing.assert_array_equal(result.outputs["modules"], native_core)
    expected = np.repeat(
        np.repeat(np.pad(native_core, 4, constant_values=255), 5, axis=0), 5, axis=1
    )
    np.testing.assert_array_equal(result.outputs["image"], expected)
    if mode == "kanji":
        zxingcpp = pytest.importorskip("zxingcpp")
        assert zxingcpp.read_barcode(result.outputs["image"]).text == text
    else:
        decoded = builtin_registry().create("inspect.read_qr", parameters={"mode": "single"})
        recovered = decoded.execute({"image": packet_for(result, "image")})
        assert recovered is not None, decoded.error
        assert recovered.outputs["codes"][0]["text"] == text


@pytest.mark.parametrize("correction", CORRECTION)
@pytest.mark.parametrize("version", [1, 5])
def test_all_correction_levels_and_explicit_versions(correction, version):
    text = "PART-42"
    _, result = encode({"text": text, "version": version, "correction_level": correction})
    p = cv2.QRCodeEncoder_Params()
    p.version, p.correction_level = version, CORRECTION[correction]
    raw = cv2.QRCodeEncoder_create(p).encode(text)
    assert result.outputs["info"]["symbols"][0]["version"] == version
    grid = result.outputs["symbols"].items[0].value
    np.testing.assert_array_equal(grid[32:-32:8, 32:-32:8], raw[2:-2, 2:-2])


def test_wired_text_keeps_ticket_and_companions_but_uses_new_label_grids():
    clock = TicketClock()
    ticket = clock.issue("PLC", "fixed", "part-17")
    context = {
        "ticket": ticket,
        "spatial": frame([640, 480], "camera"),
        "pyramid": {
            "level": 1,
            "largest_size": [1280, 960],
            "to_largest": np.diag([2, 2, 1]).tolist(),
        },
        "acquisition": {"encoder": 123},
        "raw_sensor": {"format": "raw10"},
    }
    _, result = encode({"text": "unused default"}, text=ValuePacket("PART-0017", context))
    assert result.outputs["text"] == "PART-0017"
    payload = result.payloads["image"]
    assert ticket_key(payload["ticket"]) == ticket_key(ticket)
    assert payload["acquisition"] == {"encoder": 123}
    assert payload["spatial"]["root_id"] != "camera"
    assert "pyramid" not in payload and "raw_sensor" not in payload
    assert result.outputs["symbols"].items[0].payload["spatial"] == payload["spatial"]


@pytest.mark.parametrize("number,columns", [(1, 0), (2, 2), (3, 2), (4, 1)])
def test_structured_append_native_parts_group_order_and_sheet_geometry(number, columns):
    text = "PART-0042/LOT-A-INSPECTION-1234567890"
    _, result = encode(
        {
            "text": text,
            "mode": "structured_append",
            "structure_number": number,
            "sheet_columns": columns,
            "pixels_per_module": 3,
            "quiet_zone": 5,
            "sheet_gap": 2,
        }
    )
    p = cv2.QRCodeEncoder_Params()
    p.mode, p.structure_number, p.correction_level = (
        MODES["structured_append"],
        number,
        CORRECTION["M"],
    )
    native = cv2.QRCodeEncoder_create(p).encodeStructuredAppend(text)
    symbols = result.outputs["symbols"].items
    assert result.outputs["count"] == len(symbols) == number
    assert result.outputs["info"]["requires_reassembly"]
    assert result.outputs["modules"] is NO_VALUE
    for index, (packet, raw, info) in enumerate(
        zip(symbols, native, result.outputs["info"]["symbols"], strict=True)
    ):
        expected = np.repeat(
            np.repeat(np.pad(raw[2:-2, 2:-2], 5, constant_values=255), 3, axis=0), 3, axis=1
        )
        np.testing.assert_array_equal(packet.value, expected)
        x, y, w, h = info["bounds"]
        np.testing.assert_array_equal(result.outputs["image"][y : y + h, x : x + w], expected)
        assert packet.payload["qr_label"]["position"] == index
        assert packet.payload["qr_label"]["requires_reassembly"]


def test_zero_quiet_zone_and_graphics_off_leave_encoded_pixels_untouched():
    block = EncodeQRCode(parameters={"quiet_zone": 0, "pixels_per_module": 1})
    normal = block.execute({})
    assert normal is not None, block.error
    assert len(normal.drawings["label bounds"].objects) == 1
    image = normal.outputs["image"].copy()
    block.set_graphics_mode("off")
    lean = block.execute({})
    assert not lean.drawings
    np.testing.assert_array_equal(lean.outputs["image"], image)
    assert image.shape == (21, 21)


def test_pixel_budget_is_checked_before_scaling_or_sheet_allocation(monkeypatch):
    block = EncodeQRCode(
        parameters={"version": 40, "pixels_per_module": 32, "maximum_output_pixels": 100}
    )
    monkeypatch.setattr(cv2, "resize", lambda *a, **k: pytest.fail("budget reached resize"))
    assert block.execute({}) is None
    assert block.error_kind == "invalid" and not block.current_emissions


@pytest.mark.parametrize(
    "parameters",
    [
        {"text": ""},
        {"text": "a\0b"},
        {"text": 42},
        {"text": "1" * 8193},
        {"mode": "numeric", "text": "12A"},
        {"mode": "numeric", "text": "\u0661"},
        {"mode": "alphanumeric", "text": "lowercase"},
        {"mode": "kanji", "text": "ASCII"},
        {"mode": "kanji", "text": "\U0001f600"},
        {"mode": "structured_append", "text": "\u6f22\u5b57"},
        {"mode": "structured_append", "text": "A", "structure_number": 2},
        {"version": 1, "text": "x" * 400, "mode": "byte"},
        {"version": 41},
        {"quiet_zone": -1},
        {"pixels_per_module": 1.5},
        {"structure_number": 17},
    ],
)
def test_bad_payloads_modes_capacity_and_controls_are_contained(parameters):
    block = EncodeQRCode(parameters=parameters)
    assert block.execute({}) is None
    assert block.error_kind == "invalid" and not block.current_emissions


def test_inactive_structured_count_does_not_change_single_symbol_encoding():
    _, ordinary = encode({"structure_number": 1})
    _, retained = encode({"structure_number": 16})
    np.testing.assert_array_equal(ordinary.outputs["image"], retained.outputs["image"])


def test_native_failure_and_invalid_images_are_contained(monkeypatch):
    class Broken:
        def encode(self, text):
            raise RuntimeError("synthetic native failure")

    monkeypatch.setattr(cv2, "QRCodeEncoder_create", lambda params: Broken())
    block = EncodeQRCode()
    assert block.execute({}) is None and block.error_kind == "exception"

    class Invalid:
        def encode(self, text):
            return np.full((25, 25), 255, np.uint8)

    monkeypatch.setattr(cv2, "QRCodeEncoder_create", lambda params: Invalid())
    assert block.execute({}) is None and block.error_kind == "invalid"


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize(
    "name", ["qr_roundtrip", "qr_unicode", "structured_append", "capacity_failure"]
)
def test_projects_known_payloads_binary_reference_grouping_and_xml(profile, name, tmp_path):
    graph = Graph.load(EXAMPLE / (name + ".json"), builtin_registry())
    truth = json.loads((EXAMPLE / "truth.json").read_text(encoding="utf8"))
    try:
        graph.set_run_profile(profile)
        graph.run()
        block = graph.blocks["generate"]
        if name == "capacity_failure":
            assert block.error_kind == "invalid" and not block.current_emissions
            return
        assert block.last_result is not None, block.error
        result = block.last_result
        assert bool(result.drawings) == (profile == "debug")
        assert result.outputs["text"] == truth["payloads"][name]
        if name == "structured_append":
            assert result.outputs["count"] == truth["structured_count"]
            assert len(result.outputs["symbols"].items) == truth["structured_count"]
            assert result.outputs["info"]["requires_reassembly"]
        else:
            reader = graph.blocks["read"]
            assert reader.state == "pass", reader.error
            assert reader.last_result.outputs["codes"][0]["text"] == truth["payloads"][name]
        if name == "qr_roundtrip":
            np.testing.assert_array_equal(
                result.outputs["image"],
                cv2.imread(str(EXAMPLE / "label_reference.png"), cv2.IMREAD_UNCHANGED),
            )
        path = tmp_path / "label.cv.encode_qr.xml"
        save_block_xml(block, path)
        assert read_block_xml(path) == block.to_project_item()
    finally:
        graph.close()
