"""Exercise native code readers with generated linear and matrix symbols."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.graphics_layers import RESULT_GRAPHICS
from opencv_dataflow.payloads import NO_VALUE


@pytest.mark.parametrize("mode", ["multi", "single"])
def test_qr_decode_and_mask(mode):
    reader = builtin_registry().create("inspect.read_qr", parameters={"mode": mode})
    qr = cv2.QRCodeEncoder_create().encode("dataflow-42")
    image = np.pad(np.repeat(np.repeat(qr, 8, axis=0), 8, axis=1), 24, constant_values=255)
    result = reader.execute({"image": image})
    assert result.passed
    assert result.outputs["count"] == 1
    code = result.outputs["codes"][0]
    assert (code["text"], code["symbology"]) == ("dataflow-42", "QR_CODE")
    assert np.asarray(code["corners"]).shape == (4, 2)
    assert np.allclose(code["center"], np.asarray(image.shape[::-1]) / 2, atol=2)
    assert result.outputs["overlay"].shape == (*image.shape, 3)
    assert not reader.execute({"image": image, "mask": np.zeros_like(image)}).passed
    blank = reader.execute({"image": np.full_like(image, 255)})
    assert not blank.passed
    assert blank.fail_reasons[0]["code"] == "code_read_count_below_minimum"


def ean13(check_digit, module_width):
    # 5901234123457, left digits use L/G/G/L/L/G for leading digit 5.
    modules = (
        "101"
        "0001011"
        "0100111"
        "0110011"
        "0010011"
        "0111101"
        "0011101"
        "01010"
        "1100110"
        "1101100"
        "1000010"
        "1011100"
        "1001110" + {"7": "1000100", "8": "1001000"}[check_digit] + "101"
    )
    row = np.repeat(np.array([0 if bit == "1" else 255 for bit in modules], np.uint8), module_width)
    return np.pad(np.tile(row, (120, 1)), ((100, 100), (80, 80)), constant_values=255)


def test_ean13_decode_with_geometry():
    image = ean13("7", 2)
    reader = builtin_registry().create("inspect.read_barcode", parameters={"draw_overlay": False})
    reader.exposed_outputs.append(RESULT_GRAPHICS)
    reader.set_graphics_mode("off")
    result = reader.execute({"image": image})
    assert result is not None, reader.error
    assert result.passed, result.message
    assert result.outputs["count"] == 1
    assert result.outputs["overlay"] is NO_VALUE
    layers = result.outputs[RESULT_GRAPHICS].layers
    assert len(layers) == 1 and layers[0].objects[0].closed
    assert not result.graphics
    code = result.outputs["codes"][0]
    assert code["text"] == "5901234123457"
    assert code["symbology"] == "EAN_13"
    assert np.asarray(code["corners"]).shape == (4, 2)


def test_zxing_decodes_mixed_symbols_and_respects_region():
    zxing = pytest.importorskip("zxingcpp")
    matrix = np.asarray(
        zxing.write_barcode_to_image(
            zxing.create_barcode("LOT-0042", zxing.BarcodeFormat.DataMatrix), scale=6
        )
    )
    linear = np.asarray(
        zxing.write_barcode_to_image(
            zxing.create_barcode("SERIAL42", zxing.BarcodeFormat.Code128), scale=3
        )
    )
    h, w = matrix.shape
    rows, cols = linear.shape
    scene = np.full((max(h, rows) + 40, w + cols + 80), 255, np.uint8)
    scene[20 : 20 + h, 20 : 20 + w] = matrix
    scene[20 : 20 + rows, 60 + w : 60 + w + cols] = linear
    block = builtin_registry().create("inspect.read_zxing", parameters={"format": "AllReadable"})
    result = block.execute({"image": scene})
    assert result is not None, block.error
    assert {(c["text"], c["symbology"]) for c in result.outputs["codes"]} == {
        ("LOT-0042", "DataMatrix"),
        ("SERIAL42", "Code128"),
    }
    selection = np.zeros_like(scene)
    selection[:, : w + 40] = 255
    filtered = block.execute({"image": scene, "mask": selection})
    assert filtered.outputs["count"] == 1
    code = filtered.outputs["codes"][0]
    assert code["text"] == "LOT-0042"
    assert code["bytes_hex"] == b"LOT-0042".hex()
    np.testing.assert_allclose(code["center"], [20 + w / 2, 20 + h / 2], atol=1)


def test_zxing_reads_gs1_and_keeps_checksum_errors_out_of_codes():
    zxing = pytest.importorskip("zxingcpp")
    text = "(01)09506000134352(10)LOT0042"
    code = zxing.create_barcode(text, zxing.BarcodeFormat.DataMatrix, gs1=True)
    image = np.asarray(zxing.write_barcode_to_image(code, scale=5))
    block = builtin_registry().create("inspect.read_zxing")
    result = block.execute({"image": image})
    assert result is not None, block.error
    item = result.outputs["codes"][0]
    assert item["text"] == text
    assert item["content_type"] == "GS1"
    assert item["symbology_identifier"] == "]d2"
    block.parameters.update(format="EAN13", report_decode_errors=True)
    invalid = block.execute({"image": ean13("8", 4)})
    assert invalid is not None, block.error
    assert not invalid.passed
    assert invalid.outputs["codes"] == []
    assert "ChecksumError" in invalid.outputs["errors"][0]["error"]


def test_zxing_example_verifies_ticketed_lot_code():
    pytest.importorskip("zxingcpp")
    from pathlib import Path

    project = (
        Path(__file__).resolve().parents[1]
        / "examples/applications/datamatrix_reading/datamatrix_reading.json"
    )
    graph = Graph.load(project, builtin_registry())
    try:
        graph.run()
        verifier = graph.blocks["verify"]
        assert verifier.state == "pass", verifier.error
        assert verifier.last_result.outputs["matches"] == ["LOT-0042"]
        assert verifier.last_result.payloads["matches"]["ticket"]["value"] == "datamatrix_lot.png"
    finally:
        graph.close()
