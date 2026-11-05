"""Exercise the installed native readers with generated QR and EAN-13 images."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
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


def test_ean13_decode_with_geometry():
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
        "1001110"
        "1000100"
        "101"
    )
    row = np.repeat(np.array([0 if bit == "1" else 255 for bit in modules], np.uint8), 2)
    image = np.pad(np.tile(row, (120, 1)), ((100, 100), (80, 80)), constant_values=255)
    reader = builtin_registry().create("inspect.read_barcode", parameters={"draw_overlay": False})
    result = reader.execute({"image": image})
    assert result is not None, reader.error
    assert result.passed, result.message
    assert result.outputs["count"] == 1
    assert result.outputs["overlay"] is NO_VALUE
    code = result.outputs["codes"][0]
    assert code["text"] == "5901234123457"
    assert code["symbology"] == "EAN_13"
    assert np.asarray(code["corners"]).shape == (4, 2)
