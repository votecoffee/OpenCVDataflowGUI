"""Check OpenCV local binarization and selected output semantics."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock


@pytest.mark.parametrize("method", ["niblack", "sauvola", "wolf", "nick"])
@pytest.mark.parametrize("mode", ["binary", "binary_inv", "trunc", "tozero", "tozero_inv"])
def test_local_threshold_matches_native_with_mask_and_region(method, mode):
    rng = np.random.default_rng(54)
    gray = rng.integers(0, 256, (29, 35), dtype=np.uint8)
    rgb = ImageArray(np.repeat(gray[:, :, None].astype(np.uint16) * 257, 3, axis=2), "rgb")
    mask = np.zeros(gray.shape, np.uint8)
    mask[8:22, 4:23] = 1
    region = {
        "model": "image_region_v1", "image_size": [35, 29],
        "shape": "rectangle", "rectangle": [10, 5, 19, 15],
    }
    ticket = TicketClock().issue("camera")
    source = ValuePacket(rgb, {"spatial": frame([35, 29], "inspection"), "ticket": ticket})
    block = builtin_registry().create("cv.local_threshold", parameters={
        "method": method, "mode": mode, "block_size": 9, "k": -0.17,
        "sauvola_range": 96.0, "max_value": 211,
    })
    result = block.execute({"image": source, "mask": mask, "region": region})
    assert result is not None, block.error
    reference = cv2.ximgproc.niBlackThreshold(
        gray, 211, getattr(cv2, f"THRESH_{mode.upper()}"), 9, -0.17,
        binarizationMethod=getattr(cv2.ximgproc, f"BINARIZATION_{method.upper()}"), r=96,
    )
    expected = np.zeros_like(gray)
    expected[8:20, 10:23] = reference[8:20, 10:23]
    np.testing.assert_array_equal(result.outputs["image"], expected)
    assert result.outputs["image"].dtype == np.uint8
    assert result.payloads["image"]["ticket"] == ticket
    assert result.payloads["image"]["spatial"]["root_id"] == "inspection"


@pytest.mark.parametrize(("mode", "sample"), [("binary", 0), ("binary_inv", 199), ("trunc", 73)])
def test_flat_wolf_uses_the_constant_intensity(monkeypatch, mode, sample):
    def unexpected(*_args, **_kwargs):
        pytest.fail("A flat frame reached the native Wolf formula")

    monkeypatch.setattr(cv2.ximgproc, "niBlackThreshold", unexpected)
    block = builtin_registry().create(
        "cv.local_threshold", parameters={"method": "wolf", "mode": mode, "max_value": 199}
    )
    result = block.execute({"image": np.full((8, 9), 73, np.uint8)})
    assert result is not None, block.error
    np.testing.assert_array_equal(result.outputs["image"], np.full((8, 9), sample, np.uint8))


def test_local_threshold_reports_invalid_parameters_before_processing():
    block = builtin_registry().create("cv.local_threshold", parameters={"block_size": 10})
    source = np.arange(72, dtype=np.uint8).reshape(8, 9)
    assert block.execute({"image": source}) is None
    assert "odd" in block.error
    block.parameters.update(block_size=9, sauvola_range=0)
    assert block.execute({"image": source}) is None
    assert "Sauvola range" in block.error
