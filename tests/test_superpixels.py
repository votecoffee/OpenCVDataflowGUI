"""Check native SLIC maps, output selection and input validation."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock


@pytest.mark.parametrize(
    ("algorithm", "thick"), [("slic", False), ("slico", True), ("mslic", False)]
)
def test_superpixels_match_native_lab_labels_and_boundary_mask(algorithm, thick):
    y, x = np.indices((48, 64))
    gray = (60 + x + y).astype(np.uint8)
    bgr = np.stack((gray, (50 + (x > 29) * 100).astype(np.uint8), gray + 70), axis=-1)
    rgb = ImageArray(bgr[:, :, ::-1].copy(), "rgb")
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    native = cv2.ximgproc.createSuperpixelSLIC(lab, getattr(cv2.ximgproc, algorithm.upper()), 8, 13)
    native.iterate(4)
    native.enforceLabelConnectivity(22)
    labels = native.getLabels() + 1
    boundaries = native.getLabelContourMask(thick_line=thick)
    mask = np.zeros(gray.shape, np.uint8)
    mask[5:44, 7:54] = 255
    region = {
        "model": "image_region_v1", "image_size": [64, 48],
        "shape": "rectangle", "rectangle": [4, 9, 42, 37],
    }
    selection = np.zeros(gray.shape, dtype=bool)
    selection[9:44, 7:46] = True
    ticket = TicketClock().issue("camera")
    packet = ValuePacket(rgb, {"spatial": frame([64, 48], "station"), "ticket": ticket})

    block = builtin_registry().create("cv.slic_superpixels", parameters={
        "algorithm": algorithm, "region_size": 8, "ruler": 13, "iterations": 4,
        "connectivity_percent": 22, "thick_boundaries": thick,
    })
    result = block.execute({"image": packet, "mask": mask, "region": region})
    assert result is not None, block.error
    np.testing.assert_array_equal(result.outputs["labels"], np.where(selection, labels, 0))
    np.testing.assert_array_equal(result.outputs["boundaries"], np.where(selection, boundaries, 0))
    assert result.outputs["labels"].dtype == np.int32
    assert result.outputs["boundaries"].dtype == np.uint8
    assert result.outputs["count"] == len(np.unique(labels[selection]))
    assert result.payloads["labels"]["ticket"] == ticket
    assert result.payloads["boundaries"]["spatial"]["root_id"] == "station"


def test_superpixels_gray_empty_selection_and_invalid_spacing():
    image = np.arange(35 * 27, dtype=np.uint8).reshape(27, 35)
    block = builtin_registry().create("cv.slic_superpixels", parameters={
        "region_size": 8, "iterations": 2, "connectivity_percent": 0,
    })
    result = block.execute({"image": image})
    assert result is not None, block.error
    assert result.outputs["labels"].min() > 0
    assert result.outputs["count"] == len(np.unique(result.outputs["labels"]))
    empty = block.execute({"image": image, "mask": np.zeros_like(image)})
    assert empty is not None, block.error
    assert not np.any(empty.outputs["labels"]) and not np.any(empty.outputs["boundaries"])
    assert empty.outputs["count"] == 0
    block.parameters["region_size"] = 28
    assert block.execute({"image": image}) is None
    assert "shorter image dimension" in block.error


def test_mslic_rejects_single_seed_before_calling_opencv(monkeypatch):
    def unexpected(*_args, **_kwargs):
        pytest.fail("Single-seed MSLIC reached OpenCV")

    monkeypatch.setattr(cv2.ximgproc, "createSuperpixelSLIC", unexpected)
    block = builtin_registry().create(
        "cv.slic_superpixels", parameters={"algorithm": "mslic", "region_size": 8}
    )
    assert block.execute({"image": np.zeros((8, 11), np.uint8)}) is None
    assert "at least two starting regions" in block.error
