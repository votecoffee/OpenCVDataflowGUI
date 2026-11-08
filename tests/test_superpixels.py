"""Check native superpixel maps, output selection and input validation."""

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


@pytest.fixture
def single_opencv_thread():
    # LSC's parallel floating point reductions can differ at tied boundaries.
    threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        yield
    finally:
        cv2.setNumThreads(threads)


@pytest.mark.parametrize("method", ["lsc", "seeds"])
def test_new_superpixel_methods_match_native_maps(method, single_opencv_thread):
    h, w = 48, 64
    y, x = np.indices((h, w))
    bgr = np.stack((50 + x, 60 + y * 2, 70 + (x > 26) * 80), axis=-1).astype(np.uint8)
    rgb = ImageArray(bgr[:, :, ::-1].copy(), "rgb")
    if method == "lsc":
        native = cv2.ximgproc.createSuperpixelLSC(cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab), 8, 0.13)
        native.iterate(3)
        native.enforceLabelConnectivity(17)
        options = {"region_size": 8, "ratio": 0.13, "min_component_pixels": 17}
    else:
        native = cv2.ximgproc.createSuperpixelSEEDS(w, h, 3, 70, 3, 1, 6, True)
        native.iterate(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), 3)
        options = {
            "target_count": 70, "levels": 3, "prior": 1, "histogram_bins": 6,
            "double_step": True, "color_space": "hsv",
        }
    options.update(iterations=3, thick_boundaries=True)
    labels = native.getLabels() + 1
    boundaries = native.getLabelContourMask(thick_line=True)
    ticket = TicketClock().issue("camera")
    source = ValuePacket(rgb, {"spatial": frame([w, h], "fixture"), "ticket": ticket})
    mask = np.zeros((h, w), np.uint8)
    mask[7:40, 11:51] = 255
    block = builtin_registry().create(f"cv.{method}_superpixels", parameters=options)
    result = block.execute({"image": source, "mask": mask})
    assert result is not None, block.error
    np.testing.assert_array_equal(result.outputs["labels"], np.where(mask, labels, 0))
    np.testing.assert_array_equal(result.outputs["boundaries"], np.where(mask, boundaries, 0))
    assert result.outputs["count"] == np.unique(labels[mask > 0]).size
    assert result.payloads["labels"]["ticket"] == ticket
    assert result.payloads["boundaries"]["spatial"]["root_id"] == "fixture"


def test_lsc_black_input_and_invalid_portrait_grid():
    block = builtin_registry().create(
        "cv.lsc_superpixels", parameters={"region_size": 8, "iterations": 2}
    )
    result = block.execute({"image": np.zeros((24, 32), np.uint8)})
    assert result is not None, block.error
    assert result.outputs["labels"].min() >= 1 and result.outputs["count"] > 1
    assert block.execute({"image": np.zeros((11, 8), np.uint8)}) is None
    assert "LSC seed grid" in block.error


def test_seeds_guards_grid_and_histogram_memory(monkeypatch):
    def unexpected(*_args, **_kwargs):
        pytest.fail("Invalid SEEDS settings reached the native factory")

    monkeypatch.setattr(cv2.ximgproc, "createSuperpixelSEEDS", unexpected)
    block = builtin_registry().create("cv.seeds_superpixels", parameters={"target_count": 10})
    assert block.execute({"image": np.zeros((4, 4), np.uint8)}) is None
    assert "at least 6" in block.error
    assert block.execute({"image": np.zeros((6, 100), np.uint8)}) is None
    assert "cannot form a grid" in block.error
    block.parameters.update(target_count=40)
    assert block.execute({"image": np.zeros((6, 6), np.uint8)}) is None
    assert "two pixels per starting region" in block.error
    block.parameters.update(target_count=30, levels=4)
    assert block.execute({"image": np.zeros((17, 160), np.uint8)}) is None
    assert "per axis at every level" in block.error
    block.parameters.update(target_count=4000, levels=5, histogram_bins=16)
    assert block.execute({"image": np.zeros((256, 256, 3), np.uint8)}) is None
    assert "histogram buffers" in block.error
