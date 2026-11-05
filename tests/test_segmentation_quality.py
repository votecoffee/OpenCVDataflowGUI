"""Exercise segmentation labels, selection, and quality metrics through the registry."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import NO_VALUE


def test_flood_fill_mask_is_a_barrier():
    image = np.full((9, 11), 72, np.uint8)
    selection = np.full_like(image, 255)
    selection[:, 4] = 0
    fill = builtin_registry().create("inspect.flood_fill", parameters={"seed_y": 4})
    result = fill.execute({"image": image, "mask": selection})
    assert result is not None, fill.error
    assert result.outputs["area"] == 36
    assert result.outputs["bounds"] == [0, 0, 4, 9]
    assert np.all(result.outputs["mask"][:, :4] == 255)
    assert not result.outputs["mask"][:, 4:].any()


@pytest.mark.parametrize("mode", ["binary_components", "labels"])
def test_watershed_maintains_seed_ids_and_input(mode):
    image = np.zeros((30, 40), np.uint8)
    image[:, 20:] = 255
    markers = np.zeros_like(image, dtype=np.uint8 if mode == "binary_components" else np.int32)
    values = (1, 2) if mode == "binary_components" else (7, 24)
    markers[12, 10], markers[12, 30] = values
    original = markers.copy()
    block = builtin_registry().create("inspect.watershed", parameters={"marker_mode": mode})
    result = block.execute({"image": image, "markers": markers})
    assert result is not None, block.error
    assert result.outputs["count"] == 2
    assert tuple(result.outputs["labels"][12, [10, 30]]) == values
    assert result.outputs["labels"].dtype == np.int32
    assert result.outputs["boundaries"][:, 19:22].any()
    np.testing.assert_array_equal(markers, original)


@pytest.mark.parametrize("metric", ["ssim", "mse", "psnr"])
def test_quality_defect_and_selection(metric):
    reference = np.full((40, 45), 120, np.uint8)
    sample = reference.copy()
    sample[3:10, 2:12] = 20
    selection = np.zeros_like(reference)
    selection[22:, :] = 255
    threshold = {"ssim": 0.95, "mse": 20, "psnr": 45}[metric]
    block = builtin_registry().create(
        "inspect.image_quality",
        parameters={"metric": metric, "check_threshold": True, "threshold": threshold},
    )
    full = block.execute({"image_a": reference, "image_b": sample})
    assert full is not None, block.error
    assert not full.passed
    assert full.fail_reasons[0]["code"] == "image_quality_out_of_tolerance"
    if metric == "psnr":
        assert full.outputs["map"] is NO_VALUE
    else:
        assert full.outputs["map"].shape == reference.shape
    masked = block.execute({"image_a": reference, "image_b": sample, "mask": selection})
    assert masked is not None, block.error
    assert masked.passed


@pytest.mark.parametrize("mode", ["rectangle", "binary_mask", "labels"])
def test_grabcut_modes(mode):
    image = np.full((64, 64), 20, np.uint8)
    image[24:40, 24:40] = 220
    markers = np.zeros_like(image)
    markers[16:48, 16:48] = 3 if mode == "labels" else 255
    block = builtin_registry().create(
        "inspect.grabcut", parameters={"mode": mode, "rectangle": [16, 16, 32, 32]}
    )
    result = block.execute({"image": image, "markers": markers})
    assert result is not None, block.error
    np.testing.assert_array_equal(result.outputs["mask"] != 0, image == 220)


@pytest.mark.parametrize("method", ["zhang_suen", "guo_hall"])
def test_thinning_removes_width_and_respects_selection(method):
    image = np.zeros((24, 28), np.uint8)
    image[8:15, 2:26] = 255
    selection = np.zeros_like(image)
    selection[:, :20] = 255
    block = builtin_registry().create("inspect.thin_mask", parameters={"method": method})
    result = block.execute({"image": image, "mask": selection})
    assert result is not None, block.error
    skeleton = result.outputs["mask"]
    assert 5 < np.count_nonzero(skeleton) < 20
    assert not skeleton[:, 20:].any()


def test_fill_holes_leaves_edge_connected_background_and_large_holes():
    image = np.zeros((30, 30), np.uint8)
    image[2:28, 2:28] = 255
    image[2:12, 10] = 0
    image[6:9, 14:17] = 0
    image[16:22, 16:22] = 0
    block = builtin_registry().create("inspect.fill_mask_holes", parameters={"maximum_area": 10})
    result = block.execute({"image": image})
    assert result is not None, block.error
    assert result.outputs["count"] == 1
    assert np.count_nonzero(result.outputs["holes"]) == 9
    assert result.outputs["mask"][7, 15] == 255
    assert result.outputs["mask"][7, 10] == result.outputs["mask"][17, 17] == 0
