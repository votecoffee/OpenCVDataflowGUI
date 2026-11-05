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
