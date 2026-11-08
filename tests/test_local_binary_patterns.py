"""Verify native LBP codes, neighborhood selection and comparison compatibility."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock


@pytest.mark.parametrize(
    ("method", "neighbors", "bins"),
    [("default", 12, 4096), ("ror", 8, 256), ("uniform", 24, 26), ("nri_uniform", 24, 555)],
)
def test_local_binary_patterns_match_native_on_mask_and_region(method, neighbors, bins):
    feature = pytest.importorskip("skimage.feature")
    rng = np.random.default_rng(31)
    image = rng.integers(0, 4096, (16, 18), dtype=np.uint16)
    mask = np.zeros(image.shape, np.uint8)
    mask[3:13, 6:13] = 255
    mask[5, 6] = 0
    region = {
        "model": "image_region_v1", "image_size": [18, 16],
        "shape": "rectangle", "rectangle": [0, 0, 11, 10],
    }
    expected_mask = np.zeros(image.shape, dtype=bool)
    expected_mask[3:10, 6:11] = True
    expected_mask[5, 6] = False
    ticket = TicketClock().issue("camera")
    spatial = frame([18, 16], "surface")
    block = builtin_registry().create("inspect.local_binary_patterns", parameters={
        "method": method, "neighbors": neighbors, "radius": 1.5,
    })
    result = block.execute({
        "image": ValuePacket(image, {"spatial": spatial, "ticket": ticket}),
        "region": region, "mask": mask,
    })
    assert result is not None, block.error
    native = feature.local_binary_pattern(image, neighbors, 1.5, method=method).astype(np.int32)
    np.testing.assert_array_equal(result.outputs["valid_mask"], expected_mask * 255)
    np.testing.assert_array_equal(result.outputs["codes"], np.where(expected_mask, native, -1))
    assert result.outputs["codes"].dtype == np.int32
    assert result.payloads["codes"]["spatial"] == spatial
    assert result.payloads["histogram"]["ticket"] == ticket
    channel = result.outputs["histogram"]["channels"][0]
    assert channel["counts"] == np.bincount(native[expected_mask], minlength=bins).tolist()
    assert result.outputs["sample_count"] == sum(channel["counts"]) == 34
    assert "LBP histogram" in result.graphics


def test_lbp_histogram_checks_settings_and_excludes_border():
    pytest.importorskip("skimage.feature")
    registry = builtin_registry()
    image = np.tile(np.arange(20, dtype=np.uint8), (22, 1))
    block = registry.create("inspect.local_binary_patterns")
    ref = block.execute({"image": image})
    assert ref is not None, block.error
    np.testing.assert_array_equal(ref.outputs["valid_mask"][1:-1, 1:-1], 255)
    assert ref.outputs["sample_count"] == 20 * 18
    assert np.all(ref.outputs["codes"][0] == -1)
    same = block.execute({"image": image + 70})
    compare = registry.create("inspect.compare_histograms")
    inputs = {"reference": ref.outputs["histogram"], "sample": same.outputs["histogram"]}
    result = compare.execute(inputs)
    assert result is not None and result.outputs["value"] == pytest.approx(0), compare.error
    block.parameters["radius"] = 2
    different = block.execute({"image": image})
    assert compare.execute({**inputs, "sample": different.outputs["histogram"]}) is None
    assert "same channels" in compare.error


def test_local_binary_patterns_reject_invalid_settings_and_empty_neighborhoods():
    registry = builtin_registry()
    block = registry.create("inspect.local_binary_patterns", parameters={
        "method": "default", "neighbors": 13,
    })
    image = np.zeros((8, 8), np.uint8)
    assert block.execute({"image": image}) is None
    assert "at most 12" in block.error
    block.parameters.update(method="uniform", radius=2)
    assert block.execute({"image": image, "mask": np.zeros_like(image)}) is None
    assert "complete LBP neighborhood" in block.error
