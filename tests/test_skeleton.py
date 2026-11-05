"""Verify scikit-image skeletonization and topology-preserving spur pruning."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import NO_VALUE


@pytest.mark.parametrize("method", ["zhang", "lee", "guo_hall", "medial_axis"])
def test_skeletonization_with_selection(method):
    pytest.importorskip("skimage")
    image = np.zeros((24, 40), np.uint8)
    image[8:15, 2:26] = 255
    image[8:15, 30:37] = 255
    selection = np.zeros_like(image)
    selection[:, :26] = 255
    block = builtin_registry().create("inspect.skeletonize", parameters={"method": method})
    result = block.execute({"image": image, "mask": selection})
    assert result is not None, block.error
    skeleton = result.outputs["mask"]
    assert skeleton.shape == image.shape
    assert skeleton.dtype == np.uint8
    assert not skeleton[:, 26:].any()
    assert 5 < result.outputs["pixel_count"] == np.count_nonzero(skeleton) < 7 * 24
    distance = result.outputs["distance"]
    if method == "medial_axis":
        assert distance.shape == image.shape
        assert distance.dtype == np.float32
        assert distance[11, 15] == 4
        assert not distance[:, 26:].any()
    else:
        assert distance is NO_VALUE


def test_pruning_keeps_trunk_loop_and_standalone_line():
    pytest.importorskip("skimage")
    expected = np.zeros((25, 30), np.uint8)
    expected[12, 3:25] = 255
    expected[18, 5:9] = 255
    expected[[2, 5], 20:24] = 255
    expected[2:6, [20, 23]] = 255
    image = expected.copy()
    image[9:12, 12] = 255
    block = builtin_registry().create("inspect.prune_skeleton", parameters={"max_length": 3})
    result = block.execute({"image": image})
    assert result is not None, block.error
    np.testing.assert_array_equal(result.outputs["mask"], expected)
    np.testing.assert_array_equal(result.outputs["removed"], image - expected)
    assert result.outputs["removed_branches"] == 1
    assert result.outputs["removed_pixels"] == 3
    preview = result.outputs["preview"]
    assert preview[10, 12, 2] > preview[10, 12, 1]
    assert preview[12, 12, 1] > preview[12, 12, 2]
    assert "removed spurs" in result.drawings

    # Only the first two of the three branch pixels are selected.
    selection = np.zeros_like(image)
    selection[9:11, 12] = 255
    result = block.execute({"image": image, "mask": selection})
    assert result is not None, block.error
    np.testing.assert_array_equal(result.outputs["mask"], image)
    assert result.outputs["removed_pixels"] == 0


@pytest.mark.parametrize(("limit", "removed"), [(2.7, 0), (2.9, 2)])
def test_pruning_measures_diagonal_length(limit, removed):
    pytest.importorskip("skimage")
    image = np.zeros((25, 30), np.uint8)
    image[10, 2:11] = 255
    image[10:19, 10] = 255
    image[[9, 8], [11, 12]] = 255
    block = builtin_registry().create("inspect.prune_skeleton", parameters={"max_length": limit})
    result = block.execute({"image": image})
    assert result is not None, block.error
    assert result.outputs["removed_pixels"] == removed
    assert result.outputs["mask"][10, 10] == 255
