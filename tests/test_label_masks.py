"""Check signed label selection through comparison and range blocks."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import frame, metadata


def test_compare_selects_negative_label_inside_a_mask_and_region():
    labels = np.array([[-1, -1, 1], [-1, 2, -1]], np.int32)
    spatial = frame([3, 2], "sensor")
    source = ImageArray(labels, "gray", {"spatial": spatial})
    mask = np.array([[255, 0, 255], [255, 255, 255]], np.uint8)
    region = {
        "model": "image_region_v1",
        "image_size": [3, 2],
        "shape": "rectangle",
        "rectangle": [0, 0, 3, 1],
    }
    block = builtin_registry().create("cv.compare_pixels", parameters={"operation": "equal"})
    result = block.execute({"image_a": source, "scalar_b": -1, "mask": mask, "region": region})
    assert result is not None and result.passed, block.error
    np.testing.assert_array_equal(result.outputs["mask"], [[255, 0, 0], [0, 0, 0]])
    assert result.outputs["count"] == 1 and result.outputs["fraction"] == 0.5
    assert metadata(result.outputs["mask"])["spatial"] == spatial


def test_compare_preserves_adjacent_large_ids_and_extreme_differences():
    low, high = np.iinfo(np.int32).min, np.iinfo(np.int32).max
    a = np.array([[low, high, 16_777_217, -1]], np.int32)
    b = np.array([[high, high - 1, 16_777_216, -1]], np.int32)
    block = builtin_registry().create("cv.compare_pixels", parameters={"operation": "equal"})
    result = block.execute({"image_a": a, "image_b": b})
    assert result is not None and result.passed, block.error
    np.testing.assert_array_equal(result.outputs["mask"], [[0, 0, 0, 255]])

    block.parameters["tolerance"] = 1
    result = block.execute({"image_a": a, "image_b": b})
    assert result is not None and result.passed, block.error
    np.testing.assert_array_equal(result.outputs["mask"], [[0, 255, 255, 255]])


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    (
        (-1.1, 1.9, [0, 255, 255, 255, 0]),
        (-0.8, -0.2, [0, 0, 0, 0, 0]),
        (-1e30, -(2**31), [255, 0, 0, 0, 0]),
        (2**31 - 1, 1e30, [0, 0, 0, 0, 255]),
        (2**31, 1e30, [0, 0, 0, 0, 0]),
    ),
)
def test_int32_range_intersects_real_bounds_with_the_integer_domain(lower, upper, expected):
    pixels = np.array([[-(2**31), -1, 0, 1, 2**31 - 1]], np.int32)
    block = builtin_registry().create(
        "cv.in_range", parameters={"lower": [lower], "upper": [upper]}
    )
    result = block.execute({"image": pixels})
    assert result is not None and result.passed, block.error
    np.testing.assert_array_equal(result.outputs["image"], [expected])


def test_wired_label_range_and_selection_preserve_coordinates():
    spatial = frame([4, 1], "camera")
    labels = ImageArray(np.array([[-1, 0, 1, 2]], np.int32), "gray", {"spatial": spatial})
    mask = np.array([[255, 255, 0, 255]], np.uint8)
    block = builtin_registry().create("cv.in_range")
    result = block.execute({"image": labels, "lower": -1, "upper": 1, "mask": mask})
    assert result is not None and result.passed, block.error
    np.testing.assert_array_equal(result.outputs["image"], [[255, 255, 0, 0]])
    assert metadata(result.outputs["image"])["spatial"] == spatial


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_example_masks_watershed_boundaries_and_interiors(profile):
    project = (
        Path(__file__).parents[1] / "examples/image_processing" / "label_masks" / "label_masks.json"
    )
    graph = Graph.load(project, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    watershed = graph.blocks["watershed"].last_result.outputs
    labels = watershed["labels"]
    assert labels.dtype == np.int32
    assert set(np.unique(labels)) == {-1, 1, 2}
    boundaries = graph.blocks["boundaries"].last_result.outputs
    np.testing.assert_array_equal(boundaries["mask"], watershed["boundaries"])
    assert boundaries["count"] == int(np.count_nonzero(labels == -1))
    interiors = graph.blocks["interiors"].last_result.outputs["image"]
    np.testing.assert_array_equal(interiors, watershed["mask"])
