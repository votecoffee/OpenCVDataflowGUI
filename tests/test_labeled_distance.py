"""Verify distances, seed identity, selection, and the nearest-region graph."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, frame, metadata


@pytest.mark.parametrize("distance_type", ("l1", "chessboard"))
def test_pixel_labels_identify_a_seed_at_the_minimum_distance(distance_type):
    seeds = np.zeros((5, 7), np.uint8)
    positions = np.array([[0, 0], [0, 1], [4, 6]])
    seeds[tuple(positions.T)] = 255
    block = builtin_registry().create(
        "cv.distance_labels", parameters={"distance_type": distance_type, "label_mode": "pixels"}
    )
    result = block.execute({"image": seeds})
    assert result is not None and result.passed, block.error
    outputs = result.outputs
    assert outputs["label_count"] == outputs["seed_pixel_count"] == 3
    labels = outputs["labels"]
    assert labels.dtype == np.int32 and outputs["distance"].dtype == np.float32
    np.testing.assert_array_equal(labels[tuple(positions.T)], [1, 2, 3])
    coordinates = np.indices(seeds.shape).transpose(1, 2, 0)
    offsets = np.abs(coordinates[..., None, :] - positions)
    distances = offsets.sum(axis=-1) if distance_type == "l1" else offsets.max(axis=-1)
    np.testing.assert_array_equal(outputs["distance"], distances.min(axis=-1))
    picked = np.take_along_axis(distances, (labels - 1)[..., None], axis=-1)
    np.testing.assert_array_equal(picked[..., 0], outputs["distance"])
    assert outputs["selected_mask"] is NO_VALUE


def test_component_labels_use_eight_connectivity_and_can_emit_a_zone_mask():
    seeds = np.zeros((4, 6), np.uint8)
    seeds[0, 0] = seeds[1, 1] = seeds[-1, -1] = 255
    spatial = frame([6, 4], "sensor")
    source = ImageArray(seeds, "gray", {"spatial": spatial})
    block = builtin_registry().create("cv.distance_labels", parameters={"selected_label": 2})
    result = block.execute({"image": source})
    assert result is not None and result.passed, block.error
    outputs = result.outputs
    assert outputs["label_count"] == 2 and outputs["seed_pixel_count"] == 3
    labels = outputs["labels"]
    assert labels[0, 0] == labels[1, 1] == 1
    assert labels[-1, -1] == 2
    assert outputs["distance"][2, 2] == pytest.approx(1.4, abs=1e-4)
    np.testing.assert_array_equal(outputs["selected_mask"], np.uint8(labels == 2) * 255)
    assert metadata(outputs["labels"])["spatial"] == spatial
    assert metadata(outputs["selected_mask"])["spatial"] == spatial


def test_zero_polarity_with_mask_and_region_only_restricts_seeds():
    seeds = np.full((3, 4), 255, np.uint8)
    seeds[0, 0] = seeds[0, 3] = seeds[2, 0] = 0
    mask = np.full_like(seeds, 255)
    mask[0, 3] = 0
    region = {
        "model": "image_region_v1",
        "image_size": [4, 3],
        "shape": "rectangle",
        "rectangle": [0, 0, 4, 1],
    }
    block = builtin_registry().create(
        "cv.distance_labels", parameters={"seed_polarity": "zero", "distance_type": "l1"}
    )
    result = block.execute({"image": seeds, "mask": mask, "region": region})
    assert result is not None and result.passed, block.error
    outputs = result.outputs
    assert outputs["seed_pixel_count"] == 1
    assert outputs["label_count"] == 1
    np.testing.assert_array_equal(outputs["labels"], 1)
    np.testing.assert_array_equal(outputs["distance"], [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]])


def test_empty_seed_set_and_unavailable_label_emit_distinct_failures():
    block = builtin_registry().create("cv.distance_labels", parameters={"selected_label": 2})
    missing = block.execute({"image": np.zeros((2, 3), np.uint8)})
    assert missing is not None and not missing.passed, block.error
    assert missing.fail_reasons[0]["code"] == "distance_no_seeds"
    assert missing.outputs["distance"] is missing.outputs["labels"] is NO_VALUE
    assert missing.outputs["label_count"] == missing.outputs["seed_pixel_count"] == 0

    result = block.execute({"image": np.array([[255, 0]], np.uint8)})
    assert result is not None and not result.passed, block.error
    assert result.fail_reasons[0]["code"] == "distance_label_unavailable"
    np.testing.assert_array_equal(result.outputs["labels"], [[1, 1]])
    assert result.outputs["selected_mask"] is NO_VALUE


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_example_partitions_every_pixel_between_two_seed_regions(profile):
    project = (
        Path(__file__).parents[1]
        / "examples/opencv tools"
        / "labeled_distance"
        / "labeled_distance.json"
    )
    graph = Graph.load(project, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    seeds = graph.blocks["union"].last_result.outputs["mask"] != 0
    outputs = graph.blocks["distance"].last_result.outputs
    assert outputs["label_count"] == 2
    assert outputs["seed_pixel_count"] == int(seeds.sum())
    assert outputs["labels"][180, 120] == 1 and outputs["labels"][180, 360] == 2
    np.testing.assert_array_equal(outputs["distance"][seeds], 0)
    assert outputs["distance"][180, 240] == 96
    np.testing.assert_array_equal(outputs["selected_mask"], np.uint8(outputs["labels"] == 1) * 255)
