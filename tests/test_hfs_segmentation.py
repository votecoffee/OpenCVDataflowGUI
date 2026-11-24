"""Check CPU HFS labels, selected outputs and measurement handoff."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.hfs_segmentation import HfsSegments
from opencv_dataflow.images import ImageArray
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/hfs_segmentation"


@pytest.fixture
def scene():
    return cv2.imread(str(EXAMPLE / "assets/scene.png"), cv2.IMREAD_COLOR)


@pytest.mark.parametrize(("profile", "workers"), [("debug", 1), ("lean", 3)])
def test_saved_graph_recovers_ground_truth_and_feeds_measurement(profile, workers):
    truth = cv2.imread(str(EXAMPLE / "assets/truth.png"), cv2.IMREAD_GRAYSCALE)
    graph = Graph.load(EXAMPLE / "hfs_segmentation.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.set_branch_workers(workers)
        graph.run()
        block = graph.blocks["segments"]
        assert block_entry(block).path == (
            "OpenCV", "Hierarchical Feature Selection (hfs)", "Segmentation"
        )
        result = block.last_result
        assert result and result.passed, block.error
        assert result.outputs["count"] == 3
        labels = result.outputs["labels"]
        assert labels.dtype == np.int32
        ids = np.unique(labels)
        lookup = {label: int(np.bincount(truth[labels == label]).argmax()) for label in ids}
        assert set(lookup.values()) == {0, 1, 2}
        mapped = np.array(list(lookup.values()))[np.searchsorted(ids, labels)]
        assert np.mean(truth == mapped) > 0.999
        assert result.payloads["labels"]["spatial"]["root_id"] == "scene:image"
        assert result.payloads["labels"]["ticket"]["value"] == "scene.png"
        measurements = graph.blocks["measure"].last_result
        assert measurements and measurements.passed, graph.blocks["measure"].error
        assert measurements.outputs["count"] == 3
        for item in measurements.outputs["collection"]["items"]:
            assert item["metrics"]["area"] == np.count_nonzero(labels == item["id"])
        assert graph.blocks["filmstrip"].last_result.passed
    finally:
        graph.close()


def test_native_labels_and_mean_color_output_are_on_the_same_grid(scene):
    parameters = {
        "stage1_threshold": 0.13, "stage1_min_size": 90,
        "stage2_threshold": 0.31, "stage2_min_size": 180,
        "spatial_weight": 0.77, "superpixel_size": 12,
    }
    native = cv2.hfs.HfsSegment.create(*scene.shape[:2], *parameters.values())
    labels = native.performSegmentCpu(scene, False)
    expected_colors = native.performSegmentCpu(scene, True)
    block = HfsSegments(parameters=parameters)
    result = block.execute({"image": ImageArray(scene[..., ::-1], "rgb")})
    assert result and result.passed, block.error
    np.testing.assert_array_equal(result.outputs["labels"], labels)
    assert result.outputs["image"].color_format == "rgb"
    np.testing.assert_allclose(result.outputs["image"][..., ::-1], expected_colors, atol=1)
    boundaries = result.outputs["boundaries"] != 0
    assert result.outputs["boundaries"].dtype == np.uint8
    for point in ((24, 54), (71, 108), (48, 35), (48, 126)):
        y, x = point
        assert boundaries[y - 1:y + 2, x - 1:x + 2].any()
    assert not boundaries[48, 54]


def test_region_and_mask_clip_labels_keep_native_ids_and_preserve_outer_pixels(scene):
    mask = np.zeros(scene.shape[:2], np.uint8)
    mask[22:76, 16:114] = 255
    region = {
        "model": "image_region_v1", "image_size": [145, 97],
        "shape": "rectangle", "rectangle": [24, 30, 57, 33],
    }
    selected = np.zeros(mask.shape, bool)
    selected[30:63, 24:81] = True
    mask[:, :28] = 0
    selected &= mask != 0
    block = HfsSegments()
    full = block.execute({"image": scene})
    clipped = block.execute({"image": scene, "mask": mask, "region": region})
    assert clipped and clipped.passed, block.error
    assert clipped.outputs["count"] == 2
    np.testing.assert_array_equal(
        clipped.outputs["labels"][selected], full.outputs["labels"][selected]
    )
    assert not clipped.outputs["labels"][~selected].any()
    assert not clipped.outputs["boundaries"][~selected].any()
    np.testing.assert_array_equal(clipped.outputs["image"][~selected], scene[~selected])


def test_empty_selection_and_grayscale_output():
    source = np.full((21, 27), 67, np.uint8)
    block = HfsSegments()
    result = block.execute({"image": source, "mask": np.zeros_like(source)})
    assert result and result.passed, block.error
    assert result.outputs["count"] == 0
    assert not result.outputs["labels"].any() and not result.outputs["boundaries"].any()
    np.testing.assert_array_equal(result.outputs["image"], source)
    assert result.outputs["image"].color_format == "gray"


@pytest.mark.parametrize(
    ("shape", "dtype", "spacing", "error"),
    [
        ((97, 145), np.uint8, 4, "Superpixel size must be a whole number between 8 and 4096"),
        ((7, 32), np.uint8, 8, "fit both image axes"),
        ((520, 512), np.uint8, 8, "4,096 SLIC seeds"),
        ((1300, 1700), np.uint8, 64, "2.1 million pixels"),
        ((32, 32), np.uint16, 8, "Unsupported image depth"),
    ],
)
def test_unsuitable_images_and_seed_grids_are_rejected(shape, dtype, spacing, error):
    block = HfsSegments(parameters={"superpixel_size": spacing})
    assert block.execute({"image": np.zeros(shape, dtype)}) is None
    assert error in block.error


def test_missing_hfs_has_actionable_error(scene, monkeypatch):
    monkeypatch.setattr(cv2, "hfs", None)
    block = HfsSegments()
    assert block.execute({"image": scene}) is None
    assert "requires OpenCV contrib" in block.error
