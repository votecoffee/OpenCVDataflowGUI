"""Qualify DAISY native rows, bounded sampling and known-displacement matches."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/daisy_descriptors"


def daisy(**settings):
    return builtin_registry().create("inspect.daisy_descriptor", parameters=settings)


@pytest.mark.parametrize("norm", ["none", "partial", "full", "sift"])
def test_grid_native_rows_and_selected_lattice_indices(norm):
    image = np.random.default_rng(256).integers(0, 256, (80, 104), dtype=np.uint8)
    spatial = frame([104, 80], "daisy")
    ticket = TicketClock("daisy").issue("vision", "fixed", "target")
    mask = np.zeros_like(image)
    mask[28:63, 30:80] = 255
    region = {
        "model": "image_region_v1",
        "image_size": [104, 80],
        "shape": "rectangle",
        "rectangle": [41, 0, 38, 80],
        "coordinate_context": spatial,
    }
    node = daisy(
        radius=8,
        radial_divisions=2,
        angular_divisions=5,
        histogram_bins=4,
        grid_step=16,
        grid_origin=[14, 14],
        grid_angle=35,
        use_orientation=True,
        normalization=norm,
    )
    result = node.execute(
        {
            "image": ValuePacket(image, {"spatial": spatial, "ticket": ticket}),
            "mask": ValuePacket(mask, {"spatial": spatial}),
            "region": region,
        }
    )
    assert result and result.passed, node.error
    features = result.outputs["features"]
    centers = [
        [46, 30],
        [62, 30],
        [78, 30],
        [46, 46],
        [62, 46],
        [78, 46],
        [46, 62],
        [62, 62],
        [78, 62],
    ]
    assert result.outputs["points"] == centers
    assert [k["source_index"] for k in features["keypoints"]] == [7, 8, 9, 12, 13, 14, 17, 18, 19]
    flag = getattr(cv2.xfeatures2d, "DAISY_NRM_" + norm.upper())
    native = cv2.xfeatures2d.DAISY_create(8, 2, 5, 4, flag, None, True, True)
    _, expected = native.compute(image, [cv2.KeyPoint(x, y, 16, 35) for x, y in centers])
    assert features["descriptor_size"] == 44
    np.testing.assert_array_equal(features["descriptors"], expected)
    np.testing.assert_array_equal(result.outputs["image"], image)
    assert result.payloads["features"]["spatial"] == spatial
    assert result.payloads["features"]["ticket"] == ticket
    assert len(result.drawings["DAISY"].objects) == len(centers)


def test_wired_keypoints_with_nearest_sampling_homography_match_native():
    image = np.random.default_rng(174).integers(0, 256, (88, 100), dtype=np.uint8)
    transform = [[1, 0, 3], [0, 1, 2], [0, 0, 1]]
    points = [[24, 31], [55, 45], [48, 69]]
    node = daisy(radius=7, interpolation=False, sampling_homography=transform, normalization="full")
    node.set_graphics_mode("off")
    result = node.execute({"image": image, "keypoints": points})
    assert result and result.passed, node.error
    native = cv2.xfeatures2d.DAISY_create(
        7, 3, 8, 8, cv2.xfeatures2d.DAISY_NRM_FULL, np.array(transform, float), False, False
    )
    _, expected = native.compute(image, [cv2.KeyPoint(x, y, 7) for x, y in points])
    np.testing.assert_array_equal(result.outputs["features"]["descriptors"], expected)
    assert result.outputs["points"] == points and not result.drawings


def test_grid_homography_and_native_budgets_validate_before_computing(monkeypatch):
    identity = np.eye(3).tolist()
    for settings, field in (
        ({"sampling_homography": identity}, "sampling_homography"),
        ({"grid_origin": [0.5, 0]}, "grid_origin"),
        ({"radial_divisions": 8, "angular_divisions": 32}, "radial_divisions"),
    ):
        with pytest.raises(ValidationError) as error:
            daisy(**settings).validate_parameters()
        assert field in error.value.fields
    image = np.zeros((120, 120), np.uint8)
    horizon = [[1, 0, 0], [0, 1, 0], [1, 0, -60]]
    node = daisy(interpolation=False, sampling_homography=horizon)
    assert node.execute({"image": image}) is None and "finite bounded projection" in node.error
    node = daisy(grid_step=1)
    assert node.execute({"image": image}) is None and "exceeds 4,096 points" in node.error
    node = daisy(use_orientation=True)
    assert node.execute({"image": image, "keypoints": [[50, 50]]}) is None
    assert "measured angle" in node.error

    monkeypatch.setattr("opencv_dataflow.daisy_descriptor._MAX_HISTOGRAM_VALUES", 1)
    assert node.execute({"image": image, "keypoints": []}).outputs["count"] == 0
    assert node.execute({"image": image}) is None and "histogram workspace" in node.error


@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_pair_recovers_all_known_grid_displacements(profile):
    graph = Graph.load(EXAMPLE / "daisy_descriptors.json", builtin_registry())
    graph.set_run_profile(profile)
    graph.set_branch_workers(2)
    try:
        graph.run()
        evaluation = graph.blocks["evaluate"].last_result
        assert evaluation and evaluation.passed
        operating = evaluation.outputs["metrics"]["operating"]
        assert operating["true_positives"] == 63
        assert operating["precision"] == operating["recall"] == 1
        match = graph.blocks["match"].last_result.outputs["matches"]
        shift = np.asarray(match["train_points"]) - np.asarray(match["query_points"])
        np.testing.assert_array_equal(shift, np.tile([8, 6], (63, 1)))
        query, train = [
            np.asarray(graph.blocks[side].last_result.outputs["features"]["descriptors"])
            for side in ("query", "train")
        ]
        distances = np.asarray([row["distance"] for row in match["matches"]])
        expected = [
            np.linalg.norm(query[row["query_index"]] - train[row["train_index"]])
            for row in match["matches"]
        ]
        np.testing.assert_allclose(distances, expected, atol=1e-7)
    finally:
        graph.close()
