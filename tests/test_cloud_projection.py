"""Calibrated rasterization and visibility against independent depth controls."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import camera_record, pose_record
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.pointclouds import PointCloud
from opencv_dataflow.tickets import TicketClock

TYPE = "3d.cloud_project"
EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/cloud_visibility"
K = np.array([[8.0, 0.0, 3.0], [0.0, 8.0, 2.0], [0.0, 0.0, 1.0]])
SIZE = (7, 5)


def xyz(u, v, z):
    return [(u - 3) * z / 8, (v - 2) * z / 8, z]


def test_example_matches_independent_reference_labels_and_nearest_depth():
    def asset(name):
        return cv2.imread(str(EXAMPLE / "assets" / name), cv2.IMREAD_UNCHANGED)

    graph = Graph.load(EXAMPLE / "cloud_visibility.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["project"]
        result = block.last_result
        assert result and result.passed, block.error
        outputs = result.outputs
        encoded = asset("expected_visibility.png")
        labels, overlap = encoded & 15, (encoded & 16) != 0
        truth = asset("expected_nearest_mm.png")
        known = labels > 0
        np.testing.assert_array_equal(outputs["coverage_mask"] > 0, known)
        np.testing.assert_array_equal(outputs["visible_mask"] > 0, labels == 1)
        np.testing.assert_array_equal(outputs["occluded_mask"] > 0, (labels == 2) | overlap)
        np.testing.assert_array_equal(outputs["unverified_mask"] > 0, labels >= 3)
        assert np.isnan(outputs["depth"][~known]).all()
        assert outputs["depth"].units == "m"
        np.testing.assert_allclose(outputs["depth"][known], truth[known] / 1000, atol=1e-7)
        color = outputs["color"]
        assert color.color_format == "rgb"
        assert np.all(color[overlap] == [245, 120, 35])
        assert np.all(color[known & ~overlap] == [30, 210, 140])
        stats = outputs["report"]
        assert stats["self_occluded_points"] == overlap.sum() == 560
        assert stats["reference_occluded_points"] == (labels == 2).sum() == 48
        assert stats["missing_reference_points"] == (labels == 3).sum() == 45
        assert stats["in_front_of_reference_points"] == (labels == 4).sum() == 45
        assert stats["covered_pixels"] == 2508
        assert len(outputs["visible_cloud"].points) == (labels == 1).sum()
        assert len(outputs["occluded_cloud"].points) == 608
        assert len(outputs["unverified_cloud"].points) == 90
    finally:
        graph.close()


def test_pose_units_ties_and_subset_provenance():
    camera = camera_record(K, [], SIZE, frame="target")
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    offset = np.array([0.05, -0.1, 0.2])
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, offset * 1000
    pose = pose_record(transform, "source", "target", "mm")
    target_xyz = np.array([xyz(2, 2, z) for z in (1.0, 1.0, 1.004, 1.5)] + [xyz(5, 2, 1.0)])
    points = (target_xyz - offset) @ rotation * 1000
    colors = np.array([[1, 2, 3], [200, 201, 202], [4, 5, 6], [9, 8, 7], [8, 0, 1]])
    source_grid = frame([5, 1], "source-grid")
    clock = TicketClock()
    cloud_ticket, reference_ticket = clock.issue("cloud"), clock.issue("reference")
    cloud = PointCloud(
        points,
        "mm",
        "source",
        colors=colors,
        pixels=np.c_[np.arange(5), np.zeros(5, int)],
        source_size=(5, 1),
        metadata={"spatial": source_grid, "ticket": cloud_ticket},
    )
    reference = np.zeros(SIZE[::-1], np.uint16)
    reference[2, 2] = 1000
    grid = frame(SIZE, "reference-grid")
    block = builtin_registry().create(TYPE, parameters={"surface_tolerance_m": 0.005})
    result = block.execute(
        {
            "cloud": cloud,
            "camera": camera,
            "transform": pose,
            "reference_depth": ValuePacket(
                reference, {"spatial": grid, "ticket": reference_ticket}
            ),
        }
    )
    assert result and result.passed, block.error
    out = result.outputs
    np.testing.assert_array_equal(out["color"][2, 2], colors[0])
    assert out["depth"][2, 2] == pytest.approx(1.0)
    np.testing.assert_array_equal(out["visible_cloud"].pixels[:, 0], [0, 1, 2])
    np.testing.assert_array_equal(out["occluded_cloud"].pixels[:, 0], [3])
    np.testing.assert_array_equal(out["unverified_cloud"].pixels[:, 0], [4])
    assert out["visible_cloud"].frame == "source" and out["visible_cloud"].units == "mm"
    assert result.payloads["color"]["spatial"] == grid
    assert result.payloads["color"]["ticket"] == reference_ticket
    assert result.payloads["visible_cloud"]["spatial"] == source_grid
    assert result.payloads["visible_cloud"]["ticket"] == cloud_ticket
    assert out["visible_mask"][2, 2] == out["occluded_mask"][2, 2] == 255


def test_pixel_cells_clipping_and_selection_without_reference():
    points = np.array(
        [
            xyz(-0.5, 2, 1),
            xyz(6.5, 2, 1),
            xyz(2.5, 2, 1),
            xyz(2, 2, 0.125),
            xyz(3, 2, 3),
            xyz(2, 2, -1),
            xyz(1, 1, 1),
        ]
    )
    mask = np.zeros(SIZE[::-1], np.uint8)
    mask[2, :4] = 255
    region = {
        "model": "image_region_v1",
        "shape": "rectangle",
        "image_size": list(SIZE),
        "rectangle": [0, 0, 7, 4],
    }
    block = builtin_registry().create(TYPE, parameters={"minimum_z_m": 0.25, "maximum_z_m": 2})
    result = block.execute(
        {
            "cloud": PointCloud(points),
            "camera": camera_record(K, [], SIZE),
            "mask": mask,
            "region": region,
        }
    )
    assert result and result.passed, block.error
    out = result.outputs
    assert out["color"] is NO_VALUE
    np.testing.assert_array_equal(np.nonzero(out["visible_mask"])[1], [0, 3])
    assert not out["occluded_mask"].any() and not out["unverified_mask"].any()
    assert out["report"]["behind_camera_points"] == 1
    assert out["report"]["clipped_z_points"] == 2
    assert out["report"]["outside_camera_grid_points"] == 1
    assert out["report"]["outside_selection_points"] == 1
    assert out["report"]["projected_points"] == 2
    assert not out["report"]["compared_to_reference"]


def test_selection_rejects_an_unrelated_reference_grid():
    block = builtin_registry().create(TYPE)
    values = np.ones(SIZE[::-1], np.float32)
    mask = np.full(SIZE[::-1], 255, np.uint8)
    result = block.execute(
        {
            "cloud": PointCloud([[0, 0, 1]]),
            "camera": camera_record(K, [], SIZE),
            "reference_depth": ValuePacket(values, {"spatial": frame(SIZE, "grid-a")}),
            "mask": ValuePacket(mask, {"spatial": frame(SIZE, "grid-b")}),
        }
    )
    assert result is None
    assert "Selection coordinate frame differs" in block.error


def test_projection_uses_pinhole_distortion():
    # x/z=0.5, k1=0.75 yields u=7.75 instead of 7.0 at the same pinhole intrinsics.
    camera = camera_record(K, [0.75, 0.0, 0.0, 0.0], (12, 7))
    block = builtin_registry().create(TYPE)
    result = block.execute({"cloud": PointCloud([[0.5, 0, 1]]), "camera": camera})
    assert result and result.passed, block.error
    assert result.outputs["depth"][2, 8] == 1
    assert np.isnan(result.outputs["depth"][2, 7])


def test_missing_projection_has_empty_masks_and_failure_status():
    block = builtin_registry().create(TYPE)
    result = block.execute(
        {
            "cloud": PointCloud([[0, 0, -1]]),
            "camera": camera_record(K, [], SIZE),
        }
    )
    assert result and not result.passed, block.error
    assert np.isnan(result.outputs["depth"]).all()
    assert result.fail_reasons[0]["code"] == "no_projected_cloud_samples"
    assert not result.outputs["coverage_mask"].any()
    assert not len(result.outputs["visible_cloud"].points)


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("frame", "Connect a transform"),
        ("units", "requires metric cloud units"),
        ("reference", "Camera image size differs"),
        ("oversize", "projection limit"),
        ("wrong_pose", "map the cloud frame to the camera"),
    ],
)
def test_invalid_inputs_are_rejected(mode, message):
    block = builtin_registry().create(TYPE)
    cloud = PointCloud(
        [[0, 0, 1]],
        "unitless" if mode == "units" else "m",
        "source" if mode == "frame" else "camera",
    )
    camera = camera_record(K, [], (4000, 4000) if mode == "oversize" else SIZE)
    inputs = {"cloud": cloud, "camera": camera}
    if mode == "reference":
        inputs["reference_depth"] = np.ones((7, 7), np.float32)
    if mode == "wrong_pose":
        inputs["transform"] = pose_record(np.eye(4), "camera", "other")
    assert block.execute(inputs) is None
    assert message in block.error
