"""Verify metric gauge alignment on surveyed targets independent from the fit."""

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import camera_record, pose_record
from opencv_dataflow.payloads import NO_VALUE, ValuePacket
from opencv_dataflow.pointclouds import PointCloud
from opencv_dataflow.scene_scale import RecoverSceneScale
from opencv_dataflow.tickets import TicketClock

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/scene_scale"


def _camera_xyz(points, view):
    matrix = np.asarray(view["world_to_camera"]["source_to_target"])
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _project(xyz, view):
    camera = view["camera"]
    k = np.asarray(camera["camera_matrix"])
    d = np.asarray(camera["distortion"])
    xy, _ = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), k, d)
    return xy.reshape(-1, 2)


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_four_view_workflow_recovers_metric_cloud_and_preserves_reprojection(profile):
    truth = json.loads((EXAMPLE / "assets/ground_truth.json").read_bytes())
    graph = Graph.load(EXAMPLE / "scene_scale.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        original, registered = graph.blocks["unitless"], graph.blocks["metric"]
        assert original.last_result and original.last_result.passed, original.error
        assert registered.last_result and registered.last_result.passed, registered.error
        before, after = original.last_result.outputs, registered.last_result.outputs
        report = after["report"]
        assert report["inlier_anchors"] == 9 and report["declared_anchors"] == 10
        assert [row["id"] for row in report["anchors"] if row["status"] == "outlier"] == ["P02"]
        assert report["holdout_count"] == 8 and not report["missing_holdouts"]
        scale = after["scale_m_per_unit"]
        assert scale == pytest.approx(truth["scale_m_per_unit"], abs=0.0001)
        cloud = after["cloud"]
        assert cloud.frame == "survey_frame" and cloud.units == "m"
        assert cloud.metadata["landmark_ids"] == before["cloud"].metadata["landmark_ids"]
        assert len(cloud.points) == 26
        surveyed = {row["id"]: np.asarray(row["xyz"]) / 1000 for row in truth["points"]}
        xyz = {
            key: point
            for key, point in zip(cloud.metadata["landmark_ids"], cloud.points, strict=True)
        }
        holdout_residuals = [
            np.linalg.norm(xyz[row["id"]] - surveyed[row["id"]]) for row in report["holdouts"]
        ]
        assert report["holdout_rms_m"] == pytest.approx(
            np.sqrt(np.mean(np.square(holdout_residuals))), abs=1e-7
        )
        assert report["holdout_rms_m"] < 0.0008
        assert report["holdout_max_error_m"] < 0.0013
        for raw_view, metric_view in zip(before["views"], after["views"], strict=True):
            assert metric_view["camera"]["units"] == metric_view["world_to_camera"]["units"] == "m"
            assert metric_view["world_to_camera"]["source_frame"] == cloud.frame
            assert metric_view["camera"]["camera_matrix"] == raw_view["camera"]["camera_matrix"]
            raw_xyz = _camera_xyz(before["cloud"].points, raw_view)
            metric_xyz = _camera_xyz(cloud.points, metric_view)
            np.testing.assert_allclose(metric_xyz, raw_xyz * scale, atol=1e-7)
            np.testing.assert_allclose(
                _project(metric_xyz, metric_view), _project(raw_xyz, raw_view), atol=0.0001
            )
        assert bool(registered.last_result.graphics) == (profile == "debug")
    finally:
        graph.close()


def _case():
    points = np.array(
        [[0, 0, 2], [1, 0, 2.5], [0, 2, 3], [-1, -1, 3.2], [-1, 1, 3.5], [2, -1, 2.1]],
        np.float32,
    )
    ids = [3, "A", 9, "B", 17, "C"]
    rot, _ = cv2.Rodrigues(np.array([0.15, -0.28, 0.30]))
    measured = 0.245 * (points @ rot.T) + [0.25, -0.7, 0.03]
    cloud = PointCloud(
        points,
        "unitless",
        "gauge",
        colors=np.arange(18).reshape(6, 3),
        normals=np.tile([0, 0, 1], (6, 1)),
        pixels=np.c_[np.arange(6), np.arange(6)],
        source_size=(8, 8),
        metadata={"landmark_ids": ids},
    )
    refs = [{"id": key, "xyz": xyz.tolist()} for key, xyz in zip(ids, measured, strict=True)]
    controls = {
        "model": "metric_landmark_controls_v1",
        "frame": "datum",
        "units": "m",
        "anchors": [refs[i] for i in (3, 0, 2, 1)],
        "holdouts": refs[4:],
    }
    return cloud, controls, rot, measured


def test_similarity_is_proper_and_transforms_normals_without_disturbing_pixel_provenance():
    cloud, controls, rot, measured = _case()
    block = RecoverSceneScale()
    result = block.execute({"cloud": cloud, "controls": controls})
    assert result and result.passed, block.error
    outputs = result.outputs
    assert outputs["views"] is NO_VALUE
    assert outputs["scale_m_per_unit"] == pytest.approx(0.245, abs=1e-13)
    target = outputs["cloud"]
    assert target.metadata["landmark_ids"] == cloud.metadata["landmark_ids"]
    assert target.metadata["scene_similarity"] == outputs["similarity"]
    np.testing.assert_allclose(target.points, measured, atol=1e-7)
    np.testing.assert_allclose(target.normals, cloud.normals @ rot.T, atol=1e-7)
    np.testing.assert_array_equal(target.pixels, cloud.pixels)
    np.testing.assert_array_equal(target.colors, cloud.colors)
    assert target.source_size == cloud.source_size
    assert np.linalg.det(np.asarray(outputs["similarity"]["rotation"])) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("fault", "reason"),
    [
        ("missing_holdout", "scene_scale_holdout_missing"),
        ("wrong_holdout", "scene_scale_holdout_error_above_maximum"),
        ("collinear", "scene_scale_no_supported_fit"),
        ("missing_anchor", "scene_scale_anchor_support_below_minimum"),
        ("reflection", "scene_scale_anchor_support_below_minimum"),
    ],
)
def test_invalid_survey_geometry_or_independent_evidence_withholds_outputs(fault, reason):
    cloud, controls, _, _ = _case()
    settings = {"minimum_anchor_inlier_ratio": 1}
    if fault == "missing_holdout":
        controls["holdouts"][0]["id"] = "absent"
    elif fault == "wrong_holdout":
        controls["holdouts"][0]["xyz"][0] += 0.1
    elif fault == "collinear":
        cloud = PointCloud(
            np.c_[np.arange(6), np.zeros((6, 2))],
            "unitless",
            "gauge",
            metadata=cloud.metadata,
        )
    elif fault == "missing_anchor":
        controls["anchors"][0]["id"] = "absent"
    else:
        for row in (*controls["anchors"], *controls["holdouts"]):
            row["xyz"][0] *= -1
    block = RecoverSceneScale(parameters=settings)
    result = block.execute({"cloud": cloud, "controls": controls})
    assert result and not result.passed and not block.error
    assert reason in {record["code"] for record in result.fail_reasons}
    for name in ("cloud", "views", "similarity", "scale_m_per_unit"):
        assert result.outputs[name] is NO_VALUE and not block.current_emissions.get(name)
    if fault in ("missing_holdout", "wrong_holdout"):
        assert result.outputs["report"]["anchor_inlier_rms_m"] < 1e-12
        assert result.outputs["report"]["inlier_anchors"] == 4


def test_degenerate_cross_covariance_produces_a_report_instead_of_an_opencv_exception():
    points = np.mgrid[-1:2:2, -1:2:2, -1:2:2].reshape(3, -1).T.astype(np.float64)
    target = points * points[:, [1, 2, 0]]
    np.testing.assert_array_equal(target.T @ points, np.zeros((3, 3)))
    ids = list(range(len(points)))
    cloud = PointCloud(points, "unitless", "gauge", metadata={"landmark_ids": ids})
    controls = {
        "model": "metric_landmark_controls_v1",
        "frame": "world",
        "units": "m",
        "anchors": [{"id": key, "xyz": xyz.tolist()} for key, xyz in zip(ids, target, strict=True)],
    }
    block = RecoverSceneScale(parameters={"minimum_anchor_inlier_ratio": 1})
    result = block.execute({"cloud": cloud, "controls": controls})
    assert result and not result.passed and not block.error
    assert result.outputs["report"]["inlier_anchors"] < len(ids)
    assert result.outputs["cloud"] is NO_VALUE


def _view(name, frame):
    model = camera_record([[20, 0, 15], [0, 21, 10], [0, 0, 1]], [], (32, 24), "unitless", name)
    return {
        "model": "landmark_view_v1",
        "id": name,
        "camera": model,
        "world_to_camera": pose_record(np.eye(4), frame, name, "unitless"),
        "landmark_ids": [3],
        "image_points": [[15, 10]],
    }


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("overlap", "must be disjoint"),
        ("duplicate_ids", "unique landmark IDs"),
        ("units", "unitless cloud"),
        ("wrong_view_frame", "unitless frame"),
        ("duplicate_views", "View IDs must be unique"),
        ("ticket", "same ticket"),
    ],
)
def test_id_frame_units_and_ticket_contracts(fault, message):
    cloud, controls, _, _ = _case()
    arguments = {"cloud": cloud, "controls": controls}
    if fault == "overlap":
        controls["holdouts"].append(copy.deepcopy(controls["anchors"][0]))
    elif fault == "duplicate_ids":
        cloud.metadata["landmark_ids"][0] = "A"
    elif fault == "units":
        cloud.units = "m"
    elif fault == "wrong_view_frame":
        arguments["views"] = [_view(f"cam_{i}", "other") for i in range(3)]
    elif fault == "duplicate_views":
        arguments["views"] = [_view("cam", "gauge") for _ in range(3)]
    else:
        clock = TicketClock()
        for key, value in arguments.items():
            arguments[key] = ValuePacket(value, {"ticket": clock.issue(key)})
    block = RecoverSceneScale()
    assert block.execute(arguments) is None and block.state == "error"
    assert message.lower() in block.error.lower()
