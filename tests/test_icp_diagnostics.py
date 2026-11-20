"""Check native refinement and the independent ICP quality gates."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_tools import ICPCloud
from opencv_dataflow.core import Graph
from opencv_dataflow.icp_diagnostics import evaluate_icp
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.pointclouds import PointCloud

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples/opencv tools/icp_registration/icp_registration.json"
)


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_icp_graph_recovers_known_pose_and_rejects_bad_support(profile):
    graph = Graph.load(EXAMPLE, builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        source = graph.blocks["curved_source"].last_result.outputs["cloud"]
        target = graph.blocks["curved_target"].last_result.outputs["cloud"]
        accepted = graph.blocks["align"].last_result
        assert accepted.passed
        outputs = accepted.outputs
        pose = outputs["pose"]
        matrix = np.asarray(pose["source_to_target"])
        expected_rotation, _ = cv2.Rodrigues(np.array([0.004, -0.003, 0.006]))
        np.testing.assert_allclose(matrix[:3, :3], expected_rotation, atol=5e-5)
        np.testing.assert_allclose(pose["translation"], [0.001, -0.002, 0.004], atol=2e-5)
        assert outputs["cloud"].frame == target.frame
        np.testing.assert_allclose(
            outputs["cloud"].normals,
            source.normals @ matrix[:3, :3].T,
            atol=1e-7,
        )

        xyz = source.points @ matrix[:3, :3].T + matrix[:3, 3]
        delta = xyz[:, None] - target.points[None, :]
        distances = np.linalg.norm(delta, axis=2)
        target_rows = distances.argmin(axis=1)
        distances_to_target = distances[np.arange(len(xyz)), target_rows]
        source_rows = np.flatnonzero(
            (distances_to_target <= 0.003)
            & (distances.argmin(axis=0)[target_rows] == np.arange(len(xyz)))
        )
        peers = target_rows[source_rows]
        normals = target.normals[peers].astype(float)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        pairs = outputs["inliers"]
        np.testing.assert_array_equal(pairs["source_index"], source_rows)
        np.testing.assert_array_equal(pairs["target_index"], peers)
        np.testing.assert_allclose(pairs["distance"], distances[source_rows, peers], atol=1e-12)
        np.testing.assert_allclose(
            pairs["point_to_plane"],
            (delta[source_rows, peers] * normals).sum(axis=1),
            atol=1e-12,
        )
        quality = outputs["report"]["quality"]
        assert quality["source_overlap_count"] == quality["target_overlap_count"] == 288
        assert quality["inlier_count"] == 288 and quality["constraint_rank"] == 6
        assert quality["inlier_rmse"] == pytest.approx(
            np.mean(distances[source_rows, peers] ** 2) ** 0.5
        )
        assert bool(accepted.graphics) == (profile == "debug")

        strict = graph.blocks["strict"].last_result
        planar = graph.blocks["plane"].last_result
        assert {reason["code"] for reason in strict.fail_reasons} == {
            "icp_source_overlap",
            "icp_target_overlap",
        }
        assert {reason["code"] for reason in planar.fail_reasons} == {"icp_degenerate"}
        assert planar.outputs["report"]["native_status"] == 0
        assert planar.outputs["residual"] < 0.001
        planar_quality = planar.outputs["report"]["quality"]
        assert planar_quality["constraint_rank"] == 3
        assert (
            planar_quality["source_overlap_fraction"]
            == planar_quality["target_overlap_fraction"]
            == 1
        )
        for result in (strict, planar):
            assert not result.passed and result.outputs["pose"] is NO_VALUE
            assert result.outputs["cloud"] is NO_VALUE
            assert bool(result.graphics) == (profile == "debug")
    finally:
        graph.close()


def test_tied_neighbors_count_toward_overlap_without_supplying_reciprocal_evidence():
    source = np.array([[0.0, 0, 0], [4, 0, 0], [8, 0, 0], [20, 0, 0]])
    target = PointCloud(
        [[-1, 0, 0], [1, 0, 0], [4, 0, -0.5], [8, 0, 0.5], [30, 0, 0]],
        normals=[[0, 0, 1]] * 5,
    )
    report, rows, peers, distances, planes, _ = evaluate_icp(source, target, 1, lambda: False)
    assert report["source_overlap_count"] == 3 and report["target_overlap_count"] == 4
    assert report["ambiguous_source_count"] == 1 and report["inlier_count"] == 2
    assert report["constraint_rank"] == 2
    np.testing.assert_array_equal(rows, [1, 2])
    np.testing.assert_array_equal(peers, [2, 3])
    np.testing.assert_allclose(distances, [0.5, 0.5])
    np.testing.assert_allclose(planes, [0.5, -0.5])
    assert evaluate_icp(source, target, 1, lambda: True) is None


def test_quality_gate_parameter_constraints():
    with pytest.raises(ValidationError, match="correspondence radius"):
        ICPCloud(
            parameters={"quality_checks": True, "maximum_inlier_rmse": 0.02}
        ).validate_parameters()
    with pytest.raises(ValidationError, match="point budget"):
        ICPCloud(
            parameters={"quality_checks": True, "max_points_per_cloud": 6, "minimum_inliers": 7}
        ).validate_parameters()
