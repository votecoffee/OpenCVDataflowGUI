"""Verify local-feature registration, correspondence provenance and failure gates."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_features import RegisterCloudFeatures
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import camera_record
from opencv_dataflow.images import DepthArray
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.pointclouds import PointCloud

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples/opencv tools/cloud_feature_registration/cloud_feature_registration.json"
)
ROTATION = Rotation.from_rotvec([0.19, -0.28, 0.31]).as_matrix()
TRANSLATION = np.array([0.125, -0.06, 0.09])


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_partial_scan_recovers_truth_and_reports_inlier_and_overlap_errors(profile):
    graph = Graph.load(EXAMPLE, builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        block = graph.blocks["align"]
        result = block.last_result
        assert result and result.passed, block.error
        outputs = result.outputs
        pose = outputs["pose"]
        transform = np.array(pose["source_to_target"])
        assert (pose["source_frame"], pose["target_frame"], pose["units"]) == (
            "source_scan",
            "target_scan",
            "m",
        )
        error = Rotation.from_matrix(transform[:3, :3] @ ROTATION.T).magnitude()
        assert np.degrees(error) < 0.03
        assert np.linalg.norm(transform[:3, 3] - TRANSLATION) < 0.0003
        source = graph.blocks["source"].last_result.outputs["cloud"]
        target = graph.blocks["target"].last_result.outputs["cloud"]
        a, b = np.asarray(source.points, float), np.asarray(target.points, float)
        aligned = a @ transform[:3, :3].T + transform[:3, 3]
        np.testing.assert_allclose(outputs["cloud"].points, aligned, atol=1e-7, rtol=0)
        rows = outputs["inlier_correspondences"]
        report = outputs["report"]
        assert report["reciprocal_matches"] > report["inliers"] == len(rows) > 400
        assert len({row["source_index"] for row in rows}) == len(rows)
        assert len({row["target_index"] for row in rows}) == len(rows)
        errors = np.array(
            [np.linalg.norm(aligned[row["source_index"]] - b[row["target_index"]]) for row in rows]
        )
        np.testing.assert_allclose(errors, [row["error_m"] for row in rows], atol=1e-12, rtol=0)
        assert report["inlier_rms_m"] == pytest.approx(np.sqrt(np.mean(errors**2)))
        assert report["inlier_rms_m"] < 0.0005
        for name, first, other in (("source", aligned, b), ("target", b, aligned)):
            distances = cKDTree(other).query(first)[0]
            assert report["overlap"][name]["fraction"] == np.mean(distances <= 0.004)
        assert graph.blocks["view"].last_result.passed
        if profile == "debug":
            assert result.graphics["Registered views"].ndim == 3
        else:
            assert not result.graphics
    finally:
        graph.close()


def test_partial_view_depth_reconstruction_preserves_both_pixel_correspondences():
    registry = builtin_registry()
    size = 52
    rng = np.random.default_rng(844)
    y, x = np.indices((size, size))
    z = 0.8 + 0.015 * rng.random((size, size)) + 0.08 * (x > 26)
    z += 0.05 * np.exp(-((x - 20) ** 2 + (y - 19) ** 2) / 95)
    a = np.where((x < 37) & (rng.random(z.shape) > 0.03), z, np.nan)
    b = np.rot90(np.where((x >= 14) & (rng.random(z.shape) > 0.03), z, np.nan), k=-1)
    inputs = {}
    for name, values in (("source", a), ("target", b)):
        camera = camera_record([[95, 0, 25.5], [0, 95, 25.5], [0, 0, 1]], [], z.shape, frame=name)
        block = registry.create("3d.depth_cloud", parameters={"sample_step": 1})
        result = block.execute({"depth": DepthArray(values, "m"), "camera": camera})
        assert result and result.passed, block.error
        inputs[name] = result.outputs["cloud"]
    block = RegisterCloudFeatures()
    result = block.execute(inputs)
    assert result and result.passed, block.error
    outputs = result.outputs
    rotation = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    np.testing.assert_allclose(outputs["pose"]["source_to_target"], rotation, atol=1e-8, rtol=0)
    np.testing.assert_array_equal(outputs["cloud"].pixels, inputs["source"].pixels)
    assert outputs["cloud"].source_size == inputs["source"].source_size
    for row in outputs["inlier_correspondences"]:
        a_pixel, b_pixel = row["source_pixel"], row["target_pixel"]
        assert b_pixel == [size - 1 - a_pixel[1], a_pixel[0]]
        assert row["error_m"] < 1e-8
    assert outputs["report"]["overlap"]["source"]["fraction"] < 0.65


@pytest.mark.local_examples
def test_coverage_gate_withholds_pose_but_exposes_inliers_on_the_same_clouds():
    graph = Graph.load(EXAMPLE, builtin_registry())
    try:
        graph.blocks["align"].parameters["minimum_overlap_fraction"] = 0.8
        graph.run()
        block = graph.blocks["align"]
        result = block.last_result
        assert result and not result.passed and not block.error
        assert {r["code"] for r in result.fail_reasons} == {"cloud_features_low_overlap"}
        assert result.outputs["pose"] is result.outputs["cloud"] is NO_VALUE
        assert len(result.outputs["inlier_correspondences"]) > 400
        assert not result.outputs["report"]["accepted"]
    finally:
        graph.close()


def test_registration_in_millimeters_preserves_attributes_and_is_repeatable():
    rng = np.random.default_rng(43)
    src = rng.uniform([-20, -30, 500], [30, 20, 560], (200, 3))
    dst = src @ ROTATION.T + TRANSLATION * 1000
    rgb = rng.integers(0, 256, src.shape, np.uint8)
    normal = np.broadcast_to([0.0, 0.0, 1.0], src.shape)
    inputs = {
        "source": PointCloud(src, "mm", "a", colors=rgb, normals=normal),
        "target": PointCloud(dst[rng.permutation(len(dst))], "mm", "b"),
    }
    block = RegisterCloudFeatures(parameters={"ransac_iterations": 128})
    first, second = (block.execute(inputs) for _ in range(2))
    assert first and first.passed and second and second.passed, block.error
    assert first.outputs["inlier_correspondences"] == second.outputs["inlier_correspondences"]
    pose = first.outputs["pose"]
    assert pose["units"] == first.outputs["cloud"].units == "mm"
    np.testing.assert_allclose(pose["translation"], TRANSLATION * 1000, atol=1e-3, rtol=0)
    np.testing.assert_array_equal(first.outputs["cloud"].colors, rgb)
    np.testing.assert_allclose(first.outputs["cloud"].normals, normal @ ROTATION.T, atol=1e-6)
    np.testing.assert_array_equal(inputs["source"].points, src.astype(np.float32))


def test_collinear_and_sparse_clouds_cannot_publish_a_rigid_pose():
    rng = np.random.default_rng(22)
    x = np.cumsum(rng.uniform(0.0007, 0.0025, 55))
    line = np.column_stack((x, np.zeros((len(x), 2))))
    block = RegisterCloudFeatures(parameters={"ransac_iterations": 128})
    result = block.execute(
        {"source": PointCloud(line, frame="a"), "target": PointCloud(line + 0.03, frame="b")}
    )
    assert result and not result.passed
    assert result.outputs["report"]["reciprocal_matches"] > 12
    assert result.fail_reasons[0]["code"] == "cloud_features_no_pose_consensus"
    assert result.outputs["pose"] is NO_VALUE
    empty = PointCloud(np.empty((0, 3)), frame="b")
    result = block.execute({"source": PointCloud(line, frame="a"), "target": empty})
    assert result and not result.passed
    assert result.fail_reasons[0]["code"] == "cloud_features_insufficient_matches"


def test_reflected_cloud_fails_proper_rigid_fit():
    rng = np.random.default_rng(71)
    a = rng.uniform(-0.05, 0.05, (120, 3))
    b = a.copy()
    b[:, 0] *= -1
    block = RegisterCloudFeatures(
        parameters={
            "inlier_distance_m": 0.001,
            "minimum_inlier_fraction": 0.9,
            "minimum_overlap_fraction": 0.8,
            "ransac_iterations": 256,
        }
    )
    result = block.execute({"source": PointCloud(a), "target": PointCloud(b)})
    assert result and not result.passed
    assert result.outputs["pose"] is NO_VALUE
    assert result.outputs["report"]["reciprocal_matches"] == len(a)


def test_mismatched_length_units_are_rejected():
    xyz = np.zeros((12, 3))
    block = RegisterCloudFeatures()
    assert block.execute({"source": PointCloud(xyz), "target": PointCloud(xyz, "mm")}) is None
    assert "matching metric cloud units" in block.error
