"""Verify oriented extents, pose direction, and symmetric or sparse cases."""

from itertools import product
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_extents import MeasureOrientedCloud
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import pose_matrix, pose_record
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.pointclouds import PointCloud

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/custom tools/oriented_cloud_extents"
AXES = Rotation.from_rotvec([0.23, -0.49, 0.57]).as_matrix()
CENTER = np.array([310.5, -69.75, 581.25])


def _box(extent=(82, 34, 13), units="mm"):
    corners = np.asarray(list(product((-1, 1), repeat=3))) * np.asarray(extent) / 2
    xyz = corners @ AXES.T + CENTER
    return PointCloud(xyz if units == "mm" else xyz / 1000, units, "station")


def _reference(axes=AXES, target="station"):
    return pose_record(pose_matrix(axes, [9, 8, 7]), "fixture", target, "m")


def _check_box(result, cloud, dimensions, center):
    assert result.passed
    pose = result.outputs["pose"]
    transform = np.asarray(pose["source_to_target"])
    axes = transform[:3, :3]
    assert (pose["source_frame"], pose["target_frame"], pose["units"]) == (
        "oriented_box",
        cloud.frame,
        cloud.units,
    )
    np.testing.assert_allclose(axes.T @ axes, np.eye(3), atol=1e-10)
    assert np.linalg.det(axes) == pytest.approx(1)
    np.testing.assert_allclose(transform[:3, 3], center, atol=3e-5)
    np.testing.assert_allclose(result.outputs["dimensions"], dimensions, atol=4e-5)
    local = (cloud.points - transform[:3, 3]) @ axes
    np.testing.assert_allclose(local.min(axis=0), -np.array(dimensions) / 2, atol=4e-5)
    np.testing.assert_allclose(local.max(axis=0), np.array(dimensions) / 2, atol=4e-5)
    return axes


@pytest.mark.parametrize("units", ["mm", "m"])
def test_pca_recovers_rotated_extents_pose_and_global_sign_convention(units):
    cloud = _box(units=units)
    tool = MeasureOrientedCloud()
    result = tool.execute({"cloud": cloud})
    scale = 1 if units == "mm" else 0.001
    axes = _check_box(result, cloud, np.array([82, 34, 13]) * scale, CENTER * scale)
    np.testing.assert_allclose(np.abs(axes.T @ AXES), np.eye(3), atol=2e-6)
    assert result.outputs["report"]["numerical_rank"] == 3
    assert result.outputs["report"]["box"]["volume"] == pytest.approx(
        82 * 34 * 13 * scale**3, rel=7e-6
    )
    reverse = tool.execute({"cloud": cloud.subset(np.arange(7, -1, -1))})
    np.testing.assert_allclose(
        reverse.outputs["pose"]["source_to_target"],
        result.outputs["pose"]["source_to_target"],
        atol=1e-9,
    )


def test_reference_resolves_pca_signs_but_does_not_replace_measured_center():
    cloud = _box()
    signed = AXES * [-1, 1, -1]
    tool = MeasureOrientedCloud()
    result = tool.execute({"cloud": cloud, "reference_pose": _reference(signed)})
    axes = _check_box(result, cloud, [82, 34, 13], CENTER)
    np.testing.assert_allclose(axes, signed, atol=2e-6)
    assert result.outputs["report"]["axis_direction_source"] == "reference_signs"
    straight = PointCloud(list(product([-41, 41], [-17, 17], [-6.5, 6.5])), "mm", "station")
    half_turn_tie = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    tied = tool.execute({"cloud": straight, "reference_pose": _reference(half_turn_tie)})
    assert tied.outputs["pose"] is NO_VALUE
    assert tied.fail_reasons[0]["code"] == "cloud_extents_ambiguous_reference_signs"


@pytest.mark.parametrize("dimensions", [(36, 36, 36), (82, 36, 36), (36, 35.6, 11)])
def test_symmetric_or_nearly_symmetric_data_requires_an_external_orientation(dimensions):
    cloud = _box(dimensions)
    tool = MeasureOrientedCloud()
    result = tool.execute({"cloud": cloud, "reference_pose": _reference()})
    assert not result.passed
    assert result.outputs["pose"] is NO_VALUE and result.outputs["dimensions"] is NO_VALUE
    assert result.fail_reasons[0]["code"] == "cloud_extents_ambiguous_principal_axes"
    assert "provisional_pca_box" in result.outputs["report"]
    fixed = MeasureOrientedCloud(parameters={"orientation": "reference"})
    measured = fixed.execute({"cloud": cloud, "reference_pose": _reference()})
    np.testing.assert_allclose(_check_box(measured, cloud, dimensions, CENTER), AXES, atol=1e-10)


def test_rank_deficient_and_incompatible_reference_have_distinct_failures():
    tool = MeasureOrientedCloud()
    planar = _box((82, 34, 0))
    result = tool.execute({"cloud": planar})
    assert result.passed and result.outputs["report"]["numerical_rank"] == 2
    assert result.outputs["dimensions"][2] < 4e-5
    line = _box((82, 0, 0))
    result = tool.execute({"cloud": line})
    assert result.fail_reasons[0]["code"] == "cloud_extents_rank_deficient"
    assert result.outputs["pose"] is NO_VALUE
    assert tool.execute({"cloud": planar, "reference_pose": _reference(target="other")}) is None
    assert "cloud frame" in tool.error


def test_reference_box_center_comes_from_extrema_with_uneven_sampling():
    cloud = _box()
    biased = PointCloud(
        np.r_[cloud.points, np.repeat(cloud.points[:1], 32, axis=0)], "mm", "station"
    )
    assert np.linalg.norm(biased.points.mean(axis=0) - CENTER) > 20
    tool = MeasureOrientedCloud(parameters={"orientation": "reference"})
    result = tool.execute({"cloud": biased, "reference_pose": _reference()})
    _check_box(result, biased, [82, 34, 13], CENTER)


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_example_replays_with_pose_and_dimensions_through_the_registry(profile):
    graph = Graph.load(EXAMPLE / "oriented_cloud_extents.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        pca, fixed = (graph.blocks[key].last_result for key in ("pca", "reference"))
        assert pca.passed and fixed.passed
        for result in (pca, fixed):
            np.testing.assert_allclose(result.outputs["dimensions"], [82, 34, 13], atol=6e-5)
            assert result.outputs["pose"]["target_frame"] == "measurement_station"
        assert pca.payloads["pose"]["ticket"] == fixed.payloads["pose"]["ticket"]
    finally:
        graph.close()


@pytest.mark.local_examples
def test_example_cube_control_withholds_pca_pose_and_uses_fixture_axes():
    graph = Graph.load(EXAMPLE / "oriented_cloud_extents.json", builtin_registry())
    try:
        graph.blocks["source"].parameters["path"] = "assets/rotated_cube.ply"
        graph.run()
        pca, fixed = (graph.blocks[key].last_result for key in ("pca", "reference"))
        assert pca.outputs["pose"] is NO_VALUE
        assert pca.fail_reasons[0]["code"] == "cloud_extents_ambiguous_principal_axes"
        assert fixed.passed
        np.testing.assert_allclose(fixed.outputs["dimensions"], [36, 36, 36], atol=6e-5)
    finally:
        graph.close()
