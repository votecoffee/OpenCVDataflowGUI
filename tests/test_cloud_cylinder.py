"""Known-cylinder recovery, scan provenance and degenerate-surface gates."""

from pathlib import Path
from runpy import run_path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import pose_record
from opencv_dataflow.payloads import NO_VALUE, ValuePacket
from opencv_dataflow.pointclouds import PointCloud

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/custom tools/cloud_cylinder"
TYPE = "3d.cloud_cylinder"


@pytest.fixture(scope="module")
def example():
    return run_path(str(EXAMPLE / "build_example.py"))


def _assert_truth(output, truth):
    model = output["model"]
    assert model["units"] == "m"
    assert model["kind"] == "cylinder"
    assert abs(output["diameter_m"] - truth["diameter_m"]) < 0.00012
    axis = np.array(model["axis_direction"])
    expected = np.array(truth["axis"])
    angle = np.degrees(np.arccos(np.clip(abs(axis @ expected), 0, 1)))
    assert angle < 0.15
    error = np.array(model["point_on_axis_m"]) - truth["center_m"]
    assert np.linalg.norm(np.cross(error, expected)) < 0.00006
    transform = np.array(output["axis_pose"]["source_to_target"])
    np.testing.assert_allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-10)
    np.testing.assert_allclose(transform[:3, 2], axis)
    np.testing.assert_allclose(transform[:3, 3], model["point_on_axis_m"])
    report = output["report"]
    assert report["angular_span_deg"] == pytest.approx(truth["arc_deg"], abs=2)
    assert report["axial_support_m"] == pytest.approx(truth["axial_span_m"], abs=0.0001)
    assert report["inlier_count"] == truth["wall_points"]
    assert 0.00005 < report["inlier_rms_m"] < 0.00010


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_project_recovers_partial_metric_walls(example, profile):
    graph = Graph.load(EXAMPLE / "cloud_cylinder.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        for key, mode in (("shaft", "estimated"), ("bore", "supplied")):
            _, truth, _ = example["make_scan"](key)
            block = graph.blocks[key]
            result = block.last_result
            assert result and result.passed, block.error
            _assert_truth(result.outputs, truth)
            assert result.outputs["report"]["seed_normals"] == mode
            assert ("cylinder fit" in result.graphics) == (profile == "debug")
    finally:
        graph.close()


def test_bore_residuals_and_seed_pose_preserve_row_order_and_units(example):
    scan, truth, good = example["make_scan"]("bore", normals=True)
    scan = PointCloud(
        scan.points,
        scan.units,
        scan.frame,
        scan.colors,
        normals=scan.normals,
        attributes={"source_index": np.arange(len(good))},
    )
    axis = np.array(truth["axis"])
    u = np.cross(axis, [0, 0, 1])
    u /= np.linalg.norm(u)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((u, np.cross(axis, u), axis))
    transform[:3, 3] = np.array(truth["center_m"]) * 1000
    pose = pose_record(transform, "fixture", scan.frame, "mm")
    packet = ValuePacket(scan, {"inspection_lane": "station-1"})
    settings = {"inlier_tolerance_m": 0.00035, "maximum_fit_samples": 380, "random_seed": 4}
    block = builtin_registry().create(TYPE, parameters={**settings, "surface": "bore"})
    result = block.execute({"cloud": packet, "seed_pose": pose})
    assert result and result.passed, block.error
    out = result.outputs
    _assert_truth(out, truth)
    np.testing.assert_array_equal(out["inlier_indices"], np.flatnonzero(good))
    np.testing.assert_array_equal(out["outlier_indices"], np.flatnonzero(~good))
    np.testing.assert_array_equal(out["inliers"].attributes["source_index"], np.flatnonzero(good))
    np.testing.assert_array_equal(out["outliers"].colors, scan.colors[~good])
    np.testing.assert_array_equal(out["inliers"].normals, scan.normals[good])
    assert result.payloads["inliers"]["inspection_lane"] == "station-1"
    assert np.array(out["model"]["axis_direction"]) @ axis > 0.9999
    assert out["report"]["fit_samples"] == 380
    assert out["report"]["classified_points"] == len(good)
    assert out["report"]["hypotheses_tested"] == 256
    outer = (
        builtin_registry()
        .create(TYPE, parameters=settings)
        .execute({"cloud": packet, "seed_pose": pose})
    )
    assert outer and outer.passed
    np.testing.assert_allclose(
        out["signed_residuals_m"], -np.array(outer.outputs["signed_residuals_m"])
    )


@pytest.mark.parametrize(
    "arc,length,reason",
    [
        (19, None, "angular_support"),
        (None, 0.0006, "axial_support"),
    ],
)
def test_inadequate_wall_support_is_a_failed_result(example, arc, length, reason):
    scan, _, _ = example["make_scan"]("bore", arc=arc, length=length, normals=True, clutter=False)
    block = builtin_registry().create(TYPE, parameters={"inlier_tolerance_m": 0.00035})
    result = block.execute({"cloud": scan})
    assert result and not result.passed, block.error
    assert result.fail_reasons[0]["code"] == f"cloud_cylinder_{reason}"
    assert all(value is NO_VALUE for name, value in result.outputs.items() if name != "report")


@pytest.mark.parametrize("kind", ["line", "plane", "ring"])
def test_degenerate_surfaces_do_not_publish_diameter(kind):
    x = np.linspace(-0.025, 0.025, 30)
    xx, yy = np.meshgrid(x, x)
    if kind == "line":
        points = np.column_stack((x, 2 * x, -3 * x + 0.3))
    elif kind == "plane":
        points = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, 0.3)))
    else:
        t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        points = np.column_stack((0.025 * np.cos(t), 0.025 * np.sin(t), np.full(len(t), 0.3)))
    block = builtin_registry().create(TYPE, parameters={"inlier_tolerance_m": 0.00035})
    result = block.execute({"cloud": PointCloud(points)})
    assert result and not result.passed, block.error
    assert result.outputs["diameter_m"] is NO_VALUE


def test_valid_geometry_can_fail_diameter_tolerance_and_bad_frames_fail_validation(example):
    scan, truth, _ = example["make_scan"]("shaft")
    block = builtin_registry().create(
        TYPE, parameters={"inlier_tolerance_m": 0.00035, "maximum_diameter_m": 0.04}
    )
    result = block.execute({"cloud": scan})
    assert result and not result.passed
    assert result.fail_reasons[0]["code"] == "cloud_cylinder_diameter_limit"
    _assert_truth(result.outputs, truth)
    pose = pose_record(np.eye(4), "fixture", "unrelated")
    assert block.execute({"cloud": scan, "seed_pose": pose}) is None
    assert "cloud frame" in block.error


def test_impossible_diameter_controls_are_rejected():
    block = builtin_registry().create(
        TYPE, parameters={"minimum_diameter_m": 0.04, "maximum_diameter_m": 0.03}
    )
    assert block.execute({"cloud": PointCloud(np.zeros((32, 3)))}) is None
    assert "Minimum diameter must not exceed" in block.error
