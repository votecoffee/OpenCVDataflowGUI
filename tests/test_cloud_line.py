"""Qualify native XYZ line fitting, support identity, and failure behavior."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_line import FitCloudLine
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE, ValuePacket
from opencv_dataflow.pointclouds import PointCloud
from opencv_dataflow.shape_fitting import LINE_DISTANCES
from opencv_dataflow.tickets import TicketClock

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/cloud_line_fit"


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_saved_cloud_fit_preserves_rows_and_orients_from_the_reference(profile):
    graph = Graph.load(EXAMPLE / "cloud_line_fit.json", builtin_registry())
    graph.set_run_profile(profile)
    truth = json.loads((EXAMPLE / "assets/truth.json").read_text(encoding="utf-8"))
    try:
        graph.run()
        block = graph.blocks["fit"]
        result = block.last_result
        assert result and result.passed, block.error
        source = graph.blocks["source"].last_result
        cloud = source.outputs["cloud"]
        data = result.outputs
        line = data["line"]
        assert line["units"] == "mm" and line["frame"] == "fixture"
        assert line["direction_is_oriented"] and line["direction_convention"] == "reference_z"
        axis, center = np.array(line["axis_direction"]), np.array(line["point_on_axis"])
        error = np.degrees(np.arccos(np.clip(axis @ truth["reference_orientation"], -1, 1)))
        assert error < 0.05
        assert data["span"] == pytest.approx(179, abs=0.01)
        np.testing.assert_allclose(
            data["endpoints"],
            np.array(truth["center_mm"])
            + np.array(truth["observed_limits_mm"][::-1])[:, None] * np.array(truth["axis"]),
            atol=0.1,
            rtol=0,
        )
        np.testing.assert_allclose(
            data["radial_distances"],
            np.linalg.norm(np.cross(cloud.points - center, axis), axis=1),
            atol=1e-9,
            rtol=0,
        )
        for name, index_key in (("inliers", "inlier"), ("outliers", "outlier")):
            rows = truth[f"{index_key}_rows"]
            assert data[f"{index_key}_indices"] == rows
            np.testing.assert_array_equal(data[name].points, cloud.points[rows])
            np.testing.assert_array_equal(data[name].colors, cloud.colors[rows])
            assert (
                data[name].attributes["source_index"].tolist() == truth[f"{index_key}_source_index"]
            )
            np.testing.assert_array_equal(
                data[name].attributes["source_scan"], cloud.attributes["source_scan"][rows]
            )
        assert result.payloads["line"]["ticket"] == source.payloads["cloud"]["ticket"]
        assert data["report"]["support_linearity"] > 0.9999
        assert data["report"]["inlier_radial_rms"] < 0.11
        assert bool(result.graphics) == (profile == "debug")
    finally:
        graph.close()


@pytest.mark.local_examples
def test_same_clutter_fails_least_squares_at_the_same_tolerance():
    graph = Graph.load(EXAMPLE / "least_squares_control.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["fit"]
        result = block.last_result
        assert result and not result.passed, block.error
        assert result.outputs["line"] is NO_VALUE
        assert result.outputs["report"]["inlier_count"] < 16
        assert result.fail_reasons[0]["code"] == "cloud_line_insufficient_support"
    finally:
        graph.close()


@pytest.mark.parametrize("distance", LINE_DISTANCES)
def test_all_native_losses_accept_xyz_and_match_opencv(distance):
    x = np.linspace(-44, 43, 30)
    xyz = np.column_stack((x, 55 + x / 4 + 0.025 * np.sin(x), 400 - x / 2))
    cloud = PointCloud(xyz, "mm", "fixture")
    robust = distance in ("fair", "welsch", "huber")
    scale = 0.45
    p = dict(
        distance=distance,
        inlier_tolerance=scale,
        robust_scale=0.23 if robust else 0,
        radius_epsilon=0.002,
        angle_epsilon_rad=0.0005,
    )
    tool = FitCloudLine(parameters=p)
    result = tool.execute({"cloud": cloud})
    assert result and result.passed, tool.error
    xyz = cloud.points.astype(float)
    mean = xyz.mean(axis=0)
    native = cv2.fitLine(
        ((xyz - mean) / scale).astype(np.float32).reshape(-1, 1, 3),
        LINE_DISTANCES[distance],
        p["robust_scale"] / scale,
        p["radius_epsilon"] / scale,
        p["angle_epsilon_rad"],
    ).ravel()
    fit = result.outputs["line"]
    direction = native[:3].astype(float)
    direction /= np.linalg.norm(direction)
    assert direction @ fit["axis_direction"] == pytest.approx(1.0, abs=1e-13)
    deviation = np.array(fit["point_on_axis"]) - mean - native[3:] * scale
    assert np.linalg.norm(np.cross(deviation, direction)) < 1e-6
    assert not fit["direction_is_oriented"]
    assert fit["direction_convention"] == "canonical_dominant_component"
    if distance == "l2":
        _, _, basis = np.linalg.svd(xyz - mean, full_matrices=False)
        assert abs(basis[0] @ direction) == pytest.approx(1.0, abs=1e-9)


def test_unit_scaling_and_bounded_subsample_keep_full_order_and_ticket():
    x = np.linspace(-90, 90, 200)
    xyz = np.column_stack((x, 55 + x / 4 + 0.025 * np.sin(x), 400 - x / 2))
    ticket = TicketClock("native-line").issue("scan")
    results = {}
    for units, factor in (("mm", 1), ("m", 0.001)):
        cloud = PointCloud(
            xyz * factor, units, "fixture", attributes={"source_index": np.arange(len(x)) * 2}
        )
        tool = FitCloudLine(
            parameters={
                "inlier_tolerance": 0.6 * factor,
                "maximum_fit_samples": 34,
                "random_seed": 107,
            }
        )
        result = tool.execute({"cloud": ValuePacket(cloud, {"ticket": ticket})})
        assert result and result.passed, tool.error
        assert result.payloads["line"]["ticket"] == ticket
        data = result.outputs
        assert data["inlier_indices"] == list(range(200))
        assert data["outlier_indices"] == []
        assert data["inliers"].attributes["source_index"].tolist() == list(range(0, 400, 2))
        assert data["report"]["fit_samples"] == 34
        assert data["report"]["inlier_count"] == 200
        results[units] = np.asarray(data["endpoints"]) / factor
    np.testing.assert_allclose(results["mm"], results["m"], atol=5e-4, rtol=0)


@pytest.mark.parametrize(
    ("xyz", "params", "reason"),
    [
        (np.zeros((0, 3)), {}, "insufficient_points"),
        (np.tile([12, 8, 5], (8, 1)), {}, "coincident"),
        ([[0, 0, 5], [1, 0, 5], [2, 0, 5]], {"minimum_span": 5}, "insufficient_span"),
        ([[0, 0, 5], [1, 0, 5], [0, 1, 5], [1, 1, 5]], {"inlier_tolerance": 2}, "ambiguous_axis"),
    ],
)
def test_degenerate_inputs_and_weak_geometry_withhold_outputs(xyz, params, reason):
    block = FitCloudLine(parameters=params)
    result = block.execute({"cloud": PointCloud(xyz, "mm")})
    assert result and not result.passed, block.error
    assert result.fail_reasons[0]["code"] == f"cloud_line_{reason}"
    assert all(value is NO_VALUE for key, value in result.outputs.items() if key != "report")


def test_perpendicular_reference_and_wrong_frame_have_explicit_outcomes():
    cloud = PointCloud([[0, 0, 5], [1, 0, 5], [2, 0, 5]], "mm", "fixture")
    registry = builtin_registry()
    reference = (
        registry.create(
            "3d.pose",
            parameters={"source_frame": "tool", "target_frame": "fixture", "units": "mm"},
        )
        .execute({})
        .outputs["transform"]
    )
    block = FitCloudLine()
    result = block.execute({"cloud": cloud, "reference_pose": reference})
    assert result and not result.passed, block.error
    assert result.fail_reasons[0]["code"] == "cloud_line_ambiguous_sign"
    assert result.outputs["line"] is NO_VALUE

    assert (
        block.execute({"cloud": cloud, "reference_pose": {**reference, "target_frame": "other"}})
        is None
    )
    assert "cloud frame" in block.error


def test_point_and_native_scale_budgets_report_bad_settings():
    xyz = [[0, 0, 5], [1, 0, 5], [2, 0, 5], [3, 0, 5]]
    inputs = {"cloud": PointCloud(xyz)}
    tiny_budget = FitCloudLine(parameters={"max_input_points": 3})
    assert tiny_budget.execute(inputs) is None
    assert "point budget" in tiny_budget.error
    tiny_tolerance = FitCloudLine(parameters={"inlier_tolerance": 1e-9})
    assert tiny_tolerance.execute(inputs) is None
    assert "native fit range" in tiny_tolerance.error
    wrong_loss = FitCloudLine(parameters={"distance": "l2", "robust_scale": 1})
    assert wrong_loss.execute(inputs) is None
    assert "only fair" in wrong_loss.error.lower()
