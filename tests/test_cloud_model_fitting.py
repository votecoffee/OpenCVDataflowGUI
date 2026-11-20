"""Fit controlled metric models and verify full-cloud provenance and failures."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_model_fitting import FitCloudModel
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.pointclouds import PointCloud
from opencv_dataflow.tickets import TicketClock

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples/custom tools/cloud_model_fitting/cloud_model_fitting.json"
)


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_metric_fits_recover_coefficients_and_known_scan_rows(profile):
    graph = Graph.load(EXAMPLE, builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        for kind, n in (("plane", 63), ("sphere", 72)):
            cloud = graph.blocks[f"{kind}_source"].last_result.outputs["cloud"]
            block = graph.blocks[kind]
            result = block.last_result
            assert result and result.passed, block.error
            out = result.outputs
            model = out["model"]
            points = cloud.points.astype(np.float64) / 1000
            assert model["kind"] == kind and model["frame"] == cloud.frame and model["units"] == "m"
            if kind == "plane":
                normal = np.array([-0.15, 0.2, 1])
                normal /= np.linalg.norm(normal)
                assert model["coefficients"] == pytest.approx(
                    [*normal, -0.6 * normal[2]], abs=0.0006
                )
                independently_signed = points @ np.array(model["normal"]) + model["offset_m"]
            else:
                assert model["coefficients"] == pytest.approx(
                    [-0.033, 0.055, 0.430, 0.038], abs=0.0001
                )
                independently_signed = (
                    np.linalg.norm(points - model["center_m"], axis=1) - model["radius_m"]
                )
            np.testing.assert_allclose(out["signed_residuals_m"], independently_signed, atol=1e-15)
            keep = np.all(cloud.colors == [50, 145, 235], axis=1)
            np.testing.assert_array_equal(np.abs(independently_signed) <= 0.0005, keep)
            assert out["inlier_indices"] == np.flatnonzero(keep).tolist()
            assert out["outlier_indices"] == np.flatnonzero(~keep).tolist()
            for port, indices in (("inliers", keep), ("outliers", ~keep)):
                np.testing.assert_array_equal(out[port].points, cloud.points[indices])
                np.testing.assert_array_equal(out[port].colors, cloud.colors[indices])
                assert out[port].frame == cloud.frame and out[port].units == "mm"
            report = out["report"]
            assert report["inlier_count"] == n and report["classified_points"] == len(points)
            assert report["inlier_fraction"] == pytest.approx(n / len(points))
            assert 0 < report["inlier_rms_m"] < 0.0001
            assert min(independently_signed[~keep]) < 0 < max(independently_signed[~keep])
            assert ("fit support" in result.graphics) == (profile == "debug")
    finally:
        graph.close()


def test_sample_budget_only_limits_fitting_and_keeps_ticket_and_source_pixels():
    x, y = np.meshgrid(np.arange(11), np.arange(11))
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    points = np.vstack((points, [[0, 0, -1], [1, 0, 1], [1, 1, 1]]))
    pixels = np.column_stack((np.arange(len(points)), np.zeros(len(points), dtype=int)))
    cloud = PointCloud(
        points,
        "m",
        "scan",
        normals=np.tile([0, 0, 1], (len(points), 1)),
        pixels=pixels,
        source_size=(len(points), 1),
    )
    ticket = TicketClock().issue("fit")
    spatial = frame([len(points), 1], "camera")
    packet = ValuePacket(cloud, {"ticket": ticket, "spatial": spatial})
    block = FitCloudModel(parameters={"maximum_fit_samples": 23, "minimum_inliers": 100})
    result = block.execute({"cloud": packet})
    assert result and result.passed, block.error
    out = result.outputs
    assert out["report"]["fit_samples"] == 23 and out["report"]["classified_points"] == len(points)
    assert out["model"]["coefficients"] == [0, 0, 1, 0]
    assert out["inlier_indices"] == list(range(121))
    assert out["signed_residuals_m"][-3:] == [-1, 1, 1]
    for port, rows in (("inliers", slice(0, 121)), ("outliers", slice(121, None))):
        assert result.payloads[port]["ticket"] == ticket
        assert result.payloads[port]["spatial"] == spatial
        np.testing.assert_array_equal(out[port].pixels, cloud.pixels[rows])
        np.testing.assert_array_equal(out[port].normals, cloud.normals[rows])
    again = block.execute({"cloud": packet})
    assert again.outputs["model"] == out["model"]
    assert again.outputs["inlier_indices"] == out["inlier_indices"]
    assert again.outputs["report"] == out["report"]


@pytest.mark.parametrize(
    ("cloud", "parameters", "reason"),
    [
        (PointCloud(np.empty((0, 3))), {}, "insufficient_points"),
        (PointCloud([[i, 0, 0] for i in range(20)]), {}, "no_valid_fit"),
        (
            PointCloud([[i, j, 7] for i in range(5) for j in range(5)]),
            {"model": "sphere"},
            "no_valid_fit",
        ),
        (
            PointCloud([[0, 0, 1], [0, 0, -1], [0, 1, 0], [1, 0, 0]]),
            {"model": "sphere", "minimum_sphere_radius_m": 2, "maximum_sphere_radius_m": 3},
            "no_valid_fit",
        ),
        (
            PointCloud([[i, j, 0] for i in range(4) for j in range(4)]),
            {"minimum_inliers": 17},
            "insufficient_support",
        ),
    ],
)
def test_failed_fits_withhold_model_residuals_and_cloud_outputs(cloud, parameters, reason):
    result = FitCloudModel(parameters=parameters).execute({"cloud": cloud})
    assert result is not None and not result.passed
    assert result.fail_reasons[0]["code"] == f"cloud_model_{reason}"
    assert all(value is NO_VALUE for name, value in result.outputs.items() if name != "report")
    if reason == "insufficient_support":
        assert result.outputs["report"]["inlier_count"] == len(cloud.points)
    if reason == "no_valid_fit":
        assert result.outputs["report"]["valid_hypotheses"] == 0


def test_cancellation_during_hypothesis_search_withholds_results():
    cloud = PointCloud([[i, j, i * j] for i in range(4) for j in range(4)])
    flags = iter([False] * 5)
    result = FitCloudModel().execute({"cloud": cloud}, cancelled=lambda: next(flags, True))
    assert result is not None and not result.passed
    assert result.fail_reasons[0]["code"] == "cloud_model_cancelled"
    assert result.outputs["report"]["hypotheses_tested"] > 0
    assert result.outputs["report"]["classified_points"] == 0
    assert result.outputs["model"] is NO_VALUE


@pytest.mark.parametrize(
    ("settings", "units", "message"),
    [
        ({}, "unitless", "cloud in m or mm"),
        ({"max_input_points": 3}, "m", "point budget"),
        ({"model": "sphere", "maximum_fit_samples": 3}, "m", "at least four"),
        (
            {"model": "sphere", "minimum_sphere_radius_m": 0.08, "maximum_sphere_radius_m": 0.03},
            "m",
            "radius must not exceed",
        ),
    ],
)
def test_metric_budget_and_coupled_parameter_validation(settings, units, message):
    cloud = PointCloud(np.zeros((4, 3)), units)
    block = FitCloudModel(parameters=settings)
    assert block.execute({"cloud": cloud}) is None
    assert message in block.error
