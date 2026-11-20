"""Verify local surface fits, orientation, and accepted-cloud data alignment."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_surface_quality import MeasureCloudSurfaceQuality
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.pointclouds import PointCloud
from opencv_dataflow.tickets import TicketClock

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples/custom tools/cloud_surface_quality/cloud_surface_quality.json"
)


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_example_measures_plane_and_sphere_and_feeds_aligned_regions(profile):
    graph = Graph.load(EXAMPLE, builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        source = graph.blocks["source"].last_result.outputs["cloud"]
        block = graph.blocks["quality"]
        result = block.last_result
        assert result and result.passed, block.error
        out = result.outputs
        accepted = out["cloud"]
        keep = (source.colors[:, 1] == 145) | (source.colors[:, 1] == 215)
        assert out["valid_indices"] == np.flatnonzero(keep).tolist()
        for port, mask in (("cloud", keep), ("rejected_cloud", ~keep)):
            np.testing.assert_array_equal(out[port].points, source.points[mask])
            np.testing.assert_array_equal(out[port].colors, source.colors[mask])
            assert (out[port].frame, out[port].units) == (source.frame, "mm")
        assert source.normals is None
        plane = accepted.colors[:, 1] == 145
        sphere = accepted.colors[:, 1] == 215
        np.testing.assert_array_equal(accepted.normals[plane], np.tile([0, 0, 1], (81, 1)))
        rays = accepted.points[sphere] - [60, -10, 430]
        rays /= np.linalg.norm(rays, axis=1)[:, None]
        np.testing.assert_allclose(np.linalg.norm(accepted.normals, axis=1), 1, atol=1e-7)
        assert np.all(np.einsum("ij,ij->i", accepted.normals[sphere], rays) > np.cos(np.deg2rad(6)))
        curvature = np.array(out["curvature"])
        assert np.all(curvature[plane] == 0) and np.all(curvature[sphere] > 0)
        stats = out["report"]
        assert stats["accepted_points"] == 313 and stats["rejected_points"] == 8
        assert stats["insufficient_neighbors"] == 1 and stats["ambiguous_normals"] == 7
        assert ("surface quality" in result.graphics) == (profile == "debug")

        diagnostics = out["source_diagnostics"]
        for row in np.flatnonzero(source.colors[:, 1] == 215)[::17]:
            offsets = source.points.astype(np.float64) - source.points[row]
            nearby = np.linalg.norm(offsets, axis=1) <= 22
            values = np.linalg.eigvalsh(np.cov(offsets[nearby], rowvar=False))
            assert np.count_nonzero(nearby) <= 24
            assert diagnostics["neighbor_counts"][row] == np.count_nonzero(nearby)
            assert diagnostics["curvature"][row] == pytest.approx(values[0] / values.sum())
            assert diagnostics["normal_quality"][row] == pytest.approx(
                (values[1] - values[0]) / values[2]
            )
        for row in np.flatnonzero(~keep):
            assert diagnostics["curvature"][row] is None

        regions = graph.blocks["regions"].last_result
        assert regions and regions.passed, graph.blocks["regions"].error
        assert sorted(region["point_count"] for region in regions.outputs["regions"]) == [81, 232]
        assert regions.outputs["report"]["normal_source"] == "cloud"
        assert regions.outputs["report"]["curvature_source"] == "input"
    finally:
        graph.close()


@pytest.mark.parametrize(("mode", "sign"), [("toward_viewpoint", -1), ("canonical", 1)])
def test_orientation_and_inclusive_radius_with_quality_gate_and_ticket(mode, sign):
    points = [[x, y, 1] for y in range(3) for x in range(3)]
    pixels = [[i, 0] for i in range(9)]
    cloud = PointCloud(points, frame="scan", pixels=pixels, source_size=(9, 1))
    ticket = TicketClock().issue("normal")
    spatial = frame([9, 1], "camera")
    block = MeasureCloudSurfaceQuality(
        parameters={"radius_m": 1, "minimum_normal_quality": 0.7, "orientation": mode}
    )
    result = block.execute({"cloud": ValuePacket(cloud, {"ticket": ticket, "spatial": spatial})})
    assert result and result.passed, block.error
    out = result.outputs
    assert out["source_diagnostics"]["neighbor_counts"] == [3, 4, 3, 4, 5, 4, 3, 4, 3]
    assert out["source_diagnostics"]["normal_quality"][4] == 1
    assert out["report"]["below_minimum_quality"] == 8
    assert out["valid_indices"] == [4] and out["curvature"] == [0]
    np.testing.assert_array_equal(out["cloud"].normals, [[0, 0, sign]])
    for port, rows in (("cloud", [4]), ("rejected_cloud", [0, 1, 2, 3, 5, 6, 7, 8])):
        assert result.payloads[port]["ticket"] == ticket
        assert result.payloads[port]["spatial"] == spatial
        np.testing.assert_array_equal(out[port].pixels, np.array(pixels)[rows])


@pytest.mark.parametrize(
    ("xyz", "radius", "sparse", "ambiguous", "neighbors"),
    [
        ([[0, 0, 0]], 1, 1, 0, [1]),
        ([[0, 0, 0]] * 4, 1, 0, 4, [4] * 4),
        (np.vstack((np.eye(3), -np.eye(3))), 2, 0, 6, [6] * 6),
    ],
)
def test_degenerate_cloud_diagnostics_keep_source_order(xyz, radius, sparse, ambiguous, neighbors):
    block = MeasureCloudSurfaceQuality(parameters={"radius_m": radius})
    result = block.execute({"cloud": PointCloud(xyz)})
    assert result is not None and not result.passed, block.error
    assert result.fail_reasons[0]["code"] == "cloud_surface_no_valid_fits"
    out = result.outputs
    assert out["report"]["insufficient_neighbors"] == sparse
    assert out["report"]["ambiguous_normals"] == ambiguous
    assert out["source_diagnostics"]["neighbor_counts"] == neighbors
    assert out["source_diagnostics"]["curvature"] == [None] * len(xyz)
    assert out["curvature"] == [] and out["valid_indices"] == []
    np.testing.assert_array_equal(out["rejected_cloud"].points, xyz)


@pytest.mark.parametrize(
    ("cloud", "settings", "message"),
    [
        (PointCloud([[0, 0, 0]], "unitless"), {}, "cloud in m or mm"),
        (PointCloud(np.zeros((3, 3))), {"max_input_points": 2}, "point budget"),
        (PointCloud([[0, 0, 0]]), {"viewpoint_m": [0, 0]}, "Viewpoint must be finite XYZ"),
        (
            PointCloud([[0, 0, 0]]),
            {"fit_neighbors": 3, "minimum_neighbors": 4},
            "Minimum neighbors",
        ),
    ],
)
def test_quality_measurement_validates_cloud_and_configuration(cloud, settings, message):
    block = MeasureCloudSurfaceQuality(parameters=settings)
    assert block.execute({"cloud": cloud}) is None and message in block.error


def test_cancelled_fit_does_not_emit_partial_diagnostics():
    block = MeasureCloudSurfaceQuality()
    flags = iter([False] * 2)
    cloud = PointCloud([[0, 0, 0], [1, 1, 0], [1, 0, 0]])
    result = block.execute({"cloud": cloud}, cancelled=lambda: next(flags, True))
    assert result and not result.passed
    assert result.fail_reasons[0]["code"] == "cloud_surface_cancelled"
    assert result.outputs["report"]["classified_points"] == 0
    assert result.outputs["source_diagnostics"] is NO_VALUE
