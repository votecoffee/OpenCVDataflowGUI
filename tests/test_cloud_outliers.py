"""Verify geometric outlier selection, point provenance and finite work buffers."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_outliers import RejectCloudOutliers
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.pointclouds import PointCloud
from opencv_dataflow.tickets import TicketClock

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples/custom tools/cloud_outlier_rejection/cloud_outlier_rejection.json"
)


def _pair_distances(points_m):
    differences = points_m[:, None, :] - points_m[None, :, :]
    out = np.sqrt(np.sum(differences * differences, axis=-1))
    np.fill_diagonal(out, np.inf)
    return out


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_folded_panel_removes_known_noise_and_keeps_the_fold(profile):
    graph = Graph.load(EXAMPLE, builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        source = graph.blocks["source"].last_result.outputs["cloud"]
        matrix = _pair_distances(source.points.astype(np.float64) * 0.001)
        density = (matrix <= 0.008).sum(axis=1)
        local_means = np.sort(matrix, axis=1)[:, :6].mean(axis=1)
        cutoff = local_means.mean() + 1.5 * local_means.std()
        for name, expect in (("radius", density >= 3), ("statistical", local_means <= cutoff)):
            block = graph.blocks[name]
            result = block.last_result
            assert result and result.passed, block.error
            output = result.outputs
            assert output["kept_indices"] == np.flatnonzero(expect).tolist() == list(range(198))
            assert output["removed_indices"] == list(range(198, 204))
            for attr in ("points", "colors", "normals"):
                np.testing.assert_array_equal(
                    getattr(output["cloud"], attr), getattr(source, attr)[:198]
                )
                np.testing.assert_array_equal(
                    getattr(output["removed_cloud"], attr), getattr(source, attr)[198:]
                )
            assert (output["cloud"].units, output["cloud"].frame) == ("mm", "folded_panel")
            assert output["report"]["classified_points"] == 204
            assert output["report"]["removed_points"] == 6
            if name == "radius":
                assert output["report"]["maximum_observed_neighbors"] == int(density.max())
            else:
                assert output["report"]["maximum_local_mean_m"] == pytest.approx(cutoff)
    finally:
        graph.close()


def test_inclusive_radius_excludes_self_but_counts_coincident_rows_and_keeps_pixels():
    points = [[0, 0, 0], [0, 0, 0], [1000, 0, 0], [3000, 0, 0]]
    pixels = [[0, 0], [1, 0], [2, 0], [3, 0]]
    coords = frame([4, 1], "camera")
    ticket = TicketClock().issue("scan")
    cloud = PointCloud(points, "mm", "scan", pixels=pixels, source_size=(4, 1))
    block = RejectCloudOutliers(parameters={"radius_m": 1, "minimum_neighbors": 2})
    result = block.execute({"cloud": ValuePacket(cloud, {"spatial": coords, "ticket": ticket})})
    assert result and result.passed, block.error
    assert result.outputs["kept_indices"] == [0, 1, 2]
    assert result.outputs["removed_indices"] == [3]
    np.testing.assert_array_equal(result.outputs["cloud"].pixels, pixels[:3])
    np.testing.assert_array_equal(result.outputs["removed_cloud"].pixels, pixels[3:])
    for port in ("cloud", "removed_cloud"):
        assert result.payloads[port]["ticket"] == ticket
        assert result.payloads[port]["spatial"] == coords


def test_statistical_neighbors_are_capped_and_zero_variance_keeps_duplicates():
    cloud = PointCloud(np.zeros((3, 3)), "m", "scan")
    block = RejectCloudOutliers(parameters={"method": "statistical", "statistical_neighbors": 20})
    result = block.execute({"cloud": cloud})
    assert result and result.passed, block.error
    assert result.outputs["kept_indices"] == [0, 1, 2]
    report = result.outputs["report"]
    assert report["requested_neighbors"] == 20 and report["effective_neighbors"] == 2
    assert report["population_sd_m"] == 0 and report["maximum_local_mean_m"] == 0


def test_insufficient_neighbors_and_cancellation_do_not_classify_rows():
    singleton = PointCloud([[0, 0, 0]])
    stat = RejectCloudOutliers(parameters={"method": "statistical"})
    result = stat.execute({"cloud": singleton})
    assert result is not None and not result.passed
    assert result.fail_reasons[0]["code"] == "cloud_outlier_insufficient_neighbors"
    assert result.outputs["report"]["classified_points"] == 0
    assert result.outputs["removed_indices"] == []
    radius = RejectCloudOutliers()
    isolated = radius.execute({"cloud": singleton})
    assert isolated is not None and not isolated.passed
    assert isolated.fail_reasons[0]["code"] == "cloud_outlier_no_inliers"
    assert isolated.outputs["removed_indices"] == [0]
    empty = radius.execute({"cloud": PointCloud(np.empty((0, 3)))})
    assert empty is not None and empty.fail_reasons[0]["code"] == "cloud_outlier_empty"
    flags = iter([False, False, True])
    cancelled = radius.execute({"cloud": singleton}, cancelled=lambda: next(flags, True))
    assert cancelled is not None and not cancelled.passed
    assert cancelled.outputs["report"]["classified_points"] == 0
    assert cancelled.outputs["removed_indices"] == []
    assert cancelled.fail_reasons[0]["code"] == "cloud_outlier_cancelled"


@pytest.mark.parametrize(
    ("cloud", "settings", "message"),
    [
        (PointCloud([[0, 0, 0]], "unitless"), {}, "cloud in m or mm"),
        (PointCloud([[0, 0, 0], [1, 0, 0]]), {"max_input_points": 1}, "point budget"),
        (PointCloud([[0, 0, 0]]), {"radius_m": 0}, "Radius m"),
    ],
)
def test_metric_and_budget_validation(cloud, settings, message):
    block = RejectCloudOutliers(parameters=settings)
    assert block.execute({"cloud": cloud}) is None and message in block.error
