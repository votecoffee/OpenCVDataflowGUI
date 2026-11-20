"""Verify region boundaries, fallback fits, provenance and work limits."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_region_growing import GrowCloudRegions
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.pointclouds import PointCloud
from opencv_dataflow.tickets import TicketClock

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples/custom tools/cloud_region_growing/cloud_region_growing.json"
)


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_fold_separates_faces_and_rejects_exact_seam_rows(profile):
    graph = Graph.load(EXAMPLE, builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        source = graph.blocks["source"].last_result.outputs["cloud"]
        seam = (source.points[:, 0] == 20) & (source.points[:, 2] == 500)
        expected = np.where(source.normals[:, 2] == 1, 1, 2)
        assert seam.sum() == 10
        for key, labels, counts in (
            ("join", np.ones(50, int), [50]),
            ("split", expected, [25, 25]),
            ("trim", np.where(seam, 0, expected), [20, 20]),
        ):
            block = graph.blocks[key]
            result = block.last_result
            assert result and result.passed, block.error
            out = result.outputs
            assert out["labels"] == labels.tolist()
            assert out["count"] == len(counts)
            assert [region["point_count"] for region in out["regions"]] == counts
            for label, region in enumerate(out["regions"], start=1):
                assert region["source_indices"] == np.flatnonzero(labels == label).tolist()
            assert out["report"]["normal_source"] == "cloud"
            assert out["report"]["curvature_source"] == "local_fit"
            np.testing.assert_array_equal(np.array(out["curvature"]) > 0.05, seam)
            for port, subset in (("cloud", labels != 0), ("rejected_cloud", labels == 0)):
                for attribute in ("points", "colors", "normals"):
                    np.testing.assert_array_equal(
                        getattr(out[port], attribute), getattr(source, attribute)[subset]
                    )
                assert out[port].units == "mm" and out[port].frame == "fold_fixture"
            assert ("regions" in result.graphics) == (profile == "debug")
    finally:
        graph.close()


def test_plane_fit_handles_large_origin_and_rejects_collinear_neighborhoods():
    plane = [[x, y, 1000] for y in range(3) for x in range(3)]
    line = [[15 + x, 0, 1000] for x in range(4)]
    xyz = np.array(plane + line) + [100_000, 100_000, 100_000]
    cloud = PointCloud(xyz)
    result = GrowCloudRegions(parameters={"radius_m": 1.01, "maximum_curvature": 0}).execute(
        {"cloud": cloud}
    )
    assert result and result.passed
    assert cloud.normals is None
    assert result.outputs["labels"] == [1] * 9 + [0] * 4
    normal = result.outputs["cloud"].normals
    np.testing.assert_array_equal(np.abs(normal[:, 2]), np.ones(9))
    assert result.outputs["curvature"] == [0] * 9 + [None] * 4
    assert result.outputs["report"]["invalid_geometry_points"] == 4


def test_exact_links_unsigned_normals_and_curvature_break_bridge_with_payloads():
    ticket = TicketClock().issue("fold")
    spatial = frame([5, 1], "camera")
    cloud = PointCloud(
        [[x, 0, 0] for x in range(5)],
        "m",
        "fold",
        normals=[[0, 0, (-1) ** x] for x in range(5)],
        pixels=[[x, 0] for x in range(5)],
        source_size=(5, 1),
    )
    block = GrowCloudRegions(
        parameters={"radius_m": 1, "normal_angle_deg": 0, "minimum_points": 2},
    )
    result = block.execute(
        {
            "cloud": ValuePacket(cloud, {"ticket": ticket, "spatial": spatial}),
            "curvature": [0, 0.05, 0.06, 0, 0],
        },
    )
    assert result and result.passed, block.error
    assert result.outputs["labels"] == [1, 1, 0, 2, 2]
    assert result.outputs["report"]["high_curvature_points"] == 1
    assert result.outputs["report"]["curvature_source"] == "input"
    for port, rows in (("cloud", [0, 1, 3, 4]), ("rejected_cloud", [2])):
        assert result.payloads[port]["ticket"] == ticket
        assert result.payloads[port]["spatial"] == spatial
        np.testing.assert_array_equal(result.outputs[port].pixels, cloud.pixels[rows])


@pytest.mark.parametrize("values", [[0], [0, -1, 0], [0, 0.34, 0], [0, "inf", 0]])
def test_curvature_input_validates_source_alignment_and_range(values):
    cloud = PointCloud(np.zeros((3, 3)), normals=np.tile([0, 0, 1], (3, 1)))
    block = GrowCloudRegions()
    assert block.execute({"cloud": cloud, "curvature": values}) is None
    assert "Curvature must" in block.error


@pytest.mark.parametrize(
    ("settings", "code", "cancelled"),
    [
        ({"max_neighbor_pairs": 1}, "pair_budget", lambda: False),
        ({"max_regions": 1}, "result_budget", lambda: False),
        ({}, "cancelled", iter([False, False, True]).__next__),
    ],
)
def test_limits_never_emit_partial_regions(settings, code, cancelled):
    xyz = [[0, 0, 0]] * 3 + [[1, 0, 0]] * 3
    cloud = PointCloud(xyz, normals=np.tile([0, 0, 1], (len(xyz), 1)))
    result = GrowCloudRegions(parameters=settings).execute(
        {"cloud": cloud, "curvature": [0] * len(xyz)},
        cancelled=cancelled,
    )
    assert result is not None and not result.passed
    assert result.fail_reasons[0]["code"] == f"cloud_region_{code}"
    assert result.outputs["labels"] is NO_VALUE and result.outputs["curvature"] is NO_VALUE
    assert result.outputs["report"]["classified_points"] == 0
