"""Verify metric scan merge, duplicate suppression and source provenance."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_merge import MergeCloudScans
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import pose_record
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket, frame
from opencv_dataflow.pointclouds import PointCloud
from opencv_dataflow.tickets import TicketClock

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples/custom tools/cloud_scan_merge/cloud_scan_merge.json"
)


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_scan_merge_transforms_geometry_and_retains_original_rows(profile):
    graph = Graph.load(EXAMPLE, builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        append = graph.blocks["append"].last_result
        merged = graph.blocks["merge"].last_result
        assert append.passed and merged.passed
        reference_xyz = np.array(
            [
                [0, 0, 400],
                [10, 0, 400],
                [0, 10, 400],
                [10, 10, 400],
                [0.2, 0.1, 400],
                [10.2, 0, 400],
                [20, 0, 400],
                [20, 10, 400],
                [20.2, 0.1, 400],
                [30, 0, 405],
                [30.1, 0.1, 405],
            ]
        )
        np.testing.assert_allclose(append.outputs["cloud"].points, reference_xyz, atol=5e-5)
        cloud = merged.outputs["cloud"]
        np.testing.assert_allclose(
            cloud.points, reference_xyz[[0, 1, 2, 3, 6, 7, 9, 10]], atol=5e-5
        )
        np.testing.assert_allclose(cloud.normals[4:6], [[0, 0.6, 0.8]] * 2, atol=1e-6)
        np.testing.assert_array_equal(cloud.colors[4:6], [[80, 235, 130]] * 2)
        assert cloud.frame == "fixture" and cloud.units == "mm"
        report = merged.outputs["report"]
        assert report["suppressed_points"] == 3 and report["retained_points"] == 8
        assert [row["retained_points"] for row in report["scans"]] == [4, 2, 2]
        origins = merged.outputs["provenance"]["scans"]
        assert [row["output_span"] for row in origins] == [[0, 4], [4, 6], [6, 8]]
        assert [row["source_rows"] for row in origins] == [[0, 1, 2, 3], [2, 3], [1, 2]]
        assert merged.payloads["cloud"]["ticket"]["value"] == "registered-scan-fixture"
        assert ("point cloud" in merged.graphics) == (profile == "debug")
    finally:
        graph.close()


def test_group_retains_member_ticket_and_pixel_grid_in_provenance_only():
    clock = TicketClock()
    camera_a, camera_b = frame((2, 2), "camera_a"), frame((3, 1), "camera_b")
    group_ticket = clock.issue("batch", "fixed", 17)
    a = PointCloud(
        [[0, 0, 1]],
        frame="fixture",
        pixels=[[1, 1]],
        source_size=(2, 2),
        metadata={"spatial": camera_a},
    )
    b = PointCloud(
        [[10, 0, 1000]],
        "mm",
        "fixture",
        pixels=[[2, 0]],
        source_size=(3, 1),
    )
    left = clock.issue("camera_a", "fixed", "left-42")
    right = clock.issue("camera_b", "fixed", "right-91")
    group = FlowGroup(
        (
            ValuePacket(a, {"ticket": left}),
            ValuePacket(b, {"ticket": right, "spatial": camera_b}),
        ),
        ("front", "side"),
    )
    result = MergeCloudScans().execute({"scans": ValuePacket(group, {"ticket": group_ticket})})
    assert result.passed
    assert result.outputs["cloud"].pixels is None and result.outputs["cloud"].source_size is None
    assert "spatial" not in result.payloads["cloud"]
    assert result.payloads["cloud"]["ticket"] == group_ticket
    report = result.outputs["provenance"]["scans"]
    assert [origin["scan"] for origin in report] == ["front", "side"]
    assert [origin["ticket"] for origin in report] == [left, right]
    assert [origin["spatial"] for origin in report] == [camera_a, camera_b]
    assert [origin["source_pixels"] for origin in report] == [[[1, 1]], [[2, 0]]]


def test_overlap_is_inclusive_and_only_compares_to_retained_earlier_rows():
    block = MergeCloudScans(parameters={"merge_mode": "prefer_earlier", "overlap_radius_m": 0.001})
    clouds = [PointCloud([[x, 0, 0]], "mm", "fixture") for x in (0, 1, 2)]
    block.exposed_outputs.remove("provenance")
    result = block.execute({f"cloud_{i}": cloud for i, cloud in enumerate(clouds, 1)})
    assert result.passed and result.outputs["provenance"] is NO_VALUE
    assert result.outputs["report"]["retained_points"] == 2
    np.testing.assert_allclose(result.outputs["cloud"].points, [[0, 0, 0], [0.002, 0, 0]])


def test_invalid_frames_tickets_and_attribute_contract_are_distinct():
    one = PointCloud([[0, 0, 0]], frame="fixture", colors=[[255, 0, 0]])
    two = PointCloud([[1, 0, 0]], frame="sensor")
    couple = {"cloud_1": one, "cloud_2": two}
    block = MergeCloudScans()
    assert block.execute(couple) is None and "needs a pose" in block.error
    transform = pose_record(np.eye(4), "sensor", "fixture", "m")
    couple["pose_2"] = transform
    common = block.execute(couple)
    assert common.passed and common.outputs["cloud"].colors is None
    block.parameters["attribute_policy"] = "require_matching"
    assert block.execute(couple) is None and "availability" in block.error
    too_small = MergeCloudScans(parameters={"max_input_points": 1})
    assert too_small.execute(couple) is None and "point budget" in too_small.error
    clk = TicketClock()
    couple["cloud_1"] = ValuePacket(one, {"ticket": clk.issue("front")})
    couple["cloud_2"] = ValuePacket(two, {"ticket": clk.issue("side")})
    assert block.execute(couple) is None and "shared ticket" in block.error
