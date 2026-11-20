"""Check metric clustering against known geometry and budget exits."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_clustering import ClusterPointCloud
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.pointclouds import PointCloud
from opencv_dataflow.tickets import TicketClock

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples/custom tools/cloud_clustering/cloud_clustering.json"
)


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_tray_separates_three_parts_and_rejects_small_and_large_groups(profile):
    graph = Graph.load(EXAMPLE, builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        cloud = graph.blocks["source"].last_result.outputs["cloud"]
        block = graph.blocks["cluster"]
        result = block.last_result
        assert result and result.passed, block.error
        outputs = result.outputs
        assert outputs["count"] == 3
        assert outputs["report"]["candidate_clusters"] == 5
        assert outputs["report"]["count_rejected_clusters"] == 1
        assert outputs["report"]["extent_rejected_clusters"] == 1
        assert outputs["report"]["classified_points"] == 37
        # Fixture colors identify ground-truth membership independently of the geometry code.
        expected_labels = np.zeros(len(cloud.points), int)
        for label, (record, color) in enumerate(
            zip(outputs["clusters"], ([45, 145, 245], [115, 230, 80], [205, 90, 210]), strict=True),
            start=1,
        ):
            indices = np.flatnonzero(np.all(cloud.colors == color, axis=1))
            expected_labels[indices] = label
            assert record["label"] == label and record["point_count"] == 9
            assert record["source_indices"] == indices.tolist()
            assert record["extent_xyz_m"] == pytest.approx([0.01, 0.01, 0.0])
        assert outputs["labels"] == expected_labels.tolist()
        assert outputs["report"]["accepted_points"] == 27
        assert outputs["report"]["rejected_points"] == 10
        for name, mask in (
            ("cloud", expected_labels > 0),
            ("rejected_cloud", expected_labels == 0),
        ):
            selected = outputs[name]
            assert (selected.units, selected.frame) == ("mm", "parts_tray")
            for attribute in ("points", "colors", "normals"):
                np.testing.assert_array_equal(
                    getattr(selected, attribute), getattr(cloud, attribute)[mask]
                )
        assert ("clusters" in result.graphics) == (profile == "debug")
    finally:
        graph.close()


def test_inclusive_links_keep_transitive_chain_and_preserve_ticket_and_pixels():
    pixels = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]
    cloud = PointCloud(
        [[2000, 0, 0], [5000, 0, 0], [1000, 0, 0], [0, 0, 0], [1000, 0, 0]],
        "mm",
        "scan",
        pixels=pixels,
        source_size=(5, 1),
    )
    coords = frame([5, 1], "camera")
    ticket = TicketClock().issue("scan")
    block = ClusterPointCloud(
        parameters={"radius_m": 1, "minimum_points": 4, "maximum_extent_m": "2.0"},
    )
    result = block.execute({"cloud": ValuePacket(cloud, {"ticket": ticket, "spatial": coords})})
    assert result and result.passed, block.error
    assert result.outputs["labels"] == [1, 0, 1, 1, 1]
    assert result.outputs["report"]["neighbor_pairs"] == 5
    assert result.outputs["clusters"][0]["source_indices"] == [0, 2, 3, 4]
    assert result.outputs["clusters"][0]["longest_extent_m"] == 2
    for port, rows in (("cloud", [0, 2, 3, 4]), ("rejected_cloud", [1])):
        assert result.payloads[port]["ticket"] == ticket
        assert result.payloads[port]["spatial"] == coords
        np.testing.assert_array_equal(result.outputs[port].pixels, np.array(pixels)[rows])


@pytest.mark.parametrize(
    ("cloud", "settings", "reason", "details"),
    [
        (
            PointCloud(np.zeros((16, 3))),
            {"max_neighbor_pairs": 100},
            "cloud_cluster_pair_budget",
            {"neighbor_pairs": 120},
        ),
        (
            PointCloud([[0, 0, 0], [1, 0, 0]]),
            {"minimum_points": 1, "max_clusters": 1},
            "cloud_cluster_result_budget",
            {"candidate_clusters": 2, "matching_clusters": 2},
        ),
    ],
)
def test_work_budgets_do_not_emit_partial_classifications(cloud, settings, reason, details):
    result = ClusterPointCloud(parameters=settings).execute({"cloud": cloud})
    assert result is not None and not result.passed
    assert result.fail_reasons[0]["code"] == reason
    assert result.outputs["labels"] is NO_VALUE
    assert result.outputs["count"] == 0
    assert not len(result.outputs["rejected_cloud"].points)
    assert result.outputs["report"]["classified_points"] == 0
    assert details.items() <= result.outputs["report"].items()


def test_empty_all_rejected_and_cancelled_have_distinct_reasons():
    block = ClusterPointCloud()
    empty = block.execute({"cloud": PointCloud(np.empty((0, 3)))})
    assert empty.fail_reasons[0]["code"] == "cloud_cluster_empty"
    assert empty.outputs["labels"] == []

    cloud = PointCloud([[0, 0, 0], [1, 0, 0]])
    rejected = block.execute({"cloud": cloud})
    assert rejected.fail_reasons[0]["code"] == "cloud_cluster_no_matches"
    assert rejected.outputs["labels"] == [0, 0]
    assert rejected.outputs["report"]["classified_points"] == 2

    flags = iter([False, False, True])
    cancelled = block.execute({"cloud": cloud}, cancelled=lambda: next(flags, True))
    assert cancelled.fail_reasons[0]["code"] == "cloud_cluster_cancelled"
    assert cancelled.outputs["labels"] is NO_VALUE
    assert cancelled.outputs["report"]["classified_points"] == 0


@pytest.mark.parametrize(
    ("cloud", "settings", "message"),
    [
        (PointCloud([[0, 0, 0]], "unitless"), {}, "cloud in m or mm"),
        (PointCloud([[0, 0, 0], [1, 0, 0]]), {"max_input_points": 1}, "point budget"),
        (PointCloud([[0, 0, 0]]), {"maximum_points": 2}, "minimum must not exceed"),
        (
            PointCloud([[0, 0, 0]]),
            {"minimum_extent_m": 0.03, "maximum_extent_m": 0.02},
            "minimum must not exceed",
        ),
    ],
)
def test_metric_count_and_extent_validation(cloud, settings, message):
    block = ClusterPointCloud(parameters=settings)
    assert block.execute({"cloud": cloud}) is None and message in block.error
