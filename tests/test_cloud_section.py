"""Verify metric section dimensions, coverage and original-row references."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_section import MeasureCloudSection
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import pose_record
from opencv_dataflow.payloads import NO_VALUE, ValuePacket
from opencv_dataflow.pointclouds import PointCloud
from opencv_dataflow.tickets import TicketClock

EXAMPLE = (
    Path(__file__).resolve().parents[1] / "examples/custom tools/cloud_sections/cloud_sections.json"
)


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_stepped_channel_accounts_for_plane_offset_width_and_gap(profile):
    graph = Graph.load(EXAMPLE, builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.set_branch_workers(1 if profile == "debug" else 2)
        graph.run()
        source = graph.blocks["scan"].last_result
        original = source.outputs["cloud"]
        normal = Rotation.from_rotvec([0.32, -0.27, 0.44]).as_matrix()[:, 2]
        for key, height, row_ids, widths, coverage in (
            ("lower", 0.025, np.arange(13, 21), [16, 18, None, 20, 24], 0.8),
            ("upper", 0.055, np.arange(3, 13), [28, 28, 32, 32, 36], 1.0),
        ):
            result = graph.blocks[key].last_result
            assert result is not None and result.passed
            data = result.outputs
            section, report = data["section"], data["report"]
            assert section.frame == report["cloud_frame"] == "bench_scan"
            assert section.units == report["cloud_units"] == "mm"
            assert report["reference_frame"] == "fixture"
            np.testing.assert_array_equal(section.points, original.points[row_ids])
            np.testing.assert_array_equal(section.colors, original.colors[row_ids])
            np.testing.assert_array_equal(section.attributes["source_index"], row_ids + 1000)
            np.testing.assert_array_equal(
                section.attributes["intensity"], original.attributes["intensity"][row_ids]
            )
            np.testing.assert_allclose(report["normal_cloud"], normal, atol=1e-12)
            np.testing.assert_allclose(
                report["origin_cloud_m"],
                [0.260, -0.09, 0.35] + height * normal,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                report["signed_distance_range_m"],
                [-0.0005, 0.0005],
                atol=4e-8,
            )
            assert report["rms_distance_m"] == pytest.approx(0.0005, abs=3e-8)
            assert report["coverage"] == coverage
            assert report["covered_length_m"] == pytest.approx(0.125 * coverage)
            assert report["observed_v_width_m"] == pytest.approx(
                max(w for w in widths if w) / 1000, abs=4e-8
            )
            assert report["observed_u_extent_m"] == pytest.approx(0.1, abs=4e-8)
            bins = data["profile"]["bins"]
            for cell, width in zip(bins, widths, strict=True):
                if width is None:
                    assert not cell["supported"] and cell["point_count"] == 0
                    assert cell["width_m"] is None and cell["minimum_point"] is None
                    continue
                assert cell["supported"] and cell["point_count"] == 2
                assert cell["width_m"] == pytest.approx(width / 1000, abs=4e-8)
                assert cell["v_mean_m"] == pytest.approx(0, abs=4e-8)
                for field in ("minimum_point", "maximum_point"):
                    point = cell[field]
                    assert point["input_row"] in row_ids
                    assert point["source_index"] == 1000 + point["input_row"]
            for name in ("profile", "report", "section"):
                assert result.payloads[name]["ticket"] == source.payloads["cloud"]["ticket"]
                assert result.payloads[name]["cloud_section"] == section.metadata["cloud_section"]
            assert ("Section profile" in result.graphics) == (profile == "debug")
    finally:
        graph.close()


def _controls(**overrides):
    return {
        "u_min_m": -0.125,
        "u_max_m": 0.125,
        "bins": 2,
        "thickness_m": 0.03125,
        "offset_m": 0.25,
        **overrides,
    }


@pytest.mark.parametrize(
    ("plane", "axes"), [("xy", (0, 1, 2)), ("xz", (0, 2, 1)), ("yz", (1, 2, 0))]
)
def test_cardinal_sections_include_slab_edges_and_assign_bin_edges(plane, axes):
    xyz = np.zeros((7, 3))
    xyz[:, axes] = [
        [-0.125, -0.0625, 0.234375],
        [-0.125, 0.0625, 0.265625],
        [0.0, -0.125, 0.25],
        [0.125, 0.125, 0.25],
        [0.0, 100.0, 0.26565625],
        [0.140625, 0.0, 0.25],
        [-0.125, 0.5, 0.25],
    ]
    cloud = PointCloud(xyz, "m", "scan")
    result = MeasureCloudSection(
        parameters=_controls(plane=plane, restrict_v=True, v_min_m=-0.125, v_max_m=0.125),
    ).execute({"cloud": cloud})
    assert result.passed
    np.testing.assert_array_equal(result.outputs["section"].points, cloud.points[:4])
    np.testing.assert_array_equal(
        result.outputs["section"].attributes["source_index"], [0, 1, 2, 3]
    )
    bins = result.outputs["profile"]["bins"]
    assert [b["point_count"] for b in bins] == [2, 2]
    assert [b["width_m"] for b in bins] == [0.125, 0.25]
    assert bins[1]["u_start_m"] == 0 and bins[1]["u_end_m"] == 0.125
    report = result.outputs["report"]
    assert report["normal_cloud"] == np.eye(3)[axes[-1]].tolist()
    assert report["signed_distance_range_m"] == [-0.015625, 0.015625]


def test_ties_use_earliest_input_row_and_preserve_replayed_scan_provenance():
    cloud = PointCloud(
        [[-0.06, -0.01, 0.25], [-0.04, 0.025, 0.25], [-0.03, -0.01, 0.25]],
        "m",
        "scan",
        normals=np.tile([0, 0, 1], (3, 1)),
        pixels=np.column_stack((np.arange(3), np.zeros(3))),
        source_size=(3, 1),
        attributes={"source_index": [13, 5, 9], "source_scan": [4, 4, 8]},
        metadata={"recorded_cloud": {"timestamp_ns": 1_900_000_000_000_000_003}},
    )
    ticket = TicketClock().issue("scanner")
    block = MeasureCloudSection(parameters=_controls())
    result = block.execute({"cloud": ValuePacket(cloud, {"ticket": ticket})})
    assert result.passed
    first, second = result.outputs["profile"]["bins"]
    assert first["minimum_point"] == {"input_row": 0, "source_index": 13, "source_scan": 4}
    assert first["maximum_point"] == {"input_row": 1, "source_index": 5, "source_scan": 4}
    assert not second["supported"] and second["width_m"] is None
    section = result.outputs["section"]
    np.testing.assert_array_equal(section.pixels, cloud.pixels)
    np.testing.assert_array_equal(section.normals, cloud.normals)
    assert section.source_size == cloud.source_size
    assert result.payloads["profile"]["recorded_cloud"] == cloud.metadata["recorded_cloud"]
    assert result.payloads["profile"]["ticket"] == ticket


@pytest.mark.parametrize(
    ("parameters", "status", "selected"),
    [
        ({"offset_m": 1.0}, "no_section_points", 0),
        ({"minimum_points_per_bin": 3}, "no_supported_bins", 2),
        ({"minimum_coverage": 0.75}, "insufficient_coverage", 2),
    ],
)
def test_missing_support_fails_with_explicit_bin_quality(parameters, status, selected):
    block = MeasureCloudSection(parameters=_controls(**parameters))
    result = block.execute({"cloud": PointCloud([[-0.05, 0, 0.25], [-0.05, 0.02, 0.25]])})
    assert not result.passed
    assert result.fail_reasons[0]["code"] == f"cloud_section_{status}"
    assert result.outputs["report"]["section_points"] == selected
    assert result.outputs["profile"]["bins"][1]["v_mean_m"] is None
    assert len(result.outputs["section"].points) == selected


def test_invalid_ranges_units_frame_and_cancelled_work_are_distinct():
    cloud = PointCloud([[0, 0, 0.25], [0, 0.02, 0.25]], "m", "scan")
    for config, diagnostic in (
        (_controls(u_max_m=-0.2), "distinct profile bin edges"),
        (_controls(restrict_v=True, v_min_m=2, v_max_m=1), "Maximum V"),
        (_controls(max_input_points=1), "point budget"),
    ):
        block = MeasureCloudSection(parameters=config)
        assert block.execute({"cloud": cloud}) is None and diagnostic in block.error
    block = MeasureCloudSection(parameters=_controls())
    assert block.execute({"cloud": PointCloud(cloud.points, "unitless")}) is None
    assert "cloud in m or mm" in block.error
    incorrect = pose_record(np.eye(4), "fixture", "other", "mm")
    assert block.execute({"cloud": cloud, "pose": incorrect}) is None
    assert "into the cloud frame" in block.error
    cancelled = iter((False, True))
    result = block.execute({"cloud": cloud}, cancelled=lambda: next(cancelled, True))
    assert not result.passed and result.outputs["report"]["status"] == "cancelled"
    assert result.outputs["section"] is NO_VALUE and result.outputs["profile"] is NO_VALUE
