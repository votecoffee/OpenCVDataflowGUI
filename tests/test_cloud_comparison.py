"""Independent dimensional, overlap, and provenance checks for point clouds."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_comparison import CompareCloudDimensions
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.pointclouds import PointCloud
from opencv_dataflow.tickets import TicketClock

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples/custom tools/cloud_dimensional_comparison/cloud_dimensional_comparison.json"
)


def _codes(result):
    return {entry["code"] for entry in result.fail_reasons}


def _pairs(a, b, reverse):
    delta = a[:, None, :] - b[None, :, :]
    if reverse:
        delta = -delta
    distances = np.linalg.norm(delta, axis=2)
    closest = distances.argmin(axis=1)
    count = np.arange(len(a))
    z = delta[count, closest, 2]
    distance = distances[count, closest]
    matched = distance <= 0.004
    return matched, closest, distance, z


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_inspection_matches_an_independent_pairwise_oracle(profile):
    graph = Graph.load(EXAMPLE, builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        measured = graph.blocks["measured"].last_result.outputs["cloud"]
        reference = graph.blocks["reference"].last_result.outputs["cloud"]
        assert measured.units == "m" and reference.units == "mm"
        xyz, nominal = measured.points.astype(float), reference.points.astype(float) / 1000
        assert graph.blocks["pass"].last_result.passed
        failed_size = graph.blocks["size"].last_result
        failed_overlap = graph.blocks["overlap"].last_result
        assert _codes(failed_size) == {
            "measured_tolerance_below_minimum",
            "reference_tolerance_below_minimum",
        }
        assert _codes(failed_overlap) == {
            "measured_overlap_below_minimum",
            "reference_overlap_below_minimum",
        }
        out = failed_size.outputs
        for name, a, b, reverse, count in (
            ("measured", xyz, nominal, False, 31),
            ("reference", nominal, xyz, True, 32),
        ):
            matched, closest, distances, signed = _pairs(a, b, reverse)
            good = matched & (signed >= -0.0008) & (signed <= 0.0009)
            target = np.where(matched, closest, -1)
            expected_state = np.where(matched, np.where(good, 1, 2), 0)
            pairs = out[f"{name}_pairs"]
            assert (
                pairs["nearest_index"] == target.tolist()
                and pairs["state"] == expected_state.tolist()
            )
            assert all(value is None for value in np.array(pairs["distance_m"], object)[~matched])
            np.testing.assert_allclose(
                np.array(pairs["distance_m"], float)[matched], distances[matched]
            )
            np.testing.assert_allclose(
                np.array(pairs["signed_distance_m"], float)[matched], signed[matched]
            )
            direction = out["report"][
                "reference_to_measured" if reverse else "measured_to_reference"
            ]
            assert direction["matched_points"] == direction["signed_points"] == 30
            assert direction["overlap_fraction"] == pytest.approx(30 / count)
            assert direction["in_tolerance_points"] == 27
            assert direction["signed_m"]["minimum"] == pytest.approx(-0.002, abs=5e-8)
            assert direction["signed_m"]["maximum"] == pytest.approx(0.0012, abs=5e-8)
            assert direction["euclidean_m"]["rms"] == pytest.approx(
                np.sqrt(np.mean(distances[matched] ** 2))
            )
        assert len(out["measured_defects"].points) == 3
        assert len(out["unmatched_measured"].points) == 1
        np.testing.assert_array_equal(out["missing_reference"].points, reference.points[[3, 26]])
        assert ("Measured deviations" in failed_size.graphics) == (profile == "debug")
    finally:
        graph.close()


def test_duplicate_and_equidistant_targets_withhold_sign_and_normal_conflicts_are_separate():
    reference = PointCloud(
        [[0, 0, 0], [0, 0, 0], [1, 0, 0], [3, 0, 0], [5, 0, 0], [7, 0, 0]],
        normals=[[0, 0, 1]] * 6,
    )
    measured = PointCloud(
        [[0, 0, 0.125], [2, 0, 0.125], [5, 0, 0.125], [7, 0, 0.125]],
        normals=[[0, 0, 1], [0, 0, 1], [0, 0, -1], [0, 0, 1]],
    )
    settings = {"correspondence_radius_m": 1.125, "distance_tolerance_m": 1.125}
    block = CompareCloudDimensions(parameters=settings)
    result = block.execute({"measured": measured, "reference": reference})
    assert result and result.passed, block.error
    pairs = result.outputs["measured_pairs"]
    assert pairs["nearest_index"] == [-1, -1, 4, 5]
    assert pairs["signed_distance_m"] == [None, None, None, 0.125]
    assert pairs["distance_m"] == pytest.approx([0.125, np.hypot(1, 0.125), 0.125, 0.125])
    assert pairs["state"] == [1] * 4
    report = result.outputs["report"]["measured_to_reference"]
    assert report["ambiguous_matches"] == 2 and report["normal_agreement_failures"] == 1

    signed = CompareCloudDimensions(
        parameters={
            **settings,
            "comparison": "normal",
            "positive_tolerance_m": 0.125,
        }
    )
    result = signed.execute({"measured": measured, "reference": reference})
    assert result is not None and not result.passed
    assert "measured_signed_coverage_below_minimum" in _codes(result)
    assert result.outputs["measured_pairs"]["state"] == [3, 3, 3, 1]
    assert all(entry["message"] != entry["code"] for entry in result.fail_reasons)


def test_inclusive_radius_and_original_row_payloads_are_preserved_in_both_directions():
    ticket = TicketClock().issue("dimensional")
    measured = PointCloud(
        [[0, 0, 0.5], [10, 0, 0]],
        frame="part",
        pixels=[[0, 0], [1, 0]],
        source_size=(2, 1),
    )
    reference = PointCloud(
        [[0, 0, 0], [20, 0, 0]],
        frame="part",
        normals=[[0, 0, 1]] * 2,
        pixels=[[1, 0], [2, 0]],
        source_size=(3, 1),
    )
    a = {"ticket": ticket, "spatial": frame([2, 1], "measured")}
    b = {"ticket": ticket, "spatial": frame([3, 1], "reference")}
    block = CompareCloudDimensions(
        parameters={
            "correspondence_radius_m": 0.5,
            "distance_tolerance_m": 0.25,
            "tie_tolerance_m": 0,
            "minimum_measured_overlap": 0.5,
            "minimum_reference_overlap": 0.5,
        }
    )
    result = block.execute(
        {"measured": ValuePacket(measured, a), "reference": ValuePacket(reference, b)}
    )
    assert result is not None and not result.passed
    assert _codes(result) == {
        "measured_tolerance_below_minimum",
        "reference_tolerance_below_minimum",
    }
    assert result.outputs["measured_pairs"]["distance_m"] == [0.5, None]
    assert result.outputs["reference_pairs"]["signed_distance_m"] == [0.5, None]
    for port, pixels, expected in (
        ("measured_defects", [[0, 0]], a),
        ("unmatched_measured", [[1, 0]], a),
        ("missing_reference", [[2, 0]], b),
    ):
        np.testing.assert_array_equal(result.outputs[port].pixels, pixels)
        for key in ("ticket", "spatial"):
            assert result.payloads[port][key] == expected[key]


def test_no_overlap_and_no_reference_normals_have_distinct_failures():
    comparison = CompareCloudDimensions(
        parameters={
            "comparison": "normal",
            "minimum_measured_overlap": 0.0,
            "minimum_reference_overlap": 0.0,
            "minimum_signed_fraction": 0.0,
        }
    )
    measured = PointCloud([[0, 0, 0]])
    result = comparison.execute({"measured": measured, "reference": PointCloud([[20, 0, 0]])})
    assert result is not None and _codes(result) == {"no_overlapping_pairs"}
    report = result.outputs["report"]["measured_to_reference"]
    assert report["euclidean_m"] is None and report["in_tolerance_fraction"] is None
    result = comparison.execute({"measured": measured, "reference": PointCloud([[0, 0, 0]])})
    assert result is not None and _codes(result) == {
        "measured_signed_coverage_below_minimum",
        "reference_signed_coverage_below_minimum",
    }


@pytest.mark.parametrize(
    ("settings", "cloud", "message"),
    [
        ({}, PointCloud([[0, 0, 0]], frame="other"), "same coordinate frame"),
        ({}, PointCloud([[0, 0, 0]], "unitless"), "m or mm"),
        ({"max_points_per_cloud": 1}, PointCloud([[0, 0, 0], [1, 0, 0]]), "point budget"),
    ],
)
def test_rejects_incompatible_frame_units_and_budget(settings, cloud, message):
    tool = CompareCloudDimensions(parameters=settings)
    assert tool.execute({"measured": PointCloud([[0, 0, 0]]), "reference": cloud}) is None
    assert message in tool.error


def test_cancellation_withholds_partial_correspondences():
    tool = CompareCloudDimensions()
    cloud = PointCloud([[0, 0, 0], [1, 0, 0]])
    checks = iter([False] * 3)
    result = tool.execute(
        {"measured": cloud, "reference": cloud}, cancelled=lambda: next(checks, True)
    )
    assert result is not None and _codes(result) == {"cloud_comparison_cancelled"}
    for name in ("measured_pairs", "reference_pairs", "measured_defects"):
        assert result.outputs[name] is NO_VALUE
