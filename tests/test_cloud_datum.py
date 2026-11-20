"""Qualify signed datum statistics, fixed footprint and sparse defects."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_datum import MeasureCloudDatum
from opencv_dataflow.cloud_model_fitting import FitCloudModel
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import pose_record
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.pointclouds import PointCloud
from opencv_dataflow.tickets import TicketClock

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/custom tools/cloud_datum/cloud_datum.json"


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_three_scans_distinguish_height_from_spread_and_coverage(profile):
    graph = Graph.load(EXAMPLE, builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.set_branch_workers(1 if profile == "debug" else 2)
        graph.run()
        height_cases = {
            "nominal": (True, 2, 0, 1, [], []),
            "raised": (False, 6, 0, 1, list(range(100, 116)), ["height_tolerance"]),
            "warped": (
                False,
                2,
                6,
                15 / 16,
                [100, 115],
                ["insufficient_coverage", "height_tolerance", "peak_to_valley"],
            ),
        }
        for name, (passed, mean, flatness, coverage, ids, reasons) in height_cases.items():
            result = graph.blocks[name].last_result
            source = graph.blocks[name + "_scan"].last_result
            assert result is not None and result.passed is passed
            assert [r["code"] for r in result.fail_reasons] == [f"cloud_datum_{r}" for r in reasons]
            out = result.outputs
            report = out["report"]
            assert report["input_points"] == report["selected_points"] + 1
            assert report["coverage"] == coverage
            assert report["signed_height_m"]["mean"] == pytest.approx(mean / 1000, abs=4e-8)
            assert report["datum_parallel_peak_to_valley_m"] == pytest.approx(
                flatness / 1000, abs=6e-8
            )
            np.testing.assert_array_equal(out["defects"].attributes["source_index"], ids)
            assert out["defects"].frame == report["cloud_frame"] == "station"
            assert out["defects"].units == "mm"
            assert report["reference_frame"] == "fixture"
            assert report["datum_kind"] == "fixture_pose"
            valid = out["coverage_mask"] == 255
            assert np.count_nonzero(valid) == round(16 * coverage)
            assert np.array_equal(valid, np.isfinite(out["deviation"]))
            assert np.count_nonzero(out["defect_mask"]) == len(ids)
            for port in ("coverage_mask", "deviation", "defect_mask"):
                assert result.payloads[port]["spatial"]["root_id"] == f"{name}:datum"
                assert result.payloads[port]["ticket"] == source.payloads["cloud"]["ticket"]
            if name == "warped":
                assert np.isnan(out["deviation"][2, 2])
                assert out["deviation"][0, 3] == pytest.approx(0.003, abs=4e-8)
                assert out["deviation"][3, 0] == pytest.approx(-0.003, abs=4e-8)
                assert report["nominal_deviation_m"]["rms"] == pytest.approx(
                    np.sqrt(18 / 15) / 1000,
                    abs=4e-8,
                )
            else:
                np.testing.assert_allclose(out["deviation"], (mean - 2) / 1000, atol=4e-8, rtol=0)
            assert ("Datum deviation" in result.graphics) == (profile == "debug")
    finally:
        graph.close()


def _control(**options):
    return {
        "u_min_m": 0.0,
        "u_max_m": 0.25,
        "v_min_m": 0.0,
        "v_max_m": 0.25,
        "grid_width": 2,
        "grid_height": 2,
        **options,
    }


@pytest.mark.parametrize(
    "datum",
    [
        {
            "model": "camera_depth_plane_v1",
            "frame": "scan",
            "units": "m",
            "normal": [0, 0, 1],
            "offset": -0.125,
        },
        {
            "schema": "point_cloud_model_v1",
            "kind": "plane",
            "frame": "scan",
            "units": "mm",
            "coefficients": [0, 0, 2, -250],
        },
    ],
)
def test_signed_boundaries_and_sparse_defects_use_original_points(datum):
    cloud = PointCloud(
        [
            [0, 0, 0.125 + 0.03125 + 0.0625],
            [0, 0, 0.125],
            [0.25, 0.25, 0.125 + 0.125],
            [0.125, 0.0, 0.125 + 0.015625],
            [0.25, 0, 0.125 + 0.09375],
        ],
        "m",
        "scan",
        attributes={"source_index": [18, 11, 6, 26, 21]},
        metadata={"recorded_cloud": {"timestamp_ns": 1_900_000_000_000_000_003}},
    )
    block = MeasureCloudDatum(
        parameters=_control(
            nominal_height_m=0.03125,
            minimum_points_per_cell=2,
            negative_tolerance_m=0.015625,
            positive_tolerance_m=0.0625,
        )
    )
    ticket = TicketClock().issue("scan")
    packet = ValuePacket(cloud, {"ticket": ticket, "spatial": frame([1, 5], "source_grid")})
    result = block.execute({"cloud": packet, "datum": datum})
    assert result is not None and not result.passed, block.error
    out = result.outputs
    assert np.isnan(out["deviation"][0, 1])  # A lone out-of-tolerance point.
    assert out["deviation"][1, 0] == 0.0625  # Strongest residual is itself within tolerance.
    assert out["deviation"][1, 1] == 0.0625  # Upper and lower inclusive limits.
    np.testing.assert_array_equal(out["coverage_mask"], [[0, 0], [255, 255]])
    np.testing.assert_array_equal(out["defect_mask"], [[0, 255], [255, 0]])
    np.testing.assert_array_equal(out["defects"].attributes["source_index"], [11, 6])
    np.testing.assert_array_equal(out["defects"].points, cloud.points[[1, 2]])
    assert out["report"]["coverage"] == 0.5 and out["report"]["in_tolerance_fraction"] == 0.6
    assert out["report"]["origin_cloud_m"] == [0, 0, 0.125]
    assert result.payloads["deviation"]["spatial"]["root_id"] == f"{block.id}:datum"
    assert result.payloads["defects"]["spatial"]["root_id"] == "source_grid"
    assert result.payloads["report"]["recorded_cloud"] == cloud.metadata["recorded_cloud"]
    assert result.payloads["defect_mask"]["ticket"] == ticket


def test_plane_fitted_by_builtin_block_connects_to_metric_measurement():
    cloud = PointCloud(
        [[u, v, 0.125] for u in (0.0625, 0.1875) for v in (0.0625, 0.1875)], "m", "scan"
    )
    fit = FitCloudModel(parameters={"minimum_inliers": 3})
    known = fit.execute({"cloud": cloud})
    assert known is not None and known.passed, fit.error
    result = MeasureCloudDatum(parameters=_control(minimum_coverage=1)).execute(
        {"cloud": cloud, "datum": known.outputs["model"]},
    )
    assert result is not None and result.passed
    assert result.outputs["report"]["datum_parallel_peak_to_valley_m"] == 0
    np.testing.assert_array_equal(result.outputs["deviation"], np.zeros((2, 2)))


def test_empty_grid_and_sparse_grid_fail_and_do_not_invent_deviation():
    datum = pose_record(np.eye(4), "fixture", "scan", "m")
    block = MeasureCloudDatum(parameters=_control(minimum_points_per_cell=2))
    for points, codes in (
        ([[1, 0, 0]], ["cloud_datum_empty", "cloud_datum_insufficient_coverage"]),
        ([[0, 0, 0]], ["cloud_datum_insufficient_coverage"]),
    ):
        result = block.execute({"cloud": PointCloud(points, "m", "scan"), "datum": datum})
        assert result is not None and not result.passed
        assert [reason["code"] for reason in result.fail_reasons] == codes
        assert np.isnan(result.outputs["deviation"]).all()
        assert not result.outputs["coverage_mask"].any()


def test_invalid_pose_or_plane_and_cancellation_are_reported_separately():
    cloud = PointCloud([[0.05, 0.05, 0.125], [0.15, 0.05, 0.125]], "m", "scan")
    datum = pose_record(np.eye(4), "fixture", "scan", "m")
    for params, error in (
        (_control(max_input_points=1), "point budget"),
        (_control(u_max_m=-0.1), "distinct grid edges"),
    ):
        block = MeasureCloudDatum(parameters=params)
        assert block.execute({"cloud": cloud, "datum": datum}) is None
        assert error in block.error
    block = MeasureCloudDatum(parameters=_control())
    for record, error in (
        ({**datum, "target_frame": "wrong"}, "into the cloud frame"),
        (
            {
                "model": "camera_depth_plane_v1",
                "frame": "scan",
                "units": "m",
                "normal": [0, 0, 0],
                "offset": 0,
            },
            "nonzero normal",
        ),
        (
            {"schema": "point_cloud_model_v1", "kind": "sphere", "units": "m", "frame": "scan"},
            "fitted cloud plane",
        ),
    ):
        assert block.execute({"cloud": cloud, "datum": record}) is None
        assert error in block.error
    flags = iter((False, True))
    stopped = block.execute({"cloud": cloud, "datum": datum}, cancelled=lambda: next(flags, True))
    assert not stopped.passed and stopped.outputs["report"]["status"] == "cancelled"
    assert stopped.outputs["deviation"] is NO_VALUE and stopped.outputs["defects"] is NO_VALUE
