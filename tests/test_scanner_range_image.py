"""Verify metric angular projection, return ownership and dataflow semantics."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph, Port, validate_value
from opencv_dataflow.geometry3d_common import pose_record
from opencv_dataflow.payloads import NO_VALUE, SPHERICAL_RANGE_GRID, ValuePacket
from opencv_dataflow.pointclouds import PointCloud

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/custom tools/scanner_range_image"


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_project_organizes_known_returns_across_seam(profile):
    graph = Graph.load(EXAMPLE / "scanner_range_image.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        chosen = {
            "nearest": (2.0, 10, 0, 101),
            "farthest": (4.0, 90, 1, 201),
            "strongest": (3.0, 90, 2, 20_000_003),
        }
        for mode, expected in chosen.items():
            block = graph.blocks[mode]
            result = block.last_result
            assert result is not None and result.passed, block.error
            outputs = result.outputs
            arrays = ("range", "intensity", "source_rows", "source_indices")
            assert tuple(outputs[key][1, 0] for key in arrays) == pytest.approx(expected)
            assert outputs["intensity"].dtype == np.float64
            assert outputs["source_indices"].dtype == outputs["source_rows"].dtype == np.int32
            assert outputs["source_indices"][1, 0] == expected[-1]
            assert outputs["range"].shape == (3, 6)
            assert outputs["range"][1, 1] == pytest.approx(1.5)
            assert outputs["range"][0, 2] == pytest.approx(6.5)
            assert outputs["range"][2, 3] == pytest.approx(8.0)
            assert outputs["range"][0, 5] == pytest.approx(12.0)
            assert outputs["source_rows"][2, 3] == 5 and outputs["source_indices"][2, 3] == -1
            assert np.isnan(outputs["intensity"][2, 3])
            assert outputs["valid_mask"][2, 3] == 255
            for key in arrays:
                assert result.payloads[key]["spatial"] == result.payloads["range"]["spatial"]
                assert result.payloads[key][SPHERICAL_RANGE_GRID]["axis_convention"].endswith(
                    "Z up"
                )
            assert np.isnan(outputs["range"][0, 0]) and outputs["source_rows"][0, 0] == -1
            assert outputs["return_count"][1, 0] == 4
            assert outputs["return_count"].sum() == 8
            report = outputs["report"]
            assert report["grid"]["elevation_edges_deg"] == [-15.0, -5.0, 4.0, 25.0]
            assert report["occupied_bins"] == 5 and report["empty_bins"] == 13
            assert report["range_clipped_points"] == report["outside_grid_points"] == 2
            assert report["undefined_azimuth_points"] == report["multiple_return_bins"] == 1
            assert report["strongest_fallback_bins"] == int(mode == "strongest")
            assert ("Radial range" in result.graphics) == (profile == "debug")
        empty = graph.blocks["empty"].last_result
        assert empty is not None and not empty.passed
        assert empty.outputs["report"]["status"] == "no_returns"
        assert np.isnan(empty.outputs["range"]).all() and not empty.outputs["valid_mask"].any()
    finally:
        graph.close()


def block(**params):
    return builtin_registry().create("3d.cloud_range_image", parameters=params)


def test_exact_angular_boundaries_remain_half_open_and_use_new_root_for_new_grid():
    cloud = PointCloud([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [1, 0, 1], [1, 0, -1]], "m", "scanner")
    detector = block(
        columns=1,
        rows=2,
        azimuth_start_deg=0,
        azimuth_span_deg=90,
        elevation_min_deg=-45,
        elevation_max_deg=45,
    )
    result = detector.execute({"cloud": cloud})
    assert result is not None and result.passed, detector.error
    np.testing.assert_array_equal(result.outputs["source_rows"], [[0], [4]])
    assert result.outputs["report"]["outside_grid_points"] == 3
    root = result.payloads["range"]["spatial"]["root_id"]
    detector.parameters["azimuth_start_deg"] = 90
    changed = detector.execute({"cloud": cloud})
    assert changed is not None and changed.passed
    assert changed.outputs["source_rows"][0, 0] == 1
    assert changed.payloads["range"]["spatial"]["root_id"] != root
    full = block(columns=4, rows=1, elevation_min_deg=-1, elevation_max_deg=1)
    round_trip = full.execute({"cloud": cloud})
    assert round_trip is not None and round_trip.passed
    np.testing.assert_array_equal(round_trip.outputs["source_rows"], [[2, -1, 0, 1]])


def test_missing_optional_columns_and_depth_guard_are_explicit():
    source = PointCloud([[2, 0, 0]], "m", "scanner")
    result = block(columns=1, rows=1).execute({"cloud": source})
    assert result is not None and result.passed
    assert result.outputs["intensity"] is result.outputs["source_indices"] is NO_VALUE
    range_raster = result.outputs["range"]
    validate_value(Port("image"), range_raster)
    with pytest.raises(TypeError, match="spherical scanner"):
        validate_value(Port("depth", "depth"), range_raster)
    wrapped = ValuePacket(np.asarray(range_raster), result.payloads["range"])
    with pytest.raises(TypeError, match="spherical scanner"):
        validate_value(Port("depth", "depth"), wrapped)
    validate_value(Port("depth", "depth"), np.full((1, 1), 2.0, np.float32))
    strongest = block(return_policy="strongest")
    assert strongest.execute({"cloud": source}) is None
    assert "intensity attribute" in strongest.error


def test_pose_is_required_for_mismatched_frames_and_converted_from_its_own_units():
    source = PointCloud([[1000, 0, 0]], "mm", "stand")
    detector = block(columns=1, rows=1)
    assert detector.execute({"cloud": source}) is None
    assert "cloud frame to the scanner frame" in detector.error
    transform = np.eye(4)
    transform[0, 3] = 0.25
    wrong = pose_record(transform, "stand", "camera", "m")
    assert detector.execute({"cloud": source, "transform": wrong}) is None
    correct = pose_record(transform, "stand", "scanner", "m")
    result = detector.execute({"cloud": source, "transform": correct})
    assert result is not None and result.passed
    assert result.outputs["range"][0, 0] == pytest.approx(1.25)


@pytest.mark.parametrize(
    ("params", "error"),
    [
        ({"columns": 8192, "rows": 1024}, "bins"),
        ({"elevation_edges_deg": [-10, 10, 10]}, "strictly increasing"),
    ],
)
def test_invalid_grids_fail_before_rasterization(params, error):
    detector = block(**params)
    result = detector.execute({"cloud": PointCloud([[2, 0, 0]], "m", "scanner")})
    assert result is None and error in detector.error
