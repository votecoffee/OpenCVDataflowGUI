"""Qualify datum-relative dimensions and independent silhouette edge diagnostics."""

import json
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.depth_silhouette import MeasureDepthSilhouette
from opencv_dataflow.geometry3d_common import camera_record
from opencv_dataflow.images import DepthArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/depth_silhouette"


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_dataset_measures_two_tilted_parts_and_recovers_exact_missing_boundary(profile):
    truth = json.loads((EXAMPLE / "assets/ground_truth.json").read_bytes())
    graph = Graph.load(EXAMPLE / "depth_silhouette.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        datum = graph.blocks["datum"].last_result.outputs["plane"]
        source = graph.blocks["depth"].last_result
        for name, expected in truth.items():
            block = graph.blocks[name]
            result = block.last_result
            assert result and result.passed, block.error
            report = result.outputs["report"]
            for dim in ("length_m", "width_m"):
                assert result.outputs[dim] == pytest.approx(expected[dim], abs=0.001)
            assert result.outputs["thickness_m"] == pytest.approx(expected["thickness_m"], abs=3e-6)
            assert report["height_p05_m"] < report["thickness_m"] < report["height_p95_m"]
            assert report["boundary_coverage"] >= 0.95
            assert report["missing_depth_boundary_pixels"] == len(expected["lost_yx"])
            assert (
                report["rejected_height_boundary_pixels"]
                == report["unprojectable_boundary_pixels"]
                == 0
            )
            assert report["invalid_boundary_pixels"] == len(expected["lost_yx"])
            np.testing.assert_array_equal(
                np.argwhere(result.outputs["invalid_boundary_mask"]), sorted(expected["lost_yx"])
            )
            corners = np.asarray(report["box_on_datum_xyz_m"])
            np.testing.assert_allclose(corners @ datum["normal"] + datum["offset"], 0, atol=1e-12)
            assert (
                result.payloads["invalid_boundary_mask"]["spatial"]
                == source.payloads["depth"]["spatial"]
            )
            assert result.payloads["report"]["ticket"] == source.payloads["depth"]["ticket"]
            if profile == "debug":
                drawing = result.drawings["Silhouette measurements"]
                assert drawing.source == "depth"
                assert [item.kind for item in drawing.objects[:2]] == ["polyline", "label"]
                assert len(drawing.objects) == 2 + len(expected["lost_yx"])
            else:
                assert not result.drawings
    finally:
        graph.close()


def _inputs():
    h, w = 16, 20
    depth = np.full((h, w), 1000, np.uint16)
    selection = np.zeros(depth.shape, np.uint8)
    selection[5:11, 4:13] = 255
    depth[selection != 0] = 900
    camera = camera_record([[200, 0, 9.5], [0, 250, 7.5], [0, 0, 1]], [], (w, h), "mm", "cam")
    plane = {
        "model": "camera_depth_plane_v1",
        "frame": "cam",
        "units": "mm",
        "normal": [0, 0, -1],
        "offset": 1000,
    }
    return {"depth": depth, "camera": camera, "plane": plane, "mask": selection}


def test_planar_dimensions_include_exact_pixel_footprints_and_datum_thickness():
    inputs = _inputs()
    block = MeasureDepthSilhouette(parameters={"input_units": "mm", "units_per_sample": 1})
    result = block.execute(inputs)
    assert result and result.passed, block.error
    measured = result.outputs
    assert measured["length_m"] == pytest.approx(9 * 0.9 / 200, abs=1e-8)
    assert measured["width_m"] == pytest.approx(6 * 0.9 / 250, abs=1e-8)
    assert measured["thickness_m"] == pytest.approx(0.1, abs=1e-7)
    assert measured["report"]["selected_pixels"] == 54
    assert measured["report"]["boundary_pixels"] == 26
    assert not measured["invalid_boundary_mask"].any()


def test_depth_holes_and_below_datum_are_reported_and_block_outputs_gate():
    inputs = _inputs()
    inputs["depth"][5, 4:9] = 0
    inputs["depth"][6, 4] = 1000
    inputs["depth"][7, 7] = 0
    block = MeasureDepthSilhouette(parameters={"input_units": "mm", "units_per_sample": 1})
    result = block.execute(inputs)
    assert result and not result.passed and not block.error
    report = result.outputs["report"]
    assert report["invalid_boundary_pixels"] == 6
    assert report["missing_depth_boundary_pixels"] == 5
    assert report["rejected_height_boundary_pixels"] == 1
    assert report["unprojectable_boundary_pixels"] == 0
    assert report["face_pixels"] == 47
    assert [reason["code"] for reason in result.fail_reasons] == [
        "silhouette_boundary_support_insufficient"
    ]
    assert result.outputs["width_m"] is NO_VALUE
    assert not block.current_emissions.get("width_m")
    assert result.outputs["invalid_boundary_mask"][7, 7] == 0


def test_near_grazing_boundary_marks_corner_rays_that_cross_the_datum_horizon():
    w, h = 16, 14
    matrix = [[20, 0, 7], [0, 24, 6.5], [0, 0, 1]]
    normal = [np.sqrt(0.99), 0, -0.1]
    mask = np.zeros((h, w), np.uint8)
    mask[4:9, 4:10] = 255
    denominator = normal[0] * (np.arange(w) - 7) / 20 + normal[2]
    row = np.zeros(w, np.float32)
    row[4:10] = -0.98 / denominator[4:10]
    plane = {
        "model": "camera_depth_plane_v1",
        "frame": "cam",
        "units": "m",
        "normal": normal,
        "offset": 1,
    }
    inputs = {
        "depth": DepthArray(np.broadcast_to(row, (h, w)), "m"),
        "camera": camera_record(matrix, [], (w, h), frame="cam"),
        "plane": plane,
        "mask": mask,
    }
    block = MeasureDepthSilhouette()
    result = block.execute(inputs)
    assert result and not result.passed and not block.error
    report = result.outputs["report"]
    assert report["unprojectable_boundary_pixels"] == 5
    assert report["missing_depth_boundary_pixels"] == report["rejected_height_boundary_pixels"] == 0
    np.testing.assert_array_equal(
        np.argwhere(result.outputs["invalid_boundary_mask"]), [[y, 9] for y in range(4, 9)]
    )


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        ("clipped", "silhouette_touches_image_border"),
        ("two", "silhouette_requires_one_component"),
        ("empty", "silhouette_requires_one_component"),
    ],
)
def test_clipped_and_multiple_silhouettes_withhold_measurements(scenario, expected):
    inputs = _inputs()
    if scenario == "clipped":
        inputs["mask"] = np.roll(inputs["mask"], -4, axis=1)
        inputs["depth"] = np.roll(inputs["depth"], -4, axis=1)
    elif scenario == "two":
        inputs["mask"][12:14, 13:17] = 255
        inputs["depth"][12:14, 13:17] = 900
    else:
        inputs["mask"].fill(0)
    block = MeasureDepthSilhouette(parameters={"input_units": "mm", "units_per_sample": 1})
    result = block.execute(inputs)
    assert result and not result.passed and not block.error
    assert expected in {reason["code"] for reason in result.fail_reasons}
    assert result.outputs["thickness_m"] is NO_VALUE
    if scenario == "clipped":
        block.parameters["reject_image_border"] = False
        result = block.execute(inputs)
        assert result and result.passed and result.outputs["report"]["touches_image_border"]


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("plane_frame", "depth camera frame"),
        ("unitless", "declare metric units"),
        ("bad_normal", "unit normal"),
        ("distortion", "undistorted pinhole"),
        ("ticket", "same ticket"),
        ("mask_grid", "coordinate frame differs"),
    ],
)
def test_calibration_timing_and_spatial_guards(fault, message):
    inputs = _inputs()
    if fault == "plane_frame":
        inputs["plane"]["frame"] = "wrong"
    elif fault == "unitless":
        inputs["camera"]["units"] = "unitless"
    elif fault == "bad_normal":
        inputs["plane"]["normal"] = [0, 0, -2]
    elif fault == "distortion":
        inputs["camera"]["distortion"] = [0.1, 0, 0, 0]
    elif fault == "ticket":
        clock = TicketClock()
        for key in ("depth", "plane"):
            inputs[key] = ValuePacket(inputs[key], {"ticket": clock.issue(key)})
    else:
        for key in ("depth", "mask"):
            inputs[key] = ValuePacket(inputs[key], {"spatial": frame((20, 16), key)})
    block = MeasureDepthSilhouette()
    assert block.execute(inputs) is None and block.state == "error"
    assert message in block.error
