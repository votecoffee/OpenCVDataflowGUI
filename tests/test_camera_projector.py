"""Different optical grids, metric stepped-depth truth and decoder interoperability."""

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/camera_projector"
CALIBRATE = "3d.camera_projector_calibrate"
RECONSTRUCT = "3d.camera_projector_reconstruct"


def asset(name):
    return cv2.imread(str(EXAMPLE / "assets" / name), cv2.IMREAD_UNCHANGED)


@pytest.fixture(scope="module")
def rig():
    graph = Graph.load(EXAMPLE / "camera_projector.json", builtin_registry())
    try:
        graph.run()
        assert graph.blocks["fit"].state == "pass", graph.blocks["fit"].error
        yield graph.blocks["fit"].last_result.outputs["rig"]
    finally:
        graph.close()


def test_saved_fit_and_holdout_reconstruct_both_planes():
    graph = Graph.load(EXAMPLE / "camera_projector.json", builtin_registry())
    try:
        graph.run()
        fit = graph.blocks["fit"].last_result
        assert fit and fit.passed, graph.blocks["fit"].error
        report = fit.outputs["fit_report"]
        assert report["camera_size"] == [152, 116]
        assert report["projector_size"] == [256, 192]
        assert len(report["views"]) == 14 and report["rms_pixels"] < 0.015
        assert {row["points"] for row in report["views"]} == {39, 40, 41, 42}
        assert report["baseline_m"] == pytest.approx(
            np.linalg.norm([-0.137, 0.019, 0.009]), abs=0.0003
        )
        assert fit.outputs["rig"]["transform"]["translation"] == pytest.approx(
            [-0.137, 0.019, 0.009], abs=0.0003
        )

        decoded = graph.blocks["depth"]
        result = decoded.last_result
        assert result and result.passed, decoded.error
        out = result.outputs
        truth = asset("truth_depth_mm.png")
        valid = truth > 0
        np.testing.assert_array_equal(out["valid_mask"] > 0, valid)
        depth = out["depth"]
        assert depth.units == "m"
        np.testing.assert_allclose(depth[valid], truth[valid] / 1000, atol=0.001)
        assert np.isnan(depth[~valid]).all()
        assert out["report"]["rejected_points"] == 72
        cloud = out["cloud"]
        assert cloud.frame == "camera" and cloud.units == "m"
        assert len(cloud.points) == valid.sum()
        x, y = cloud.pixels.T
        np.testing.assert_array_equal(cloud.points[:, 2], depth[y, x])
        assert result.payloads["valid_mask"]["ticket"]["value"] == "stepped_surface"
        assert result.payloads["depth"]["spatial"]["image_size"] == [152, 116]
    finally:
        graph.close()


def test_projector_refinement_and_fit_quality_gate(rig):
    cameras = {k: copy.deepcopy(rig[k]) for k in ("camera", "projector")}
    projector = cameras["projector"]
    projector["camera_matrix"][0][0] *= 0.92
    projector["camera_matrix"][1][1] *= 1.09
    projector["distortion"] = [0] * 5
    paths = EXAMPLE / "assets" / "calibration_views.json"
    observations = json.loads(paths.read_bytes())
    block = builtin_registry().create(CALIBRATE, parameters={"projector_intrinsics": "refine"})
    inputs = {**cameras, **observations}
    result = block.execute(inputs)
    assert result and result.passed, block.error
    assert result.outputs["rig"]["camera"] == cameras["camera"]
    fitted = result.outputs["rig"]["projector"]["camera_matrix"]
    assert fitted[0][0] == pytest.approx(224, abs=0.3)
    assert fitted[1][1] == pytest.approx(231, abs=0.3)
    block.parameters["max_view_rms_px"] = 0.001
    result = block.execute(inputs)
    assert result and not result.passed
    assert result.outputs["rig"] is NO_VALUE
    assert result.fail_reasons[0]["code"] == "camera_projector_view_rms"


def test_recovers_same_metric_sparse_depth_when_calibration_is_in_millimeters(rig):
    mm = copy.deepcopy(rig)
    mm["camera"]["units"] = mm["projector"]["units"] = mm["transform"]["units"] = "mm"
    for row in mm["transform"]["source_to_target"][:3]:
        row[3] *= 1000
    xmap, ymap = asset("projector_x.tiff"), asset("projector_y.tiff")
    pixels = [[51, 65], [100, 71], [73, 13]]
    matches = [[float(xmap[y, x]), float(ymap[y, x])] for x, y in pixels[:2]] + [None]
    source = frame([152, 116], "station")
    ticket = TicketClock().issue("camera")
    result = (
        builtin_registry()
        .create(RECONSTRUCT)
        .execute(
            {
                "rig": mm,
                "camera_points": ValuePacket(pixels, {"spatial": source, "ticket": ticket}),
                "projector_points": matches,
                "valid": [True, True, False],
            }
        )
    )
    assert result and result.passed
    np.testing.assert_allclose(result.outputs["cloud"].points[:, 2], [1.1, 1.33], atol=0.001)
    assert np.isnan(result.outputs["depth"][13, 73])
    assert result.payloads["cloud"]["ticket"] == ticket


def test_existing_graycode_decoder_supplies_sparse_projector_coordinates(rig):
    registry = builtin_registry()
    pattern = registry.create("3d.graycode", parameters={"width": 256, "height": 192}).execute({})
    x, y = asset("projector_x.tiff"), asset("projector_y.tiff")
    lookup_x = np.rint(np.maximum(x, 0)).astype(int)
    lookup_y = np.rint(np.maximum(y, 0)).astype(int)
    truth = asset("truth_depth_mm.png")
    valid = truth > 0
    source = pattern.outputs["patterns"]
    captures = FlowGroup(
        tuple(
            ValuePacket(packet.value[lookup_y, lookup_x], packet.payload) for packet in source.items
        ),
        source.labels,
    )
    decoder = registry.create(
        "3d.graycode_decode", parameters={"width": 256, "height": 192, "sample_step": 12}
    )
    pairs = decoder.execute(
        {
            "patterns": captures,
            "white": valid.astype(np.uint8) * 255,
            "black": np.zeros_like(valid, np.uint8),
        }
    )
    assert pairs and pairs.passed, decoder.error
    block = registry.create(RECONSTRUCT)
    result = block.execute({"rig": rig, **pairs.outputs})
    assert result and result.passed, block.error
    cloud = result.outputs["cloud"]
    assert len(cloud.points) > 85
    xx, yy = cloud.pixels.T
    np.testing.assert_allclose(cloud.points[:, 2], truth[yy, xx] / 1000, atol=0.035)


def test_phase_maps_preserve_the_camera_grid_and_support(rig):
    registry = builtin_registry()
    same_grid = frame([152, 116], "fixed-camera")
    maps = {}
    for axis, extent in (("x", 256), ("y", 192)):
        raw = asset(f"projector_{axis}.tiff")
        config = (
            registry.create(
                "3d.phase_shift_patterns", parameters={"width": 256, "height": 192, "axis": axis}
            )
            .execute({})
            .outputs["specification"]
        )
        images = tuple(
            ValuePacket(
                np.where(raw >= 0, 0.5 + 0.2 * np.sin(2 * np.pi * raw / period + phase), 0),
                {"spatial": same_grid},
            )
            for period in (extent, extent // 8)
            for phase in (0, 2 * np.pi / 3, 4 * np.pi / 3)
        )
        result = registry.create("3d.phase_shift_decode").execute(
            {
                "patterns": FlowGroup(images, tuple(config["order"])),
                "specification": config,
            }
        )
        assert result and result.passed
        port = "projector_coordinate"
        maps[f"projector_{axis}"] = ValuePacket(result.outputs[port], result.payloads[port])
    block = registry.create(RECONSTRUCT, parameters={"max_reprojection_error_px": 0.25})
    result = block.execute({"rig": rig, **maps})
    assert result and result.passed, block.error
    truth = asset("truth_depth_mm.png")
    valid = truth > 0
    np.testing.assert_array_equal(result.outputs["valid_mask"] != 0, valid)
    np.testing.assert_allclose(result.outputs["depth"][valid], truth[valid] / 1000, atol=0.001)
    assert result.payloads["depth"]["spatial"] == same_grid


def test_invalid_pair_modes_axis_and_geometry_reject_explicitly(rig):
    block = builtin_registry().create(RECONSTRUCT)
    x, y = asset("projector_x.tiff"), asset("projector_y.tiff")
    assert block.execute({"rig": rig, "projector_x": x}) is None
    assert "both dense" in block.error
    wrong = ValuePacket(x, {"phase_shift": {"axis": "y", "projector_size": [256, 192]}})
    assert block.execute({"rig": rig, "projector_x": wrong, "projector_y": y}) is None
    assert "axis or projector dimensions" in block.error
    wrong_rig = copy.deepcopy(rig)
    wrong_rig["transform"]["target_frame"] = "other"
    assert block.execute({"rig": wrong_rig, "projector_x": x, "projector_y": y}) is None
    assert "Rig pose" in block.error
    block.parameters["minimum_ray_angle_degrees"] = 89
    result = block.execute({"rig": rig, "projector_x": x, "projector_y": y})
    assert result and not result.passed
    assert not result.outputs["valid_mask"].any()


def test_projection_selection_and_saved_rig_roundtrip(rig, tmp_path):
    registry = builtin_registry()
    path = str(tmp_path / "camera_projector.json")
    writer = registry.create("output.calibration", parameters={"path": path})
    saved = writer.execute({"profile": rig})
    assert saved and saved.passed, writer.error
    reader = registry.create("input.calibration", parameters={"path": path})
    loaded = reader.execute({})
    assert loaded and loaded.passed, reader.error
    assert loaded.outputs["profile"] == rig

    x, y = asset("projector_x.tiff"), asset("projector_y.tiff")
    grid = frame([152, 116], "camera-grid")
    mask = np.zeros(x.shape, np.uint8)
    mask[14:24, 40:65] = 255
    region = {
        "model": "image_region_v1",
        "shape": "rectangle",
        "image_size": [152, 116],
        "rectangle": [44, 16, 10, 6],
        "coordinate_context": grid,
    }
    block = registry.create(RECONSTRUCT, parameters={"sample_step": 2})
    inputs = {
        "rig": loaded.outputs["profile"],
        "projector_x": ValuePacket(x, {"spatial": grid}),
        "projector_y": ValuePacket(y, {"spatial": grid}),
        "mask": ValuePacket(mask, {"spatial": grid}),
        "region": region,
    }
    result = block.execute(inputs)
    assert result and result.passed, block.error
    assert result.outputs["report"]["accepted_points"] == 15
    allowed = np.zeros(x.shape, bool)
    allowed[16:22:2, 44:54:2] = True
    np.testing.assert_array_equal(result.outputs["valid_mask"] != 0, allowed)
    block.parameters["max_points"] = 14
    assert block.execute(inputs) is None and "Too many correspondences" in block.error
