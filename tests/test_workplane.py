"""Check workplane rectification, frame provenance and calibrated strip judgments."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock


def calibration(shape, *, matrix=None):
    if matrix is None:
        matrix = np.diag([0.25, 0.25, 1])
    return {
        "model": "planar_homography_v1", "pixel_to_physical": matrix.tolist(),
        "units": "mm", "image_size": list(shape[::-1]), "rms_error": 0.01, "origin_pixel": [0, 0],
    }


def test_rectified_workplane_maps_physical_origin_and_valid_mask_into_source_frame():
    image = np.arange(100, dtype=np.uint16).reshape(10, 10)
    space = frame([10, 10], "camera")
    ticket = TicketClock().issue("camera")
    mask = np.full(image.shape, 255, np.uint8)
    mask[4, 3] = 0
    tool = builtin_registry().create("calibration.rectify_workplane", parameters={
        "origin": [0.5, 0.75], "width": 4, "height": 3, "units_per_pixel": 0.25,
    })
    result = tool.execute({
        "image": ValuePacket(image, {"spatial": space, "ticket": ticket}),
        "mapping": ValuePacket(calibration(image.shape), {"spatial": space}), "mask": mask,
    })
    assert result is not None and result.passed, tool.error
    expected = image[3:6, 2:6].copy()
    expected[1, 1] = 0
    np.testing.assert_array_equal(result.outputs["image"], expected)
    assert result.outputs["image"].dtype == np.uint16
    assert result.outputs["valid_mask"][1, 1] == 0
    scale = result.outputs["scale"]
    assert scale["units"] == "mm" and scale["units_per_pixel"] == 0.25
    assert scale["origin"] == [0.5, 0.75]
    assert scale["spatial"] == result.payloads["image"]["spatial"]
    assert scale["spatial"]["root_id"] == "camera"
    np.testing.assert_array_equal(scale["spatial"]["to_root"], [[1, 0, 2], [0, 1, 3], [0, 0, 1]])
    assert result.payloads["scale"]["ticket"] == ticket


def test_workplane_rejects_missing_registration_and_out_of_range_calibration_error():
    image = np.full((12, 16), 255, np.uint8)
    mapping = calibration(image.shape)
    tool = builtin_registry().create("calibration.rectify_workplane", parameters={
        "width": 12, "height": 8,
    })
    assert tool.execute({"image": image, "mapping": mapping}) is None
    assert "coordinate frame" in tool.error
    tool.parameters["registration"] = "fixed_grid"
    assert tool.execute({"image": image, "mapping": mapping}).passed
    tool.parameters["maximum_rms_error"] = 0.005
    assert tool.execute({"image": image, "mapping": mapping}) is None
    assert "RMS error" in tool.error


def test_workplane_rejects_calibration_horizon_and_reports_empty_coverage():
    image = np.ones((12, 16), np.uint8)
    tool = builtin_registry().create("calibration.rectify_workplane", parameters={
        "width": 7, "height": 5, "registration": "fixed_grid", "units_per_pixel": 1,
    })
    projective = np.array([[1, 0, 0], [0, 1, 0], [0.25, 0, 1]])
    mapping = calibration(image.shape, matrix=projective)
    assert tool.execute({"image": image, "mapping": mapping}) is None
    assert "horizon" in tool.error
    tool.parameters["origin"] = [100, 100]
    result = tool.execute({"image": image, "mapping": calibration(image.shape)})
    assert not result.passed and not result.outputs["valid_mask"].any()
    assert result.fail_reasons[0]["code"] == "workplane_empty_coverage"


def test_strip_width_and_defect_length_use_workplane_scale_and_reject_stale_scale():
    registry = builtin_registry()
    image = np.zeros((15, 30), np.uint8)
    image[4:11] = 255
    image[4:11, 10:14] = 0
    image[6:9, 10:14] = 255
    workplane = registry.create("calibration.rectify_workplane", parameters={
        "width": 30, "height": 15, "registration": "fixed_grid", "interpolation": "nearest",
    })
    clock = TicketClock()
    ticket = clock.issue("camera")
    result = workplane.execute({
        "image": ValuePacket(image, {"spatial": frame([30, 15], "camera"), "ticket": ticket}),
        "mapping": calibration(image.shape),
    })
    unwrap = registry.create("cv.unwrap_path_strip", parameters={
        "vertices": [[2, 7], [27, 7]], "half_width": 5, "cross_samples": 11,
    })
    wires = {"image": result.outputs["image"], "mask": result.outputs["valid_mask"]}
    scale = result.outputs["scale"]
    wrong = {**scale, "spatial": frame([30, 15], "other")}
    assert unwrap.execute({**wires, "scale": wrong}) is None
    assert "current input grid" in unwrap.error
    old = ValuePacket(scale, {"ticket": clock.issue("camera")})
    assert unwrap.execute({**wires, "scale": old}) is None
    assert "share one ticket" in unwrap.error
    data = unwrap.execute({**wires, "scale": scale}).outputs
    assert data["sampling"]["physical_scale"] == {"units": "mm", "units_per_pixel": 0.25}

    inspect = registry.create("inspect.strip_width", parameters={"minimum_width": 1.25})
    measurement = inspect.execute({
        "image": data["image"], "mask": data["valid_mask"], "sampling": data["sampling"],
    })
    assert measurement.outputs["measurements"]["units"] == "mm"
    assert measurement.outputs["measurements"]["maximum"] == 1.75
    group = registry.create("inspect.strip_defects")
    result = group.execute({
        "profile": ValuePacket(measurement.outputs["profile"], measurement.payloads["profile"]),
        "sampling": data["sampling"],
    })
    run, = result.outputs["defects"]
    assert run["kind"] == "narrow"
    assert run["units"] == run["width_units"] == "mm"
    assert run["length"] == pytest.approx(1)
    assert run["minimum_width"] == pytest.approx(0.75)
    judge = registry.create("inspect.judge_strip_defects", parameters={
        "length_units": "mm", "limits": {"narrow": {"max_count": 1, "max_length": 1}},
    })
    assert judge.execute({"inspection_report": result.outputs["inspection_report"]}).passed
