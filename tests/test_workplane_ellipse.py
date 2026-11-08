"""Check ellipse dimensions, tolerances and coverage on a calibrated grid."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock


def capture(**parameters):
    tool = builtin_registry().create("inspect.workplane_ellipse", parameters=parameters)
    clock = TicketClock()
    space = frame([100, 90], "plane")
    payload = {"spatial": space, "ticket": clock.issue("camera")}
    return tool, {
        "image": ValuePacket(np.zeros((90, 100), np.uint8), payload),
        "fit": ValuePacket({
            "shape": "ellipse", "center": [44, 38], "axis_lengths": [12, 32],
            "angle_degrees": 90, "rmse": 0.8,
        }, payload),
        "scale": ValuePacket({
            "model": "rectified_workplane_v1", "spatial": space, "units": "mm",
            "units_per_pixel": 0.25, "origin": [-4, 2],
        }, payload),
        "valid_mask": ValuePacket(np.full((90, 100), 255, np.uint8), payload),
    }, clock


def test_diameters_residual_and_center_are_in_calibrated_units():
    tool, wires, _ = capture(
        minimum_diameter=3, maximum_diameter=8, maximum_axis_difference=5, maximum_fit_rmse=0.2,
    )
    tool.exposed_outputs.append("result_graphics")
    wires["valid_mask"].value[44, 58] = 0  # In the bounding box, outside the ellipse.
    result = tool.execute(wires)
    assert result is not None and result.passed, tool.error
    assert result.outputs["measurements"] == {
        "units": "mm", "status": "measured", "center": [7, 11.5],
        "major_diameter": 8, "minor_diameter": 3, "axis_difference": 5, "fit_rmse": 0.2,
    }
    assert result.payloads["major_diameter"]["ticket"] == wires["image"].payload["ticket"]
    assert result.payloads["result_graphics"]["spatial"] == wires["image"].payload["spatial"]


def test_dimension_ovality_and_fit_error_have_separate_failure_reasons():
    tool, wires, _ = capture(
        minimum_diameter=4, maximum_diameter=7, maximum_axis_difference=2, maximum_fit_rmse=0.1,
    )
    result = tool.execute(wires)
    assert not result.passed and result.outputs["major_diameter"] == 8
    assert [item["code"] for item in result.fail_reasons] == [
        "workplane_ellipse_minor_diameter_out_of_range",
        "workplane_ellipse_major_diameter_out_of_range",
        "workplane_ellipse_axis_difference_out_of_range",
        "workplane_ellipse_fit_rmse_out_of_range",
    ]


@pytest.mark.parametrize("clipping", ["mask", "edge"])
def test_incomplete_footprint_withholds_numeric_outputs(clipping):
    tool, wires, _ = capture()
    if clipping == "mask":
        wires["valid_mask"].value[38, 59] = 0
    else:
        wires["fit"].value["center"] = [16, 38]
    result = tool.execute(wires)
    assert not result.passed and result.outputs["measurements"]["status"] == "unmeasurable"
    assert result.fail_reasons[0]["code"] == "workplane_ellipse_unmeasurable"
    assert all(result.outputs[key] is NO_VALUE for key in (
        "major_diameter", "minor_diameter", "axis_difference", "fit_rmse",
    ))


def test_ellipse_must_match_current_workplane_frame_and_ticket():
    tool, wires, clock = capture()
    fit = wires["fit"]
    wires["fit"] = ValuePacket(fit.value, {**fit.payload, "ticket": clock.issue("camera")})
    assert tool.execute(wires) is None and "share one ticket" in tool.error
    wires["fit"] = ValuePacket(fit.value, {"spatial": frame([100, 90], "foreign")})
    assert tool.execute(wires) is None and "workplane image frame" in tool.error
