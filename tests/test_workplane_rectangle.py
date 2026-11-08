"""Check calibrated dimensions, long-axis symmetry and complete coverage."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock


def fixture(**parameters):
    tool = builtin_registry().create("inspect.workplane_rectangle", parameters=parameters)
    clock = TicketClock()
    space = frame([100, 90], "workplane")
    payload = {"spatial": space, "ticket": clock.issue("camera")}
    wires = {
        "image": ValuePacket(np.zeros((90, 100), np.uint8), payload),
        "rectangle": ValuePacket({
            "center": [44, 38], "size": [16, 40], "angle_degrees": 89,
        }, payload),
        "scale": ValuePacket({
            "model": "rectified_workplane_v1", "spatial": space,
            "units": "mm", "units_per_pixel": 0.25, "origin": [-4, 2],
        }, payload),
        "valid_mask": ValuePacket(np.full((90, 100), 255, np.uint8), payload),
    }
    return tool, wires, clock


def test_workplane_rectangle_canonicalizes_long_side_and_treats_opposite_axes_as_equivalent():
    tool, wires, _ = fixture(
        minimum_length=10, maximum_length=10, minimum_width=4, maximum_width=4,
        check_angle=True, target_angle=179, angle_tolerance=0,
    )
    result = tool.execute(wires)
    assert result is not None and result.passed, tool.error
    assert result.outputs["measurements"] == {
        "units": "mm", "status": "measured", "center": [7, 11.5],
        "length": 10, "width": 4, "angle_degrees": -1,
    }
    assert result.payloads["length"]["ticket"] == wires["image"].payload["ticket"]
    assert result.payloads["length"]["spatial"] == wires["image"].payload["spatial"]


def test_rectangle_has_separate_width_length_and_orientation_failures():
    tool, wires, _ = fixture(
        maximum_length=9, minimum_width=5,
        check_angle=True, target_angle=20, angle_tolerance=4,
    )
    result = tool.execute(wires)
    assert not result.passed
    assert [item["code"] for item in result.fail_reasons] == [
        "workplane_rectangle_length_out_of_range", "workplane_rectangle_width_out_of_range",
        "workplane_rectangle_angle_out_of_range",
    ]
    assert result.outputs["width"] == 4


def test_near_square_preserves_dimensions_but_withholds_ambiguous_angle():
    tool, wires, _ = fixture(check_angle=True)
    wires["rectangle"].value["size"] = [20, 20.3]
    result = tool.execute(wires)
    assert not result.passed
    assert result.outputs["width"] == 5 and result.outputs["angle"] is NO_VALUE
    assert result.fail_reasons[0]["code"] == "workplane_rectangle_orientation_ambiguous"


@pytest.mark.parametrize("source", ["mask", "edge"])
def test_incomplete_coverage_withholds_numeric_dimensions(source):
    tool, wires, _ = fixture()
    if source == "mask":
        wires["valid_mask"].value[38, 44] = 0
    else:
        wires["rectangle"].value["center"] = [8, 38]
    result = tool.execute(wires)
    assert not result.passed
    assert result.outputs["measurements"]["status"] == "unmeasurable"
    assert all(result.outputs[name] is NO_VALUE for name in ("width", "length", "angle"))


def test_rectangle_rejects_wrong_units_frame_and_ticket():
    tool, wires, clock = fixture(units="cm")
    assert tool.execute(wires) is None
    assert "units" in tool.error
    tool.parameters["units"] = "mm"
    rect = wires["rectangle"]
    old = ValuePacket(rect.value, {**rect.payload, "ticket": clock.issue("camera")})
    assert tool.execute({**wires, "rectangle": old}) is None
    assert "share one ticket" in tool.error
    foreign = ValuePacket(rect.value, {**rect.payload, "spatial": frame([100, 90], "foreign")})
    assert tool.execute({**wires, "rectangle": foreign}) is None
    assert "share the workplane image frame" in tool.error
