"""Exercise the point spacing gauge on the original camera grid."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock


def capture(**parameters):
    tool = builtin_registry().create("calibration.inspect_planar_distance", parameters=parameters)
    clock = TicketClock()
    payload = {"spatial": frame([64, 48], "camera"), "ticket": clock.issue("camera")}
    mapping = {
        "model": "planar_homography_v1",
        "pixel_to_physical": [[0.5, 0, 0], [0, 0.5, 0], [0.01, 0, 1]],
        "units": "mm", "image_size": [64, 48], "origin_pixel": [0, 0], "rms_error": 0.05,
    }
    return tool, {
        "image": ValuePacket(np.zeros((48, 64), np.uint8), payload),
        "mapping": ValuePacket(mapping, payload),
        "points": ValuePacket([[0, 0], [25, 0]], payload),
    }, clock


def test_spacing_maps_both_endpoints_with_perspective_and_keeps_ticket():
    tool, inputs, _ = capture(minimum_distance=10, maximum_distance=10)
    tool.exposed_outputs.append("result_graphics")
    result = tool.execute(inputs)
    assert result is not None and result.passed, tool.error
    assert result.outputs["distance"] == 10
    assert result.outputs["report"]["physical"] == [[0, 0], [10, 0]]
    assert result.payloads["distance"]["ticket"] == inputs["image"].payload["ticket"]
    assert result.payloads["result_graphics"]["spatial"] == inputs["image"].payload["spatial"]


@pytest.mark.parametrize("limits", [{"minimum_distance": 11}, {"maximum_distance": 9}])
def test_out_of_tolerance_measurement_is_still_available(limits):
    tool, inputs, _ = capture(**limits)
    result = tool.execute(inputs)
    assert not result.passed
    assert result.outputs["distance"] == 10
    assert result.fail_reasons[0]["code"] == "planar_distance_out_of_range"


@pytest.mark.parametrize("points", [[], [[0, 0]], [[0, 0], [0, 0]], [[0, 0], [64, 12]]])
def test_unusable_pair_reports_no_measurement(points):
    tool, inputs, _ = capture()
    inputs["points"] = ValuePacket(points, inputs["points"].payload)
    result = tool.execute(inputs)
    assert not result.passed
    assert result.outputs["distance"] is NO_VALUE
    assert result.outputs["report"]["status"] == "unmeasurable"
    assert result.fail_reasons[0]["code"] == "planar_distance_endpoints_unavailable"


@pytest.mark.parametrize("port", ["points", "mapping"])
def test_ticketed_inputs_must_be_from_current_capture(port):
    tool, inputs, clock = capture()
    packet = inputs[port]
    inputs[port] = ValuePacket(packet.value, {**packet.payload, "ticket": clock.issue("camera")})
    assert tool.execute(inputs) is None
    assert "current capture" in tool.error


def test_endpoint_grid_must_match_even_with_saved_calibration():
    tool, inputs, _ = capture(registration="fixed_grid")
    inputs["mapping"] = inputs["mapping"].value
    assert tool.execute(inputs).passed
    inputs["points"] = ValuePacket(inputs["points"].value, {"spatial": frame([64, 48], "crop")})
    assert tool.execute(inputs) is None
    assert "coordinate frame" in tool.error


def test_unacceptable_calibration_and_horizon_are_rejected():
    tool, inputs, _ = capture(maximum_mapping_rms=0.01)
    assert tool.execute(inputs) is None
    assert "maximum RMS" in tool.error
    tool.parameters["maximum_mapping_rms"] = 1
    inputs["mapping"].value["pixel_to_physical"][2][0] = -0.1
    assert tool.execute(inputs) is None
    assert "calibration horizon" in tool.error
