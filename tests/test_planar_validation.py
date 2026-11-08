"""Exercise held-out planar validation and the downstream mapping gate."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock


def calibration():
    return {
        "model": "planar_homography_v1", "pixel_to_physical": np.diag([0.25, 0.25, 1]).tolist(),
        "units": "mm", "image_size": [40, 32], "origin_pixel": [0, 0], "rms_error": 0,
    }


def parameters(**overrides):
    return {
        "registration": "fixed_grid",
        "physical_points": [[1, 1], [5, 1], [2, 4]],
        "pixel_points": [[4, 4], [20, 4], [8, 16]],
        **overrides,
    }


def create(**overrides):
    return builtin_registry().create(
        "calibration.validate_planar", parameters=parameters(**overrides)
    )


def test_planar_check_passes_the_original_mapping_with_current_ticket_and_frame():
    tool = create()
    source = np.zeros((32, 40), np.uint8)
    mapping = calibration()
    ticket = TicketClock().issue("camera")
    space = frame([40, 32], "camera")
    result = tool.execute({
        "image": ValuePacket(source, {"spatial": space, "ticket": ticket}),
        "mapping": mapping,
    })
    assert result is not None and result.passed, tool.error
    assert result.outputs["checked_mapping"] == mapping
    assert result.outputs["rms"] == 0
    assert result.outputs["max_error"] == 0
    assert result.outputs["report"]["points"][2]["physical"] == [2, 4]
    assert result.payloads["checked_mapping"]["spatial"] == space
    assert result.payloads["checked_mapping"]["ticket"] == ticket


@pytest.mark.parametrize(("rms_limit", "point_limit", "code"), [
    (0.2, 1, "planar_validation_rms_exceeds_limit"),
    (1, 0.4, "planar_validation_point_exceeds_limit"),
])
def test_planar_check_applies_independent_rms_and_worst_limits(rms_limit, point_limit, code):
    tool = create(
        pixel_points=[[4, 4], [20, 4], [10, 16]],
        maximum_rms=rms_limit, maximum_point_error=point_limit,
    )
    result = tool.execute({"image": np.zeros((32, 40), np.uint8), "mapping": calibration()})
    assert result is not None and not result.passed
    assert result.outputs["checked_mapping"] is NO_VALUE
    assert result.outputs["rms"] == pytest.approx(0.5 / np.sqrt(3))
    assert result.outputs["max_error"] == 0.5
    assert result.outputs["report"]["worst_point_index"] == 2
    assert result.outputs["report"]["points"][2]["residual"] == [0.5, 0]
    assert [reason["code"] for reason in result.fail_reasons] == [code]


def test_planar_check_reports_incomplete_fiducials_and_rejects_stale_inputs():
    tool = create()
    clock = TicketClock()
    ticket = clock.issue("camera")
    space = frame([40, 32], "camera")
    payload = {"spatial": space, "ticket": ticket}
    data = {
        "image": ValuePacket(np.zeros((32, 40), np.uint8), payload), "mapping": calibration(),
    }
    result = tool.execute({**data, "points": ValuePacket([[4, 4], [20, 4]], payload)})
    assert not result.passed and result.outputs["rms"] is NO_VALUE
    assert result.outputs["report"]["observed_count"] == 2
    assert result.fail_reasons[0]["code"] == "planar_validation_observation_unavailable"
    points = parameters()["pixel_points"]
    assert tool.execute({
        **data, "points": ValuePacket(points, {**payload, "ticket": clock.issue("camera")}),
    }) is None
    assert "current capture" in tool.error
    assert tool.execute({
        **data, "points": ValuePacket(points, {"spatial": frame([40, 32], "different")}),
    }) is None
    assert "coordinate frame" in tool.error


def test_planar_check_withholds_workplane_run_after_failure():
    registry = builtin_registry()
    graph = Graph()
    tool = graph.add(create())
    plane = graph.add(registry.create("calibration.rectify_workplane", parameters={
        "width": 40, "height": 32,
    }))
    graph.connect(tool.id, "checked_mapping", plane.id, "mapping")
    clock = TicketClock()
    try:
        for expected in (True, False):
            payload = {"ticket": clock.issue("camera"), "spatial": frame([40, 32], "camera")}
            image = ValuePacket(np.zeros((32, 40), np.uint8), payload)
            if not expected:
                tool.parameters["pixel_points"] = [[8, 4], [24, 4], [12, 16]]
            graph.run(injected={
                (tool.id, "image"): image, (tool.id, "mapping"): calibration(),
                (plane.id, "image"): image,
            })
            assert tool.last_result.passed is expected
            assert plane.state == ("pass" if expected else "inactive")
            assert (plane.last_result is not None) is expected
    finally:
        graph.close()
