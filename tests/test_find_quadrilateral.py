"""Verify page-corner selection, stable ordering, and rejection gates."""

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.graphics_layers import RESULT_GRAPHICS
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame


def test_quad_skips_larger_clipped_candidate_and_orders_diamond_without_duplicates():
    mask = np.zeros((66, 120), np.uint8)
    cv2.rectangle(mask, (0, 8), (54, 58), 255, -1)
    expected = [[85, 15], [105, 35], [85, 55], [65, 35]]
    cv2.fillConvexPoly(mask, np.int32(expected), 255)
    block = builtin_registry().create("inspect.find_quadrilateral")
    block.exposed_outputs.append(RESULT_GRAPHICS)
    context = {"spatial": frame([120, 66], "inspection")}
    result = block.execute({"image": ValuePacket(mask, context)})
    assert result is not None and result.passed, block.error
    assert result.outputs["corners"] == expected
    assert result.payloads["corners"]["spatial"] == context["spatial"]
    assert [item["status"] for item in result.outputs["measurements"]["candidates"]] == [
        "border_margin",
        "accepted",
    ]
    (layer,) = result.outputs[RESULT_GRAPHICS].layers
    assert layer.payload["spatial"] == context["spatial"]
    assert [item.text for item in layer.objects[1:]] == ["1", "2", "3", "4"]


def test_quad_withholds_points_when_the_only_contour_touches_the_border():
    block = builtin_registry().create("inspect.find_quadrilateral")
    mask = np.zeros((40, 50), np.uint8)
    mask[0:28, 4:35] = 255
    result = block.execute({"image": mask})
    assert result is not None and not result.passed
    assert result.outputs["corners"] is NO_VALUE
    assert result.fail_reasons[0]["code"] == "quadrilateral_not_found"
    assert not block.current_emissions.get("corners")


def test_quad_area_gate_rejects_an_approximation_that_fills_a_large_notch():
    mask = np.zeros((100, 100), np.uint8)
    cv2.rectangle(mask, (10, 10), (88, 88), 255, -1)
    mask[10:43, 33:62] = 0
    block = builtin_registry().create(
        "inspect.find_quadrilateral",
        parameters={
            "perimeter_fraction": 0.11,
            "minimum_area_agreement": 0.95,
        },
    )
    result = block.execute({"image": mask})
    assert result is not None and not result.passed
    (record,) = result.outputs["measurements"]["candidates"]
    assert record["vertices"] == 4
    assert record["status"] == "area_agreement"
    assert result.outputs["corners"] is NO_VALUE
