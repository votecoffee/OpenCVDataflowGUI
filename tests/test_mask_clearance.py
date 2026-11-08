"""Check mask clearance accuracy, selections, ticket metadata, and fail reasons."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.graphics_layers import draw_object
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock


def block(**parameters):
    return builtin_registry().create("inspect.mask_clearance", parameters=parameters)


def test_clearance_separates_gaps_and_out_of_range_and_preserves_ticket():
    source = np.zeros((16, 24), np.uint8)
    source[2:13, :] = 255
    source[:, 11:13] = 0
    source[:7, 17:] = 0
    source[8:, 17:] = 0
    reference = np.zeros_like(source)
    reference[7, [4, 5, 11, 19]] = 255
    coords = frame([24, 16], "registered-part")
    ticket = TicketClock().issue("camera")
    tool = block(minimum_clearance=3, maximum_clearance=5)
    result = tool.execute({
        "image": ValuePacket(source, {"spatial": coords, "ticket": ticket}),
        "reference": ValuePacket(reference, {"spatial": coords}),
    })
    assert result is not None, tool.error
    assert result.outputs["coverage"] == 0.75 and result.outputs["pass_fraction"] == 0.25
    np.testing.assert_array_equal(result.outputs["clearance"][7, [4, 5, 11, 19]], [5, 6, 0, 1])
    assert np.count_nonzero(result.outputs["clearance"]) == 3
    assert np.flatnonzero(result.outputs["missing_mask"][7]).tolist() == [11]
    assert np.flatnonzero(result.outputs["out_of_range_mask"][7]).tolist() == [5, 19]
    assert result.outputs["measurements"]["mean"] == 4
    assert [reason["code"] for reason in result.fail_reasons] == [
        "mask_coverage_below_minimum", "mask_clearance_out_of_range",
    ]
    for key in ("clearance", "missing_mask", "out_of_range_mask"):
        assert result.payloads[key]["spatial"] == coords
        assert result.payloads[key]["ticket"] == ticket
    canvas = np.zeros((*source.shape, 3), np.uint8)
    for drawing in result.drawings.values():
        for item in drawing.objects:
            draw_object(canvas, item, np.eye(3))
    assert canvas[7, 11].any() and canvas[7, 19].any()


def test_clearance_intersects_selection_without_cropping_distance_neighborhood():
    image = np.ones((15, 15), np.uint8)
    reference = image.copy()
    mask = np.zeros_like(image)
    mask[7:9, 7:9] = 255
    region = {
        "model": "image_region_v1", "image_size": [15, 15],
        "shape": "rectangle", "rectangle": [7, 7, 1, 1],
    }
    tool = block(minimum_clearance=8)
    result = tool.execute({"image": image, "reference": reference, "mask": mask, "region": region})
    assert result is not None and result.passed
    assert np.count_nonzero(result.outputs["clearance"]) == 1
    assert result.outputs["clearance"][7, 7] == 8
    assert result.outputs["measurements"]["sample_count"] == 1


def test_clearance_uses_precise_diagonal_euclidean_distance():
    image = np.ones((15, 15), np.uint8)
    image[6, 6] = 0
    reference = np.zeros_like(image)
    reference[8, 7] = 255
    result = block().execute({"image": image, "reference": reference})
    assert result.outputs["measurements"]["minimum"] == pytest.approx(np.sqrt(5))


def test_clearance_rejects_unregistered_reference_unless_grid_alignment_is_requested():
    image = np.ones((8, 8), np.uint8)
    inputs = {
        "image": ValuePacket(image, {"spatial": frame([8, 8], "a")}),
        "reference": ValuePacket(image, {"spatial": frame([8, 8], "b")}),
    }
    tool = block()
    assert tool.execute(inputs) is None
    assert "share a coordinate frame" in tool.error
    tool.parameters["alignment"] = "pixel_grid"
    assert tool.execute(inputs).passed


def test_clearance_fails_an_entirely_missing_foreground_with_no_distance_statistics():
    image = np.zeros((6, 8), np.uint8)
    tool = block(minimum_coverage=0, minimum_in_range_fraction=0)
    result = tool.execute({"image": image, "reference": np.ones_like(image)})
    assert not result.passed
    assert result.outputs["coverage"] == 0
    assert result.outputs["measurements"]["minimum"] is None
    assert result.fail_reasons[0]["code"] == "mask_coverage_below_minimum"


def test_clearance_rejects_reversed_limits_and_empty_reference():
    image = np.zeros((6, 8), np.uint8)
    tool = block(minimum_clearance=5, maximum_clearance=3)
    assert tool.execute({"image": image, "reference": image}) is None
    assert "Maximum clearance" in tool.error
    tool.parameters["maximum_clearance"] = 0
    assert tool.execute({"image": image, "reference": image}) is None
    assert "at least one reference sample" in tool.error
