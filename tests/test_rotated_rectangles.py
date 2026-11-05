"""Exercise native rotated rectangles through the inspection block contracts."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import ValuePacket, frame


def rectangle(x, y=0, size=(10, 6), angle=0):
    return {"center": [x, y], "size": list(size), "angle_degrees": angle}


@pytest.mark.parametrize(
    ("other", "classification", "area", "vertices"),
    [
        (rectangle(4), "partial", 36, 4),
        (rectangle(10), "touching", 0, 2),
        (rectangle(10, 6), "touching", 0, 1),
        (rectangle(11), "disjoint", 0, 0),
        (rectangle(0, size=(2, 2), angle=20), "contained", 4, 4),
        (rectangle(0, size=(10, 3)), "contained", 30, 4),
        (rectangle(0, size=(6, 10), angle=90), "contained", 60, 4),
        (rectangle(0), "contained", 60, 4),
    ],
)
def test_rotated_overlap_measures_and_classifies(other, classification, area, vertices):
    block = builtin_registry().create(
        "inspect.rotated_rectangle_overlap",
        parameters={"check_iou": True, "minimum_iou": 0.3},
    )
    block.exposed_outputs.append("result_graphics")
    result = block.execute({"rectangle_a": rectangle(0), "rectangle_b": other})
    assert result is not None, block.error
    metrics = result.outputs["measurements"]
    assert metrics["classification"] == classification
    assert len(result.outputs["points"]) == vertices
    assert result.outputs["area"] == pytest.approx(area, abs=1e-5)
    assert metrics["fraction_a"] == pytest.approx(area / 60)
    area_b = other["size"][0] * other["size"][1]
    assert metrics["fraction_b"] == pytest.approx(area / area_b)
    iou = area / (60 + area_b - area)
    assert result.score == result.outputs["iou"] == pytest.approx(iou)
    assert result.passed == (iou >= 0.3)
    assert [reason["code"] for reason in result.fail_reasons] == (
        [] if result.passed else ["rectangle_overlap_below_minimum"]
    )
    if vertices == 1:
        drawing = result.drawings["intersection"].objects[0]
        assert drawing.kind == "marker"


def test_rotated_bounds_feed_overlap_with_pixel_coordinates():
    registry = builtin_registry()
    bounds = registry.create("inspect.bounding_rotated_rectangle")
    overlap = registry.create("inspect.rotated_rectangle_overlap")
    center, size, angle = (140, 230), (80, 32), 27
    points = cv2.boxPoints((center, size, angle)).tolist()
    image = np.zeros((350, 350), np.uint8)
    payload = {"spatial": frame((350, 350), "camera")}
    result = bounds.execute({"points": ValuePacket(points, payload)})
    assert result is not None, bounds.error
    assert result.outputs["area"] == pytest.approx(2560, rel=1e-6)
    np.testing.assert_allclose(result.outputs["rectangle"]["center"], center, atol=1e-5)
    packet = ValuePacket(result.outputs["rectangle"], result.payloads["rectangle"])
    checked = overlap.execute(
        {
            "rectangle_a": packet,
            "rectangle_b": rectangle(140, 230, size, angle),
            "image": ValuePacket(image, payload),
        }
    )
    assert checked is not None, overlap.error
    assert checked.outputs["iou"] == pytest.approx(1, abs=1e-6)
    assert checked.payloads["points"]["spatial"]["root_id"] == "camera"
    assert len(checked.drawings["rectangles"].objects) == 2

    wrong_frame = ValuePacket(image, {"spatial": frame((350, 350), "other_camera")})
    assert overlap.execute({"rectangle_a": packet, "image": wrong_frame}) is None
    assert "same coordinate frame" in overlap.error


def test_rotated_blocks_reject_degenerate_geometry_and_accept_defaults():
    registry = builtin_registry()
    bounds = registry.create("inspect.bounding_rotated_rectangle")
    assert bounds.execute({"points": [[1, 1], [2, 2], [3, 3]]}) is None
    assert bounds.error_kind == "invalid"
    overlap = registry.create("inspect.rotated_rectangle_overlap")
    result = overlap.execute({})
    assert result is not None, overlap.error
    assert result.outputs["iou"] == pytest.approx(0.6)
    assert overlap.execute({"rectangle_a": rectangle(0, size=(10, 0))}) is None
    assert overlap.error_kind == "invalid"
