"""Check continuous polygon overlap, boundary contact, and input provenance."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import ValuePacket, frame

_SQUARE = [[0, 0], [4, 0], [4, 4], [0, 4]]


def test_polygon_overlap_preserves_small_geometry_at_a_large_coordinate_offset():
    offset = np.array([999_900, 999_900])
    a = np.array([[0, 0], [2, 0], [4, 0], [4, 4], [0, 4], [0, 0]]) + offset
    b = (np.array(_SQUARE)[::-1] + 2) + offset
    spatial = frame([200, 200], "camera")
    block = builtin_registry().create(
        "inspect.convex_polygon_overlap", parameters={"check_iou": True, "minimum_iou": 0.2}
    )
    result = block.execute(
        {"points_a": ValuePacket(a.tolist(), {"spatial": spatial}), "points_b": b.tolist()}
    )
    assert result is not None, block.error
    assert not result.passed and result.outputs["classification"] == "partial"
    assert result.outputs["area"] == 4 and result.outputs["iou"] == pytest.approx(1 / 7)
    assert result.outputs["measurements"]["fraction_a"] == 0.25
    assert result.payloads["points"]["spatial"] == spatial
    assert set(map(tuple, np.array(result.outputs["points"]) - offset)) == {
        (2, 2),
        (4, 2),
        (4, 4),
        (2, 4),
    }
    assert set(result.drawings) == {"polygons", "intersection"}


@pytest.mark.parametrize(
    ("offset", "classification", "points"),
    (
        ([4, 0], "touching", {(4, 0), (4, 4)}),
        ([4, 4], "touching", {(4, 4)}),
        ([5, 5], "none", set()),
    ),
)
def test_zero_area_intersection_keeps_only_real_contact_points(offset, classification, points):
    block = builtin_registry().create("inspect.convex_polygon_overlap")
    result = block.execute({"points_a": _SQUARE, "points_b": (np.array(_SQUARE) + offset).tolist()})
    assert result is not None, block.error
    assert result.outputs["area"] == result.outputs["iou"] == 0
    assert result.outputs["classification"] == classification
    assert set(map(tuple, result.outputs["points"])) == points


def test_nested_polygons_follow_the_containment_setting():
    block = builtin_registry().create("inspect.convex_polygon_overlap")
    inputs = {"points_a": _SQUARE, "points_b": (np.array(_SQUARE) / 4 + 1).tolist()}
    result = block.execute(inputs)
    assert result is not None, block.error
    assert result.outputs["classification"] == "contained" and result.outputs["area"] == 1
    block.parameters["include_containment"] = False
    result = block.execute(inputs)
    assert result is not None, block.error
    assert result.outputs["classification"] == "none" and result.outputs["points"] == []


def test_unordered_vertices_require_hull_mode_and_frames_must_agree():
    block = builtin_registry().create("inspect.convex_polygon_overlap")
    crossed = [_SQUARE[i] for i in (0, 2, 1, 3)]
    inputs = {"points_a": crossed, "points_b": _SQUARE}
    assert block.execute(inputs) is None
    assert "convex boundary" in block.error
    block.parameters["input_mode"] = "convex_hull"
    result = block.execute(inputs)
    assert result is not None, block.error
    assert result.outputs["iou"] == 1
    assert (
        block.execute(
            {
                "points_a": ValuePacket(crossed, {"spatial": frame([20, 20], "first")}),
                "points_b": ValuePacket(_SQUARE, {"spatial": frame([20, 20], "second")}),
            }
        )
        is None
    )
    assert "same coordinate frame" in block.error
