"""Verify side-count, area-budget, containment, and overlay contracts."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ValuePacket, frame, metadata

TYPE_ID = "inspect.approximate_convex_contour"
_PROJECT = (
    Path(__file__).parents[1]
    / "examples/convex_contour_approximation/convex_contour_approximation.json"
)
_CHAMFER = [[15, 25], [25, 15], [115, 15], [125, 25], [125, 85], [115, 95], [25, 95], [15, 85]]


def _encloses(vertices, source=_CHAMFER):
    polygon = np.asarray(vertices, np.float32)
    assert cv2.isContourConvex(polygon)
    assert all(cv2.pointPolygonTest(polygon, tuple(point), True) >= -1e-4 for point in source)


def test_target_side_count_and_area_are_measured():
    block = builtin_registry().create(TYPE_ID)
    result = block.execute({"contour": _CHAMFER})
    assert result and result.passed, block.error
    assert result.outputs["vertex_count"] == 4
    _encloses(result.outputs["points"])
    assert result.outputs["measurements"] == {
        "target_sides": 4,
        "source_area": 8600,
        "hull_area": 8600,
        "area": 8800,
        "added_area_percent": pytest.approx(100 * 200 / 8600),
        "status": "target_reached",
    }


def test_budget_returns_a_valid_partial_polygon_and_fail_reason():
    block = builtin_registry().create(TYPE_ID, parameters={"max_added_area_percent": 1})
    result = block.execute({"contour": _CHAMFER})
    assert result and not result.passed, block.error
    assert result.outputs["vertex_count"] == 7
    assert result.outputs["measurements"]["added_area_percent"] <= 1
    assert result.fail_reasons[0]["code"] == "target_not_reached"
    _encloses(result.outputs["points"])


def test_candidate_with_larger_area_that_cuts_through_hull_is_rejected(monkeypatch):
    bad = np.array([[[-20, -120], [20, -120], [20, 120], [-20, 120]]], np.float32)
    assert cv2.contourArea(bad) > 8600
    monkeypatch.setattr(cv2, "approxPolyN", lambda *args, **kwargs: bad)
    block = builtin_registry().create(TYPE_ID)
    result = block.execute({"contour": _CHAMFER})
    assert result and not result.passed, block.error
    assert result.fail_reasons[0]["code"] == "invalid_native_enclosure"
    assert result.outputs["vertex_count"] == 8
    _encloses(result.outputs["points"])


def test_convexity_choice_uses_original_contour_area_for_budget():
    notched = _CHAMFER[:2] + [[70, 45]] + _CHAMFER[2:]
    block = builtin_registry().create(TYPE_ID, parameters={"max_added_area_percent": 2.5})
    result = block.execute({"contour": notched})
    assert result and not result.passed, block.error
    metrics = result.outputs["measurements"]
    assert metrics["source_area"] == pytest.approx(7250, abs=0.01)
    assert metrics["hull_area"] == pytest.approx(8600, abs=0.01)
    assert metrics["added_area_percent"] <= 2.5
    _encloses(result.outputs["points"])

    block.parameters["convexity"] = "require_convex"
    assert block.execute({"contour": notched}) is None
    assert "strictly convex" in block.error
    star = [[0, -10], [6, 8], [-10, -3], [10, -3], [-6, 8]]
    assert block.execute({"contour": star}) is None
    assert "strictly convex" in block.error
    result = block.execute({"contour": _CHAMFER[::-1]})
    assert result and result.passed, block.error
    _encloses(result.outputs["points"])


def test_rejects_overlay_with_different_coordinate_frame():
    source_frame = frame([144, 112], "camera")
    image = ImageArray(
        np.zeros((112, 144), np.uint8), "gray", {"spatial": frame([144, 112], "other")}
    )
    block = builtin_registry().create(TYPE_ID)
    inputs = {"contour": ValuePacket(_CHAMFER, {"spatial": source_frame}), "image": image}
    assert block.execute(inputs) is None
    assert "share the contour frame" in block.error


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_example_has_two_enclosing_footprints_with_overlays(profile):
    graph = Graph.load(_PROJECT, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    source = graph.blocks["source"].last_result.outputs["image"]
    for name, sides in (("four", 4), ("six", 6)):
        result = graph.blocks[name].last_result
        assert result.outputs["vertex_count"] == sides
        assert result.payloads["points"]["spatial"] == metadata(source)["spatial"]
        _encloses(result.outputs["points"])
        assert np.any(result.outputs["overlay"] != source[..., None])
