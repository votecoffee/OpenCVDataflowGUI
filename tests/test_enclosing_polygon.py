"""Verify enclosing geometry, method selection, and native-result safeguards."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import ValuePacket, frame

_PENTAGON = [[0, 0], [3, 0], [5, 2], [3, 4], [0, 4]]


def test_enclosing_triangle_uses_local_coordinates_and_preserves_the_input_frame():
    block = builtin_registry().create("inspect.bounding_convex_polygon")
    offset = np.array([9_999_900, 9_999_900])
    rectangle = np.array([[0, 0], [6, 0], [6, 4], [0, 4]])
    spatial = frame([20, 20], "camera")
    result = block.execute(
        {"points": ValuePacket((rectangle + offset).tolist(), {"spatial": spatial})}
    )
    assert result is not None, block.error
    assert result.outputs["measurements"] == {
        "method": "triangle",
        "side_count": 3,
        "area": 48,
        "hull_area": 24,
        "fill_ratio": 0.5,
    }
    local = (np.array(result.outputs["points"]) - offset).astype(np.float32)
    assert all(cv2.pointPolygonTest(local, tuple(map(float, p)), False) >= 0 for p in rectangle)
    assert result.payloads["points"]["spatial"] == spatial
    assert len(result.drawings["enclosing polygon"].objects[0].points) == 3


def test_enclosing_quadrilateral_improves_on_the_rectangle_for_the_example_pentagon():
    block = builtin_registry().create(
        "inspect.bounding_convex_polygon", parameters={"maximum_sides": 4}
    )
    result = block.execute({"points": _PENTAGON})
    assert result is not None, block.error
    measure = result.outputs["measurements"]
    assert measure["method"] == "convex_polygon" and measure["side_count"] == 4
    assert measure["area"] == pytest.approx(19.5)
    assert measure["hull_area"] == 16
    assert result.outputs["fill_ratio"] == pytest.approx(16 / 19.5)


def test_existing_hull_is_used_when_it_fits_the_side_limit():
    block = builtin_registry().create(
        "inspect.bounding_convex_polygon", parameters={"maximum_sides": 8}
    )
    result = block.execute({"points": [[2, 2], *_PENTAGON, _PENTAGON[0]]})
    assert result is not None, block.error
    assert result.outputs["measurements"]["method"] == "hull"
    assert set(map(tuple, result.outputs["points"])) == set(map(tuple, _PENTAGON))
    assert result.outputs["area"] == 16 and result.outputs["fill_ratio"] == 1
    assert block.execute({"points": [[0, 0], [1, 1], [2, 2]]}) is None
    assert "nonzero area" in block.error


@pytest.mark.parametrize("extent", (0.1, 10))
def test_rectangle_fallback_rejects_non_enclosing_or_oversized_native_polygon(monkeypatch, extent):
    block = builtin_registry().create(
        "inspect.bounding_convex_polygon", parameters={"maximum_sides": 4}
    )

    def inaccurate(hull, _sides):
        radius = float(np.max(np.abs(hull))) * extent
        return 4 * radius**2, np.array(
            [[-radius, -radius], [radius, -radius], [radius, radius], [-radius, radius]], np.float32
        )

    monkeypatch.setattr(cv2, "minEnclosingConvexPolygon", inaccurate)
    result = block.execute({"points": _PENTAGON})
    assert result is not None, block.error
    assert result.outputs["measurements"]["method"] == "rectangle"
    assert result.outputs["area"] == pytest.approx(20)
