"""Validate native contour measurements and collection interoperability."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame

CONTOUR = [[0, 0], [8, 0], [8, 8], [4, 4], [0, 8]]


def test_compare_contours_accepts_detector_collections_and_shape_tolerance():
    registry = builtin_registry()
    detector = registry.create("inspect.contours", parameters={"min_area": 20, "expected_count": 1})
    mask = np.zeros((64, 64), np.uint8)
    mask[8:24, 8:24] = 255
    reference = detector.execute({"image": mask}).outputs["collection"]
    mask.fill(0)
    mask[12:44, 12:44] = 255
    sample = detector.execute({"image": mask}).outputs["collection"]
    block = registry.create(
        "inspect.compare_contours", parameters={"check_distance": True, "maximum_distance": 0.01}
    )
    equal = block.execute({"reference": reference, "sample": sample})
    assert equal is not None, block.error
    assert equal.passed and equal.outputs["distance"] < 1e-6
    changed = block.execute({"reference": reference, "sample": CONTOUR})
    assert changed is not None, block.error
    assert not changed.passed


def test_hull_measures_concavity_in_pixels():
    block = builtin_registry().create("inspect.contour_hull")
    result = block.execute({"contour": [*CONTOUR, CONTOUR[0]]})
    assert result is not None, block.error
    assert result.outputs["count"] == 1
    assert result.outputs["solidity"] == pytest.approx(0.75)
    defect = result.outputs["defects"][0]
    assert defect["deepest"] == [4, 4]
    assert defect["depth"] == 4


def test_point_contour_distance_filters_points_and_checks_frames():
    block = builtin_registry().create("inspect.contour_points", parameters={"minimum_distance": 1})
    points = [[2, 2], [9, 4], [8, 4]]
    payload = {"spatial": frame([16, 16], "frame-a")}
    contour = ValuePacket(CONTOUR, payload)
    result = block.execute({"contour": contour, "points": points})
    assert result is not None, block.error
    assert result.outputs["points"] == [[2, 2]]
    assert [row["signed_distance"] for row in result.outputs["measurements"]] == [2, -1, 0]
    assert result.payloads["points"]["spatial"] == payload["spatial"]
    other = ValuePacket(points, {"spatial": frame([16, 16], "frame-b")})
    assert block.execute({"contour": contour, "points": other}) is None
    assert "different coordinate frames" in block.error


def test_contour_approximation_uses_collection_frame_and_aligned_overlay():
    registry = builtin_registry()
    detect = registry.create(
        "inspect.contours", parameters={"approximation": "none", "expected_count": 1}
    )
    mask = np.zeros((64, 64), np.uint8)
    mask[8:24, 12:32] = 255
    spatial = frame([64, 64], "inspection")
    image = ValuePacket(mask, {"spatial": spatial})
    result = detect.execute({"image": image})
    assert result is not None and result.passed, detect.error
    collection = ValuePacket(result.outputs["collection"], result.payloads["collection"])
    approx = registry.create(
        "inspect.approximate_contour",
        parameters={"epsilon_mode": "length_fraction", "length_fraction": 0.01},
    )
    result = approx.execute({"contour": collection, "image": image})
    assert result is not None and result.passed, approx.error
    assert result.outputs["vertex_count"] == 4
    measurements = result.outputs["measurements"]
    assert measurements["source_vertices"] == 68
    assert measurements["epsilon_pixels"] == pytest.approx(0.68)
    assert measurements["length"] == 68 and measurements["area"] == 285
    assert result.payloads["points"]["spatial"] == spatial
    assert result.payloads["overlay"]["spatial"] == spatial
    assert tuple(result.outputs["overlay"][8, 12]) == (70, 220, 140)
    assert result.drawings["simplified contour"].objects[0].closed
    wrong_frame = ValuePacket(mask, {"spatial": frame([64, 64], "other")})
    assert approx.execute({"contour": collection, "image": wrong_frame}) is None
    assert approx.error_kind == "invalid"


def test_open_contour_preserves_endpoints_and_skips_area():
    points = [[0.5, 0], [1, 0.1], [2, 0], [3.5, 0]]
    block = builtin_registry().create(
        "inspect.approximate_contour", parameters={"closed": False, "epsilon_pixels": 0.2}
    )
    result = block.execute({"contour": points})
    assert result is not None and result.passed, block.error
    assert result.outputs["points"] == [points[0], points[-1]]
    assert result.outputs["measurements"]["length"] == 3
    assert result.outputs["measurements"]["area"] is None
    assert result.outputs["overlay"] is NO_VALUE
    assert not result.drawings["simplified contour"].objects[0].closed


def test_contour_approximation_reports_collapsed_polygon():
    block = builtin_registry().create(
        "inspect.approximate_contour", parameters={"epsilon_pixels": 100}
    )
    result = block.execute({"contour": CONTOUR})
    assert result is not None, block.error
    assert not result.passed
    assert result.fail_reasons[0]["code"] == "contour_approximation_degenerate"
    assert result.outputs["vertex_count"] < 3
