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


@pytest.mark.parametrize("direction", [1, -1])
def test_contour_profile_catches_notch_and_protrusion(direction):
    reference = [[0, 0], [20, 0], [20, 20], [0, 20]]
    sample = [[0, 0], [10, direction * 5], *reference[1:]]
    block = builtin_registry().create(
        "inspect.contour_profile", parameters={"tolerance": 4.0, "sample_spacing": 2.0}
    )
    result = block.execute({"reference": reference, "sample": sample})
    assert result is not None, block.error
    assert not result.passed
    assert result.outputs["maximum_deviation"] == pytest.approx(5)
    assert 0 < result.outputs["rms_deviation"] < 5
    assert result.outputs["violation_count"] > 0
    signed = result.outputs["profiles"]["sample_to_reference"]
    assert [10, direction * 5, direction * 5] in signed
    assert result.fail_reasons[0]["code"] == "contour_profile_out_of_tolerance"
    assert "out of tolerance" in result.drawings

    block.parameters["tolerance"] = 5.0
    result = block.execute({"reference": reference, "sample": sample})
    assert result is not None and result.passed, block.error
    assert result.outputs["violation_count"] == 0


def test_contour_profile_checks_alignment_and_bounds_sampling():
    points = [[0, 0], [20, 0], [20, 20], [0, 20]]
    reference = ValuePacket(points, {"spatial": frame([32, 32], "reference")})
    sample = ValuePacket(points, {"spatial": frame([32, 32], "sample")})
    block = builtin_registry().create("inspect.contour_profile")
    assert block.execute({"reference": reference, "sample": sample}) is None
    assert "same coordinate frame" in block.error
    block.parameters["alignment"] = "pixel_grid"
    block.exposed_outputs.append("result_graphics")
    result = block.execute({"reference": reference, "sample": sample})
    assert result is not None and result.passed, block.error
    assert result.outputs["maximum_deviation"] == 0
    assert result.payloads["profiles"]["spatial"]["root_id"] == "sample"
    assert all(
        layer.payload["spatial"]["root_id"] == "sample"
        for layer in result.outputs["result_graphics"].layers
    )
    wrong_size = ValuePacket(points, {"spatial": frame([64, 32], "sample")})
    assert block.execute({"reference": reference, "sample": wrong_size}) is None
    assert "matching pixel dimensions" in block.error

    too_large = [[0, 0], [20_000, 0], [20_000, 20_000], [0, 20_000]]
    assert block.execute({"reference": points, "sample": too_large}) is None
    assert "8192 samples" in block.error

    x = np.arange(2000)
    dense = np.vstack(
        (np.column_stack((x, np.zeros_like(x))), np.column_stack((x[::-1], np.ones_like(x))))
    ).tolist()
    assert block.execute({"reference": dense, "sample": dense}) is None
    assert "too dense" in block.error


def test_hull_measures_concavity_in_pixels():
    block = builtin_registry().create("inspect.contour_hull")
    result = block.execute({"contour": [*CONTOUR, CONTOUR[0]]})
    assert result is not None, block.error
    assert result.outputs["count"] == 1
    assert result.outputs["solidity"] == pytest.approx(0.75)
    defect = result.outputs["defects"][0]
    assert defect["deepest"] == [4, 4]
    assert defect["depth"] == 4


def test_hull_depth_gate_marks_shallow_and_qualifying_defects_in_frame():
    points = [[0, 0], [4, 2], [8, 0], [8, 8], [4, 4], [0, 8]]
    payload = {"spatial": frame([16, 16], "source")}
    packet = ValuePacket(points, payload)
    block = builtin_registry().create(
        "inspect.contour_hull",
        parameters={"minimum_depth": 3.0, "check_defect_count": True, "maximum_defects": 0},
    )
    block.exposed_outputs.append("result_graphics")
    result = block.execute({"contour": packet})
    assert result is not None, block.error
    assert not result.passed and result.outputs["count"] == 1
    assert result.outputs["maximum_depth"] == 4
    assert result.fail_reasons[0]["code"] == "too_many_convexity_defects"
    assert result.payloads["maximum_depth"]["spatial"] == payload["spatial"]
    layers = {layer.name: layer for layer in result.outputs["result_graphics"].layers}
    assert set(layers) == {"convex hull", "shallow defects", "qualifying defects"}
    assert layers["qualifying defects"].objects[0].points == ((4, 8), (4, 4))
    assert layers["shallow defects"].objects[0].points == ((4, 0), (4, 2))
    assert layers["convex hull"].payload["spatial"] == payload["spatial"]

    block.parameters["minimum_depth"] = 4.5
    result = block.execute({"contour": packet})
    assert result is not None and result.passed, block.error
    assert result.outputs["defects"] == [] and result.outputs["maximum_depth"] == 0

    block.parameters["measure_defects"] = False
    assert block.execute({"contour": packet}) is None
    assert block.error_kind == "invalid"
    block.parameters["check_defect_count"] = False
    result = block.execute({"contour": packet})
    assert result is not None and result.passed, block.error
    assert list(result.drawings) == ["convex hull"]


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


def test_contour_hierarchy_keeps_nested_holes_islands_and_sibling_links():
    image = np.zeros((90, 130), np.uint8)
    image[5:80, 5:80] = 255
    image[15:48, 15:48] = 0
    image[24:41, 24:41] = 255
    image[29:35, 29:35] = 0
    image[54:70, 54:70] = 0
    image[8:30, 90:112] = 255
    spatial = frame([130, 90], "source")
    block = builtin_registry().create("inspect.contour_hierarchy")
    result = block.execute({"image": ValuePacket(image, {"spatial": spatial})})
    assert result is not None and result.passed, block.error
    hierarchy = result.outputs["hierarchy"]
    rows = hierarchy["nodes"]
    assert result.outputs["count"] == 6 and result.outputs["hole_count"] == 3
    assert sorted(row["depth"] for row in rows) == [0, 0, 1, 1, 2, 3]
    assert {rows[index]["kind"] for index in hierarchy["root_ids"]} == {"outer"}
    assert len(hierarchy["root_ids"]) == 2
    for row in rows:
        if row["next_id"] is not None:
            assert rows[row["next_id"]]["previous_id"] == row["id"]
            assert rows[row["next_id"]]["parent_id"] == row["parent_id"]
        if row["child_ids"]:
            assert row["first_child_id"] in row["child_ids"]
        for child in row["child_ids"]:
            assert rows[child]["parent_id"] == row["id"]
            assert rows[child]["depth"] == row["depth"] + 1
    island = next(row for row in rows if row["kind"] == "island")
    assert [rows[child]["kind"] for child in island["child_ids"]] == ["hole"]
    assert [item["id"] for item in result.outputs["collection"]["items"]] == list(range(6))
    assert result.payloads["collection"]["spatial"] == spatial
    assert result.payloads["overlay"]["spatial"] == spatial
    assert len(result.drawings["contour hierarchy"].objects) == 6
    assert tuple(result.outputs["overlay"][5, 5]) == (80, 230, 140)

    selection = np.full_like(image, 255)
    selection[:, 85:] = 0
    result = block.execute({"image": image, "mask": selection})
    assert result is not None, block.error
    assert result.outputs["count"] == 5 and result.outputs["hole_count"] == 3
    assert len(result.outputs["hierarchy"]["root_ids"]) == 1


def test_contour_hierarchy_handles_empty_masks_and_optional_overlay():
    block = builtin_registry().create(
        "inspect.contour_hierarchy", parameters={"draw_overlay": False}
    )
    result = block.execute({"image": np.zeros((12, 16), bool)})
    assert result is not None and result.passed, block.error
    assert result.outputs["hierarchy"]["nodes"] == []
    assert result.outputs["collection"]["items"] == []
    assert result.outputs["count"] == result.outputs["hole_count"] == 0
    assert result.outputs["overlay"] is NO_VALUE
