"""Verify native polar results, point selection, and line collection wiring."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ValuePacket, frame, metadata

TYPE_ID = "inspect.hough_point_lines"
_PROJECT = Path(__file__).parents[1] / "examples/geometry/point_set_lines/point_set_lines.json"


def _samples():
    vertical = [[12, y] for y in range(5, 65, 3)]
    horizontal = [[x, 52] for x in range(18, 110, 4)]
    return vertical + horizontal + [[87, 7], [105, 15], [17, 41], [39, 68]]


def test_image_values_do_not_vote_and_segments_span_nearby_points():
    pixels = np.full((80, 128), 255, np.uint8)
    spatial = frame([128, 80], "camera")
    image = ImageArray(pixels, "gray", {"spatial": spatial})
    block = builtin_registry().create(TYPE_ID, parameters={"minimum_votes": 16, "maximum_lines": 2})
    block.exposed_outputs.append("result_graphics")
    result = block.execute(
        {"image": image, "points": ValuePacket(_samples(), {"spatial": spatial})}
    )
    assert result and result.passed, block.error
    assert result.outputs["point_count"] == 47
    assert result.outputs["count"] == 2
    horizontal, vertical = result.outputs["polar_lines"]
    assert [horizontal["votes"], vertical["votes"]] == [23, 20]
    assert [horizontal["support_count"], vertical["support_count"]] == [23, 20]
    assert [horizontal["rho_pixels"], vertical["rho_pixels"]] == [52, 12]
    assert [horizontal["normal_angle_degrees"], vertical["normal_angle_degrees"]] == pytest.approx(
        [90, 0], abs=1e-5
    )
    data = result.outputs["collection"]
    assert data["context"]["spatial"] == spatial and data["kind"] == "lines"
    assert [line["metrics"]["strength"] for line in data["items"]] == [23, 20]
    segments = [line["geometry"]["points"] for line in data["items"]]
    np.testing.assert_allclose(segments, [[[106, 52], [18, 52]], [[12, 5], [12, 62]]], atol=1e-5)
    assert len(result.outputs["result_graphics"].layers[0].objects) == 2


def test_mask_and_region_restrict_voting_and_minimum_votes_is_inclusive():
    image = np.zeros((80, 128), np.uint8)
    mask = np.full_like(image, 255)
    mask[5, 12] = 0
    region = {
        "model": "image_region_v1",
        "image_size": [128, 80],
        "shape": "rectangle",
        "rectangle": [9, 0, 6, 70],
    }
    registry = builtin_registry()
    block = registry.create(TYPE_ID, parameters={"minimum_votes": 19})
    inputs = {"image": image, "points": _samples(), "mask": mask, "region": region}
    result = block.execute(inputs)
    assert result and result.passed, block.error
    assert result.outputs["point_count"] == 19
    assert result.outputs["polar_lines"][0]["votes"] == 19
    assert result.outputs["collection"]["items"][0]["geometry"]["points"] == [[12, 8], [12, 62]]
    block.parameters["minimum_votes"] = 20
    result = block.execute(inputs)
    assert result and not result.passed
    assert result.outputs["polar_lines"] == [] and result.outputs["count"] == 0
    assert result.fail_reasons[0]["code"] == "hough_point_lines_below_minimum"


def test_custom_signed_rho_and_normal_angle_range():
    block = builtin_registry().create(
        TYPE_ID,
        parameters={
            "rho_mode": "custom",
            "rho_min": -10,
            "rho_max": -5,
            "rho_step": 0.5,
            "theta_min_degrees": 135,
            "theta_max_degrees": 136,
            "theta_step_degrees": 2,
            "minimum_votes": 20,
        },
    )
    points = [[y + 12, y] for y in range(3, 23)]
    result = block.execute({"image": np.zeros((48, 48), np.uint8), "points": points})
    assert result and result.passed, block.error
    line = result.outputs["polar_lines"][0]
    assert line["rho_pixels"] == -8.5 and line["votes"] == 20
    assert line["normal_angle_degrees"] == pytest.approx(135, abs=1e-5)
    endpoints = np.asarray(result.outputs["collection"]["items"][0]["geometry"]["points"])
    np.testing.assert_allclose(endpoints @ (np.array([-1, 1]) / np.sqrt(2)), -8.5, atol=1e-5)


def test_empty_votes_and_invalid_accumulator_or_frame():
    registry = builtin_registry()
    source = ImageArray(np.zeros((80, 128), np.uint8), "gray", {"spatial": frame([128, 80], "a")})
    block = registry.create(TYPE_ID, parameters={"minimum_lines": 0})
    empty = block.execute({"image": source, "points": []})
    assert empty and empty.passed and empty.outputs["count"] == 0
    points = ValuePacket(_samples(), {"spatial": frame([128, 80], "b")})
    assert block.execute({"image": source, "points": points}) is None
    assert "coordinate frame" in block.error
    assert block.execute({"image": source, "points": [[128, 3]]}) is None
    assert "within the input image" in block.error
    block.parameters.update(rho_step=0.01, theta_step_degrees=0.01)
    assert block.execute({"image": source, "points": _samples()}) is None
    assert "Hough accumulator" in block.error


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_saved_example_sorts_and_renders_line_collection(profile):
    graph = Graph.load(_PROJECT, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    source = graph.blocks["source"].last_result.outputs["image"]
    ranked = graph.blocks["sort"].last_result.outputs["collection"]
    assert [item["metrics"]["strength"] for item in ranked["items"]] == [23, 20]
    assert ranked["context"]["spatial"] == metadata(source)["spatial"]
    drawn = graph.blocks["draw"].last_result.outputs
    assert drawn["count"] == 2
    assert source[52, 20] == 0 and np.any(drawn["overlay"][52, 20] != 0)
