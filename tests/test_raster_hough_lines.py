"""Verify raster Hough voting, selections, and downstream line overlays."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import metadata

TYPE_ID = "inspect.hough_polar_lines"
_PROJECT = Path(__file__).parents[1] / "examples/geometry/hough_polar_lines/hough_polar_lines.json"


def _edges():
    image = np.zeros((96, 144), np.uint8)
    image[22, 10:118] = 40
    image[40:88, 130] = 220
    return image


@pytest.mark.parametrize(
    ("mode", "votes", "rho", "angle", "support", "endpoints"),
    [
        ("binary", 108, 22, 90, 108, [[117, 22], [10, 22]]),
        ("weighted", 10560, 130, 0, 48, [[130, 40], [130, 87]]),
    ],
)
def test_edge_strength_changes_peak_order(mode, votes, rho, angle, support, endpoints):
    image = _edges()
    block = builtin_registry().create(TYPE_ID, parameters={"vote_mode": mode, "maximum_lines": 1})
    block.exposed_outputs.append("result_graphics")
    result = block.execute({"image": image})
    assert result and result.passed, block.error
    assert result.outputs["count"] == 1 and result.outputs["edge_pixel_count"] == 156
    (peak,) = result.outputs["polar_lines"]
    assert peak["votes"] == votes and peak["support_count"] == support
    assert peak["rho_pixels"] == rho
    assert peak["normal_angle_degrees"] == pytest.approx(angle, abs=1e-5)
    (line,) = result.outputs["collection"]["items"]
    assert line["metrics"]["strength"] == votes
    np.testing.assert_allclose(line["geometry"]["points"], endpoints, atol=1e-5)
    assert len(result.outputs["result_graphics"].layers[0].objects) == 1
    np.testing.assert_array_equal(image, _edges())


def test_mask_and_region_apply_before_voting_with_inclusive_weighted_threshold():
    image = _edges()
    mask = np.full_like(image, 255)
    mask[40, 130] = 0
    region = {
        "model": "image_region_v1",
        "image_size": [144, 96],
        "shape": "rectangle",
        "rectangle": [125, 39, 10, 55],
    }
    block = builtin_registry().create(
        TYPE_ID,
        parameters={
            "vote_mode": "weighted",
            "minimum_votes": 47 * 220,
            "theta_max_degrees": 1,
            "theta_step_degrees": 2,
        },
    )
    inputs = {"image": image, "mask": mask, "region": region}
    result = block.execute(inputs)
    assert result and result.passed, block.error
    assert result.outputs["edge_pixel_count"] == 47
    assert result.outputs["polar_lines"] == [
        {"votes": 10340, "rho_pixels": 130.0, "normal_angle_degrees": 0.0, "support_count": 47}
    ]
    (line,) = result.outputs["collection"]["items"]
    assert line["geometry"]["points"] == [[130, 41], [130, 87]]
    block.parameters["minimum_votes"] += 1
    result = block.execute(inputs)
    assert result and not result.passed
    assert result.outputs["polar_lines"] == [] and result.outputs["count"] == 0
    assert result.fail_reasons[0]["code"] == "hough_polar_lines_below_minimum"


def test_empty_edges_and_invalid_input_or_search_settings():
    block = builtin_registry().create(TYPE_ID, parameters={"minimum_lines": 0})
    result = block.execute({"image": np.zeros_like(_edges())})
    assert result and result.passed and result.outputs["count"] == 0
    assert block.execute({"image": np.zeros((96, 144, 3), np.uint8)}) is None
    assert "8-bit monochrome" in block.error
    block.parameters.update(rho_step=0.01, theta_step_degrees=0.01)
    assert block.execute({"image": _edges()}) is None
    assert "Hough accumulator" in block.error
    block.parameters.update(theta_min_degrees=60, theta_max_degrees=60)
    assert block.execute({"image": _edges()}) is None
    assert "Theta maximum" in block.error


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_example_selects_and_draws_different_lines(profile):
    graph = Graph.load(_PROJECT, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    source = graph.blocks["source"].last_result.outputs["image"]
    for name, votes, (x, y) in (("binary", 108, (12, 22)), ("weighted", 10560, (130, 42))):
        ranked = graph.blocks[name].last_result.outputs["collection"]
        assert ranked["context"]["spatial"] == metadata(source)["spatial"]
        assert ranked["items"][0]["metrics"]["strength"] == votes
        drawn = graph.blocks[f"draw_{name}"].last_result.outputs
        assert drawn["count"] == 1
        assert np.any(drawn["overlay"][y, x] != source[y, x])
