"""Exercise ranking, clearance and diagnostics of image-plane pick candidates."""

from pathlib import Path

import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.poses import local_offset, make_pose


def match(x, y, *, score=0.95, angle=0):
    return {"score": score, "pose": make_pose(local_offset(x, y, angle), [128, 96], {})}


def test_pick_selection_checks_neighbors_angle_occupancy_and_score():
    image = np.zeros((96, 128), np.uint8)
    occupancy = np.zeros_like(image)
    occupancy[63, 109] = 255
    occupancy[50, 20] = 255  # Occupied pixels within the selected part are expected.
    matches = [
        match(22, 50),
        match(55, 74),
        match(68, 74, score=0.2),
        match(100, 74, angle=90),
        match(100, 63),
    ]
    block = builtin_registry().create(
        "inspect.select_pick",
        parameters={
            "footprint": [-5, -5, 10, 10],
            "clearance": 4,
            "angle_tolerance": 10,
            "require_occupancy": True,
        },
    )
    result = block.execute({"image": image, "matches": matches, "occupancy": occupancy})
    assert result is not None, block.error
    candidates = result.outputs["candidates"]
    assert result.outputs["count"] == 1
    assert result.outputs["pose"]["origin"] == [22, 50]
    assert candidates[1]["blocking_indices"] == [2]
    assert "score_below_minimum" in candidates[2]["reasons"]
    assert "angle_out_of_range" in candidates[3]["reasons"]
    assert "occupancy_in_clearance" in candidates[4]["reasons"]
    assert candidates[0]["occupied_pixels"] == 0


def test_pick_selection_ranking_symmetry_and_no_pick():
    image = np.zeros((96, 128), np.uint8)
    matches = [match(24, 24, score=0.99), match(50, 64, angle=179), match(90, 64, angle=179)]
    block = builtin_registry().create(
        "inspect.select_pick",
        parameters={"footprint": [-4, -4, 8, 8], "angle_period": "180", "angle_tolerance": 2},
    )
    inputs = {
        "image": image,
        "matches": {"model": "pattern_matches_v1", "items": matches, "count": 3},
    }
    result = block.execute(inputs)
    assert result is not None, block.error
    assert result.outputs["selected"]["index"] == 1
    block.parameters["ranking"] = "score"
    assert block.execute(inputs).outputs["selected"]["index"] == 0
    block.parameters["clearance"] = 70
    failure = block.execute(inputs)
    assert failure is not None, block.error
    assert not failure.passed
    assert failure.outputs["selected"] is NO_VALUE
    assert failure.outputs["pose"] is NO_VALUE
    assert failure.fail_reasons[0]["code"] == "no_pick_candidate"


def test_pick_project_runs_with_masked_downstream_part():
    path = Path(__file__).resolve().parents[1] / "examples/conveyor_pick/conveyor_pick.json"
    graph = Graph.load(path, builtin_registry())
    try:
        graph.run()
        result = graph.blocks["pick"].last_result
        assert result is not None, graph.blocks["pick"].error
        assert result.passed
        assert result.outputs["count"] == 24
        assert result.outputs["selected"]["pose"]["origin"][1] == 259.5
        rejected = [candidate for candidate in result.outputs["candidates"] if candidate["reasons"]]
        assert [candidate["reasons"] for candidate in rejected] == [["occupancy_in_clearance"]]
    finally:
        graph.close()
