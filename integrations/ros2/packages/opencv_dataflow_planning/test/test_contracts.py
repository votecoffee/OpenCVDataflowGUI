import copy
import json

import pytest

from opencv_dataflow_planning.benchmark import summarize
from opencv_dataflow_planning.contracts import (
    decode,
    moveit_request,
    plan,
    request,
    scene_digest,
    trajectory_preview,
)


def configuration():
    return {
        "group_name": "arm",
        "frame_id": "base_link",
        "joint_names": ["a", "b"],
        "joint_types": ["revolute", "prismatic"],
        "start": [0.0, 0.0],
        "goal": [0.2, 0.1],
        "tolerance": [0.001, 0.001],
        "start_tolerance": 0.0001,
        "workspace_min": [-1.0, -1.0, -1.0],
        "workspace_max": [1.0, 1.0, 1.0],
        "allowed_planning_time": 0.1,
        "velocity_scale": 0.2,
        "acceleration_scale": 0.2,
        "pipeline_id": "",
        "planner_id": "",
    }


def scene():
    return {
        "robot_model_name": "fixture",
        "robot_state": {
            "joint_state": {
                "header": {"frame_id": "base_link", "stamp": {"sec": 4, "nanosec": 5}},
                "name": ["a", "b", "finger"],
                "position": [0.0, 0.0, 0.04],
            },
            "attached_collision_objects": [{"object": {"id": "fixture_tool"}}],
        },
        "world": {"collision_objects": [{"id": "bin", "height": 0.2}]},
    }


def response():
    points = [
        {"positions": [0.0, 0.0], "time_from_start": {"sec": 0, "nanosec": 0}},
        {"positions": [0.1, 0.2], "time_from_start": {"sec": 1, "nanosec": 0}},
    ]
    return {
        "motion_plan_response": {
            "error_code": {"val": 1},
            "planning_time": 0.02,
            "group_name": "arm",
            "trajectory_start": scene()["robot_state"],
            "trajectory": {"joint_trajectory": {"joint_names": ["b", "a"], "points": points}},
        }
    }


def test_full_scene_start_and_attached_tool_are_retained():
    original = scene()
    before = copy.deepcopy(original)
    actual = moveit_request(configuration(), original)["motion_plan_request"]
    assert original == before
    assert actual["start_state"]["joint_state"]["position"] == [0.0, 0.0, 0.04]
    assert (
        actual["start_state"]["attached_collision_objects"]
        == before["robot_state"]["attached_collision_objects"]
    )
    assert actual["start_state"]["is_diff"] is False
    assert actual["max_velocity_scaling_factor"] == 0.2


@pytest.mark.parametrize(
    "changes",
    [
        {"velocity_scale": 0.0},
        {"acceleration_scale": 1.1},
        {"joint_names": ["a", "a"]},
        {"goal": [0.0]},
        {"goal": [float("nan"), 0.0]},
        {"joint_types": ["revolute", "degrees"]},
        {"workspace_max": [-2.0, 1.0, 1.0]},
    ],
)
def test_invalid_configuration_cannot_reach_moveit(changes):
    value = configuration()
    value.update(changes)
    with pytest.raises(ValueError):
        plan(value)


def test_zero_velocity_scale_is_not_silently_full_speed():
    value = configuration()
    value["velocity_scale"] = 0
    with pytest.raises(ValueError):
        moveit_request(value, scene())


def test_changed_robot_start_is_rejected():
    value = scene()
    value["robot_state"]["joint_state"]["position"][0] = 0.2
    with pytest.raises(ValueError, match="start does not match"):
        moveit_request(configuration(), value)


def test_scene_revision_ignores_header_time_but_catches_geometry_acm_and_state():
    original = scene()
    digest = scene_digest(original)
    value = copy.deepcopy(original)
    value["robot_state"]["joint_state"]["header"]["stamp"]["sec"] += 1
    assert scene_digest(value) == digest
    for key, replacement in (
        ("world", {"collision_objects": []}),
        ("allowed_collision_matrix", {"entry_names": ["tool", "bin"]}),
        ("robot_state", {"joint_state": {"position": [1.0]}}),
    ):
        value = copy.deepcopy(original)
        value[key] = replacement
        assert scene_digest(value) != digest


def test_reordered_trajectory_units_and_preview_metrics():
    original = response()
    metrics, result = trajectory_preview(configuration(), original)
    assert metrics["status"] == "preview" and metrics["planned_duration_s"] == 1
    assert metrics["joint_travel"] == [
        {"name": "a", "units": "rad", "distance": 0.2},
        {"name": "b", "units": "m", "distance": 0.1},
    ]
    assert metrics["execution_authorized"] is False
    result["group_name"] = "changed"
    assert original["motion_plan_response"]["group_name"] == "arm"


@pytest.mark.parametrize("case", ["time", "start", "goal", "joints", "group", "nan", "multidof"])
def test_invalid_success_responses_are_not_published_as_previews(case):
    value = response()
    r = value["motion_plan_response"]
    joint = r["trajectory"]["joint_trajectory"]
    if case == "time":
        joint["points"][1]["time_from_start"]["sec"] = 0
    elif case in ("start", "goal", "nan"):
        joint["points"][0 if case == "start" else 1]["positions"][0] = (
            float("nan") if case == "nan" else 3.0
        )
    elif case == "joints":
        joint["joint_names"] = ["a", "c"]
    elif case == "group":
        r["group_name"] = "other"
    else:
        r["trajectory"]["multi_dof_joint_trajectory"] = {"points": [{}]}
    with pytest.raises(ValueError):
        trajectory_preview(configuration(), value)


def test_no_plan_remains_a_provider_failure():
    value = response()
    value["motion_plan_response"]["error_code"]["val"] = -12
    metrics, preview = trajectory_preview(configuration(), value)
    assert metrics == {"status": "no_plan", "moveit_error_code": -12} and preview is None


def test_request_schema_rejects_duplicates_nonfinite_and_extra_commands():
    with pytest.raises(ValueError):
        decode('{"request_id":"a","request_id":"b"}')
    with pytest.raises(ValueError):
        decode('{"x":1e999}')
    value = {
        "schema": "opencv_dataflow_plan_request_v1",
        "operation": "snapshot",
        "request_id": "a",
        "run_id": "b",
    }
    assert request(value) == value
    value["execute"] = True
    with pytest.raises(ValueError):
        request(value)


def test_benchmark_does_not_count_a_preview_or_unknown_as_a_successful_pick():
    row = {
        "episode_id": "a",
        "case_id": "frozen_case",
        "seed": 12,
        "backend": "moveit",
        "warm": True,
        "planning_status": "preview",
        "planning_wall_s": 0.2,
        "simulated_outcome": "not_run",
        "simulated_cycle_s": None,
    }
    summary = summarize([row])["groups"][0]
    assert summary["preview_count"] == 1 and summary["simulated_attempts"] == 0
    assert summary["simulated_success_fraction"] is None and summary["not_run_or_unknown"] == 1
    second = {
        **row,
        "episode_id": "b",
        "planning_wall_s": 0.4,
        "simulated_outcome": "success",
        "simulated_cycle_s": 3.0,
    }
    summary = summarize([row, second])["groups"][0]
    assert summary["planning_latency_s"]["p50"] == pytest.approx(0.3)
    assert summary["simulated_attempts"] == 1 and summary["simulated_success_fraction"] == 1
    with pytest.raises(ValueError):
        summarize([row, row])
    assert json.dumps(summarize([row]), allow_nan=False)
