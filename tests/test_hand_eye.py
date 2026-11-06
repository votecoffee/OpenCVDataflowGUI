"""Verify transform direction, ticket pairing and hand-eye quality gates."""

import copy
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import pose_matrix, pose_record, rigid
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket
from opencv_dataflow.tickets import TicketClock

_PROJECT = (
    Path(__file__).parents[1]
    / "examples/calibration/hand_eye_calibration/hand_eye_calibration.json"
)


def _expected(rvec, translation):
    return pose_matrix(cv2.Rodrigues(np.asarray(rvec, dtype=float))[0], translation)


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_saved_eye_in_hand_graph_recovers_camera_mount_and_current_world_pose(profile):
    graph = Graph.load(_PROJECT, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    outputs = graph.blocks["fit"].last_result.outputs
    camera = outputs["camera_to_gripper"]
    assert (camera["source_frame"], camera["target_frame"], camera["units"]) == (
        "camera",
        "gripper",
        "mm",
    )
    mount = _expected([0.18, -0.1, 0.22], [40, -25, 90])
    np.testing.assert_allclose(rigid(camera), mount, atol=1e-8)
    np.testing.assert_allclose(
        rigid(outputs["target_to_base"]), _expected([0.15, 0.05, -0.2], [170, 90, 820]), atol=1e-8
    )
    assert outputs["translation_rms_mm"] < 1e-8 and outputs["rotation_rms_degrees"] < 1e-8
    world = graph.blocks["world"].last_result.outputs["transform"]
    assert (world["source_frame"], world["target_frame"]) == ("camera", "base")
    expected = _expected([0.24, -0.07, 0.11], [215, 65, 267]) @ mount
    np.testing.assert_allclose(rigid(world), expected, atol=1e-8)


def test_ticketed_groups_pair_by_identity_even_when_one_arrives_in_reverse_order():
    registry = builtin_registry()
    fit = Graph.load(_PROJECT, registry).blocks["fit"]
    clock = TicketClock(session="calibration")
    robot, board = [], []
    for index, (gripper, target) in enumerate(
        zip(fit.parameters["gripper_poses"][:3], fit.parameters["target_poses"][:3])
    ):
        robot.append(ValuePacket(gripper, {"ticket": clock.issue("robot", "fixed", index)}))
        board.append(ValuePacket(target, {"ticket": clock.issue("camera", "fixed", index)}))
    result = fit.execute(
        {
            "gripper_poses": FlowGroup(tuple(robot), ("robot",) * 3),
            "target_poses": FlowGroup(tuple(reversed(board)), ("camera",) * 3),
        }
    )
    assert result is not None and result.passed, fit.error
    np.testing.assert_allclose(result.outputs["camera_to_gripper"]["translation"], [40, -25, 90])
    assert [sample["ticket"] for sample in result.outputs["report"]["samples"]] == [0, 1, 2]
    board[-1].payload["ticket"] = clock.issue("camera", "fixed", 99)
    assert (
        fit.execute(
            {
                "gripper_poses": FlowGroup(tuple(robot), ("robot",) * 3),
                "target_poses": FlowGroup(tuple(board), ("camera",) * 3),
            }
        )
        is None
    )
    assert "same sample tickets" in fit.error


def test_single_axis_motion_is_rejected():
    fit = builtin_registry().create("calibration.hand_eye")
    mount = _expected([0.18, -0.1, 0.22], [40, -25, 90])
    target = _expected([0.15, 0.05, -0.2], [170, 90, 820])
    grippers = [_expected([0, 0, angle], [200, 30, 250]) for angle in (0.1, 0.4, 0.7, 0.9)]
    assert (
        fit.execute(
            {
                "gripper_poses": [pose_record(g, "gripper", "base", "mm") for g in grippers],
                "target_poses": [
                    pose_record(
                        np.linalg.inv(mount) @ np.linalg.inv(g) @ target, "target", "camera", "mm"
                    )
                    for g in grippers
                ],
            }
        )
        is None
    )
    assert "two independent axes" in fit.error


@pytest.mark.parametrize("unit", ("mm", "m"))
def test_inconsistent_target_positions_fail_in_millimeters_and_suppress_transforms(unit):
    fit = Graph.load(_PROJECT, builtin_registry()).blocks["fit"]
    robot = copy.deepcopy(fit.parameters["gripper_poses"])
    target = copy.deepcopy(fit.parameters["target_poses"])
    target[-1]["source_to_target"][0][3] += 25
    if unit == "m":
        for pose in robot + target:
            matrix = rigid(pose)
            matrix[:3, 3] /= 1000
            pose.update(source_to_target=matrix.tolist(), units="m")
    result = fit.execute({"gripper_poses": robot, "target_poses": target})
    assert result is not None and not result.passed, fit.error
    assert result.outputs["translation_rms_mm"] > 2
    assert result.outputs["rotation_rms_degrees"] < 1e-8
    assert result.fail_reasons[0]["code"] == "hand_eye_translation_rms_exceeds_limit"
    assert result.outputs["camera_to_gripper"] is result.outputs["target_to_base"] is NO_VALUE


def test_inconsistent_target_orientation_has_its_own_failure_reason():
    fit = Graph.load(_PROJECT, builtin_registry()).blocks["fit"]
    fit.parameters["max_translation_rms_mm"] = 0
    target = copy.deepcopy(fit.parameters["target_poses"])
    disturbed = _expected([0.12, -0.08, 0.14], [0, 0, 0]) @ rigid(target[-1])
    target[-1] = pose_record(disturbed, "target", "camera", "mm")
    result = fit.execute({"target_poses": target})
    assert result is not None and not result.passed, fit.error
    assert result.outputs["rotation_rms_degrees"] > 1
    assert [reason["code"] for reason in result.fail_reasons] == [
        "hand_eye_rotation_rms_exceeds_limit"
    ]


@pytest.mark.parametrize("change", ({"source_frame": "camera"}, {"units": "m"}))
def test_wrong_target_direction_or_units_are_rejected(change):
    fit = Graph.load(_PROJECT, builtin_registry()).blocks["fit"]
    target = copy.deepcopy(fit.parameters["target_poses"])
    target[1].update(change)
    assert fit.execute({"target_poses": target}) is None
    assert "pose from target to camera in mm" in fit.error
