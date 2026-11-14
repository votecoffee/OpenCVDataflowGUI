"""Explicit private-domain DDS fixture; no MoveIt planner, simulator or motion is run."""

import json
import os
import sys
import time
import uuid
from pathlib import Path

os.environ["ROS_DOMAIN_ID"] = "129"
os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
os.environ.pop("ROS_STATIC_PEERS", None)

import rclpy  # noqa: E402
from moveit_msgs.msg import DisplayTrajectory  # noqa: E402
from moveit_msgs.srv import GetMotionPlan, GetPlanningScene  # noqa: E402
from rclpy.callback_groups import ReentrantCallbackGroup  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from trajectory_msgs.msg import JointTrajectoryPoint  # noqa: E402

from opencv_dataflow_planning.bridge import make_node  # noqa: E402


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


def main():
    tag = "/planning_fixture_" + uuid.uuid4().hex
    arguments = ["--ros-args"]
    for key, value in {
        "request_topic": tag + "/request",
        "result_topic": tag + "/result",
        "preview_topic": tag + "/preview",
        "scene_service": tag + "/scene",
        "plan_service": tag + "/plan",
        "robot_revision": "fixture-v1",
        "max_wait_seconds": "0.5",
    }.items():
        arguments += ["-p", f"{key}:={value}"]
    rclpy.init(args=arguments)
    fixture = Node("planning_fixture")
    bridge = make_node()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(fixture)
    executor.add_node(bridge)
    replies, previews = [], []
    publisher = fixture.create_publisher(String, tag + "/request", 10)
    fixture.create_subscription(
        String, tag + "/result", lambda m: replies.append(json.loads(m.data)), 10
    )
    fixture.create_subscription(DisplayTrajectory, tag + "/preview", previews.append, 10)
    controls = {"bad_time": False, "slow": False, "calls": 0}

    def get_scene(req, result):
        result.scene.robot_model_name = "fixture"
        result.scene.robot_state.joint_state.name = ["a", "b", "finger"]
        result.scene.robot_state.joint_state.position = [0.0, 0.0, 0.04]
        return result

    def get_plan(req, result):
        controls["calls"] += 1
        if controls["slow"]:
            time.sleep(0.9)
        value = req.motion_plan_request
        assert value.start_state.joint_state.name == ["a", "b", "finger"]
        assert value.start_state.joint_state.position[-1] == 0.04
        response = result.motion_plan_response
        response.group_name = value.group_name
        response.trajectory_start = value.start_state
        response.error_code.val = 1
        response.planning_time = 0.01
        joint = response.trajectory.joint_trajectory
        joint.joint_names = ["a", "b"]
        for positions, seconds in (([0.0, 0.0], 0), ([0.2, 0.1], 0 if controls["bad_time"] else 1)):
            point = JointTrajectoryPoint()
            point.positions = positions
            point.time_from_start.sec = seconds
            joint.points.append(point)
        return result

    fixture.create_service(GetPlanningScene, tag + "/scene", get_scene)
    fixture.create_service(
        GetMotionPlan, tag + "/plan", get_plan, callback_group=ReentrantCallbackGroup()
    )

    def until(condition, timeout=6):
        end = time.monotonic() + timeout
        while not condition() and time.monotonic() < end:
            executor.spin_once(timeout_sec=0.02)
        assert condition(), {"replies": replies, "job": bridge.job}

    def send(operation, identity, run="run1", **kwargs):
        row = {
            "schema": "opencv_dataflow_plan_request_v1",
            "operation": operation,
            "request_id": identity,
            "run_id": run,
            **kwargs,
        }
        publisher.publish(String(data=json.dumps(row)))

    def reply(identity, status):
        return any(r["request_id"] == identity and r["status"] == status for r in replies)

    try:
        until(
            lambda: (
                bridge.scene_client.service_is_ready()
                and bridge.plan_client.service_is_ready()
                and publisher.get_subscription_count() == 1
                and bridge.results.get_subscription_count() == 1
            )
        )
        send("snapshot", "s1")
        until(lambda: reply("s1", "snapshot"))
        snapshot = next(r for r in replies if r["request_id"] == "s1")
        common = {
            "server_id": snapshot["server_id"],
            "robot_revision": "fixture-v1",
            "scene_revision": snapshot["scene_revision"],
            "plan": configuration(),
        }
        send("plan", "p1", **common)
        until(lambda: reply("p1", "preview") and len(previews) == 1)
        send("plan", "p1", **common)
        until(lambda: reply("p1", "duplicate_request"))
        send("plan", "changed", **{**common, "scene_revision": "0" * 64})
        until(lambda: reply("changed", "scene_changed"))
        assert controls["calls"] == 1
        controls["bad_time"] = True
        send("plan", "bad", **common)
        until(lambda: reply("bad", "invalid_result"))
        controls.update(bad_time=False, slow=True)
        send("plan", "slow", **common)
        until(lambda: reply("slow", "timed_out"))
        send("reset", "reset_busy", run="run2", server_id=snapshot["server_id"])
        until(lambda: reply("reset_busy", "busy"))
        until(lambda: bridge.job is None)
        assert len(previews) == 1 and not reply("slow", "preview")
        send("reset", "reset", run="run2", server_id=snapshot["server_id"])
        until(lambda: reply("reset", "reset"))
        send("snapshot", "old_run")
        until(lambda: reply("old_run", "run_changed"))
        send("snapshot", "s2", run="run2")
        until(lambda: reply("s2", "snapshot"))
        send("plan", "cancelled", run="run2", **common)
        until(
            lambda: (
                bridge.job is not None
                and bridge.job["stage"] == "plan"
                and bridge.job["future"] is not None
            )
        )
        send("cancel", "cancelled", run="run2", server_id=snapshot["server_id"])
        until(lambda: reply("cancelled", "cancelled"))
        until(lambda: bridge.job is None)
        assert len(previews) == 1 and not reply("cancelled", "preview")
        receipt = {
            "native_ros": True,
            "domain": 129,
            "fixture_only": True,
            "real_planner": False,
            "robot_motion": False,
            "simulator": False,
            "snapshot_and_preview": True,
            "retained_other_joints": True,
            "duplicate_and_scene_drift_rejected": True,
            "invalid_success_rejected": True,
            "timeout_cancel_drain_without_late_preview": True,
            "run_reset_checked": True,
            "plan_service_calls": controls["calls"],
            "published_previews": len(previews),
        }
        if len(sys.argv) > 1:
            Path(sys.argv[1]).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(receipt), flush=True)
    finally:
        executor.shutdown(timeout_sec=5)
        bridge.destroy_node()
        fixture.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
