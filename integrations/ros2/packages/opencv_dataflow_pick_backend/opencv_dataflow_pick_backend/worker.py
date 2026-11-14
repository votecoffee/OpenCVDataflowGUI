"""Bounded JSON pipe server. The application, not this node, owns pick sequencing.

This executable deliberately supports only the allocated simulation fixture.
Stdout is a JSON-lines transport; ROS diagnostics remain on stderr.
"""

import argparse
import base64
import copy
import hashlib
import json
import os
import queue
import sys
import threading
import time
import uuid
from collections import deque

import numpy as np

from .kinematics import SerialChain, joint_samples

MAX_LINE = 24 * 1024 * 1024
FINGERS = ("gripper_left_finger_joint", "gripper_right_finger_joint")


class CandidateCollision(RuntimeError):
    """A completed read-only path candidate failed post-retiming sampling."""


def encode(value):
    return json.dumps(value, allow_nan=False, separators=(",", ":"))


def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)) or type(value).__module__ == "array":
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def matrix(position, quaternion):
    x, y, z, w = quaternion
    if not np.isfinite([*position, x, y, z, w]).all() or abs(np.linalg.norm(quaternion) - 1) > 0.01:
        raise ValueError("Use finite poses and a unit quaternion")
    result = np.eye(4)
    result[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    result[:3, 3] = position
    return result


def quaternion(rotation):
    # Stable eigensystem formulation; native adapter does not need SciPy.
    r = np.asarray(rotation, float)
    k = (
        np.array(
            [
                [
                    r[0, 0] - r[1, 1] - r[2, 2],
                    r[0, 1] + r[1, 0],
                    r[0, 2] + r[2, 0],
                    r[2, 1] - r[1, 2],
                ],
                [
                    r[0, 1] + r[1, 0],
                    r[1, 1] - r[0, 0] - r[2, 2],
                    r[1, 2] + r[2, 1],
                    r[0, 2] - r[2, 0],
                ],
                [
                    r[0, 2] + r[2, 0],
                    r[1, 2] + r[2, 1],
                    r[2, 2] - r[0, 0] - r[1, 1],
                    r[1, 0] - r[0, 1],
                ],
                [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1], np.trace(r)],
            ]
        )
        / 3
    )
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1]
    return q if q[3] >= 0 else -q


def image_array(message):
    types = {
        "rgb8": (np.dtype("u1"), 3),
        "bgr8": (np.dtype("u1"), 3),
        "32FC1": (np.dtype(">f4" if message.is_bigendian else "<f4"), 1),
    }
    if message.encoding not in types:
        raise ValueError("Fixture must publish rgb8/bgr8 and 32FC1 axial metres")
    dtype, channels = types[message.encoding]
    width, height = message.width, message.height
    if not 1 <= width <= 1920 or not 1 <= height <= 1080:
        raise ValueError("Image dimensions exceed the fixture limit")
    row = width * channels * dtype.itemsize
    raw = bytes(message.data)
    if message.step < row or len(raw) != height * message.step:
        raise ValueError("Image stride or byte count is invalid")
    rows = np.frombuffer(raw, np.uint8).reshape(height, message.step)[:, :row].copy()
    result = rows.view(dtype).reshape(
        (height, width, channels) if channels > 1 else (height, width)
    )
    if message.encoding == "rgb8":
        result = result[:, :, ::-1]
    return np.ascontiguousarray(result, dtype=np.float32 if channels == 1 else np.uint8)


def packed(array):
    value = np.ascontiguousarray(array)
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "base64": base64.b64encode(value.tobytes()).decode("ascii"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation-only", action="store_true", required=True)
    _, ros_args = parser.parse_known_args()
    if os.environ.get("ROS_DOMAIN_ID") != "130" or not os.environ.get(
        "GZ_PARTITION", ""
    ).startswith("opencv_bin_pick_"):
        raise RuntimeError(
            "Simulation backend requires allocated domain 130 and opencv_bin_pick_ partition"
        )
    import rclpy
    from action_msgs.msg import GoalStatus
    from control_msgs.action import FollowJointTrajectory
    from geometry_msgs.msg import Pose
    from moveit_msgs.action import ExecuteTrajectory
    from moveit_msgs.msg import (
        AllowedCollisionEntry,
        AttachedCollisionObject,
        CollisionObject,
        Constraints,
        JointConstraint,
        PlanningScene,
    )
    from moveit_msgs.srv import (
        ApplyPlanningScene,
        GetCartesianPath,
        GetMotionPlan,
        GetPlannerParams,
        GetPlanningScene,
        GetPositionFK,
        GetPositionIK,
        GetStateValidity,
        SetPlannerParams,
    )
    from rcl_interfaces.srv import GetParameters
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from rosidl_runtime_py.convert import message_to_ordereddict
    from sensor_msgs.msg import CameraInfo, Image, JointState
    from shape_msgs.msg import SolidPrimitive
    from std_srvs.srv import Trigger
    from tf2_ros import Buffer, TransformListener
    from trajectory_msgs.msg import JointTrajectoryPoint

    rclpy.init(args=ros_args)
    requests = queue.Queue(maxsize=4)
    cancelled = threading.Event()

    def read():
        while True:
            line = sys.stdin.buffer.readline(MAX_LINE + 1)
            if not line:
                cancelled.set()
                return
            try:
                if len(line) > MAX_LINE:
                    raise ValueError("Request exceeds transport limit")
                item = json.loads(
                    line,
                    parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")),
                )
                if item.get("operation") == "cancel":
                    cancelled.set()
                else:
                    requests.put_nowait(item)
            except Exception as error:
                print(str(error), file=sys.stderr, flush=True)

    class Backend(Node):
        def __init__(self):
            super().__init__("opencv_dataflow_pick_backend")
            self.server_id = str(uuid.uuid4())
            self.run_id = None
            self.environment = None
            self.seen = deque(maxlen=512)
            self.plans = {}
            self.frames = {}
            self.information = None
            self.active_goal = None
            self.poisoned = False
            self.chain = None
            self.robot_description_sha256 = None
            self.fk_verified = None
            self.collect_motion = False
            self.latest_joint_feedback = None
            self.motion_samples = []
            self.motion_dropped = 0
            self.motion_rewind = False
            self.planner_profile = None
            self.tf = Buffer(node=self)
            self.listener = TransformListener(self.tf, self)
            self.service_clients = {
                "scene": self.create_client(GetPlanningScene, "/get_planning_scene"),
                "apply": self.create_client(ApplyPlanningScene, "/apply_planning_scene"),
                "ik": self.create_client(GetPositionIK, "/compute_ik"),
                "fk": self.create_client(GetPositionFK, "/compute_fk"),
                "parameters": self.create_client(GetParameters, "/move_group/get_parameters"),
                "get_planner": self.create_client(GetPlannerParams, "/get_planner_params"),
                "set_planner": self.create_client(SetPlannerParams, "/set_planner_params"),
                "validity": self.create_client(GetStateValidity, "/check_state_validity"),
                "plan": self.create_client(GetMotionPlan, "/plan_kinematic_path"),
                "cartesian": self.create_client(GetCartesianPath, "/compute_cartesian_path"),
            }
            self.execute_client = ActionClient(self, ExecuteTrajectory, "/execute_trajectory")
            self.gripper_client = ActionClient(
                self,
                FollowJointTrajectory,
                "/gripper_controller/follow_joint_trajectory",
            )
            self.subs = []
            self.subs.append(
                self.create_subscription(
                    JointState, "/joint_states", self.joint_feedback, qos_profile_sensor_data
                )
            )
            for key in ("rgb", "depth"):
                self.frames[key] = deque(maxlen=4)
                self.subs.append(
                    self.create_subscription(
                        Image,
                        "/opencv_fixture/source/" + key,
                        lambda msg, k=key: self.frames[k].append((time.monotonic(), msg)),
                        qos_profile_sensor_data,
                    )
                )
            self.subs.append(
                self.create_subscription(
                    CameraInfo,
                    "/opencv_fixture/source/camera_info",
                    lambda msg: setattr(self, "information", (time.monotonic(), msg)),
                    qos_profile_sensor_data,
                )
            )

        def wait(self, future, seconds=30):
            until = time.monotonic() + seconds
            while not future.done() and time.monotonic() < until and not cancelled.is_set():
                rclpy.spin_once(self, timeout_sec=0.02)
            if not future.done() or cancelled.is_set():
                if self.active_goal is not None:
                    stop = self.active_goal.cancel_goal_async()
                    end = time.monotonic() + 3
                    while not stop.done() and time.monotonic() < end:
                        rclpy.spin_once(self, timeout_sec=0.02)
                # No automatic reuse after timeout/cancel; services may still complete.
                self.poisoned = True
                raise RuntimeError("cancelled_or_timed_out_outcome_requires_reconciliation")
            return future.result()

        def service(self, key, request, seconds=30):
            client = self.service_clients[key]
            if not client.wait_for_service(timeout_sec=3):
                raise RuntimeError("Service unavailable: " + client.srv_name)
            return self.wait(client.call_async(request), seconds)

        def trigger(self, name):
            if name not in self.service_clients:
                self.service_clients[name] = self.create_client(Trigger, name)
            reply = self.service(name, Trigger.Request())
            if not reply.success:
                raise RuntimeError(reply.message)
            return json.loads(reply.message)

        def guard(self):
            deadline = time.monotonic() + 2
            while True:
                status = self.trigger("/opencv_fixture/status")
                if (
                    status.get("simulation_only") is not True
                    or status.get("ros_domain_id") != "130"
                    or status.get("gz_partition") != os.environ["GZ_PARTITION"]
                ):
                    raise RuntimeError("Fixture isolation identity differs")
                identity = (status["session_id"], status["run_id"], status["scene_sha256"])
                if self.environment is not None and self.environment != identity:
                    self.poisoned = True
                    raise RuntimeError(
                        "Fixture restarted/reset/scene changed; start a new application run"
                    )
                self.environment = identity
                if status.get("ready") is True and status.get("physics", {}).get(
                    "time_recently_advancing"
                ):
                    return status
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "Fresh camera/robot/TF readiness did not recover within 2 seconds"
                    )
                # Reacquire readiness observations only; never replay a motion request.
                rclpy.spin_once(self, timeout_sec=0.03)

        def model(self):
            request = GetParameters.Request()
            request.names = ["robot_description"]
            response = self.service("parameters", request)
            if len(response.values) != 1 or response.values[0].type != 4:
                raise RuntimeError("MoveIt robot_description string is unavailable")
            text = response.values[0].string_value
            digest = hashlib.sha256(text.encode()).hexdigest()
            if self.chain is not None and digest != self.robot_description_sha256:
                self.poisoned = True
                raise RuntimeError("MoveIt robot description changed during this worker session")
            if self.chain is None:
                self.chain = SerialChain(text)
                self.robot_description_sha256 = digest
            return {
                **self.chain.record(),
                "robot_description_sha256": digest,
                "source": "MoveIt robot_description parameter; no mesh geometry",
            }

        def joint_feedback(self, message):
            self.latest_joint_feedback = (time.monotonic(), message)
            if not self.collect_motion or self.chain is None:
                return
            stamp = message.header.stamp.sec * 10**9 + message.header.stamp.nanosec
            if self.motion_samples:
                previous = self.motion_samples[-1]["stamp_ns"]
                if stamp < previous:
                    self.motion_rewind = True
                    return
                if stamp - previous < 20_000_000:
                    return
            try:
                positions = dict(zip(message.name, message.position))
                pose, _ = self.chain.forward(positions)
                if len(self.motion_samples) >= 2048:
                    self.motion_dropped += 1
                    return
                self.motion_samples.append(
                    {
                        "stamp_ns": stamp,
                        "position_m": pose[:3, 3].tolist(),
                        "joint_positions": [float(positions[n]) for n in self.chain.required],
                    }
                )
            except (ValueError, KeyError):
                self.motion_dropped += 1

        def validate_fk(self, solution, start):
            if self.fk_verified is not None:
                return self.fk_verified
            points = solution.joint_trajectory.points
            errors = []
            for index in sorted(set([0, len(points) // 2, len(points) - 1])):
                request = GetPositionFK.Request()
                request.header.frame_id = "world"
                request.fk_link_names = ["gripper_tcp"]
                request.robot_state = copy.deepcopy(start)
                positions = dict(zip(start.joint_state.name, start.joint_state.position))
                positions.update(
                    zip(solution.joint_trajectory.joint_names, points[index].positions)
                )
                request.robot_state.joint_state.position = [
                    float(positions[n]) for n in request.robot_state.joint_state.name
                ]
                response = self.service("fk", request)
                if (
                    response.error_code.val != 1
                    or response.fk_link_names != ["gripper_tcp"]
                    or len(response.pose_stamped) != 1
                ):
                    raise RuntimeError("Native FK service returned incomplete TCP geometry")
                native = response.pose_stamped[0]
                if native.header.frame_id != "world":
                    raise RuntimeError("Native FK returned another coordinate frame")
                p, q = native.pose.position, native.pose.orientation
                expected = matrix([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])
                actual, _ = self.chain.forward(positions)
                error = float(np.max(np.abs(expected - actual)))
                if error > 1e-7:
                    raise RuntimeError(f"Review FK differs from MoveIt: {error}")
                errors.append(error)
            self.fk_verified = {
                "source": "native GetPositionFK",
                "states_checked": len(errors),
                "maximum_matrix_error": max(errors),
            }
            return self.fk_verified

        def scene(self):
            request = GetPlanningScene.Request()
            request.components.components = 1023
            return self.service("scene", request).scene

        def transform(self, target, source, when):
            end = time.monotonic() + 5
            while not self.tf.can_transform(target, source, when):
                if cancelled.is_set() or time.monotonic() >= end:
                    raise RuntimeError(f"Fresh TF unavailable: {source} to {target}")
                rclpy.spin_once(self, timeout_sec=0.02)
            return self.tf.lookup_transform(target, source, when)

        def apply(self, scene):
            scene.is_diff = True
            request = ApplyPlanningScene.Request()
            request.scene = scene
            if not self.service("apply", request).success:
                raise RuntimeError("MoveIt rejected planning scene diff")

        def pose(self, transform):
            value = np.asarray(transform, float)
            if (
                value.shape != (4, 4)
                or not np.isfinite(value).all()
                or not np.allclose(value[3], [0, 0, 0, 1])
            ):
                raise ValueError("Expected finite rigid world-from-TCP matrix")
            if not np.allclose(
                value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-6
            ) or not np.isclose(np.linalg.det(value[:3, :3]), 1):
                raise ValueError("Pose rotation must be orthonormal and right handed")
            msg = Pose()
            msg.position.x, msg.position.y, msg.position.z = map(float, value[:3, 3])
            (
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ) = map(float, quaternion(value[:3, :3]))
            return msg

        def box(self, name, size, transform, frame="world"):
            obj = CollisionObject()
            obj.id = name
            obj.header.frame_id = frame
            obj.operation = obj.ADD
            primitive = SolidPrimitive()
            primitive.type = primitive.BOX
            primitive.dimensions = list(map(float, size))
            obj.primitives = [primitive]
            obj.primitive_poses = [self.pose(transform)]
            return obj

        def geometry_hash(self, scene):
            return hashlib.sha256(
                encode(
                    plain(
                        {
                            "world": message_to_ordereddict(scene.world),
                            "attachments": [
                                message_to_ordereddict(x)
                                for x in scene.robot_state.attached_collision_objects
                            ],
                            "acm": message_to_ordereddict(scene.allowed_collision_matrix),
                        }
                    )
                ).encode()
            ).hexdigest()

        def remember(self, solution, start, reference_digest, fraction=1.0):
            self.model()
            points = solution.joint_trajectory.points
            if not points or not solution.joint_trajectory.joint_names:
                raise RuntimeError("Planner returned an empty trajectory")
            times = [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in points]
            if any(b <= a for a, b in zip(times, times[1:])) or times[-1] <= 0:
                raise RuntimeError("Trajectory has invalid timing")
            current = self.scene()
            if self.geometry_hash(current) != reference_digest:
                raise RuntimeError("Planning geometry changed during the solve")
            states = joint_samples(
                plain(message_to_ordereddict(solution.joint_trajectory)),
                dict(zip(start.joint_state.name, start.joint_state.position)),
            )
            deadline = time.monotonic() + 20
            for state in states:
                if time.monotonic() > deadline:
                    raise RuntimeError("Post-retiming validity deadline exceeded")
                request = GetStateValidity.Request()
                request.group_name = (
                    ""  # Additional check covers the whole robot, including fingers.
                )
                request.robot_state = copy.deepcopy(start)
                request.robot_state.joint_state.position = [
                    float(state[n]) for n in start.joint_state.name
                ]
                checked = self.service("validity", request)
                if not checked.valid:
                    pairs = [f"{c.contact_body_1}/{c.contact_body_2}" for c in checked.contacts[:3]]
                    raise CandidateCollision("Retimed candidate collision: " + ", ".join(pairs))
            if self.geometry_hash(self.scene()) != reference_digest:
                raise RuntimeError("Planning geometry changed during post-retiming checks")
            key = str(uuid.uuid4())
            self.plans.clear()
            self.plans[key] = (
                solution,
                copy.deepcopy(start),
                self.geometry_hash(current),
            )
            review = self.chain.trajectory(
                plain(message_to_ordereddict(solution.joint_trajectory)),
                dict(zip(start.joint_state.name, start.joint_state.position)),
            )
            review["native_validation"] = self.validate_fk(solution, start)
            return {
                "plan_id": key,
                "trajectory": plain(message_to_ordereddict(solution)),
                "start_state": plain(message_to_ordereddict(start)),
                "fraction": fraction,
                "geometry_sha256": self.geometry_hash(current),
                "point_count": len(points),
                "duration_s": times[-1],
                "tcp_path": review,
                "post_retiming_checks": {
                    "states": len(states),
                    "maximum_joint_step": 0.01,
                    "interpolation": "linear in joint space",
                    "valid": True,
                    "continuous_collision_certificate": False,
                },
            }

        def action(self, client, goal, seconds=90):
            if not client.wait_for_server(timeout_sec=5):
                raise RuntimeError("Action server unavailable")
            handle = self.wait(client.send_goal_async(goal), 10)
            if not handle.accepted:
                raise RuntimeError("Action goal rejected")
            self.active_goal = handle
            try:
                result = self.wait(handle.get_result_async(), seconds)
            finally:
                self.active_goal = None
            if result.status != GoalStatus.STATUS_SUCCEEDED:
                raise RuntimeError("Action did not succeed: " + str(result.status))
            return result.result

        def dispatch(self, item):
            operation = item["operation"]
            p = item.get("parameters", {})
            status = self.guard()
            if operation == "hello":
                return {
                    "fixture": status,
                    "native_python": sys.executable,
                    "assisted": True,
                    "kinematic_model": self.model(),
                }
            if operation == "fixture_status":
                return status
            if operation == "configure_planner":
                name = "opencv_rrt_connect"
                before = GetPlannerParams.Request()
                before.pipeline_id = "ompl"
                before.planner_config = name
                before.group = status["planning_group"]
                previous = self.service("get_planner", before).params
                request = SetPlannerParams.Request()
                request.pipeline_id = "ompl"
                request.planner_config = name
                request.group = status["planning_group"]
                request.replace = True
                request.params.keys = ["type", "range", "longest_valid_segment_fraction"]
                request.params.values = ["geometric::RRTConnect", "0.25", "0.0001"]
                self.service("set_planner", request)
                actual = self.service("get_planner", before).params
                values = dict(zip(actual.keys, actual.values))
                if (
                    values.get("type") != "geometric::RRTConnect"
                    or float(values.get("range", 0)) != 0.25
                    or float(values.get("longest_valid_segment_fraction", 0)) != 0.0001
                ):
                    raise RuntimeError("Named OMPL profile read-back differs")
                self.planner_profile = {
                    "id": name,
                    "pipeline": "ompl",
                    "parameters": values,
                    "previous": dict(zip(previous.keys, previous.values)),
                    "scope": "new named profile in this isolated server; defaults unchanged",
                }
                return self.planner_profile
            if operation == "evaluation":
                return self.trigger("/opencv_fixture/evaluation/snapshot")
            if operation == "inspect_scene":
                scene = self.scene()
                tf = self.transform("world", status["tcp"], Time())
                t, q = tf.transform.translation, tf.transform.rotation
                deadline = time.monotonic() + 1
                while self.latest_joint_feedback is None and time.monotonic() < deadline:
                    rclpy.spin_once(self, timeout_sec=0.02)
                stationary = False
                if self.latest_joint_feedback is not None:
                    received, joints = self.latest_joint_feedback
                    velocities = dict(zip(joints.name, joints.velocity))
                    stationary = time.monotonic() - received <= 0.5 and all(
                        n in velocities and np.isfinite(velocities[n]) and abs(velocities[n]) < 0.02
                        for n in self.chain.required
                    )
                return {
                    "world_ids": [o.id for o in scene.world.collision_objects],
                    "attached_ids": [
                        o.object.id for o in scene.robot_state.attached_collision_objects
                    ],
                    "robot_state": plain(message_to_ordereddict(scene.robot_state)),
                    "world_from_tcp": matrix([t.x, t.y, t.z], [q.x, q.y, q.z, q.w]).tolist(),
                    "arm_stationary": bool(stationary),
                }
            if operation == "capture":
                start = time.monotonic()
                until = start + 8
                while time.monotonic() < until and not cancelled.is_set():
                    rclpy.spin_once(self, timeout_sec=0.03)
                    for received, depth in reversed(self.frames["depth"]):
                        if received <= start:
                            continue
                        stamp = (depth.header.stamp.sec, depth.header.stamp.nanosec)
                        matched = next(
                            (
                                rgb
                                for t, rgb in reversed(self.frames["rgb"])
                                if t > start
                                and (rgb.header.stamp.sec, rgb.header.stamp.nanosec) == stamp
                            ),
                            None,
                        )
                        if matched is None or self.information is None:
                            continue
                        info_time, info = self.information
                        if time.monotonic() - info_time > 0.5 or (
                            info.header.frame_id != depth.header.frame_id
                            or matched.header.frame_id != depth.header.frame_id
                            or info.width != depth.width
                            or info.height != depth.height
                        ):
                            continue
                        tf = self.transform(
                            "world",
                            depth.header.frame_id,
                            Time.from_msg(depth.header.stamp),
                        )
                        t, q = tf.transform.translation, tf.transform.rotation
                        return {
                            "capture_id": str(uuid.uuid4()),
                            "stamp_ns": stamp[0] * 10**9 + stamp[1],
                            "frame": depth.header.frame_id,
                            "rgb_bgr": packed(image_array(matched)),
                            "depth_m": packed(image_array(depth)),
                            "camera_k": list(info.k),
                            "world_from_camera": matrix(
                                [t.x, t.y, t.z], [q.x, q.y, q.z, q.w]
                            ).tolist(),
                            "source": "synchronized first-return simulator RGB-D; no object poses",
                        }
                raise RuntimeError("Fresh exact-stamp RGB-D/CameraInfo capture unavailable")
            if operation == "sync_scene":
                truth = self.trigger("/opencv_fixture/evaluation/snapshot")
                current = self.scene()
                attached = {o.object.id for o in current.robot_state.attached_collision_objects}
                diff = PlanningScene()
                for box in truth["static_boxes"]:
                    x, y, z, roll, pitch, yaw = box["pose"]
                    if abs(roll) + abs(pitch) + abs(yaw) > 1e-8:
                        raise ValueError("Static fixture boxes must be axis aligned")
                    transform = np.eye(4)
                    transform[:3, 3] = [x, y, z]
                    diff.world.collision_objects.append(self.box(box["id"], box["size"], transform))
                for obj in truth["objects"]:
                    if obj["object_id"] in attached:
                        continue
                    if obj["world_pose"] is None or obj["receive_age_s"] > 0.5:
                        raise RuntimeError("Fresh privileged collision geometry unavailable")
                    transform = matrix(
                        obj["world_pose"]["position_m"],
                        obj["world_pose"]["quaternion_xyzw"],
                    )
                    diff.world.collision_objects.append(
                        self.box(obj["object_id"], obj["size_m"], transform)
                    )
                self.apply(diff)
                return {
                    "collision_geometry": "privileged simulator geometry",
                    "scene_sha256": self.geometry_hash(self.scene()),
                }
            if operation == "touch":
                scene = self.scene()
                acm = scene.allowed_collision_matrix
                pairs = p["pairs"]
                for a, b in pairs:
                    for name in (a, b):
                        if name not in acm.entry_names:
                            acm.entry_names.append(name)
                            for row in acm.entry_values:
                                row.enabled.append(False)
                            acm.entry_values.append(
                                AllowedCollisionEntry(enabled=[False] * len(acm.entry_names))
                            )
                    i, j = acm.entry_names.index(a), acm.entry_names.index(b)
                    acm.entry_values[i].enabled[j] = bool(p["allowed"])
                    acm.entry_values[j].enabled[i] = bool(p["allowed"])
                diff = PlanningScene()
                diff.allowed_collision_matrix = acm
                self.apply(diff)
                return {"pairs": pairs, "allowed": bool(p["allowed"])}
            if operation in {"plan_pose", "plan_cartesian", "inspect_pose"}:
                self.plans.clear()
                scene = self.scene()
                reference_digest = self.geometry_hash(scene)
                start = scene.robot_state
                start.is_diff = False
                target = self.pose(p["world_from_tcp"])
                velocity = float(p.get("velocity_scale", 0.3))
                acceleration = float(p.get("acceleration_scale", 0.3))
                if not 0 < velocity <= 0.5 or not 0 < acceleration <= 0.5:
                    raise ValueError("Use scales in (0,.5]")
                direct_probe = None
                if operation in {"plan_cartesian", "plan_pose"}:
                    request = GetCartesianPath.Request()
                    request.header.frame_id = "world"
                    request.start_state = start
                    request.group_name = status["planning_group"]
                    request.link_name = status["tcp"]
                    request.waypoints = [target]
                    request.max_step = 0.005
                    request.jump_threshold = 2.0
                    request.avoid_collisions = True
                    request.max_velocity_scaling_factor = velocity
                    request.max_acceleration_scaling_factor = acceleration
                    reply = self.service("cartesian", request)
                    if reply.error_code.val == 1 and reply.fraction >= 0.999999:
                        try:
                            result = self.remember(
                                reply.solution, start, reference_digest, reply.fraction
                            )
                            result["planner_route"] = (
                                "required_cartesian"
                                if operation == "plan_cartesian"
                                else "direct_cartesian"
                            )
                            return result
                        except CandidateCollision as error:
                            if operation == "plan_cartesian":
                                raise
                            direct_probe = {"fraction": reply.fraction, "reason": str(error)}
                    elif operation == "plan_cartesian":
                        raise RuntimeError(
                            f"Incomplete Cartesian path: {reply.fraction}, "
                            f"code {reply.error_code.val}"
                        )
                    else:
                        direct_probe = {
                            "fraction": reply.fraction,
                            "error_code": reply.error_code.val,
                        }
                request = GetPositionIK.Request()
                ik = request.ik_request
                ik.group_name = status["planning_group"]
                ik.ik_link_name = status["tcp"]
                ik.robot_state = start
                ik.avoid_collisions = True
                ik.pose_stamped.header.frame_id = "world"
                ik.pose_stamped.pose = target
                ik.timeout.sec = 3
                reply = self.service("ik", request)
                if reply.error_code.val != 1 and operation == "inspect_pose":
                    raise RuntimeError("Collision-aware IK failed: " + str(reply.error_code.val))
                if operation == "inspect_pose":
                    samples = []
                    before = dict(zip(start.joint_state.name, start.joint_state.position))
                    goal = dict(
                        zip(
                            reply.solution.joint_state.name,
                            reply.solution.joint_state.position,
                        )
                    )
                    for fraction in np.linspace(0, 1, 11):
                        request = GetStateValidity.Request()
                        request.group_name = status["planning_group"]
                        request.robot_state = copy.deepcopy(start)
                        request.robot_state.joint_state.position = [
                            float(before[n] + fraction * (goal[n] - before[n]))
                            for n in start.joint_state.name
                        ]
                        validity = self.service("validity", request)
                        samples.append(
                            {
                                "fraction": float(fraction),
                                "valid": validity.valid,
                                "contacts": plain(
                                    [message_to_ordereddict(c) for c in validity.contacts]
                                ),
                            }
                        )
                    return {
                        "start_joints": before,
                        "ik_joints": goal,
                        "interpolated_validity": samples,
                        "planning_group": status["planning_group"],
                        "ik_link": status["tcp"],
                    }
                request = GetMotionPlan.Request()
                motion = request.motion_plan_request
                motion.group_name = status["planning_group"]
                if self.planner_profile is not None:
                    motion.pipeline_id = self.planner_profile["pipeline"]
                    motion.planner_id = self.planner_profile["id"]
                motion.start_state = start
                budget = float(p.get("planning_time_s", 10.0))
                if not 1 <= budget <= 30:
                    raise ValueError("Planning budget must be 1..30 seconds")
                motion.allowed_planning_time = budget
                motion.num_planning_attempts = 5
                motion.max_velocity_scaling_factor = velocity
                motion.max_acceleration_scaling_factor = acceleration
                # Equivalent bounded joint windings reach the same Cartesian pose
                # by different routes around obstacles; preserve those alternatives.
                positions = dict(
                    zip(reply.solution.joint_state.name, reply.solution.joint_state.position)
                )
                current = dict(zip(start.joint_state.name, start.joint_state.position))
                solutions = [positions] if reply.error_code.val == 1 else []
                # Explore kinematic branches with bounded single-axis seed changes.
                # These are robot-space seeds, independent of image or part tuning.
                for name in self.chain.required:
                    for direction in (-1, 1):
                        seeded = copy.deepcopy(start)
                        index = list(seeded.joint_state.name).index(name)
                        values = list(seeded.joint_state.position)
                        values[index] += direction * np.pi / 2
                        joint = next(j for j in self.chain.record()["joints"] if j["name"] == name)
                        if joint["bounds"] is not None:
                            values[index] = float(np.clip(values[index], *joint["bounds"]))
                        seeded.joint_state.position = values
                        extra = GetPositionIK.Request()
                        extra.ik_request = copy.deepcopy(ik)
                        extra.ik_request.robot_state = seeded
                        extra.ik_request.timeout.sec = 0
                        extra.ik_request.timeout.nanosec = 150000000
                        solved = self.service("ik", extra)
                        if solved.error_code.val == 1:
                            solutions.append(
                                dict(
                                    zip(
                                        solved.solution.joint_state.name,
                                        solved.solution.joint_state.position,
                                    )
                                )
                            )
                if not solutions:
                    raise RuntimeError("Collision-aware IK failed for all bounded seeds")
                candidates = []
                seen = set()
                families = [self.chain.joint_candidates(s, current) for s in solutions]
                # Take one from each branch before adding winding alternatives.
                for level in range(max(map(len, families))):
                    for family in families:
                        if level >= len(family):
                            continue
                        candidate = family[level]
                        key = tuple(round(candidate[n], 3) for n in self.chain.required)
                        if key not in seen:
                            candidates.append(candidate)
                            seen.add(key)
                candidates = candidates[:64]
                goals = []
                for candidate in candidates:
                    goal = Constraints()
                    goal.joint_constraints = [
                        JointConstraint(
                            joint_name=name,
                            position=float(candidate[name]),
                            tolerance_above=0.001,
                            tolerance_below=0.001,
                            weight=1.0,
                        )
                        for name in self.chain.required
                    ]
                    goals.append(goal)
                motion.goal_constraints = goals
                reply = self.service("plan", request, budget + 10).motion_plan_response
                if reply.error_code.val != 1:
                    raise RuntimeError("Motion planning failed: " + str(reply.error_code.val))
                result = self.remember(reply.trajectory, start, reference_digest)
                result["ik_winding_candidates"] = len(candidates)
                result["ik_seed_queries"] = 1 + 2 * len(self.chain.required)
                result["planner_route"] = "global_rrt"
                result["direct_probe"] = direct_probe
                return result
            if operation == "execute_simulation":
                self.model()
                solution, start, digest = self.plans.pop(p["plan_id"])
                scene = self.scene()
                if self.geometry_hash(scene) != digest:
                    raise RuntimeError("Planning scene changed after planning")
                expected = dict(zip(start.joint_state.name, start.joint_state.position))
                actual = dict(
                    zip(
                        scene.robot_state.joint_state.name,
                        scene.robot_state.joint_state.position,
                    )
                )
                if any(
                    name not in actual or abs(actual[name] - value) > 0.02
                    for name, value in expected.items()
                ):
                    raise RuntimeError("Robot moved after planning")
                goal = ExecuteTrajectory.Goal()
                goal.trajectory = solution
                self.motion_samples = []
                self.motion_dropped = 0
                self.motion_rewind = False
                self.collect_motion = True
                try:
                    result = self.action(self.execute_client, goal)
                finally:
                    self.collect_motion = False
                if result.error_code.val != 1:
                    raise RuntimeError("Trajectory execution failed: " + str(result.error_code.val))
                return {
                    "executed": True,
                    "error_code": result.error_code.val,
                    "simulation_only": True,
                    "observed_tcp_path": {
                        "model": "tcp_feedback_review_v1",
                        "frame": "world",
                        "units": "m",
                        "source": "measured_joint_feedback_fk",
                        "kinematic_sha256": self.chain.sha256,
                        "joint_names": list(self.chain.required),
                        "samples": self.motion_samples,
                        "discarded_samples": self.motion_dropped,
                        "clock_rewind": self.motion_rewind,
                        "status": "available"
                        if self.motion_samples and not self.motion_rewind
                        else "unavailable",
                        "independent_pose_measurement": False,
                    },
                }
            if operation == "gripper":
                opening = float(p["opening_m"])
                if not 0 <= opening <= 0.08:
                    raise ValueError("Jaw opening must be 0..80 mm")
                goal = FollowJointTrajectory.Goal()
                goal.trajectory.joint_names = list(FINGERS)
                point = JointTrajectoryPoint()
                point.positions = [opening / 2] * 2
                point.time_from_start.sec = 1
                goal.trajectory.points = [point]
                result = self.action(self.gripper_client, goal, 15)
                if result.error_code != 0:
                    raise RuntimeError("Gripper action failed: " + result.error_string)
                return {"opening_m": opening, "error_code": result.error_code}
            if operation in {"attach", "detach"}:
                name = p["object_id"]
                if name not in status["attachments"]:
                    raise ValueError("Unknown fixture object")
                acknowledgement = self.trigger(f"/opencv_fixture/objects/{name}/{operation}")
                expected = "attached" if operation == "attach" else "detached"
                until = time.monotonic() + 3
                while time.monotonic() < until:
                    observed = self.trigger(f"/opencv_fixture/objects/{name}/state")
                    if observed["attachment"] == expected and observed.get("pending") is None:
                        break
                    rclpy.spin_once(self, timeout_sec=0.05)
                else:
                    self.poisoned = True
                    raise RuntimeError("Attachment outcome unknown; reconcile before moving")
                truth = self.trigger("/opencv_fixture/evaluation/snapshot")
                obj = next(x for x in truth["objects"] if x["object_id"] == name)
                pose = obj["world_pose"]
                world_obj = matrix(pose["position_m"], pose["quaternion_xyzw"])
                diff = PlanningScene()
                diff.robot_state.is_diff = True
                attached = AttachedCollisionObject()
                attached.link_name = "gripper_base_link"
                attached.touch_links = [
                    "gripper_left_finger_link",
                    "gripper_right_finger_link",
                    "gripper_base_link",
                ]
                if operation == "attach":
                    tf = self.transform("world", attached.link_name, Time())
                    t, q = tf.transform.translation, tf.transform.rotation
                    relative = (
                        np.linalg.inv(matrix([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])) @ world_obj
                    )
                if self.motion_rewind:
                    self.poisoned = True
                    raise RuntimeError(
                        "Simulation time rewound during execution; reconcile effects"
                    )
                    attached.object = self.box(name, obj["size_m"], relative, attached.link_name)
                    # MoveIt attaches by removing the world object itself.
                    # A second explicit REMOVE makes the diff report failure
                    # after the attachment has already been applied.
                else:
                    attached.object.id = name
                    attached.object.operation = attached.object.REMOVE
                    diff.world.collision_objects.append(self.box(name, obj["size_m"], world_obj))
                diff.robot_state.attached_collision_objects.append(attached)
                self.apply(diff)
                return {
                    "acknowledgement": acknowledgement,
                    "observed": observed,
                    "assisted": True,
                    "moveit_attachment_updated": True,
                    "world_from_object": world_obj.tolist(),
                }
            raise ValueError("Unknown simulation primitive: " + operation)

    node = Backend()
    threading.Thread(target=read, daemon=True).start()
    try:
        while rclpy.ok():
            try:
                item = requests.get_nowait()
            except queue.Empty:
                rclpy.spin_once(node, timeout_sec=0.02)
                if cancelled.is_set():
                    break
                continue
            result = {
                "schema": "opencv_sim_pick_primitive_v1",
                "server_id": node.server_id,
                "request_id": item.get("request_id"),
                "run_id": item.get("run_id"),
                "simulation_only": True,
            }
            try:
                if node.poisoned:
                    raise RuntimeError(
                        "Backend held after uncertain outcome; reconcile and restart"
                    )
                if not isinstance(item.get("request_id"), str) or not isinstance(
                    item.get("run_id"), str
                ):
                    raise ValueError("Request/run IDs required")
                if item["request_id"] in node.seen:
                    raise ValueError("Duplicate request ID")
                if node.run_id is not None and item["run_id"] != node.run_id:
                    raise ValueError("Run identity changed")
                if item["operation"] != "hello" and item.get("server_id") != node.server_id:
                    raise ValueError("Server identity changed")
                node.run_id = item["run_id"]
                node.seen.append(item["request_id"])
                result.update(status="ok", value=node.dispatch(item))
            except Exception as error:
                result.update(status="error", reason=str(error), held=node.poisoned)
            result["environment"] = node.environment
            print(encode(result), flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
