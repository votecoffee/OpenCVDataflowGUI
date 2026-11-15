"""Collision-checked simulation pose sweep and exact-image-time robot capture."""

import argparse
import copy
import hashlib
import json
import math
import os
import time
import uuid
from collections import deque
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--overlay", type=Path, required=True)
    p.add_argument("--partition", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--observe-only", action="store_true")
    args = p.parse_args()
    from .run import environment

    preparation, values = environment(args.overlay, args.partition)
    # This child joins only its explicitly named, generation-bound example launcher.
    launch = json.loads((args.overlay / "session-launch.json").read_text())
    if launch["partition"] != args.partition or (args.overlay / "session-exit.json").exists():
        raise RuntimeError("Requested workcell is not the active launch generation")
    os.environ.update(values)
    if args.output.exists():
        raise ValueError("Choose a new capture destination")
    scene = json.loads((args.overlay / "scene.json").read_text())
    import rclpy
    from geometry_msgs.msg import Pose
    from moveit_msgs.action import ExecuteTrajectory
    from moveit_msgs.msg import CollisionObject, Constraints, JointConstraint
    from moveit_msgs.srv import (
        ApplyPlanningScene,
        GetMotionPlan,
        GetPlanningScene,
        GetStateValidity,
    )
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from rosgraph_msgs.msg import Clock
    from rosidl_runtime_py.convert import message_to_ordereddict as plain
    from sensor_msgs.msg import CameraInfo, Image, JointState
    from shape_msgs.msg import SolidPrimitive
    from tf2_ros import Buffer, TransformListener

    rclpy.init()
    node = Node(
        "hand_eye_capture",
        parameter_overrides=[rclpy.parameter.Parameter("use_sim_time", value=True)],
    )
    images, infos = deque(maxlen=10), {}
    feedback = [None]
    active = [None]
    clock = [0]
    rewound = [False]

    def clock_update(message):
        value = message.clock.sec * 1_000_000_000 + message.clock.nanosec
        if clock[0] and value < clock[0]:
            rewound[0] = True
        clock[0] = value

    def stamp(msg):
        return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    def info(msg):
        infos[stamp(msg)] = msg
        while len(infos) > 20:
            del infos[min(infos)]

    def joints(msg):
        feedback[0] = (time.monotonic(), msg)

    subscriptions = [
        node.create_subscription(
            Image,
            "/hand_eye/camera/image",
            lambda m: images.append((time.monotonic(), m)),
            qos_profile_sensor_data,
        ),
        node.create_subscription(
            CameraInfo, "/hand_eye/camera/camera_info", info, qos_profile_sensor_data
        ),
        node.create_subscription(JointState, "/joint_states", joints, qos_profile_sensor_data),
        node.create_subscription(Clock, "/clock", clock_update, qos_profile_sensor_data),
    ]
    buffer = Buffer(node=node)
    listener = TransformListener(buffer, node)
    node._hand_eye_resources = (subscriptions, listener)
    clients = {
        key: node.create_client(kind, name)
        for key, kind, name in [
            ("scene", GetPlanningScene, "/get_planning_scene"),
            ("apply", ApplyPlanningScene, "/apply_planning_scene"),
            ("plan", GetMotionPlan, "/plan_kinematic_path"),
            ("valid", GetStateValidity, "/check_state_validity"),
        ]
    }
    execute = ActionClient(node, ExecuteTrajectory, "/execute_trajectory")

    def wait(future, seconds=30):
        end = time.monotonic() + seconds
        while not future.done() and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.02)
        if not future.done():
            if active[0] is not None:
                cancel = active[0].cancel_goal_async()
                cancel_end = time.monotonic() + 3
                while not cancel.done() and time.monotonic() < cancel_end:
                    rclpy.spin_once(node, timeout_sec=0.02)
            raise RuntimeError("Timed out; outcome unknown, do not automatically retry")
        return future.result()

    def call(key, request, seconds=30):
        if not clients[key].wait_for_service(timeout_sec=5):
            raise RuntimeError("Unavailable service " + key)
        return wait(clients[key].call_async(request), seconds)

    def spin_until(predicate, seconds=30):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.03)
            if rewound[0]:
                raise RuntimeError("Simulation clock rewound; start a new isolated capture session")
            result = predicate()
            if result:
                return result
        raise RuntimeError("Required fresh synchronized observations unavailable")

    args.output.mkdir(parents=True)
    manifest = dict(
        schema="opencv.hand_eye_captures/v1",
        run_id=str(uuid.uuid4()),
        launch_id=launch["launch_id"],
        simulation_only=True,
        scene_sha256=preparation["scene_sha256"],
        base_frame="base_link",
        gripper_frame="tool0",
        camera_frame="hand_eye_camera_optical_frame",
        board_frame="charuco_board",
        units="m",
        board_definition=scene["board_definition"],
        gates=scene["gates"],
        captures=[],
        complete=False,
    )

    def save_manifest():
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    def acquire(after):
        if not feedback[0] or time.monotonic() - feedback[0][0] > 1:
            return None
        for received, image in reversed(images):
            key = stamp(image)
            if key <= after or key not in infos or time.monotonic() - received > 1:
                continue
            camera = infos[key]
            if (
                image.header.frame_id != manifest["camera_frame"]
                or camera.header.frame_id != image.header.frame_id
            ):
                raise RuntimeError("Camera optical frame mismatch")
            if (camera.width, camera.height) != (image.width, image.height):
                raise RuntimeError("CameraInfo grid mismatch")
            try:
                tf = buffer.lookup_transform(
                    "base_link", "tool0", Time.from_msg(image.header.stamp)
                )
            except Exception:
                continue
            if stamp(tf) != key:
                raise RuntimeError("TF lookup did not preserve image time")
            if key <= max([c["stamp_ns"] for c in manifest["captures"]], default=0):
                raise RuntimeError("Clock rewind/duplicate capture")
            return image, camera, tf
        return None

    try:
        spin_until(
            lambda: feedback[0] and images and infos and node.get_clock().now().nanoseconds > 0, 60
        )
        if args.observe_only:
            captures = [dict(id="observe", role="preview")]
        else:
            # Known fixture collision geometry is planning input only, never calibration input.
            request = ApplyPlanningScene.Request()
            request.scene.is_diff = True
            item = CollisionObject()
            item.id = "hand_eye_target"
            item.header.frame_id = "base_link"
            item.operation = CollisionObject.ADD
            shape = SolidPrimitive()
            shape.type = SolidPrimitive.BOX
            shape.dimensions = [0.25, 0.19, 0.006]
            pose = Pose()
            x, y, z, r, pit, yaw = scene["board_pose_xyzrpy"]
            cr, sr = math.cos(r / 2), math.sin(r / 2)
            cp, sp = math.cos(pit / 2), math.sin(pit / 2)
            cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
            pose.orientation.x = sr * cp * cy - cr * sp * sy
            pose.orientation.y = cr * sp * cy + sr * cp * sy
            pose.orientation.z = cr * cp * sy - sr * sp * cy
            pose.orientation.w = cr * cp * cy + sr * sp * sy
            # Backing center follows the full texture center, not the pattern origin.
            q = pose.orientation
            from .transforms import rotation

            rotated = rotation([q.x, q.y, q.z, q.w])
            centre = [x, y, z] + rotated @ [0.105, 0.075, 0.002]
            pose.position.x, pose.position.y, pose.position.z = map(float, centre)
            item.primitives = [shape]
            item.primitive_poses = [pose]
            request.scene.world.collision_objects = [item]
            if not call("apply", request).success:
                raise RuntimeError("Target collision geometry unavailable")
            captures = scene["captures"]
        save_manifest()
        for sample in captures:
            motion = None
            if not args.observe_only:
                request = GetPlanningScene.Request()
                request.components.components = 2
                state = call("scene", request).scene.robot_state
                if time.monotonic() - feedback[0][0] > 1:
                    raise RuntimeError("Joint feedback stale")
                plan = GetMotionPlan.Request()
                m = plan.motion_plan_request
                m.group_name = "ur_manipulator"
                m.pipeline_id = "ompl"
                m.planner_id = "RRTConnectkConfigDefault"
                m.start_state = state
                m.num_planning_attempts = 3
                m.allowed_planning_time = 5.0
                m.max_velocity_scaling_factor = 0.25
                m.max_acceleration_scaling_factor = 0.25
                goal = Constraints()
                for name, value in zip(scene["joint_names"], sample["joints"]):
                    goal.joint_constraints.append(
                        JointConstraint(
                            joint_name=name,
                            position=value,
                            tolerance_above=0.001,
                            tolerance_below=0.001,
                            weight=1.0,
                        )
                    )
                m.goal_constraints = [goal]
                response = call("plan", plan, 20).motion_plan_response
                if response.error_code.val != 1 or not response.trajectory.joint_trajectory.points:
                    raise RuntimeError(
                        f"Planning rejected {sample['id']}: {response.error_code.val}"
                    )
                trajectory = response.trajectory.joint_trajectory
                previous = dict(zip(state.joint_state.name, state.joint_state.position))
                checks = 0
                for point in trajectory.points:
                    target = dict(zip(trajectory.joint_names, point.positions))
                    count = max(
                        1, math.ceil(max(abs(target[n] - previous[n]) for n in target) / 0.05)
                    )
                    for step in range(1, count + 1):
                        checks += 1
                        if checks > 4096:
                            raise RuntimeError("Trajectory checking limit exceeded")
                        check = GetStateValidity.Request()
                        check.group_name = "ur_manipulator"
                        check.robot_state = copy.deepcopy(state)
                        values = {
                            **previous,
                            **{
                                n: previous[n] + (v - previous[n]) * step / count
                                for n, v in target.items()
                            },
                        }
                        check.robot_state.joint_state.position = [
                            values[n] for n in check.robot_state.joint_state.name
                        ]
                        if not call("valid", check).valid:
                            raise RuntimeError(
                                "Post-retiming sampled collision/constraint rejection"
                            )
                    previous.update(target)
                if not execute.wait_for_server(timeout_sec=5):
                    raise RuntimeError("Execution server unavailable")
                sent = ExecuteTrajectory.Goal()
                sent.trajectory = response.trajectory
                active[0] = wait(execute.send_goal_async(sent))
                if not active[0].accepted:
                    raise RuntimeError("Execution rejected")
                result = wait(active[0].get_result_async(), 60)
                active[0] = None
                if result.status != 4 or result.result.error_code.val != 1:
                    raise RuntimeError("Execution failed/unknown")
                motion = dict(
                    planned=plain(response.trajectory),
                    execution_status=result.status,
                    execution_code=result.result.error_code.val,
                    sampled_states_checked=checks,
                    continuous_collision_certificate=False,
                )
                stable = [None]

                def settled():
                    if not feedback[0] or time.monotonic() - feedback[0][0] > 1:
                        stable[0] = None
                        return False
                    msg = feedback[0][1]
                    observed = dict(zip(msg.name, msg.position))
                    velocities = dict(zip(msg.name, msg.velocity))
                    good = all(
                        abs(observed[n] - v) < 0.004 and abs(velocities.get(n, math.inf)) < 0.01
                        for n, v in zip(scene["joint_names"], sample["joints"])
                    )
                    if not good:
                        stable[0] = None
                        return False
                    if stable[0] is None:
                        stable[0] = time.monotonic()
                    return time.monotonic() - stable[0] > 0.5

                spin_until(settled, 15)
            after = node.get_clock().now().nanoseconds
            image, camera, tf = spin_until(lambda: acquire(after), 15)
            directory = args.output / sample["id"]
            directory.mkdir()
            raw = bytes(image.data)
            (directory / "image.bin").write_bytes(raw)
            record = dict(
                id=sample["id"],
                role=sample["role"],
                run_id=manifest["run_id"],
                stamp_ns=stamp(image),
                image=dict(
                    header=plain(image.header),
                    width=image.width,
                    height=image.height,
                    step=image.step,
                    encoding=image.encoding,
                    is_bigendian=image.is_bigendian,
                    path="image.bin",
                    sha256=hashlib.sha256(raw).hexdigest(),
                ),
                camera_info=plain(camera),
                robot_transform=plain(tf),
                joint_feedback=plain(feedback[0][1]),
                tf_lookup="exact_acquisition_time",
                settled_capture=not args.observe_only,
                motion=motion,
            )
            encoded = (json.dumps(record, indent=2, allow_nan=False) + "\n").encode()
            (directory / "capture.json").write_bytes(encoded)
            manifest["captures"].append(
                dict(
                    id=sample["id"],
                    role=sample["role"],
                    stamp_ns=stamp(image),
                    path=f"{sample['id']}/capture.json",
                    sha256=hashlib.sha256(encoded).hexdigest(),
                )
            )
            save_manifest()
            print(json.dumps(dict(captured=sample["id"], stamp=stamp(image))), flush=True)
        manifest["complete"] = True
        save_manifest()
    except BaseException as error:
        manifest["error"] = str(error)
        save_manifest()
        raise
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
