"""Read-only native camera/fixture/MoveIt qualification; sends no motion goals."""
import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import time

from .node import require_isolation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require_isolation()
    if args.output.exists(): raise FileExistsError("Choose a new evidence directory")
    import numpy as np
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, CameraInfo
    from std_srvs.srv import Trigger
    from moveit_msgs.srv import GetPlanningScene
    from moveit_msgs.msg import PlanningSceneComponents
    from moveit_msgs.action import ExecuteTrajectory
    from control_msgs.action import FollowJointTrajectory

    rclpy.init(); node = rclpy.create_node("opencv_fixture_readiness_verifier")
    streams = {"rgb": OrderedDict(), "depth": OrderedDict()}; info = []
    subscriptions = []

    def receive(message, stream):
        stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        streams[stream][stamp] = (message, time.monotonic())
        while len(streams[stream]) > 8: streams[stream].popitem(last=False)

    for stream in streams:
        subscriptions.append(node.create_subscription(Image, f"/opencv_fixture/source/{stream}",
            lambda message, name=stream: receive(message, name), qos_profile_sensor_data))
    subscriptions.append(node.create_subscription(CameraInfo, "/opencv_fixture/source/camera_info",
        lambda message: info.__setitem__(slice(None), [message]), qos_profile_sensor_data))

    def request(service, service_type, body):
        client = node.create_client(service_type, service)
        if not client.wait_for_service(timeout_sec=10): raise RuntimeError(f"Missing service {service}")
        future = client.call_async(body)
        rclpy.spin_until_future_complete(node, future, timeout_sec=10)
        if not future.done(): raise RuntimeError(f"Service timeout {service}")
        result = future.result(); node.destroy_client(client)
        return result

    try:
        started = time.monotonic()
        while time.monotonic() - started < 4:
            rclpy.spin_once(node, timeout_sec=0.1)
        common = sorted(set(streams["rgb"]) & set(streams["depth"]))
        if len(common) < 2 or not info: raise RuntimeError("Need multiple synchronized original RGB/depth frames and CameraInfo")
        stamp = common[-1]; rgb, rgb_received = streams["rgb"][stamp]; depth, depth_received = streams["depth"][stamp]
        camera = info[-1]
        if depth.encoding != "32FC1": raise RuntimeError(f"Unexpected original depth encoding {depth.encoding}")
        if rgb.encoding not in {"rgb8", "bgr8"}: raise RuntimeError(f"Unexpected original RGB encoding {rgb.encoding}")
        if not (rgb.width == depth.width == camera.width and rgb.height == depth.height == camera.height):
            raise RuntimeError("Original image and CameraInfo dimensions disagree")
        if not (rgb.header.frame_id == depth.header.frame_id == camera.header.frame_id == "opencv_fixture_camera_optical_frame"):
            raise RuntimeError("Original camera frames disagree")
        if max(time.monotonic() - rgb_received, time.monotonic() - depth_received) > 0.5:
            raise RuntimeError("Source frames are not fresh")
        raw = bytes(depth.data)
        if depth.step < depth.width * 4 or len(raw) != depth.step * depth.height: raise RuntimeError("Invalid depth stride/length")
        image = np.ndarray((depth.height, depth.width), dtype=">f4" if depth.is_bigendian else "<f4", buffer=raw,
                           strides=(depth.step, 4))
        if np.count_nonzero(np.isfinite(image) & (image > 0)) < image.size // 4:
            raise RuntimeError("Source depth does not contain enough finite visible geometry")
        status_response = request("/opencv_fixture/status", Trigger, Trigger.Request())
        if not status_response.success: raise RuntimeError(status_response.message)
        status = json.loads(status_response.message)
        if not status["ready"] or not status["simulation_only"]: raise RuntimeError("Fixture is not ready for simulated execution")
        truth_response = request("/opencv_fixture/evaluation/snapshot", Trigger, Trigger.Request())
        if not truth_response.success: raise RuntimeError(truth_response.message)
        truth = json.loads(truth_response.message)
        if truth["session_id"] != status["session_id"] or truth["run_id"] != status["run_id"]:
            raise RuntimeError("Session changed during qualification")

        # Privileged geometry is used only to verify first-return source-camera consistency.
        camera_pose = status["source"]["world_from_source_camera"]
        x, y, z, w = camera_pose["quaternion_xyzw"]
        rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                             [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                             [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        origin = np.array(camera_pose["translation_m"]); k = np.array(camera.k).reshape(3, 3)
        projections = []
        for item in truth["objects"]:
            if item["world_pose"] is None: raise RuntimeError("Missing privileged object pose for camera evaluation")
            top = np.array(item["world_pose"]["position_m"]) + [0, 0, item["size_m"][2]/2]
            optical = rotation.T @ (top - origin); pixel = k @ optical; u, v = np.rint(pixel[:2]/pixel[2]).astype(int)
            if not 2 <= u < image.shape[1]-2 or not 2 <= v < image.shape[0]-2:
                raise RuntimeError("Projected part centre is outside the original source image")
            measured = float(np.median(image[v-1:v+2, u-1:u+2]))
            difference = measured - float(optical[2])
            classification = "top_surface_visible" if abs(difference) < 0.008 else "occluded_by_nearer_surface" if difference < -0.015 else "unexpected"
            if classification == "unexpected": raise RuntimeError(f"Camera/depth/world inconsistency at {item['object_id']}")
            projections.append({"object_id": item["object_id"], "pixel_uv": [int(u), int(v)],
                                "evaluation_top_depth_m": float(optical[2]), "source_first_return_m": measured,
                                "classification": classification})
        visible = sum(p["classification"] == "top_surface_visible" for p in projections)
        if visible != 3: raise RuntimeError(f"Expected three visible initial stack tops, observed {visible}")

        planning_request = GetPlanningScene.Request()
        planning_request.components.components = PlanningSceneComponents.ROBOT_STATE | PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
        planning = request("/get_planning_scene", GetPlanningScene, planning_request)
        if len(planning.scene.robot_state.joint_state.name) < 8: raise RuntimeError("MoveIt lacks full arm/gripper state")
        service_names = {name for name, _ in node.get_service_names_and_types()}
        required = {"/compute_ik", "/compute_cartesian_path", "/plan_kinematic_path", "/get_planning_scene", "/apply_planning_scene"}
        if not required <= service_names: raise RuntimeError(f"Missing MoveIt services: {required - service_names}")
        actions = {}
        for name, action_type in [("/execute_trajectory", ExecuteTrajectory),
                                  ("/gripper_controller/follow_joint_trajectory", FollowJointTrajectory),
                                  ("/joint_trajectory_controller/follow_joint_trajectory", FollowJointTrajectory)]:
            client = ActionClient(node, action_type, name)
            actions[name] = client.wait_for_server(timeout_sec=5)
            if not actions[name]: raise RuntimeError(f"Missing action server {name}")
            client.destroy()

        args.output.mkdir(parents=True)
        for name, data in [("rgb.bin", bytes(rgb.data)), ("depth.bin", raw)]:
            (args.output / name).write_bytes(data)
        receipt = {"scope": "native headless simulation; source-camera capture and read-only fixture/MoveIt checks; no motion goals",
            "status": status, "source_stamp_ns": stamp, "synchronized_frame_count": len(common),
            "images": {"width": depth.width, "height": depth.height, "rgb_encoding": rgb.encoding,
                "rgb_step": rgb.step, "depth_encoding": depth.encoding, "depth_step": depth.step,
                "depth_bigendian": bool(depth.is_bigendian), "k": list(camera.k),
                "rgb_sha256": hashlib.sha256(bytes(rgb.data)).hexdigest(), "depth_sha256": hashlib.sha256(raw).hexdigest()},
            "evaluation_only_first_return_checks": projections, "visible_stack_tops": visible,
            "moveit_services": sorted(required), "actions": actions,
            "planning_scene_world_objects_before_app_setup": len(planning.scene.world.collision_objects),
            "limits": "No contact/friction qualification, attach/lift/release/reset execution, full-shape occlusion or completed picks established"}
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"receipt": str(args.output / 'receipt.json'), "ready": True,
            "session_id": status["session_id"], "visible_stack_tops": visible, "actions_available": len(actions)}))
    finally:
        node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__": main()
