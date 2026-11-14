"""ROS adapter for the explicitly assisted, simulation-only fixture."""
import argparse
import json
import math
import os
from pathlib import Path
import time

from .scene import camera_transform, load
from .state import FixtureState, Observation


def require_isolation():
    if os.environ.get("ROS_DOMAIN_ID") != "130" or not os.environ.get("GZ_PARTITION", "").startswith("opencv_bin_pick_"):
        raise RuntimeError("Fixture requires its allocated ROS domain 130 and opencv_bin_pick_ Gazebo partition")
    if os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE") != "LOCALHOST":
        raise RuntimeError("Fixture requires localhost-only ROS discovery")


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument("--simulation-only", action="store_true", required=True)
    arguments, ros_args = parser.parse_known_args(args)
    require_isolation()
    # Lazy imports: offline scene/state inspection never initializes ROS.
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from rclpy.clock import Clock as RclpyClock, ClockType
    from sensor_msgs.msg import CameraInfo, Image, JointState
    from std_msgs.msg import Empty, String
    from std_srvs.srv import Trigger
    from tf2_msgs.msg import TFMessage
    from tf2_ros import Buffer, TransformListener, StaticTransformBroadcaster
    from geometry_msgs.msg import TransformStamped
    from rosgraph_msgs.msg import Clock

    scene = load(arguments.overlay / "scene.json")

    class FixtureNode(Node):
        def __init__(self):
            super().__init__("opencv_bin_picking_fixture")
            self.core = FixtureState(scene)
            self.source = {}
            self.camera_info = None
            self.last_error = None
            self.clock_ns = None
            self.clock_advanced = None
            self.tf = Buffer(node=self)
            self.listener = TransformListener(self.tf, self)
            self.broadcaster = StaticTransformBroadcaster(self)
            static = camera_transform(scene)
            tf = TransformStamped(); tf.header.frame_id = static["parent"]; tf.child_frame_id = static["child"]
            tf.header.stamp = self.get_clock().now().to_msg()
            tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = static["translation_m"]
            tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w = static["quaternion_xyzw"]
            self.broadcaster.sendTransform(tf)
            self.status_pub = self.create_publisher(String, "/opencv_fixture/status", 10)
            self.truth_pub = self.create_publisher(String, "/opencv_fixture/evaluation/poses", 10)
            self.subscriptions_owned = []
            self.services_owned = []
            self.commands = {}
            self.create_subscription(JointState, "/joint_states", self.joints, qos_profile_sensor_data)
            self.create_subscription(Clock, "/clock", self.clock, 10)
            for name in ("rgb", "depth"):
                self.subscriptions_owned.append(self.create_subscription(Image, f"/opencv_fixture/source/{name}",
                    lambda message, n=name: self.image(message, n), qos_profile_sensor_data))
            self.create_subscription(CameraInfo, "/opencv_fixture/source/camera_info", self.information, qos_profile_sensor_data)
            self.service("/opencv_fixture/status", lambda: self.snapshot())
            self.service("/opencv_fixture/evaluation/snapshot", lambda: self.truth())
            self.service("/opencv_fixture/reset", self.reset)
            for part in scene["parts"]:
                part_id = part["id"]; prefix = f"/opencv_fixture/assistance/{part_id}"
                for operation in ("attach", "detach"):
                    self.commands[(part_id, operation)] = self.create_publisher(Empty, prefix + "/" + operation, 10)
                    self.service(f"/opencv_fixture/objects/{part_id}/{operation}",
                                 lambda p=part_id, o=operation: self.command(p, o))
                self.service(f"/opencv_fixture/objects/{part_id}/state", lambda p=part_id: self.object_state(p))
                self.subscriptions_owned.append(self.create_subscription(String, prefix + "/state",
                    lambda message, p=part_id: self.acknowledge(p, message.data), 10))
                self.subscriptions_owned.append(self.create_subscription(TFMessage, f"/opencv_fixture/evaluation/{part_id}/pose",
                    lambda message, p=part_id: self.pose(p, message), 10))
            self.timer = self.create_timer(0.05, self.poll, clock=RclpyClock(clock_type=ClockType.STEADY_TIME))

        def service(self, name, action):
            def callback(_request, response):
                try:
                    value = action(); response.success = True
                except (ValueError, RuntimeError) as error:
                    value = {"error": str(error)}; response.success = False
                response.message = json.dumps({"schema": "opencv.bin_picking_fixture/v1",
                    "session_id": self.core.session_id, "run_id": self.core.run_id,
                    "simulation_only": True, **value}, allow_nan=False)
                return response
            self.services_owned.append(self.create_service(Trigger, name, callback))

        def clock(self, message):
            stamp = message.clock.sec * 1_000_000_000 + message.clock.nanosec
            if self.clock_ns != stamp:
                self.clock_advanced = time.monotonic()
                self.clock_ns = stamp

        def joints(self, message):
            self.core.joints = dict(zip(message.name, message.position))
            self.core.joints_received = time.monotonic()
            arm = [index for index, name in enumerate(message.name) if name not in
                   {"gripper_left_finger_joint", "gripper_right_finger_joint"}]
            self.core.arm_stationary = len(arm) == 6 and all(index < len(message.velocity)
                and math.isfinite(message.velocity[index]) and abs(message.velocity[index]) < 0.02 for index in arm)

        def image(self, message, name):
            if message.header.frame_id == scene["camera"]["optical_frame"] and message.width > 0 and message.height > 0:
                self.source[name] = time.monotonic()
            else:
                self.last_error = f"Source {name} frame/geometry differs from the scene camera"

        def information(self, message):
            if message.header.frame_id != scene["camera"]["optical_frame"] or message.k[0] <= 0 or message.k[4] <= 0:
                self.last_error = "Source CameraInfo has an unexpected frame or intrinsics"; return
            self.camera_info = {"width": message.width, "height": message.height, "k": list(message.k),
                                "d": list(message.d), "distortion_model": message.distortion_model,
                                "origin": "original simulator CameraInfo; not fitted from evaluation object poses"}
            self.source["camera_info"] = time.monotonic()

        def pose(self, part_id, message):
            for transform in message.transforms:
                if transform.child_frame_id != part_id or transform.header.frame_id not in {scene["world"], scene["world_frame"]}:
                    self.last_error = f"Unexpected privileged pose frame for {part_id}"; continue
                t, q = transform.transform.translation, transform.transform.rotation
                if not all(math.isfinite(v) for v in (t.x, t.y, t.z, q.x, q.y, q.z, q.w)):
                    self.last_error = f"Non-finite privileged pose for {part_id}"; continue
                self.core.poses[part_id] = Observation((t.x, t.y, t.z), (q.x, q.y, q.z, q.w), time.monotonic(),
                    transform.header.stamp.sec * 1_000_000_000 + transform.header.stamp.nanosec)

        def acknowledge(self, part_id, state):
            try: self.core.acknowledge(part_id, state)
            except ValueError as error: self.last_error = str(error)

        def command(self, part_id, operation):
            if self.commands[(part_id, operation)].get_subscription_count() == 0:
                raise ValueError("ROS/Gazebo command bridge is not subscribed")
            requested = self.core.request(operation, part_id)
            self.commands[(part_id, operation)].publish(Empty())
            return {"operation": requested, "assistance": "pose_gated_fixed_joint",
                    "note": "Request sent; wait for observed attachment state before moving. Native output carries no operation UUID."}

        def object_state(self, part_id):
            return {"object_id": part_id, "attachment": self.core.state[part_id], "pending": self.core.pending}

        def truth(self):
            objects = []
            for item in scene["parts"]:
                pose = self.core.poses.get(item["id"])
                objects.append({"object_id": item["id"], "size_m": item["size"],
                    "world_pose": None if pose is None else {"position_m": pose.xyz, "quaternion_xyzw": pose.quaternion},
                    "source_stamp_ns": None if pose is None else pose.stamp_ns,
                    "receive_age_s": None if pose is None else time.monotonic() - pose.received,
                    "attachment": self.core.state[item["id"]]})
            return {"privileged": True, "usage": "collision scene setup and evaluation; never source perception",
                    "world_frame": scene["world_frame"], "objects": objects, "static_boxes": scene["static_boxes"]}

        def snapshot(self):
            state = self.core.snapshot()
            ready = (self.core.camera_ready and self.core._fresh(self.clock_advanced)
                     and self.core.tcp is not None and self.core._fresh(self.core.tcp.received)
                     and self.core._fresh(self.core.joints_received)
                     and all(state["pose_availability"].values()) and not self.core.resetting and self.core.pending is None)
            return {**state, "ready": ready,
                    "phase": "resetting" if self.core.resetting else "operation_pending" if self.core.pending
                        else "ready" if ready else "waiting_for_observations",
                    "overlay": str(arguments.overlay.resolve()),
                    "ros_domain_id": os.environ["ROS_DOMAIN_ID"], "gz_partition": os.environ["GZ_PARTITION"],
                    "camera_info": self.camera_info, "last_error": self.last_error,
                    "reset_supported": False,
                    "physics": {"simulation_stamp_ns": self.clock_ns,
                        "time_recently_advancing": self.core._fresh(self.clock_advanced),
                        "reset_phase": None}}

        def reset(self):
            raise ValueError("In-place reset is disabled: native clock rewind interrupted robot/TF feedback. "
                             "Stop this isolated fixture and prepare a new overlay/session to repeat.")

        def poll(self):
            self.core.camera_ready = self.camera_info is not None and all(
                self.core._fresh(self.source.get(key)) for key in ("rgb", "depth"))
            try:
                tf = self.tf.lookup_transform(scene["world_frame"], scene["tcp"], Time())
                stamp = tf.header.stamp.sec * 1_000_000_000 + tf.header.stamp.nanosec
                if self.core.tcp is None or stamp != self.core.tcp.stamp_ns:
                    t, q = tf.transform.translation, tf.transform.rotation
                    self.core.tcp = Observation((t.x, t.y, t.z), (q.x, q.y, q.z, q.w), time.monotonic(), stamp)
            except Exception:
                pass  # Missing/stale TF remains unavailable; cached lookups never refresh age.
            self.status_pub.publish(String(data=json.dumps(self.snapshot(), allow_nan=False)))
            self.truth_pub.publish(String(data=json.dumps({"session_id": self.core.session_id, "run_id": self.core.run_id,
                **self.truth()}, allow_nan=False)))

    rclpy.init(args=ros_args)
    node = FixtureNode()
    try: rclpy.spin(node)
    finally: node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
