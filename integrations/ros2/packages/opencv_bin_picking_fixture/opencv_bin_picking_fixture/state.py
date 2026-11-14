"""Pure assisted-grasp state machine. Truth is used for fixture eligibility only."""
from dataclasses import dataclass
import math
import time
import uuid

from . import SCHEMA
from .scene import digest, camera_transform


@dataclass(frozen=True)
class Observation:
    xyz: tuple
    quaternion: tuple  # xyzw
    received: float  # monotonic host receive time, never refreshed by a cache lookup
    stamp_ns: int


def axis_z(q):
    x, y, z, w = q
    norm = x*x + y*y + z*z + w*w
    if not math.isfinite(norm) or abs(norm - 1) > 0.01:
        raise ValueError("Pose quaternion must be finite and normalized")
    return (2*(x*z + y*w), 2*(y*z - x*w), 1 - 2*(x*x + y*y))


def yaw(q):
    axis_z(q)
    x, y, z, w = q
    return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


class FixtureState:
    def __init__(self, scene, now=time.monotonic):
        self.scene = scene
        self.scene_sha256 = digest(scene)
        self.session_id = str(uuid.uuid4())
        self.run_id = str(uuid.uuid4())
        self.now = now
        self.poses = {}
        self.tcp = None
        self.joints = {}
        self.joints_received = None
        self.arm_stationary = False
        self.camera_ready = False
        self.state = {item["id"]: "configured_detached_unobserved" for item in scene["parts"]}
        self.pending = None
        self.last_operation = None
        self.resetting = False

    def _fresh(self, received):
        return received is not None and 0 <= self.now() - received <= 0.5

    def gate(self, part_id):
        if part_id not in self.state:
            raise ValueError("Unknown object ID")
        if self.resetting or self.pending is not None:
            raise ValueError("Wait for the pending fixture operation or reconcile its outcome")
        if any(state not in {"configured_detached_unobserved", "detached"} for state in self.state.values()):
            raise ValueError("Another attachment is active or its outcome is unknown")
        pose = self.poses.get(part_id)
        if pose is None or self.tcp is None or not self._fresh(pose.received) or not self._fresh(self.tcp.received):
            raise ValueError("Fresh world-object and TCP observations are required")
        if not all(math.isfinite(v) for v in pose.xyz + self.tcp.xyz):
            raise ValueError("Finite world-object and TCP positions are required")
        if not self._fresh(self.joints_received) or not self.arm_stationary:
            raise ValueError("Fresh stationary-arm joint feedback is required")
        if not self.camera_ready:
            raise ValueError("Source RGB/depth/CameraInfo must be observed before assisted picking")
        part = next(item for item in self.scene["parts"] if item["id"] == part_id)
        if abs(part["size"][0] - part["size"][1]) > 1e-6 or abs(part["size"][2] - 0.04) > 1e-6:
            raise ValueError("Initial grasp assistance requires a square footprint and 40 mm height")
        # This initial assistance mode supports upright boxes and downward jaws.
        if axis_z(pose.quaternion)[2] < 0.98 or axis_z(self.tcp.quaternion)[2] > -0.98:
            raise ValueError("Expected an upright box and a downward-facing TCP")
        if math.hypot(self.tcp.xyz[0] - pose.xyz[0], self.tcp.xyz[1] - pose.xyz[1]) > 0.006:
            raise ValueError("TCP is outside the 6 mm horizontal grasp gate")
        # Native fingertips extend 25 mm beyond TCP; +10 mm clears a 40 mm part's support.
        if not 0.006 <= self.tcp.xyz[2] - pose.xyz[2] <= 0.018:
            raise ValueError("TCP must be 6..18 mm above the box centre")
        error = (yaw(self.tcp.quaternion) - yaw(pose.quaternion) + math.pi/4) % (math.pi/2) - math.pi/4
        if abs(error) > math.radians(5):
            raise ValueError("Jaws are not aligned with a box face within 5 degrees")
        for name in ("gripper_left_finger_joint", "gripper_right_finger_joint"):
            if name not in self.joints or not math.isfinite(self.joints[name]):
                raise ValueError("Both gripper finger positions are required")
            if abs(self.joints[name] - part["size"][0]/2) > 0.002:
                raise ValueError("Observed jaw opening does not match the box width")

    def request(self, operation, part_id):
        if operation == "attach":
            self.gate(part_id)
        elif operation == "detach":
            if part_id not in self.state or self.resetting:
                raise ValueError("Unknown object or reset in progress")
            if self.pending is not None:
                raise ValueError("Wait for the pending operation; use state to inspect uncertain outcomes")
            if self.state[part_id] != "attached":
                raise ValueError("Detach requires an observed attached state")
        else:
            raise ValueError("Unsupported fixture operation")
        self.pending = {"operation_id": str(uuid.uuid4()), "operation": operation, "object_id": part_id,
                        "requested_monotonic": self.now(), "outcome": "requested"}
        return dict(self.pending)

    def acknowledge(self, part_id, state):
        if part_id not in self.state or state not in {"attached", "detached"}:
            raise ValueError("Unexpected native attachment state")
        self.state[part_id] = state
        pending = self.pending
        if pending is not None and pending["object_id"] == part_id and state == pending["operation"] + "ed":
            self.last_operation = {**pending, "outcome": "observed", "observed_monotonic": self.now()}
            self.pending = None

    def expire(self):
        if self.pending and self.now() - self.pending["requested_monotonic"] > 2:
            self.pending["outcome"] = "unknown"
            self.state[self.pending["object_id"]] = "outcome_unknown"

    def snapshot(self):
        self.expire()
        return {"schema": SCHEMA, "session_id": self.session_id, "run_id": self.run_id,
                "scene_sha256": self.scene_sha256, "simulation_only": True,
                "assistance": "pose_gated_fixed_joint", "friction_grasp_validated": False,
                "truth_usage": "fixture eligibility, privileged collision geometry and evaluation only",
                "world_frame": self.scene["world_frame"], "robot_model": self.scene["robot_model"],
                "planning_group": self.scene["group"], "tcp": self.scene["tcp"],
                "arm_action": "/joint_trajectory_controller/follow_joint_trajectory",
                "execute_action": "/execute_trajectory",
                "gripper_action": "/gripper_controller/follow_joint_trajectory",
                "finger_joints": ["gripper_left_finger_joint", "gripper_right_finger_joint"],
                "grasp_gate": {"tcp_above_box_centre_m": [0.006, 0.018], "horizontal_error_m": 0.006,
                               "yaw_error_degrees": 5, "fresh_feedback_max_age_s": 0.5},
                "source": {"rgb": "/opencv_fixture/source/rgb", "depth": "/opencv_fixture/source/depth",
                           "camera_info": "/opencv_fixture/source/camera_info", "optical_frame": self.scene["camera"]["optical_frame"],
                           "world_from_source_camera": camera_transform(self.scene)},
                "camera_observed": self.camera_ready, "resetting": self.resetting,
                "attachments": dict(self.state), "pending": None if self.pending is None else dict(self.pending),
                "last_operation": self.last_operation,
                "pose_availability": {part: part in self.poses and self._fresh(self.poses[part].received) for part in self.state}}
