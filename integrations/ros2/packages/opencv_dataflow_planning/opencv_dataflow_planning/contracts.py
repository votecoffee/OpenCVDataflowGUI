"""Bounded app-owned planning requests and independently checked preview results."""

import copy
import hashlib
import json
import math
import re

MAX_BYTES = 1024 * 1024
MAX_SCENE_BYTES = 16 * MAX_BYTES
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}\Z")


def encoded(value, limit=MAX_BYTES):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(raw) > limit:
        raise ValueError("Planning data exceeds its byte budget")
    return raw


def decode(raw):
    if not isinstance(raw, str) or len(raw.encode()) > MAX_BYTES:
        raise ValueError("Request must be bounded JSON text")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    value = json.loads(raw, object_pairs_hook=pairs)
    encoded(value)
    return value


def fields(value, names):
    if not isinstance(value, dict) or set(value) != set(names.split()):
        raise ValueError("Unexpected or missing fields: " + names)


def identifier(value):
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError("Use a nonempty plain identifier of at most 128 characters")
    return value


def number(value, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError("Numeric field is nonfinite or outside its range")
    return float(value)


def vector(value, count, low=-1e6, high=1e6):
    if not isinstance(value, list) or len(value) != count:
        raise ValueError("Vector length does not match")
    return [number(x, low, high) for x in value]


def request(value):
    encoded(value)
    if not isinstance(value, dict):
        raise ValueError("Request must be an object")
    op = value.get("operation")
    base = "schema operation request_id run_id"
    fields(
        value,
        base
        if op == "snapshot"
        else base + " server_id"
        if op in ("cancel", "reset")
        else base + " server_id robot_revision scene_revision plan",
    )
    if value["schema"] != "opencv_dataflow_plan_request_v1" or op not in (
        "snapshot",
        "plan",
        "cancel",
        "reset",
    ):
        raise ValueError("Unknown request schema or operation")
    for key in ("request_id", "run_id"):
        identifier(value[key])
    if op != "snapshot":
        identifier(value["server_id"])
    if op == "plan":
        identifier(value["robot_revision"])
        if not isinstance(value["scene_revision"], str) or not re.fullmatch(
            "[0-9a-f]{64}", value["scene_revision"]
        ):
            raise ValueError("Use the scene digest returned by Snapshot")
        plan(value["plan"])
    return copy.deepcopy(value)


def plan(value):
    fields(
        value,
        "group_name frame_id joint_names joint_types start goal tolerance "
        "start_tolerance workspace_min workspace_max allowed_planning_time "
        "velocity_scale acceleration_scale pipeline_id planner_id",
    )
    for key in ("group_name", "frame_id"):
        identifier(value[key])
    for key in ("pipeline_id", "planner_id"):
        if value[key] != "":
            identifier(value[key])
    names = value["joint_names"]
    if (
        not isinstance(names, list)
        or not 1 <= len(names) <= 64
        or len(set(identifier(x) for x in names)) != len(names)
    ):
        raise ValueError("Provide unique planning joint names")
    n = len(names)
    if (
        not isinstance(value["joint_types"], list)
        or len(value["joint_types"]) != n
        or any(x not in ("revolute", "continuous", "prismatic") for x in value["joint_types"])
    ):
        raise ValueError("Declare each joint's URDF type; angles are radians, lengths meters")
    for key in ("start", "goal"):
        vector(value[key], n)
    vector(value["tolerance"], n, 1e-9, 1)
    number(value["start_tolerance"], 1e-9, 0.1)
    minimum = vector(value["workspace_min"], 3, -1000, 1000)
    maximum = vector(value["workspace_max"], 3, -1000, 1000)
    if any(a >= b for a, b in zip(minimum, maximum)):
        raise ValueError("Workspace bounds must be ordered in meters")
    number(value["allowed_planning_time"], 0.01, 60)
    for key in ("velocity_scale", "acceleration_scale"):
        number(value[key], 1e-6, 1)
    return copy.deepcopy(value)


def scene_digest(scene):
    """Ignore changing header stamps, retain state, frames, geometry, maps and ACM."""

    def normalize(value, depth=0):
        if depth > 48:
            raise ValueError("Scene nesting exceeds its budget")
        if isinstance(value, dict):
            return {
                k: {"sec": 0, "nanosec": 0} if k == "stamp" else normalize(v, depth + 1)
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [normalize(v, depth + 1) for v in value]
        return value

    return hashlib.sha256(encoded(normalize(scene), MAX_SCENE_BYTES)).hexdigest()


def state_positions(state):
    joint = state["joint_state"]
    names, positions = joint["name"], joint["position"]
    if (
        len(names) != len(set(names))
        or len(names) != len(positions)
        or len(names) > 256
        or state.get("multi_dof_joint_state", {}).get("joint_names")
    ):
        raise ValueError("Use a complete fixed-base robot joint state")
    return dict(zip(names, vector(positions, len(names))))


def moveit_request(configuration, scene):
    p = plan(configuration)
    state = copy.deepcopy(scene["robot_state"])
    current = state_positions(state)
    if any(
        name not in current or abs(current[name] - start) > p["start_tolerance"]
        for name, start in zip(p["joint_names"], p["start"])
    ):
        raise ValueError("Requested start does not match the planning-scene robot state")
    # Preserve non-group joints and attached bodies from the actual scene snapshot.
    positions = dict(zip(p["joint_names"], p["start"]))
    state["joint_state"]["position"] = [
        positions.get(name, current[name]) for name in state["joint_state"]["name"]
    ]
    state["is_diff"] = False
    return {
        "motion_plan_request": {
            "workspace_parameters": {
                "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": p["frame_id"]},
                "min_corner": dict(zip(("x", "y", "z"), p["workspace_min"])),
                "max_corner": dict(zip(("x", "y", "z"), p["workspace_max"])),
            },
            "start_state": state,
            "goal_constraints": [
                {
                    "name": "app_joint_goal",
                    "joint_constraints": [
                        {
                            "joint_name": name,
                            "position": float(goal),
                            "tolerance_above": float(tolerance),
                            "tolerance_below": float(tolerance),
                            "weight": 1.0,
                        }
                        for name, goal, tolerance in zip(
                            p["joint_names"], p["goal"], p["tolerance"]
                        )
                    ],
                }
            ],
            "pipeline_id": p["pipeline_id"],
            "planner_id": p["planner_id"],
            "group_name": p["group_name"],
            "num_planning_attempts": 1,
            "allowed_planning_time": float(p["allowed_planning_time"]),
            "max_velocity_scaling_factor": float(p["velocity_scale"]),
            "max_acceleration_scaling_factor": float(p["acceleration_scale"]),
        }
    }


def trajectory_preview(configuration, response):
    """Check fixed-base joint trajectories; never infer collision/physical validity."""
    encoded(response)
    p = plan(configuration)
    response = response["motion_plan_response"]
    code = response["error_code"]["val"]
    if type(code) is not int:
        raise ValueError("Invalid MoveIt error code")
    if code != 1:
        return {"status": "no_plan", "moveit_error_code": code}, None
    if response["group_name"] != p["group_name"]:
        raise ValueError("Planner returned another group")
    trajectory = response["trajectory"]
    if trajectory.get("multi_dof_joint_trajectory", {}).get("points"):
        raise ValueError("Multi-DOF trajectories are not supported by this preview adapter")
    joint = trajectory["joint_trajectory"]
    names = joint["joint_names"]
    if len(names) != len(set(names)) or set(names) != set(p["joint_names"]):
        raise ValueError("Trajectory joints differ from the requested group joints")
    indices = [names.index(name) for name in p["joint_names"]]
    points = joint["points"]
    if not 2 <= len(points) <= 10000:
        raise ValueError("A preview needs 2..10000 trajectory points")
    positions, times = [], []
    for row in points:
        values = vector(row["positions"], len(names))
        positions.append([values[i] for i in indices])
        for field in ("velocities", "accelerations", "effort"):
            if row.get(field):
                vector(row[field], len(names))
        stamp = row["time_from_start"]
        if (
            type(stamp["sec"]) is not int
            or not 0 <= stamp["sec"] <= 86400
            or type(stamp["nanosec"]) is not int
            or not 0 <= stamp["nanosec"] < 10**9
        ):
            raise ValueError("Invalid trajectory duration")
        times.append(stamp["sec"] * 10**9 + stamp["nanosec"])
    if times[0] != 0 or any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError("Require a zero-time start and strictly increasing trajectory times")
    if any(abs(a - b) > p["start_tolerance"] for a, b in zip(positions[0], p["start"])):
        raise ValueError("Trajectory does not start at the requested state")
    if any(
        abs(a - b) > tolerance for a, b, tolerance in zip(positions[-1], p["goal"], p["tolerance"])
    ):
        raise ValueError("Trajectory endpoint is outside the requested goal tolerances")
    start = state_positions(response["trajectory_start"])
    if any(
        name not in start or abs(start[name] - value) > p["start_tolerance"]
        for name, value in zip(p["joint_names"], p["start"])
    ):
        raise ValueError("Returned trajectory_start differs from the request")
    metrics = {
        "status": "preview",
        "moveit_error_code": code,
        "points": len(points),
        "planned_duration_s": times[-1] / 1e9,
        "provider_planning_s": number(response["planning_time"], 0, 86400),
        "joint_travel": [
            {
                "name": name,
                "units": "m" if kind == "prismatic" else "rad",
                "distance": sum(abs(a[i] - b[i]) for a, b in zip(positions, positions[1:])),
            }
            for i, (name, kind) in enumerate(zip(p["joint_names"], p["joint_types"]))
        ],
        "trajectory_sha256": hashlib.sha256(encoded(trajectory)).hexdigest(),
        "collision_validity": "provider_report_only",
        "execution_authorized": False,
        "physical_outcome": "not_measured",
    }
    return metrics, copy.deepcopy(response)
