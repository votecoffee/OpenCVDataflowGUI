"""Bounded serial-chain FK for trajectory review, not a collision representation."""

import hashlib
import json
import math
from xml.etree import ElementTree as ET

import numpy as np


def vector(text, default):
    values = np.asarray([float(v) for v in text.split()] if text else default, float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("URDF vectors must have three finite values")
    return values


def rotation(axis, angle):
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def origin(element):
    result = np.eye(4)
    if element is not None:
        result[:3, 3] = vector(element.get("xyz"), [0, 0, 0])
        r, p, y = vector(element.get("rpy"), [0, 0, 0])
        result[:3, :3] = rotation([0, 0, 1], y) @ rotation([0, 1, 0], p) @ rotation([1, 0, 0], r)
    return result


def joint_samples(trajectory, start, maximum_step=0.01, maximum_states=2048):
    names = list(trajectory["joint_names"])
    if not names or len(set(names)) != len(names):
        raise ValueError("Unique trajectory joints required")
    previous = np.array([start[n] for n in names], float)
    samples = []
    for point in trajectory["points"]:
        values = np.asarray(point["positions"], float)
        if values.shape != previous.shape or not np.isfinite(values).all():
            raise ValueError("Invalid retimed joint state")
        count = max(1, int(np.ceil(np.max(np.abs(values - previous)) / maximum_step)))
        if len(samples) + count > maximum_states:
            raise ValueError("Post-retiming state limit exceeded")
        for alpha in np.linspace(1 / count, 1, count):
            samples.append(
                {**start, **dict(zip(names, (previous + alpha * (values - previous)).tolist()))}
            )
        previous = values
    return samples


class SerialChain:
    def __init__(self, urdf, root="world", tip="gripper_tcp"):
        if not isinstance(urdf, str) or len(urdf.encode()) > 2 * 1024 * 1024:
            raise ValueError("Robot description exceeds 2 MiB")
        if "<!DOCTYPE" in urdf.upper() or "<!ENTITY" in urdf.upper():
            raise ValueError("URDF entities are unsupported")
        robot = ET.fromstring(urdf)
        if robot.tag != "robot":
            raise ValueError("Expected a URDF robot")
        by_child, names = {}, set()
        for joint in robot.findall("joint"):
            name = joint.get("name")
            parent, child = joint.find("parent"), joint.find("child")
            if not name or name in names or parent is None or child is None:
                raise ValueError("URDF joints require unique names and parent/child links")
            names.add(name)
            child_name = child.get("link")
            if not child_name or child_name in by_child:
                raise ValueError("URDF links must have one parent")
            by_child[child_name] = joint
        chain, seen = [], set()
        current = tip
        while current != root:
            if current in seen or current not in by_child or len(chain) >= 64:
                raise ValueError("Tip must have a bounded acyclic chain to the root")
            seen.add(current)
            joint = by_child[current]
            kind = joint.get("type")
            if kind not in {"fixed", "revolute", "continuous", "prismatic"}:
                raise ValueError("Unsupported serial-chain joint type")
            axis_element = joint.find("axis")
            axis = vector(None if axis_element is None else axis_element.get("xyz"), [1, 0, 0])
            if np.linalg.norm(axis) < 1e-12:
                raise ValueError("Joint axis must be nonzero")
            axis /= np.linalg.norm(axis)
            parent = joint.find("parent").get("link")
            limit = joint.find("limit")
            bounds = None
            if kind in {"revolute", "prismatic"} and limit is not None:
                bounds = [float(limit.get("lower")), float(limit.get("upper"))]
                if not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
                    raise ValueError("Joint limits must be finite and ordered")
            chain.append(
                dict(
                    name=joint.get("name"),
                    kind=kind,
                    parent=parent,
                    child=current,
                    origin=origin(joint.find("origin")).tolist(),
                    axis=axis.tolist(),
                    bounds=bounds,
                )
            )
            current = parent
        if not chain:
            raise ValueError("Use a nonempty root-to-tip chain")
        self._record = dict(
            model="serial_chain_review_v1",
            units="m",
            root=root,
            tip=tip,
            joints=list(reversed(chain)),
            collision_geometry=False,
        )
        encoded = json.dumps(self._record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.sha256 = hashlib.sha256(encoded.encode()).hexdigest()
        self.required = tuple(j["name"] for j in self._record["joints"] if j["kind"] != "fixed")

    def record(self):
        return json.loads(json.dumps({**self._record, "sha256": self.sha256}))

    def forward(self, positions):
        transform = np.eye(4)
        skeleton = [transform[:3, 3].copy()]
        for joint in self._record["joints"]:
            transform = transform @ np.asarray(joint["origin"])
            motion = np.eye(4)
            if joint["kind"] != "fixed":
                value = float(positions[joint["name"]])
                if not math.isfinite(value):
                    raise ValueError("Joint positions must be finite")
                if joint["kind"] == "prismatic":
                    motion[:3, 3] = np.asarray(joint["axis"]) * value
                else:
                    motion[:3, :3] = rotation(joint["axis"], value)
            transform = transform @ motion
            skeleton.append(transform[:3, 3].copy())
        return transform, np.asarray(skeleton)

    def trajectory(self, trajectory, start, maximum=512):
        names = list(trajectory["joint_names"])
        points = trajectory["points"]
        if len(set(names)) != len(names) or not 1 <= len(names) <= 64:
            raise ValueError("Trajectory needs unique joint names")
        if not 1 <= len(points) <= 20000 or not 2 <= maximum <= 2048:
            raise ValueError("Trajectory review sample limit exceeded")
        times = [
            p["time_from_start"]["sec"] + p["time_from_start"]["nanosec"] * 1e-9 for p in points
        ]
        if (
            not np.isfinite(times).all()
            or times[0] < 0
            or any(b <= a for a, b in zip(times, times[1:]))
        ):
            raise ValueError("Trajectory times must increase")
        # Validate all states before thinning; a skipped invalid waypoint stays invalid.
        for point in points:
            if len(point["positions"]) != len(names) or not np.isfinite(point["positions"]).all():
                raise ValueError("Trajectory joint positions are invalid")
        indices = np.unique(np.linspace(0, len(points) - 1, min(maximum, len(points)), dtype=int))
        xyz, skeletons = [], []
        for index in indices:
            state = {**start, **dict(zip(names, points[index]["positions"]))}
            pose, skeleton = self.forward(state)
            xyz.append(pose[:3, 3].tolist())
            skeletons.append(skeleton.tolist())
        return dict(
            model="tcp_path_review_v1",
            frame=self._record["root"],
            units="m",
            kinematic_sha256=self.sha256,
            source="planned_joint_waypoint_fk",
            times_s=[times[i] for i in indices],
            positions_m=xyz,
            skeleton_start=skeletons[0],
            skeleton_end=skeletons[-1],
            total_waypoints=len(points),
            displayed_waypoints=len(indices),
            decimated=len(indices) < len(points),
            collision_certificate=False,
        )

    def joint_candidates(self, solution, current):
        """Equivalent revolute configurations preserve distinct bounded winding routes."""
        nearest = dict(solution)
        alternatives = {}
        for joint in self._record["joints"]:
            if joint["kind"] not in {"revolute", "continuous"}:
                continue
            name = joint["name"]
            value = solution[name]
            values = [value + 2 * math.pi * k for k in range(-2, 3)]
            if joint["bounds"] is not None:
                low, high = joint["bounds"]
                values = [v for v in values if low - 1e-9 <= v <= high + 1e-9]
            if not values:
                raise ValueError("IK solution is outside joint limits")
            values.sort(key=lambda v: abs(v - current[name]))
            nearest[name] = values[0]
            alternatives[name] = values[1:]
        result = [nearest]
        for name, values in alternatives.items():
            for value in values:
                result.append({**nearest, name: value})
        return result[:32]
