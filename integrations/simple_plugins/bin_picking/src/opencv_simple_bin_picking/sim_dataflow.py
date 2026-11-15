"""Explicit simulation demo blocks; loaded by the example's trusted plugin."""

import hashlib
import json
import uuid
from pathlib import Path

import cv2
import numpy as np
from opencv_dataflow.core import Block, Port, Result
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket
from opencv_dataflow.pointclouds import PointCloud
from opencv_dataflow.tickets import DIRECT_CLOCK

from .motion import FEEDBACK, STAGES, path_graphics
from .parallel_jaw import BoxTopSettings
from .sim_client import SimClient
from .sim_workflow import pick_one
from .specs import number
from .top_template import TopTemplate


def cloud(capture, report):
    depth = capture["depth"]
    camera = capture["camera"]
    k = np.asarray(camera["camera_matrix"])
    valid = np.isfinite(depth) & (depth > 0)
    y, x = np.nonzero(valid)
    y, x = y[::4], x[::4]
    z = depth[y, x]
    points = np.c_[(x - k[0, 2]) * z / k[0, 0], (y - k[1, 2]) * z / k[1, 1], z]
    transform = np.asarray(capture["world_from_camera"])
    points = points @ transform[:3, :3].T + transform[:3, 3]
    colors = capture["rgb"][y, x, ::-1].copy()
    extra, extra_colors = [], []
    for candidate in report.get("detections", {}).get("candidates", []):
        for xy, matched in zip(
            candidate.get("expected_surface_xy_m", []),
            candidate.get("expected_surface_matched", []),
        ):
            extra.append([*xy, candidate["top_z_m"] + 0.001])
            extra_colors.append([50, 240, 100] if matched else [255, 70, 70])
        t = np.asarray(candidate["world_from_tcp"])
        for axis, color in zip(range(3), ([255, 70, 70], [70, 255, 70], [70, 120, 255])):
            extra.extend(t[:3, 3] + np.linspace(0, 0.04, 30)[:, None] * t[:3, axis])
            extra_colors.extend([color] * 30)
    if extra:
        points = np.vstack([points, extra])
        colors = np.vstack([colors, extra_colors])
    return PointCloud(
        points,
        units="m",
        frame="world",
        colors=colors,
        metadata={
            "simulation_only": True,
            "source": "measured first-return RGB-D",
            "capture_id": capture["capture_id"],
            "pose_axes": "candidate tool XYZ; not executed path",
        },
    )


def overlay(capture, report):
    image = capture["rgb"].copy()
    support = capture.get("support")
    if support is not None:
        selected = support > 0
        image[selected] = (image[selected] * 0.45 + np.array([30, 220, 50]) * 0.55).astype(np.uint8)
    text = f"Cell {report['cell']}: {report['status']}  {report.get('object_id', '')}"
    cv2.rectangle(image, (0, 0), (image.shape[1], 54), (25, 25, 25), -1)
    cv2.putText(
        image, text, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA
    )
    cv2.putText(
        image,
        "SIMULATION / assisted grasp / depth perception",
        (12, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (80, 210, 250),
        1,
        cv2.LINE_AA,
    )
    return image


def motion_cloud(capture, report):
    base = cloud(capture, report)
    points, colors, metadata = path_graphics(report)
    if not len(points):
        return None
    return PointCloud(
        np.vstack([base.points, points]),
        units="m",
        frame="world",
        colors=np.vstack([base.colors, colors]),
        metadata={
            **base.metadata,
            "motion_review": metadata,
            "scene_time": "recorded source timestamp; historical paths, not collision geometry",
        },
    )


def motion_summary(report):
    _, _, metadata = path_graphics(report)
    rows = metadata["paths"]
    image = np.full((150 + 30 * max(1, len(rows)), 960, 3), 26, np.uint8)
    cv2.putText(
        image,
        f"Cell {report['cell']} motion review: {report.get('status', 'unknown')}",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "Colored lines: planned FK | pink dots: measured-joint FK",
        (18, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "Model-based review; no independent pose measurement or collision certificate",
        (18, 91),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (80, 210, 250),
        1,
        cv2.LINE_AA,
    )
    if not rows:
        cv2.putText(
            image,
            "No motion paths recorded in this older episode",
            (18, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (80, 210, 250),
            1,
        )
    for index, row in enumerate(rows):
        measured = row["kind"] != "planned"
        color = (FEEDBACK if measured else STAGES.get(row["stage"], STAGES["transfer"]))[::-1]
        kind = "feedback" if measured else "planned"
        text = f"{row['stage']:12} {kind:9} {row['samples']:4} samples"
        if row.get("duration_s") is not None:
            text += f"   {row['duration_s']:.2f} s"
        if measured:
            text += f"   {row.get('status', 'unknown')}"
        cv2.putText(
            image,
            text,
            (18, 135 + 30 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            color,
            1,
            cv2.LINE_AA,
        )
    return image


def save_pick(root, report, captures):
    root.mkdir(parents=True, exist_ok=False)
    for name, capture in captures.items():
        cv2.imwrite(str(root / f"{name}.png"), capture["rgb"])
        cv2.imwrite(str(root / f"{name}-overlay.png"), overlay(capture, report))
        np.save(root / f"{name}-depth.npy", capture["depth"], allow_pickle=False)
        result = cloud(capture, report)
        np.savez_compressed(root / f"{name}-cloud.npz", points=result.points, colors=result.colors)
        meta = {k: v for k, v in capture.items() if k not in {"rgb", "depth", "support"}}
        (root / f"{name}-capture.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
    if captures:
        latest = captures.get("after", captures.get("before"))
        motion = motion_cloud(latest, report)
        if motion is not None:
            np.savez_compressed(
                root / "motion-cloud.npz", points=motion.points, colors=motion.colors
            )
            cv2.imwrite(str(root / "motion-summary.png"), motion_summary(report))
    (root / "outcome.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


class GridTasks(Block):
    type_id, title = "input.sim_bin_grid", "Bin to table grid"
    category, block_type, subtype = "3D", "3D", "Simulation examples"
    inputs = (Port("run_id", "text", required=False),)
    outputs = (Port("group", "any"),)
    defaults = {
        "rows": 2,
        "columns": 3,
        "origin_x_m": 0.35,
        "origin_y_m": 0.36,
        "pitch_x_m": 0.10,
        "pitch_y_m": 0.14,
        "part_center_z_m": 0.22,
        "park_x_m": 0.15,
        "park_y_m": -0.40,
        "park_z_m": 0.60,
    }
    parameter_help = {
        "part_center_z_m": "Desired object centre: table top plus half part height, metres.",
        "origin_x_m": "First destination cell in world metres; check fixture table/reach.",
        "run_id": "Optional external RunIdentity token; otherwise a new UUID per graph run.",
    }
    preview_safe = False

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if (
            any(type(p[k]) is not int or not 1 <= p[k] <= 8 for k in ("rows", "columns"))
            or p["rows"] * p["columns"] > 64
        ):
            raise ValidationError("Use a grid of 1..64 cells", "rows", "columns")
        if (
            not np.isfinite([p[k] for k in p if k not in {"rows", "columns"}]).all()
            or min(p["pitch_x_m"], p["pitch_y_m"]) < 0.06
        ):
            raise ValidationError(
                "Use finite metre coordinates and at least 60 mm pitch", "pitch_x_m", "pitch_y_m"
            )

    def process(self, inputs):
        p = self.parameters
        run = inputs.get("run_id") or str(uuid.uuid4())
        entries = []
        for row in range(p["rows"]):
            for column in range(p["columns"]):
                ticket = getattr(self, "_clock", DIRECT_CLOCK).issue(self.id)
                task = {
                    "run_id": run,
                    "cell": len(entries) + 1,
                    "row": row,
                    "column": column,
                    "xyz_m": [
                        p["origin_x_m"] + column * p["pitch_x_m"],
                        p["origin_y_m"] + row * p["pitch_y_m"],
                        p["part_center_z_m"],
                    ],
                }
                entries.append(ValuePacket(task, {"ticket": ticket}))
        park = np.diag([1.0, -1.0, -1.0, 1.0])
        park[:3, 3] = [p["park_x_m"], p["park_y_m"], p["park_z_m"]]
        entries[-1].value["park_world_from_tcp"] = park.tolist()
        entries[0].value["observation_world_from_tcp"] = park.tolist()
        return Result(
            {
                "group": FlowGroup(
                    tuple(entries), tuple(f"Cell {i + 1}" for i in range(len(entries)))
                )
            }
        )


class SimPickOne(Block):
    type_id, title = "3d.sim_bin_pick_one", "Simulated pick and grid place"
    category, block_type, subtype = "3D", "3D", "Simulation examples"
    inputs = (Port("task", "data"),)
    outputs = (
        Port("outcome", "data"),
        Port("before", "image", required=False),
        Port("after", "image", required=False),
        Port("graphics", "cloud", required=False),
        Port("motion", "cloud", required=False),
        Port("motion_summary", "image", required=False),
    )
    defaults = {
        "enabled": False,
        "bin_x_min_m": 0.335,
        "bin_x_max_m": 0.565,
        "bin_y_min_m": -0.235,
        "bin_y_max_m": 0.235,
        "floor_z_m": 0.2,
        "part_height_m": 0.04,
        "approach_height_m": 0.14,
        "archive_directory": ".local/bin-grid-runs",
        "planning_time_s": 10.0,
        "velocity_scale": 0.3,
        "acceleration_scale": 0.3,
        "top_template_path": "",
        "top_template_sha256": "",
    }
    parameter_help = {
        "enabled": "Explicit simulation execution; false emits disabled outcomes without I/O.",
        "archive_directory": "Local receipts, source images, depth and flyaround clouds.",
        "approach_height_m": "Retract above source/destination in metres; collision checked.",
        "planning_time_s": "Global planning budget per move, 1..30 seconds.",
        "velocity_scale": "Fraction of the configured robot's velocity limits, at most 0.5.",
        "acceleration_scale": "Fraction of configured acceleration limits, at most 0.5.",
        "top_template_path": "Expected surface samples; empty requires a filled rectangular top.",
    }
    project_asset_parameters = ("top_template_path",)
    requires_serial_execution = True
    parallel_safe = False
    preview_safe = False
    process_inputs_readonly = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = None
        self.run = None
        self.held = None
        self.completed = []
        self.preview_output = "motion"

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if type(p["enabled"]) is not bool:
            raise ValidationError("Enabled must be boolean", "enabled")
        if p["top_template_path"] and (
            len(p["top_template_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in p["top_template_sha256"])
        ):
            raise ValidationError(
                "Surface template requires its SHA256", "top_template_path", "top_template_sha256"
            )
        for field, label, low, high in (
            ("planning_time_s", "Planning budget", 1, 30),
            ("velocity_scale", "Velocity scale", 0.01, 0.5),
            ("acceleration_scale", "Acceleration scale", 0.01, 0.5),
        ):
            try:
                number(p[field], label, low, high)
            except (ValueError, TypeError) as error:
                raise ValidationError(str(error), field) from error
        try:
            BoxTopSettings(
                minimum=(p["bin_x_min_m"], p["bin_y_min_m"]),
                maximum=(p["bin_x_max_m"], p["bin_y_max_m"]),
                floor_z=p["floor_z_m"],
                part_height=p["part_height_m"],
                approach_height=p["approach_height_m"],
            )
        except (ValueError, TypeError) as error:
            raise ValidationError(
                str(error),
                "bin_x_min_m",
                "bin_x_max_m",
                "bin_y_min_m",
                "bin_y_max_m",
                "floor_z_m",
                "part_height_m",
                "approach_height_m",
            ) from error
        try:
            relative = Path(p["archive_directory"])
            if relative.is_absolute() or ".." in relative.parts or ":" in str(relative):
                raise ValueError("Archive directory must stay in the project")
        except (ValueError, TypeError) as error:
            raise ValidationError(str(error), "archive_directory") from error

    def process(self, inputs):
        task = inputs["task"]
        p = self.parameters
        captures = {}
        if task["run_id"] != self.run:
            self.close()
            self.run = task["run_id"]
            self.held = None
            self.completed = []
        if not p["enabled"] or self.held:
            report = {
                "schema": "bin_grid_pick_v1",
                **task,
                "status": "skipped" if self.held else "disabled",
                "reason": self.held or "Enable only in the allocated simulation environment",
                "simulation_only": True,
            }
        else:
            try:
                if self.client is None:
                    self.client = SimClient(self.run, getattr(self, "_cancelled", lambda: False))
                settings = BoxTopSettings(
                    minimum=(p["bin_x_min_m"], p["bin_y_min_m"]),
                    maximum=(p["bin_x_max_m"], p["bin_y_max_m"]),
                    floor_z=p["floor_z_m"],
                    part_height=p["part_height_m"],
                    approach_height=p["approach_height_m"],
                )
                report, captures = pick_one(
                    self.client,
                    task,
                    settings,
                    lambda phase, _: self.logger.info("Cell %s: %s", task["cell"], phase),
                    self.completed,
                    motion_settings={
                        k: p[k] for k in ("planning_time_s", "velocity_scale", "acceleration_scale")
                    },
                    top_template=TopTemplate.load(
                        Path(self.project_dir or ".") / p["top_template_path"],
                        p["top_template_sha256"],
                    )
                    if p["top_template_path"]
                    else None,
                )
                if report["status"] == "placed":
                    self.completed.append(report["object_id"])
                else:
                    self.held = report.get("reason", "Unresolved pick")
            except Exception as error:
                self.held = str(error)
                report = {
                    "schema": "bin_grid_pick_v1",
                    **task,
                    "status": "held",
                    "reason": self.held,
                    "simulation_only": True,
                }
        if p["enabled"]:
            project = Path(self.project_dir or ".").resolve()
            root = (
                project
                / p["archive_directory"]
                / hashlib.sha256(self.run.encode()).hexdigest()[:24]
                / f"cell-{task['cell']:02d}"
            ).resolve()
            try:
                if not root.is_relative_to(project):
                    raise ValueError("Archive resolves outside the project")
                report["archive"] = str(root)
                save_pick(root, report, captures)
            except (OSError, ValueError) as error:
                self.held = f"Archive failure after {report['status']}: {error}"
                report = {**report, "status": "held", "reason": self.held}
        after = captures.get("after", captures.get("before"))
        outputs = {
            "outcome": report,
            "before": overlay(captures["before"], report) if "before" in captures else NO_VALUE,
            "after": overlay(after, report) if after else NO_VALUE,
            "graphics": cloud(after, report) if after and self.generate_graphics else NO_VALUE,
            "motion": (motion_cloud(after, report) or NO_VALUE)
            if after and self.generate_graphics
            else NO_VALUE,
            "motion_summary": motion_summary(report) if self.generate_graphics else NO_VALUE,
        }
        # The summary owns overall judgment so every attempted/skipped cell reaches Batch.
        return Result(outputs, message=f"Cell {task['cell']}: {report['status']}")

    def close(self):
        if self.client is not None:
            self.client.close()
            self.client = None


class GridSummary(Block):
    type_id, title = "3d.sim_bin_grid_summary", "Grid pick summary"
    category, block_type, subtype = "3D", "3D", "Simulation examples"
    inputs = (Port("group", "any"),)
    outputs = (Port("report", "data"), Port("placed", "integer"), Port("image", "image"))
    defaults = {}
    preview_safe = True

    def process(self, inputs):
        group = inputs["group"]
        rows = [p.value for p in group.items]
        passed = all(row["status"] == "placed" for row in rows)
        report = {
            "schema": "bin_grid_summary_v1",
            "simulation_only": True,
            "complete": passed,
            "attempted_cells": len(rows),
            "placed": sum(row["status"] == "placed" for row in rows),
            "outcomes": rows,
        }
        image = np.full((120 + len(rows) * 50, 1000, 3), 26, np.uint8)
        cv2.putText(
            image,
            "SIMULATION / bin to table grid",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (240, 240, 240),
            2,
        )
        cv2.putText(
            image,
            "Assisted fixed-joint grasp; friction grasp not validated",
            (25, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (80, 210, 250),
            1,
        )
        for index, row in enumerate(rows):
            text = f"Cell {row['cell']}  {row['status']}  {row.get('object_id', '')}"
            if "placement_error_m" in row:
                text += f"  error {row['placement_error_m'] * 1000:.2f} mm"
            cv2.putText(
                image,
                text,
                (25, 130 + index * 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (80, 230, 100) if row["status"] == "placed" else (70, 160, 255),
                1,
            )
        return Result(
            {"report": report, "placed": report["placed"], "image": image},
            passed=passed,
            message=f"{report['placed']}/{len(rows)} placed",
        )


class PickReview(Block):
    type_id, title = "input.sim_pick_review", "Recorded bin-grid pick"
    category, block_type, subtype = "3D", "3D", "Simulation examples"
    outputs = (
        Port("outcome", "data"),
        Port("before", "image", required=False),
        Port("after", "image", required=False),
        Port("graphics", "cloud", required=False),
        Port("motion", "cloud", required=False),
        Port("motion_summary", "image", required=False),
    )
    defaults = {"archive_directory": "assets/run", "cell": 6}
    project_path_parameters = ("archive_directory",)
    preview_safe = True
    parameter_help = {
        "cell": "Select an independently recorded pick; the cloud thumbnail supports flyaround.",
        "archive_directory": "Retained run folder containing cell-NN outcomes and camera evidence.",
    }

    def process(self, inputs):
        p = self.parameters
        if type(p["cell"]) is not int or not 1 <= p["cell"] <= 64:
            raise ValueError("Select cell 1..64")
        root = Path(self.project_dir or ".") / p["archive_directory"] / f"cell-{p['cell']:02d}"
        report = json.loads((root / "outcome.json").read_text(encoding="utf-8"))
        if report.get("schema") != "bin_grid_pick_v1" or report.get("simulation_only") is not True:
            raise ValueError("Expected an identified simulation pick outcome")
        outputs = {
            "outcome": report,
            "before": NO_VALUE,
            "after": NO_VALUE,
            "graphics": NO_VALUE,
            "motion": NO_VALUE,
            "motion_summary": NO_VALUE,
        }
        for name in ("before", "after"):
            if (root / f"{name}-overlay.png").is_file():
                outputs[name] = cv2.imread(str(root / f"{name}-overlay.png"))
        name = "after" if (root / "after-cloud.npz").is_file() else "before"
        if self.generate_graphics and (root / f"{name}-cloud.npz").is_file():
            with np.load(root / f"{name}-cloud.npz", allow_pickle=False) as arrays:
                outputs["graphics"] = PointCloud(
                    arrays["points"],
                    units="m",
                    frame="world",
                    colors=arrays["colors"],
                    metadata={"simulation_only": True, "recorded": True, "cell": p["cell"]},
                )
        if self.generate_graphics:
            outputs["motion_summary"] = motion_summary(report)
            if (root / "motion-cloud.npz").is_file():
                with np.load(root / "motion-cloud.npz", allow_pickle=False) as arrays:
                    outputs["motion"] = PointCloud(
                        arrays["points"],
                        units="m",
                        frame="world",
                        colors=arrays["colors"],
                        metadata={
                            "simulation_only": True,
                            "recorded": True,
                            "cell": p["cell"],
                            "paths": "planned FK lines / joint-feedback FK dots",
                        },
                    )
        return Result(
            outputs,
            passed=report["status"] == "placed",
            message=f"Recorded cell {p['cell']}: {report['status']}",
        )


def register(registry):
    for cls in (GridTasks, SimPickOne, GridSummary, PickReview):
        registry.register(cls)
