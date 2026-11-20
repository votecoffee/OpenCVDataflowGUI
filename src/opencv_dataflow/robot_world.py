"""Offline fixed-camera and robot-world/hand-eye fitting with independent closure checks."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .core import Block, Port, Result
from .geometry3d_common import pose_matrix, pose_record
from .hand_eye import _paired_samples, _proper_rotation, _rotation_errors, _solve_park, _transforms
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .tickets import ticket_key

ARRANGEMENTS = ("fixed_camera", "robot_mounted_camera")
SAMPLE_FIELDS = ("gripper_poses", "target_poses", "heldout_gripper_poses", "heldout_target_poses")


def read_sample_inputs(block, inputs):
    path = block.parameters["path"]
    if not path:
        return inputs
    if inputs:
        raise ValidationError("Use an observations file or the pose connectors")
    path = Path(path)
    if not path.is_absolute() and block.parameters["path_base"] == "project":
        if block.project_dir is None:
            raise ValidationError("Open or save the project before resolving relative observations")
        path = block.project_dir / path
    with path.open("rb") as stream:
        raw = stream.read(2 * 1024 * 1024 + 1)
    if len(raw) > 2 * 1024 * 1024:
        raise ValidationError("Robot-world observations file exceeds 2 MiB")
    values = json.loads(raw)
    if not isinstance(values, dict) or set(values) - set(SAMPLE_FIELDS):
        raise ValidationError("Observations file must contain the named robot-world pose arrays")
    return values


def pose_pairs(inputs, *, heldout=False):
    prefix = "heldout_" if heldout else ""
    robot, target = (inputs.get(prefix + name) for name in SAMPLE_FIELDS[:2])
    if heldout and robot is None and target is None:
        return None
    if robot is None or target is None:
        raise ValidationError("Supply both gripper and target poses for each requested split")
    return _paired_samples(robot, target, minimum=1 if heldout else 3)


def matrix_pairs(poses, frames, units):
    robots, boards, tickets = poses
    g, b, w, c = frames
    grippers = _transforms(robots, g, b, units)
    boards = _transforms(boards, w, c, units)
    if units == "m":
        grippers[:, :3, 3] *= 1000
        boards[:, :3, 3] *= 1000
    return grippers, boards, tickets


def packing(transform):
    return np.r_[Rotation.from_matrix(transform[:3, :3]).as_rotvec(), transform[:3, 3]]


def unpacking(params):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(params[:3]).as_matrix()
    transform[:3, 3] = params[3:]
    return transform


def closure(grippers, boards, mount, station):
    left = grippers @ mount
    right = station @ boards
    angular = left[:, :3, :3].transpose(0, 2, 1) @ right[:, :3, :3]
    return left[:, :3, 3] - right[:, :3, 3], angular


def fit_joint(grippers, boards, weight, evaluations):
    """Fit G X = Z B after a constrained relative-motion initialization."""
    inverse = np.linalg.inv(boards)
    mount, rotation_pairs = _solve_park(grippers, inverse)
    stationary = grippers @ mount @ inverse
    rotation, _ = _proper_rotation(stationary[:, :3, :3].mean(axis=0))
    station = pose_matrix(rotation, stationary[:, :3, 3].mean(axis=0))

    def residual(params):
        x, z = unpacking(params[:6]), unpacking(params[6:])
        translation, angular = closure(grippers, boards, x, z)
        return np.column_stack(
            (translation, Rotation.from_matrix(angular).as_rotvec() * weight)
        ).ravel()

    initial = np.r_[packing(mount), packing(station)]
    fit = least_squares(
        residual, initial, x_scale="jac", max_nfev=evaluations, ftol=1e-10, xtol=1e-10, gtol=1e-10
    )
    jacobian = np.asarray(fit.jac)
    # Check rank in comparable parameter scales regardless of the rotation lever arm.
    scale = np.linalg.norm(jacobian, axis=0)
    if np.any(scale == 0) or np.linalg.matrix_rank(jacobian / scale, tol=1e-8) < 12:
        raise ValidationError("Pose motion does not independently constrain both rigid transforms")
    return unpacking(fit.x[:6]), unpacking(fit.x[6:]), fit, rotation_pairs


def split_report(pairs, mount, station):
    grippers, boards, tickets = pairs
    offsets, angular = closure(grippers, boards, mount, station)
    t = np.linalg.norm(offsets, axis=1)
    r = _rotation_errors(angular, np.eye(3))
    if not np.isfinite(t).all() or not np.isfinite(r).all():
        raise ValidationError("Robot-world closure contains nonfinite errors")
    return {
        "sample_count": len(t),
        "translation_rms_mm": float(np.sqrt(np.mean(t**2))),
        "rotation_rms_degrees": float(np.sqrt(np.mean(r**2))),
        "samples": [
            {
                "index": i,
                "ticket": tickets[i]["value"] if tickets is not None else None,
                "translation_error_mm": float(shift),
                "rotation_error_degrees": float(angle),
            }
            for i, (shift, angle) in enumerate(zip(t, r, strict=True))
        ],
    }


class RobotWorldHandEye(Block):
    type_id, title, category = (
        "calibration.robot_world_hand_eye",
        "Calibrate robot-world and hand–eye",
        "Calibration",
    )
    block_type, subtype = "Calibration", "Robot coordinates"
    parallel_safe = True
    process_inputs_readonly = True
    inputs = tuple(
        Port(name, "any", required=False, required_when_connected=True) for name in SAMPLE_FIELDS
    )
    outputs = (
        Port("mount_to_gripper", "data", required=False, visual_type="pose"),
        Port("station_to_base", "data", required=False, visual_type="pose"),
        Port("report", "data"),
    )
    defaults = {
        "path": "",
        "path_base": "project",
        "arrangement": "fixed_camera",
        "gripper_frame": "gripper",
        "base_frame": "base",
        "target_frame": "target",
        "camera_frame": "camera",
        "rotation_weight_mm": 100.0,
        "maximum_evaluations": 200,
        "max_translation_rms_mm": 2.0,
        "max_rotation_rms_degrees": 1.0,
    }
    parameter_choices = {"path_base": ("project", "working_directory"), "arrangement": ARRANGEMENTS}
    parameter_ranges = {
        "rotation_weight_mm": NumericRange(0.001, 100_000),
        "maximum_evaluations": NumericRange(1, 1000, whole=True),
        "max_translation_rms_mm": NumericRange(0, 100_000),
        "max_rotation_rms_degrees": NumericRange(0, 180),
    }
    parameter_help = {
        "path": "Optional JSON with paired training and optional held-out pose arrays.",
        "path_base": "Base for a relative observations file.",
        "arrangement": "Fixed camera and moving target, or moving camera and fixed target.",
        "rotation_weight_mm": "Translation-equivalent residual per radian during joint refinement.",
        "maximum_evaluations": "Limit joint least-squares evaluations.",
        "max_translation_rms_mm": "Translation RMS limit on each supplied split. Zero disables it.",
        "max_rotation_rms_degrees": "Rotation RMS limit on each supplied split. Zero disables it.",
    }
    input_help = {
        "gripper_poses": "Gripper-to-base poses in training order or a ticketed group.",
        "target_poses": "Target-to-camera poses from corresponding training images.",
        "heldout_gripper_poses": "Independent gripper-to-base poses for validation.",
        "heldout_target_poses": "Independent target-to-camera observations for validation.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not isinstance(p["path"], str):
            raise ValidationError("Enter a JSON observations path or leave it empty", "path")
        frames = [p[f"{name}_frame"] for name in ("gripper", "base", "target", "camera")]
        if any(not isinstance(f, str) or not f.strip() for f in frames) or len(set(frames)) != 4:
            raise ValidationError(
                "Use four distinct frame names for gripper, base, target and camera"
            )

    def process(self, inputs):
        p = self.parameters
        values = read_sample_inputs(self, inputs)
        training_poses = pose_pairs(values)
        robot = training_poses[0]
        units = robot[0].get("units") if isinstance(robot[0], dict) else None
        if units not in ("m", "mm"):
            raise ValidationError("Robot-world poses require consistent meter or millimeter units")
        frames = tuple(p[f"{name}_frame"] for name in ("gripper", "base", "target", "camera"))
        training = matrix_pairs(training_poses, frames, units)
        heldout_poses = pose_pairs(values, heldout=True)
        heldout = matrix_pairs(heldout_poses, frames, units) if heldout_poses else None
        if heldout and training[2] and heldout[2]:
            if {ticket_key(v) for v in training[2]} & {ticket_key(v) for v in heldout[2]}:
                raise ValidationError("Training and held-out pose tickets must be disjoint")
        gripper, base, target, camera = frames
        moving_camera = p["arrangement"] == "robot_mounted_camera"
        if moving_camera:
            training = training[0], np.linalg.inv(training[1]), training[2]
            if heldout:
                heldout = heldout[0], np.linalg.inv(heldout[1]), heldout[2]
        mount_frame, station_frame = (camera, target) if moving_camera else (target, camera)
        mount, station, solver, pairs = fit_joint(
            training[0], training[1], float(p["rotation_weight_mm"]), int(p["maximum_evaluations"])
        )
        fit_report = split_report(training, mount, station)
        tests = {"fit": fit_report}
        if heldout:
            tests["heldout"] = split_report(heldout, mount, station)
        reasons = [] if solver.success else ["robot_world_optimizer_did_not_converge"]
        ratios = []
        for split, report in tests.items():
            for kind, unit in (("translation", "mm"), ("rotation", "degrees")):
                metric = f"{kind}_rms_{unit}"
                limit = float(p[f"max_{metric}"])
                if limit:
                    value = report[metric]
                    ratios.append(value / limit)
                    if value > limit:
                        reasons.append(
                            {
                                "code": f"robot_world_{split}_{kind}_rms",
                                "message": (
                                    f"{split} {kind} RMS {value:.4g} exceeds {limit:.4g} {unit}"
                                ),
                            }
                        )
        report = {
            "model": "robot_world_hand_eye_report_v1",
            "arrangement": p["arrangement"],
            "solver": "park_init_scipy_least_squares",
            "evaluations": solver.nfev,
            "converged": bool(solver.success),
            "rotation_pairs": pairs,
            "mount_frame": mount_frame,
            "station_frame": station_frame,
            **tests,
        }
        if units == "m":
            mount[:3, 3] /= 1000
            station[:3, 3] /= 1000
        return Result(
            {
                "mount_to_gripper": NO_VALUE
                if reasons
                else pose_record(mount, mount_frame, gripper, units),
                "station_to_base": NO_VALUE
                if reasons
                else pose_record(station, station_frame, base, units),
                "report": report,
            },
            passed=not reasons,
            fail_reasons=reasons,
            score=0.0 if reasons else max(0.0, 1.0 - max(ratios, default=0.0)),
            message=f"{len(training[0])} poses · {fit_report['translation_rms_mm']:.4g} mm fit RMS",
        )
