"""Eye-in-hand calibration from paired robot and target poses."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .geometry3d_common import pose_matrix, pose_record, rigid
from .outcomes import failed_input
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, FlowGroup
from .tickets import ticket_key


def _samples(value, label):
    if isinstance(value, FlowGroup):
        if any(failed_input(item) for item in value.items):
            raise ValidationError(f"All {label} samples must pass")
        tickets = [item.payload.get("ticket") for item in value.items]
        if any(ticket is None for ticket in tickets):
            raise ValidationError(f"Every {label} sample in a group needs a ticket")
        poses = [item.value for item in value.items]
    elif isinstance(value, list):
        poses, tickets = value, None
    else:
        raise ValidationError(f"Provide {label} poses as a list or a ticketed group")
    if not 3 <= len(poses) <= 64:
        raise ValidationError(f"Provide 3 to 64 {label} poses")
    return poses, tickets


def _paired_samples(robot, target):
    gripper, robot_tickets = _samples(robot, "gripper-to-base")
    board, target_tickets = _samples(target, "target-to-camera")
    if len(gripper) != len(board):
        raise ValidationError("Gripper and target pose counts must match")
    if robot_tickets is None and target_tickets is None:
        return gripper, board, None
    if robot_tickets is None or target_tickets is None:
        raise ValidationError("Supply two pose lists or two ticketed pose groups")
    robot_keys = [ticket_key(ticket) for ticket in robot_tickets]
    target_keys = [ticket_key(ticket) for ticket in target_tickets]
    if len(set(robot_keys)) != len(robot_keys) or len(set(target_keys)) != len(target_keys):
        raise ValidationError("Each hand-eye pose group needs unique sample tickets")
    if set(robot_keys) != set(target_keys):
        raise ValidationError("Gripper and target groups must contain the same sample tickets")
    by_ticket = dict(zip(target_keys, board))
    return gripper, [by_ticket[key] for key in robot_keys], robot_tickets


def _transforms(poses, source, target, units):
    result = []
    for index, pose in enumerate(poses, start=1):
        if (
            not isinstance(pose, dict)
            or pose.get("model") != "pose_3d_v1"
            or pose.get("source_frame") != source
            or pose.get("target_frame") != target
            or pose.get("units") != units
        ):
            raise ValidationError(
                f"Sample {index} must be a rigid pose from {source} to {target} in {units}"
            )
        result.append(rigid(pose))
    return np.asarray(result)


def _proper_rotation(matrix):
    left, singular, right = np.linalg.svd(matrix)
    if np.linalg.det(left @ right) < 0:
        left[:, -1] *= -1
    return left @ right, singular


def _solve_park(gripper, target):
    # With G_i X C_i constant, A = G_j^-1 G_i and B = C_j C_i^-1 satisfy A X = X B.
    i, j = np.triu_indices(len(gripper), k=1)
    a = np.linalg.inv(gripper)[j] @ gripper[i]
    b = target[j] @ np.linalg.inv(target)[i]
    alpha = np.array([cv2.Rodrigues(r)[0].reshape(3) for r in a[:, :3, :3]])
    beta = np.array([cv2.Rodrigues(r)[0].reshape(3) for r in b[:, :3, :3]])
    # The principal axis-angle sign is ambiguous near a half turn.
    usable = (np.linalg.norm(alpha, axis=1) < np.pi - 1e-3) & (
        np.linalg.norm(beta, axis=1) < np.pi - 1e-3
    )
    rotation, singular = _proper_rotation(alpha[usable].T @ beta[usable])
    if singular[1] <= 1e-10 * max(1.0, singular[0]):
        raise ValidationError("Hand-eye fitting needs rotation about at least two independent axes")
    design = (a[:, :3, :3] - np.eye(3)).reshape(-1, 3)
    offset = (b[:, :3, 3] @ rotation.T - a[:, :3, 3]).reshape(-1)
    translation, _, rank, _ = np.linalg.lstsq(design, offset, rcond=1e-8)
    if rank != 3:
        raise ValidationError("Hand-eye motion does not constrain translation on all three axes")
    return pose_matrix(rotation, translation), int(np.count_nonzero(usable))


def _rotation_errors(rotation, mean):
    relative = mean.T @ rotation
    skew = np.column_stack(
        (
            relative[:, 2, 1] - relative[:, 1, 2],
            relative[:, 0, 2] - relative[:, 2, 0],
            relative[:, 1, 0] - relative[:, 0, 1],
        )
    )
    return np.degrees(
        np.arctan2(np.linalg.norm(skew, axis=1), np.trace(relative, axis1=1, axis2=2) - 1)
    )


class HandEyeCalibration(Block):
    type_id, title, category = "calibration.hand_eye", "Calibrate hand–eye", "Calibration"
    block_type, subtype = "Calibration", "Robot coordinates"
    inputs = (
        Port("gripper_poses", "any", required=False, required_when_connected=True),
        Port("target_poses", "any", required=False, required_when_connected=True),
    )
    outputs = (
        Port("camera_to_gripper", "data"),
        Port("target_to_base", "data"),
        Port("report", "data"),
        Port("translation_rms_mm", "number"),
        Port("rotation_rms_degrees", "number"),
    )
    defaults = {
        "gripper_poses": [],
        "target_poses": [],
        "gripper_frame": "gripper",
        "base_frame": "base",
        "target_frame": "target",
        "camera_frame": "camera",
        "max_translation_rms_mm": 2.0,
        "max_rotation_rms_degrees": 1.0,
    }
    parameter_ranges = {
        "max_translation_rms_mm": NumericRange(0),
        "max_rotation_rms_degrees": NumericRange(0, 180),
    }
    parameter_help = {
        "gripper_poses": "Gripper-to-base rigid pose records. A wired input takes precedence.",
        "target_poses": "Target-to-camera pose records for the same captures. A wired input wins.",
        "max_translation_rms_mm": "Maximum RMS spread of target positions in base. Zero skips it.",
        "max_rotation_rms_degrees": "Maximum RMS spread of target orientations. Zero skips it.",
    }
    process_inputs_readonly = True

    def process(self, inputs):
        p = self.parameters
        gripper_frame, base_frame, target_frame, camera_frame = (
            p[name] for name in ("gripper_frame", "base_frame", "target_frame", "camera_frame")
        )
        frames = (gripper_frame, base_frame, target_frame, camera_frame)
        if (
            any(not isinstance(name, str) or not name.strip() for name in frames)
            or len(set(frames)) != 4
        ):
            raise ValidationError(
                "Use four distinct frame names for gripper, base, target and camera"
            )
        robot, board, tickets = _paired_samples(
            inputs.get("gripper_poses", p["gripper_poses"]),
            inputs.get("target_poses", p["target_poses"]),
        )
        units = robot[0].get("units") if isinstance(robot[0], dict) else None
        if units not in ("m", "mm"):
            raise ValidationError("Hand-eye poses require consistent meter or millimeter units")
        gripper = _transforms(robot, gripper_frame, base_frame, units)
        target = _transforms(board, target_frame, camera_frame, units)
        transform, rotation_pairs = _solve_park(gripper, target)

        target_in_base = gripper @ transform @ target
        rotation, _ = _proper_rotation(target_in_base[:, :3, :3].mean(axis=0))
        translations = target_in_base[:, :3, 3]
        translation = translations.mean(axis=0)
        mm = 1000 if units == "m" else 1
        translation_errors = np.linalg.norm((translations - translation) * mm, axis=1)
        rotation_errors = _rotation_errors(target_in_base[:, :3, :3], rotation)
        translation_rms = float(np.sqrt(np.mean(translation_errors**2)))
        rotation_rms = float(np.sqrt(np.mean(rotation_errors**2)))
        if not np.isfinite([translation_rms, rotation_rms]).all():
            raise ValidationError("Hand-eye fitting produced nonfinite residuals")
        reasons = []
        ratios = []
        for code, value, limit, suffix in (
            ("translation", translation_rms, float(p["max_translation_rms_mm"]), "mm"),
            ("rotation", rotation_rms, float(p["max_rotation_rms_degrees"]), "degrees"),
        ):
            if limit:
                ratios.append(value / limit)
                if value > limit:
                    reasons.append(
                        {
                            "code": f"hand_eye_{code}_rms_exceeds_limit",
                            "message": (
                                f"Target {code} RMS {value:.4g} {suffix} exceeds "
                                f"{limit:.4g} {suffix}"
                            ),
                        }
                    )
        report = {
            "model": "hand_eye_report_v1",
            "method": "park",
            "sample_count": len(robot),
            "rotation_pairs": rotation_pairs,
            "translation_rms_mm": translation_rms,
            "rotation_rms_degrees": rotation_rms,
            "samples": [
                {
                    "index": index,
                    "ticket": tickets[index]["value"] if tickets is not None else None,
                    "translation_error_mm": float(t),
                    "rotation_error_degrees": float(r),
                }
                for index, (t, r) in enumerate(zip(translation_errors, rotation_errors))
            ],
        }
        return Result(
            {
                "camera_to_gripper": NO_VALUE
                if reasons
                else pose_record(transform, camera_frame, gripper_frame, units),
                "target_to_base": NO_VALUE
                if reasons
                else pose_record(
                    pose_matrix(rotation, translation), target_frame, base_frame, units
                ),
                "report": report,
                "translation_rms_mm": translation_rms,
                "rotation_rms_degrees": rotation_rms,
            },
            passed=not reasons,
            fail_reasons=reasons,
            score=max(0.0, 1 - max(ratios, default=0)),
            message=(
                f"{len(robot)} poses · target RMS {translation_rms:.4g} mm, {rotation_rms:.4g}°"
            ),
        )
