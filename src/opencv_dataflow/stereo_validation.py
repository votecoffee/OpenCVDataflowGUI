"""Verify a fixed pinhole stereo rig with independent target observations."""

import cv2
import numpy as np

from .core import Port, Result
from .fail_reasons import merge_reasons
from .geometry3d_common import Tool3D, require_pinhole
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .stereo3d import STEREO_VIEW_INPUTS, stereo_observation_lists, validate_rig


def _residuals(xyz, pixels, optics, transform):
    rotation, translation = transform[:3, :3], transform[:3, 3]
    depth = xyz @ rotation[2] + translation[2]
    if not np.isfinite(depth).all() or np.any(depth <= 0):
        return None
    k, d, _ = optics
    projected = cv2.projectPoints(xyz, cv2.Rodrigues(rotation)[0], translation, k, d)[0]
    errors = np.linalg.norm(projected.reshape(-1, 2) - pixels, axis=1)
    return errors if np.isfinite(errors).all() else None


def _anchor_pose(xyz, pixels, optics):
    k, d, _ = optics
    if (
        len(np.unique(xyz, axis=0)) < 4
        or len(np.unique(pixels, axis=0)) < 4
        or np.linalg.matrix_rank(xyz - xyz[0]) < 2
        or np.linalg.matrix_rank(pixels - pixels[0]) < 2
    ):
        return None
    ok, rv, tv = cv2.solvePnP(xyz, pixels, k, d, flags=cv2.SOLVEPNP_SQPNP)
    if not ok or not np.isfinite(rv).all() or not np.isfinite(tv).all():
        return None
    rv, tv = cv2.solvePnPRefineLM(xyz, pixels, k, d, rv, tv)
    if not np.isfinite(rv).all() or not np.isfinite(tv).all():
        return None
    matrix = np.eye(4)
    matrix[:3, :3] = cv2.Rodrigues(rv)[0]
    matrix[:3, 3] = tv.reshape(3)
    return matrix


def _rms(errors):
    if errors is None:
        return None
    maximum = float(errors.max())
    return maximum * float(np.sqrt(np.mean((errors / maximum) ** 2))) if maximum else 0.0


def _capture_metrics(xyz, ul, ur, left, right, left_to_right, right_to_left):
    l_pose = _anchor_pose(xyz, ul, left)
    r_pose = _anchor_pose(xyz, ur, right)
    # Refit each target in just one view. The opposite prediction uses the fixed rig.
    return tuple(
        None if pose is None else _residuals(xyz, pixels, optics, pose)
        for pose, pixels, optics in (
            (l_pose, ul, left),
            (r_pose, ur, right),
            (None if r_pose is None else right_to_left @ r_pose, ul, left),
            (None if l_pose is None else left_to_right @ l_pose, ur, right),
        )
    )


class ValidateStereoCalibration(Tool3D):
    type_id, title, subtype = (
        "3d.validate_stereo_calibration",
        "Validate stereo calibration",
        "Stereo",
    )
    inputs = (Port("rig", "data"), *STEREO_VIEW_INPUTS)
    outputs = (Port("report", "data"), Port("rms", "number", required=False))
    defaults = {"max_rms": 1.0, "max_view_rms": 1.5, "max_anchor_rms": 1.0}
    parameter_ranges = {
        "max_rms": NumericRange(0),
        "max_view_rms": NumericRange(0),
        "max_anchor_rms": NumericRange(0),
    }
    parameter_help = {
        "max_rms": "Maximum pooled cross-camera reprojection RMS, in pixels.",
        "max_view_rms": "Maximum cross-camera RMS for one camera and capture. Zero disables it.",
        "max_anchor_rms": "Maximum single-camera pose fit RMS per capture. Zero disables it.",
    }
    input_help = {
        "rig": "Previously fitted pinhole rig, with a fixed left-to-right pose.",
        "object_views": "Held-out target landmarks, one ordered array per stereo capture.",
        "left_views": "Left observed pixels, in the same capture and landmark order.",
        "right_views": "Right observed pixels, in the same capture and landmark order.",
    }
    parallel_safe = process_inputs_readonly = True

    def process(self, inputs):
        rig = inputs["rig"]
        left, right, l_to_r = validate_rig(rig)
        for side in ("left", "right"):
            require_pinhole(rig[side])
        r_to_l = np.linalg.inv(l_to_r)
        units = self.input_payloads["object_views"].get("object_point_units")
        if units is not None and units != rig["units"]:
            raise ValidationError("Held-out object point units must match the stereo rig")
        sources = stereo_observation_lists(inputs, minimum=1)
        p = self.parameters
        reasons, rows = [], []
        predicted_left, predicted_right = [], []
        for i, (xyz, ul, ur) in enumerate(zip(*sources, strict=True)):
            if any(
                np.any(xy < 0) or np.any(xy >= optics[2])
                for xy, optics in ((ul, left), (ur, right))
            ):
                raise ValidationError(f"Held-out view {i} has points outside its camera grid")
            errors = _capture_metrics(xyz, ul, ur, left, right, l_to_r, r_to_l)
            a, b, cross_l, cross_r = (_rms(e) for e in errors)
            row = {
                "index": i,
                "points": len(xyz),
                "left_anchor_rms_pixels": a,
                "right_anchor_rms_pixels": b,
                "left_rms_pixels": cross_l,
                "right_rms_pixels": cross_r,
                "left_max_error_pixels": None if cross_l is None else float(errors[2].max()),
                "right_max_error_pixels": None if cross_r is None else float(errors[3].max()),
            }
            rows.append(row)
            if any(v is None for v in (a, b, cross_l, cross_r)):
                reasons.append(
                    {
                        "code": "stereo_validation_pose",
                        "message": f"View {i} lacks positive-depth reprojections in both cameras",
                    }
                )
                continue
            predicted_left.append(errors[2])
            predicted_right.append(errors[3])
            for group, values, limit in (
                ("anchor", (a, b), float(p["max_anchor_rms"])),
                ("cross_camera", (cross_l, cross_r), float(p["max_view_rms"])),
            ):
                for side, rms in zip(("left", "right"), values, strict=True):
                    if limit and rms > limit:
                        reasons.append(
                            {
                                "code": f"stereo_validation_{group}_rms",
                                "message": (
                                    f"View {i} {side} {group.replace('_', ' ')} RMS "
                                    f"{rms:.4g} px exceeds {limit:.4g} px"
                                ),
                            }
                        )
        all_left = np.concatenate(predicted_left) if predicted_left else None
        all_right = np.concatenate(predicted_right) if predicted_right else None
        rms_left, rms_right = _rms(all_left), _rms(all_right)
        rms = None if rms_left is None else float(np.hypot(rms_left, rms_right) / np.sqrt(2))
        if rms is not None and rms > float(p["max_rms"]):
            reasons.append("stereo_validation_rms")
        report = {
            "model": "stereo_pinhole_validation_v1",
            "left_frame": rig["left"]["frame"],
            "right_frame": rig["right"]["frame"],
            "units": rig["units"],
            "rms_pixels": rms,
            "left_rms_pixels": rms_left,
            "right_rms_pixels": rms_right,
            "captures_measured": len(predicted_left),
            "views": rows,
            "failures": reasons,
        }
        return Result(
            {"report": report, "rms": NO_VALUE if rms is None else rms},
            passed=not reasons,
            fail_reasons=merge_reasons(*reasons),
            message=(
                "Stereo holdout poses unavailable"
                if rms is None
                else f"{len(predicted_left)} held-out views, {rms:.4g} px cross RMS"
            ),
        )
