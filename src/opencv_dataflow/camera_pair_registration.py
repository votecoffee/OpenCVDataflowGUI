"""Fixed-intrinsic registration for cameras that share a rigid target frame."""

import cv2
import numpy as np

from .cloud_projection import METERS
from .core import Port, Result
from .geometry3d_common import Tool3D, pose_matrix, pose_record, rigid
from .multicamera_calibration import (
    _anchor_pose,
    _observations,
    _optics,
    _optional_heatmap,
    _project_error,
)
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE

_LENS_FLAGS = {"pinhole": cv2.CALIB_MODEL_PINHOLE, "fisheye": cv2.CALIB_MODEL_FISHEYE}


def _pair_observations(value, minimum, field):
    observations = _observations(value, minimum_cameras=2, maximum_cameras=2)
    paired = np.flatnonzero(observations.visible.all(axis=0))
    if len(paired) < minimum:
        raise ValidationError(
            f"Use at least {minimum} paired captures of the same target poses", field
        )
    return observations, paired


def _check_holdout(fit, held):
    if (
        fit.frames != held.frames
        or fit.sizes != held.sizes
        or fit.units != held.units
        or not np.array_equal(fit.objects, held.objects)
    ):
        raise ValidationError(
            "Fit and holdout must use the same optics frames and target geometry", "holdout"
        )
    if not set(fit.captures).isdisjoint(held.captures):
        raise ValidationError("Fit and holdout capture IDs must be disjoint", "holdout")
    for camera in range(2):
        for f in np.flatnonzero(fit.visible[camera]):
            for h in np.flatnonzero(held.visible[camera]):
                shared = fit.point_visible[camera, f] & held.point_visible[camera, h]
                if shared.sum() >= 6:
                    shift = fit.pixels[camera, f, shared] - held.pixels[camera, h, shared]
                    if float(np.mean(np.sum(shift**2, axis=1))) < 0.25**2:
                        raise ValidationError(
                            "Holdout includes a view that repeats a fit observation", "holdout"
                        )


def _native_seed(pose, obs):
    if pose is None:
        return None, None, 0
    if (
        not isinstance(pose, dict)
        or pose.get("model") != "pose_3d_v1"
        or pose.get("source_frame") != obs.frames[0]
        or pose.get("target_frame") != obs.frames[1]
        or pose.get("units") not in METERS
    ):
        raise ValidationError(
            "Initial pose must map the first camera to the second in metric units", "initial_pose"
        )
    motion = rigid(pose)
    scale = METERS[pose["units"]] / METERS[obs.units]
    # The pinned binding updates a column vector supplied as an in/out argument.
    translation = np.ascontiguousarray(motion[:3, 3:4] * scale)
    return np.ascontiguousarray(motion[:3, :3]), translation, cv2.CALIB_USE_EXTRINSIC_GUESS


def _selected_points(obs, camera, capture):
    visible = obs.point_visible[camera, capture]
    return obs.objects[visible], obs.pixels[camera, capture, visible]


def _fit_cameras(obs, paired, optics, lenses, initial_pose, criteria):
    native_views = [[_selected_points(obs, c, f) for f in paired] for c in range(2)]
    (xyz_a, xy_a), (xyz_b, xy_b) = [list(zip(*rows, strict=True)) for rows in native_views]
    (k1, d1), (k2, d2) = optics
    r, t, flags = _native_seed(initial_pose, obs)
    return cv2.registerCamerasExtended(
        [xyz.astype(np.float32) for xyz in xyz_a],
        [xyz.astype(np.float32) for xyz in xyz_b],
        list(xy_a),
        list(xy_b),
        k1.copy(),
        d1.copy() if d1.size else np.zeros(5),
        _LENS_FLAGS[lenses[0]],
        k2.copy(),
        d2.copy() if d2.size else np.zeros(5),
        _LENS_FLAGS[lenses[1]],
        r,
        t,
        flags=flags,
        criteria=criteria,
    )


def _pair_capture_record(obs, capture):
    visibility = obs.point_visible[:, capture]
    return {
        "capture_id": obs.captures[capture],
        "point_counts": visibility.sum(axis=1).tolist(),
        "common_points": int(np.count_nonzero(visibility[0] & visibility[1])),
    }


def _fit_report(native, obs, paired):
    rms, r, t, essential, fundamental, rvecs, tvecs, errors = native
    errors = np.asarray(errors, np.float64)
    rvecs = np.asarray(rvecs, np.float64).reshape(-1, 3)
    tvecs = np.asarray(tvecs, np.float64).reshape(-1, 3)
    if (
        not np.isfinite(rms)
        or rms < 0
        or errors.shape != (len(paired), 2)
        or not np.isfinite(errors).all()
        or (errors < 0).any()
        or len(rvecs) != len(paired)
        or len(tvecs) != len(paired)
        or not (np.isfinite(rvecs).all() and np.isfinite(tvecs).all())
    ):
        raise ValueError("OpenCV returned invalid pair fit diagnostics")
    transform = pose_matrix(r, t)
    fit_rows = [
        {
            **_pair_capture_record(obs, f),
            "rms_px": errors[i].tolist(),
            "target_to_first_rvec": rvecs[i].tolist(),
            "target_to_first_translation": tvecs[i].tolist(),
        }
        for i, f in enumerate(paired)
    ]
    return transform, fit_rows, errors.T, essential, fundamental


def _holdout_report(obs, paired, optics, lenses, transform, settings, cancelled):
    first_to_second = transform
    second_to_first = np.linalg.inv(transform)
    records, failures = [], set()
    errors = np.full((2, len(paired)), np.nan)
    counts = np.zeros_like(errors)
    for index, capture in enumerate(paired):
        if cancelled():
            return None, None, set()
        points = [_selected_points(obs, c, capture) for c in range(2)]
        anchors = [
            _anchor_pose(xyz, xy, optical, lens)
            for (xyz, xy), optical, lens in zip(points, optics, lenses, strict=True)
        ]
        record = _pair_capture_record(obs, capture)
        anchor_rms = [
            None if anchor is None else _project_error(xyz, xy, optical, anchor, lens)
            for (xyz, xy), optical, lens, anchor in zip(
                points, optics, lenses, anchors, strict=True
            )
        ]
        record.update(
            anchor_rms_px=anchor_rms,
            cross_rms_px=[None, None],
            frame_closure_translation=None,
            frame_closure_rotation_deg=None,
        )
        records.append(record)
        if any(value is None for value in anchor_rms):
            failures.add("camera_pair_holdout_pose")
            continue
        if settings["max_anchor_rms_px"] and max(anchor_rms) > settings["max_anchor_rms_px"]:
            failures.add("camera_pair_holdout_anchor_rms")
        predicted = (second_to_first @ anchors[1], first_to_second @ anchors[0])
        for c, (xyz, pixels) in enumerate(points):
            error = _project_error(xyz, pixels, optics[c], predicted[c], lenses[c])
            record["cross_rms_px"][c] = error
            if error is None:
                failures.add("camera_pair_holdout_projection")
            else:
                errors[c, index], counts[c, index] = error, len(xyz)
                if settings["max_view_rms_px"] and error > settings["max_view_rms_px"]:
                    failures.add("camera_pair_holdout_view_rms")
        record["frame_closure_translation"] = float(
            np.linalg.norm(predicted[1][:3, 3] - anchors[1][:3, 3])
        )
        difference = predicted[1][:3, :3] @ anchors[1][:3, :3].T
        cosine = np.clip((np.trace(difference) - 1.0) / 2.0, -1.0, 1.0)
        record["frame_closure_rotation_deg"] = float(np.degrees(np.arccos(cosine)))

    samples = counts.sum()
    rms = float(np.sqrt(np.nansum(errors**2 * counts) / samples)) if samples else None
    if rms is None or (settings["max_holdout_rms_px"] and rms > settings["max_holdout_rms_px"]):
        failures.add("camera_pair_holdout_rms")
    return {"rms_px": rms, "views": records}, errors, failures


def _epipolar_matrix(matrix):
    if np.shape(matrix) != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("OpenCV returned invalid epipolar geometry")
    return matrix.tolist()


class RegisterCameraPair(Tool3D):
    type_id = "3d.register_cameras"
    title, subtype = "Register cameras with disjoint views", "Calibration"
    description = (
        "Fit fixed-intrinsic camera extrinsics using independent observations of one target."
    )
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = (
        Port("observations", "data"),
        Port("first_camera", "data"),
        Port("second_camera", "data"),
        Port(
            "initial_pose", "data", required=False, required_when_connected=True, visual_type="pose"
        ),
        Port("holdout", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("pose", "data", required=False, visual_type="pose"),
        Port("essential", "data", required=False),
        Port("fundamental", "data", required=False),
        Port("report", "data"),
    )
    defaults = {
        "max_iterations": 100,
        "epsilon": 1e-6,
        "max_fit_rms_px": 1.0,
        "max_view_rms_px": 2.0,
        "max_holdout_rms_px": 1.0,
        "max_anchor_rms_px": 0.8,
    }
    parameter_ranges = {
        "max_iterations": NumericRange(1, 500, whole=True),
        "epsilon": NumericRange(0, 1),
        "max_fit_rms_px": NumericRange(0),
        "max_view_rms_px": NumericRange(0),
        "max_holdout_rms_px": NumericRange(0),
        "max_anchor_rms_px": NumericRange(0),
    }
    parameter_help = {
        "max_iterations": "Maximum OpenCV registration solver iterations.",
        "epsilon": "OpenCV solver convergence epsilon.",
        "max_fit_rms_px": "Maximum pooled native fit RMS. Zero disables the limit.",
        "max_view_rms_px": "Maximum individual camera RMS in a fit or holdout. Zero disables.",
        "max_holdout_rms_px": "Maximum pooled cross-camera holdout RMS. Zero disables.",
        "max_anchor_rms_px": "Maximum fit RMS for each holdout anchor pose. Zero disables.",
    }
    input_help = {
        "observations": "Two-camera rig observations with at least three paired target poses.",
        "first_camera": "Fixed pinhole or fisheye model for the first observation frame.",
        "second_camera": "Fixed pinhole or fisheye model for the second observation frame.",
        "initial_pose": "Optional initial first-to-second camera pose in meters or millimeters.",
        "holdout": "Optional independent pairs with distinct IDs and matching target geometry.",
    }
    output_help = {
        "pose": "Candidate rigid pose from first-camera coordinates to second-camera coordinates.",
        "essential": "Epipolar matrix for normalized undistorted camera rays.",
        "fundamental": "Epipolar matrix for ideal undistorted pixels of a pinhole pair.",
        "report": "Native per-view fit errors, held-out cross-projections and frame closure.",
    }

    def process(self, inputs):
        p = self.parameters
        fit, paired = _pair_observations(inputs["observations"], 3, "observations")
        hold, held_pairs = None, None
        if "holdout" in inputs:
            hold, held_pairs = _pair_observations(inputs["holdout"], 1, "holdout")
            _check_holdout(fit, hold)
        records = (inputs["first_camera"], inputs["second_camera"])
        optics = _optics(records, fit, lens=None)
        lenses = tuple(record.get("lens") for record in records)
        if any(lens not in _LENS_FLAGS for lens in lenses):
            raise ValidationError(
                "Register pinhole or fisheye camera records", "first_camera", "second_camera"
            )
        initial = inputs.get("initial_pose")
        criteria = (
            cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
            int(p["max_iterations"]),
            float(p["epsilon"]),
        )
        if self._cancelled():
            return self._cancelled_result()
        native = _fit_cameras(fit, paired, optics, lenses, initial, criteria)
        if self._cancelled():
            return self._cancelled_result()
        transform, fit_rows, fit_errors, e, f = _fit_report(native, fit, paired)
        rms = float(native[0])
        failures = set()
        if p["max_fit_rms_px"] and rms > p["max_fit_rms_px"]:
            failures.add("camera_pair_fit_rms")
        if p["max_view_rms_px"] and float(fit_errors.max()) > p["max_view_rms_px"]:
            failures.add("camera_pair_fit_view_rms")

        held, held_errors = None, None
        if hold is not None:
            held, held_errors, hold_failures = _holdout_report(
                hold, held_pairs, optics, lenses, transform, p, self._cancelled
            )
            if held is None or self._cancelled():
                return self._cancelled_result()
            held["unpaired_capture_ids"] = _unpaired_ids(hold)
            failures.update(hold_failures)
        has_baseline = bool(np.linalg.norm(transform[:3, 3]) > 0)
        result = {
            "pose": pose_record(transform, fit.frames[0], fit.frames[1], fit.units),
            "essential": _epipolar_matrix(e) if has_baseline else NO_VALUE,
            "fundamental": _epipolar_matrix(f)
            if has_baseline and lenses == ("pinhole", "pinhole")
            else NO_VALUE,
            "report": {
                "model": "camera_pair_registration_v1",
                "status": "failed" if failures else "ok",
                "camera_frames": list(fit.frames),
                "lens_models": list(lenses),
                "units": fit.units,
                "initial_pose_used": initial is not None,
                "fit_rms_px": rms,
                "fit_views": fit_rows,
                "unpaired_fit_capture_ids": _unpaired_ids(fit),
                "holdout": held,
            },
        }
        graphics = {}
        for label, errors in (
            ("Fit residuals", fit_errors),
            ("Held-out cross-projection", held_errors),
        ):
            if errors is not None and self.capture_graphics:
                limit = p["max_view_rms_px"] or float(np.nanmax(errors, initial=1e-6))
                graphics[label] = _optional_heatmap(self, errors, limit)["reprojection_error"]
        message = f"{len(paired)} paired poses, fit RMS {rms:.4g} px"
        if held is not None:
            cross_rms = held["rms_px"]
            message += (
                ", held-out projection unavailable"
                if cross_rms is None
                else f", held-out RMS {cross_rms:.4g} px"
            )
        return Result(
            result,
            passed=not failures,
            fail_reasons=sorted(failures),
            graphics=graphics,
            message=message,
        )

    def _cancelled_result(self):
        return Result(
            {
                **dict.fromkeys((port.name for port in self.outputs), NO_VALUE),
                "report": {"status": "cancelled"},
            },
            passed=False,
            fail_reasons=["camera_pair_cancelled"],
            message="Camera registration cancelled",
        )


def _unpaired_ids(observations):
    return [
        capture
        for capture, both in zip(
            observations.captures, observations.visible.all(axis=0), strict=True
        )
        if not both
    ]
