"""Native multi-camera rig calibration and independent cross-camera validation."""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .core import Block, Port, Result
from .geometry3d_common import (
    Tool3D,
    camera,
    camera_record,
    image_size,
    pose_matrix,
    pose_record,
    rigid,
)
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE

OBSERVATIONS = "multicamera_observations_v1"
RIG = "multicamera_rig_v1"
_MAX_CAMERAS = 8
_MAX_CAPTURES = 64
_MAX_POINTS = 128
_MAX_FILE_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class Observations:
    units: str
    frames: tuple
    sizes: tuple
    captures: tuple
    objects: np.ndarray
    pixels: np.ndarray
    visible: np.ndarray
    point_visible: np.ndarray

    @property
    def partial(self):
        return not self.point_visible[self.visible].all()


def _names(values, label, minimum, maximum):
    if (
        not isinstance(values, (tuple, list))
        or not minimum <= len(values) <= maximum
        or any(not isinstance(v, str) or not v.strip() or len(v) > 128 for v in values)
        or len(set(values)) != len(values)
    ):
        raise ValidationError(f"{label} need {minimum}..{maximum} distinct names")
    return tuple(values)


def _observations(record):
    if not isinstance(record, dict) or record.get("model") != OBSERVATIONS:
        raise ValidationError("Connect a multicamera_observations_v1 record")
    units = record.get("units")
    if units not in ("m", "mm"):
        raise ValidationError("Target spacing must declare meters or millimeters")
    frames = _names(record.get("camera_frames"), "Camera frames", 3, _MAX_CAMERAS)
    captures = _names(record.get("capture_ids"), "Capture IDs", 1, _MAX_CAPTURES)
    sizes = record.get("image_sizes")
    if not isinstance(sizes, (tuple, list)) or len(sizes) != len(frames):
        raise ValidationError("Provide one image size per camera")
    sizes = tuple(image_size(size) for size in sizes)
    raw = record.get("object_points")
    if (
        not isinstance(raw, (tuple, list, np.ndarray))
        or (isinstance(raw, np.ndarray) and raw.ndim == 0)
        or not 6 <= len(raw) <= _MAX_POINTS
    ):
        raise ValidationError(f"Provide 6..{_MAX_POINTS} shared target points")
    objects = np.asarray(raw, dtype=np.float64)
    if (
        objects.shape != (len(raw), 3)
        or not np.isfinite(objects).all()
        or np.max(np.abs(objects)) > 1e6
        or np.unique(objects, axis=0).shape[0] != len(raw)
        or np.linalg.matrix_rank(objects - objects[0]) < 2
    ):
        raise ValidationError("Target points must be distinct finite noncollinear XYZ coordinates")
    mask = np.asarray(record.get("visibility"))
    if (
        mask.shape != (len(frames), len(captures))
        or mask.dtype.kind not in "biu"
        or np.any((mask != 0) & (mask != 1))
    ):
        raise ValidationError("Visibility must be a camera-by-capture matrix of zeroes and ones")
    visible = mask.astype(bool)
    input_pixels = record.get("image_points")
    if (
        not isinstance(input_pixels, (tuple, list))
        or len(input_pixels) != len(frames)
        or any(
            not isinstance(row, (tuple, list)) or len(row) != len(captures) for row in input_pixels
        )
    ):
        raise ValidationError("Supply a camera-by-capture table of ordered image points")
    pixels = np.zeros((*visible.shape, len(objects), 2), np.float32)
    point_visible = np.zeros(pixels.shape[:-1], bool)
    for c, f in zip(*np.where(visible), strict=True):
        label = f"{frames[c]} / {captures[f]}"
        raw_view = input_pixels[c][f]
        if (
            not isinstance(raw_view, (list, tuple, np.ndarray))
            or (isinstance(raw_view, np.ndarray) and raw_view.ndim == 0)
            or len(raw_view) != len(objects)
        ):
            raise ValidationError(f"Visible view {label} needs one slot per ordered target point")
        observed = np.array([point is not None for point in raw_view], bool)
        view = np.asarray([point for point in raw_view if point is not None], dtype=np.float32)
        if (
            view.shape != (int(observed.sum()), 2)
            or len(view) < 6
            or not np.isfinite(view).all()
            or np.any(view < 0)
            or np.any(view >= sizes[c])
            or np.unique(view, axis=0).shape[0] != len(view)
            or np.linalg.matrix_rank(view - view[0]) < 2
            or np.linalg.matrix_rank(objects[observed] - objects[observed][0]) < 2
        ):
            raise ValidationError(
                f"Visible view {label} needs a full ordered table with at least six distinct, "
                "noncollinear in-frame correspondences. Use null for missing points"
            )
        point_visible[c, f] = observed
        pixels[c, f, observed] = view
    return Observations(units, frames, sizes, captures, objects, pixels, visible, point_visible)


class LoadMulticameraObservations(Block):
    type_id, title, category = "input.multicamera_observations", "Load rig observations", "Input"
    block_type, subtype = "Input", "Calibration"
    outputs = (Port("observations", "data"),)
    defaults = {"path": ""}
    project_path_parameters = ("path",)
    parameter_help = {"path": "JSON observations file. Relative paths use the project directory."}

    def process(self, inputs):
        if not self.parameters["path"]:
            raise ValidationError("Choose a multi-camera observations JSON file", "path")
        path = Path(self.parameters["path"])
        if not path.is_absolute():
            if self.project_dir is None:
                raise ValidationError("Save the project before using a relative path", "path")
            path = self.project_dir / path
        with path.open("rb") as stream:
            raw = stream.read(_MAX_FILE_BYTES + 1)
        if len(raw) > _MAX_FILE_BYTES:
            raise ValidationError("Observations JSON exceeds 8 MiB", "path")
        record = json.loads(raw)
        parsed = _observations(record)
        return Result(
            {"observations": record},
            message=f"{len(parsed.frames)} cameras, {len(parsed.captures)} captures",
        )


def _optics(records, obs, lens):
    if not isinstance(records, (list, tuple)) or len(records) != len(obs.frames):
        raise ValidationError("Supply an ordered camera model for every frame")
    checked = []
    for record, frame, size in zip(records, obs.frames, obs.sizes, strict=True):
        k, d, record_size = camera(record)
        if (
            record.get("model") != "camera_model_v1"
            or record.get("frame") != frame
            or record.get("units") != obs.units
            or record.get("lens") != lens
            or record_size != size
        ):
            raise ValidationError(
                "Camera records must match observation frames, sizes, units and lens"
            )
        checked.append((k, d))
    return checked


def _overlaps(visibility, minimum):
    counts = visibility.astype(np.int32)
    shared = counts @ counts.T
    reachable = np.eye(len(shared), dtype=bool)[0]
    for _ in range(1, len(shared)):
        reachable |= np.any(shared[reachable] >= minimum, axis=0)
    if not reachable.all():
        raise ValidationError("Camera visibility graph lacks connected shared captures")
    return shared


def _optional_heatmap(block, rms, limit):
    if not block.generate_graphics:
        return {}
    visible = np.isfinite(rms)
    peak = max(float(limit), 1e-6)
    tiles = np.uint8(np.rint(np.clip(np.nan_to_num(rms) / peak, 0, 1) * 255))
    graphic = cv2.applyColorMap(tiles, cv2.COLORMAP_TURBO)
    graphic[~visible] = 0
    return {
        "reprojection_error": cv2.resize(
            graphic, None, fx=12, fy=24, interpolation=cv2.INTER_NEAREST
        )
    }


def _report_table(obs, rms):
    return {
        "camera_frames": list(obs.frames),
        "capture_ids": list(obs.captures),
        "visibility": obs.visible.astype(int).tolist(),
        "point_counts": obs.point_visible.sum(axis=2).tolist(),
        "per_camera_capture_rms_px": [
            [float(v) if np.isfinite(v) else None for v in row] for row in rms
        ],
    }


def _native_fit(obs, settings, fixed):
    lens = settings["lens"]
    pinhole = lens == "pinhole"
    model = cv2.CALIB_MODEL_PINHOLE if pinhole else cv2.CALIB_MODEL_FISHEYE
    controls = (
        (cv2.CALIB_RATIONAL_MODEL if settings["pinhole_distortion"] == "rational" else 0)
        if pinhole
        else cv2.CALIB_RECOMPUTE_EXTRINSIC | cv2.CALIB_FIX_SKEW
    )
    if fixed:
        # Keep exact tagged optics. Supply a conventional zero array for no pinhole distortion.
        k = [item[0].copy() for item in fixed]
        d = [
            coeff.reshape(1, -1).copy()
            if pinhole and coeff.size
            else np.zeros((1, 5))
            if pinhole
            else coeff.reshape(4, 1).copy()
            for _, coeff in fixed
        ]
    else:
        k = d = None
    flags = cv2.CALIB_USE_INTRINSIC_GUESS if fixed else 0
    if settings["initialization"] == "stereo_calibrate":
        flags |= cv2.CALIB_STEREO_REGISTRATION
    empty = np.empty((0, 2), np.float32)
    # Fisheye takes one row of two-channel points in the OpenCV 5 binding.
    views = [[arr if pinhole else arr.reshape(1, -1, 2) for arr in row] for row in obs.pixels]
    images = [
        [view if visible else empty for view, visible in zip(row, mask, strict=True)]
        for row, mask in zip(views, obs.visible, strict=True)
    ]
    # Keep the native board scale stable across meter and millimeter inputs.
    # Camera-to-camera translations are independent of the board coordinate origin.
    scale = 2 * float(np.linalg.norm(np.ptp(obs.objects, axis=0)))
    centered = obs.objects - obs.objects.mean(axis=0)
    objects = np.ascontiguousarray(centered / scale, dtype=np.float32)
    result = cv2.calibrateMultiviewExtended(
        [objects] * len(obs.captures),
        images,
        obs.sizes,
        obs.visible.astype(np.uint8),
        np.full((len(obs.frames), 1), model, np.uint8),
        k,
        d,
        None,
        None,
        flagsForIntrinsics=np.full((len(obs.frames), 1), controls, np.int32),
        flags=flags,
        criteria=(
            cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
            int(settings["max_iterations"]),
            float(settings["epsilon"]),
        ),
    )
    return result, scale


class CalibrateMulticameraRig(Tool3D):
    type_id, title, subtype = (
        "3d.calibrate_multicamera",
        "Calibrate multi-camera rig",
        "Calibration",
    )
    inputs = (
        Port("observations", "data"),
        Port("camera_models", "data", required=False, required_when_connected=True),
    )
    outputs = (Port("rig", "data", required=False), Port("report", "data"))
    defaults = {
        "lens": "pinhole",
        "intrinsics": "fit",
        "pinhole_distortion": "standard",
        "initialization": "register_cameras",
        "minimum_shared_views": 3,
        "max_iterations": 60,
        "epsilon": 1e-9,
        "max_rms_px": 1.0,
        "max_view_rms_px": 2.0,
    }
    parameter_choices = {
        "lens": ("pinhole", "fisheye"),
        "intrinsics": ("fit", "fixed"),
        "pinhole_distortion": ("standard", "rational"),
        "initialization": ("register_cameras", "stereo_calibrate"),
    }
    parameter_ranges = {
        "minimum_shared_views": NumericRange(1, _MAX_CAPTURES, whole=True),
        "max_iterations": NumericRange(1, 200, whole=True),
        "epsilon": NumericRange(1e-15, 0.1),
        "max_rms_px": NumericRange(0),
        "max_view_rms_px": NumericRange(0),
    }
    parameter_help = {
        "lens": "All cameras must use the same lens family.",
        "intrinsics": "Fit per-camera optics first, or use the connected models unchanged.",
        "pinhole_distortion": "Fit the five-coefficient or eight-coefficient pinhole model.",
        "initialization": "Use Register cameras when any target view is partial.",
        "minimum_shared_views": "Minimum shared captures required for each initializing link.",
        "max_iterations": "Iteration limit for rig adjustment and per-camera optics.",
        "epsilon": "Rig adjustment termination tolerance.",
        "max_rms_px": "Maximum fitted overall reprojection RMS. Zero disables the limit.",
        "max_view_rms_px": "Maximum fitted camera/capture RMS. Zero disables the limit.",
    }
    input_help = {
        "observations": "Named target captures, with a camera-by-capture visibility matrix.",
        "camera_models": "Ordered existing camera records for Fixed intrinsics mode.",
    }
    parallel_safe = process_inputs_readonly = True

    def process(self, inputs):
        p = self.parameters
        obs = _observations(inputs["observations"])
        fixed = p["intrinsics"] == "fixed"
        if fixed != ("camera_models" in inputs):
            raise ValidationError("Connect camera models in fixed mode only", "intrinsics")
        minimum = 3 if fixed else 6
        if np.any(obs.visible.sum(axis=1) < minimum) or not obs.visible.any(axis=0).all():
            raise ValidationError(
                f"Each camera needs at least {minimum} captures and every capture needs one view"
            )
        if not fixed and np.any(obs.objects[:, 2] != 0):
            raise ValidationError("Fitting intrinsics needs a target in the XY plane")
        optics = _optics(inputs["camera_models"], obs, p["lens"]) if fixed else None
        shared = _overlaps(obs.visible, int(p["minimum_shared_views"]))
        converged = True
        if obs.partial:
            if p["initialization"] != "register_cameras":
                raise ValidationError(
                    "Partial target views require Register cameras", "initialization"
                )
            from .multicamera_partial import fit_partial_rig

            fit = fit_partial_rig(obs, p, optics, shared)
            rms, ks, ds, rvs, ts, pairs, view_rms, board_scale, converged = fit
        else:
            if not hasattr(cv2, "calibrateMultiviewExtended"):
                raise ValidationError("This OpenCV build lacks calibrateMultiviewExtended")
            native, board_scale = _native_fit(obs, p, optics)
            rms, ks, ds, rvs, ts, pairs, _, _, view_rms = native
        if (
            not np.isfinite(rms)
            or rms < 0
            or view_rms.shape != obs.visible.shape
            or not np.isfinite(view_rms[obs.visible]).all()
            or np.any(view_rms[obs.visible] < 0)
            or pairs.shape != (len(obs.frames) - 1, 2)
            or np.any((pairs < 0) | (pairs >= len(obs.frames)))
        ):
            raise ValidationError("Calibration returned incomplete or invalid diagnostics")
        errors = np.where(obs.visible, view_rms, np.nan)
        worst = float(np.nanmax(errors))
        report = {
            "model": "multicamera_fit_v1",
            "solver": "partial_observations" if obs.partial else "native_multiview",
            **({"converged": converged} if obs.partial else {}),
            "rms_px": float(rms),
            "worst_view_rms_px": worst,
            "initialization_pairs": [[obs.frames[i], obs.frames[j]] for i, j in pairs],
            "shared_captures": shared.tolist(),
            **_report_table(obs, errors),
        }
        reasons = [] if converged else ["multicamera_fit_iteration_limit"]
        if (limit := float(p["max_rms_px"])) and rms > limit:
            reasons.append("multicamera_fit_rms")
        if (view_limit := float(p["max_view_rms_px"])) and worst > view_limit:
            reasons.append("multicamera_fit_view_rms")
        if any(shared[i, j] < int(p["minimum_shared_views"]) for i, j in pairs):
            reasons.append("multicamera_initialization_support")
        rig = NO_VALUE
        if not reasons:
            camera_models = (
                list(inputs["camera_models"])
                if fixed
                else [
                    camera_record(k, d, size, obs.units, frame, p["lens"])
                    for k, d, size, frame in zip(ks, ds, obs.sizes, obs.frames, strict=True)
                ]
            )
            transforms = [
                pose_record(
                    pose_matrix(cv2.Rodrigues(rv)[0], t * board_scale),
                    obs.frames[0],
                    frame,
                    obs.units,
                )
                for rv, t, frame in zip(rvs, ts, obs.frames, strict=True)
            ]
            rig = {
                "model": RIG,
                "units": obs.units,
                "lens": p["lens"],
                "anchor_frame": obs.frames[0],
                "cameras": camera_models,
                "anchor_to_camera": transforms,
                "training_capture_ids": list(obs.captures),
            }
        return Result(
            {"rig": rig, "report": report},
            graphics=_optional_heatmap(self, errors, view_limit or max(worst, 1e-6)),
            passed=not reasons,
            message=f"{len(obs.frames)}-camera fit, RMS {rms:.5g} px",
            fail_reasons=reasons,
        )


def _rig_geometry(rig, obs):
    if (
        not isinstance(rig, dict)
        or rig.get("model") != RIG
        or rig.get("anchor_frame") != obs.frames[0]
        or rig.get("units") != obs.units
        or rig.get("lens") not in ("pinhole", "fisheye")
    ):
        raise ValidationError("Rig and validation observations must agree on anchor and units")
    ids = _names(rig.get("training_capture_ids"), "Training capture IDs", 1, _MAX_CAPTURES)
    if not set(ids).isdisjoint(obs.captures):
        raise ValidationError("Fit and held-out capture IDs must be disjoint")
    records = rig.get("anchor_to_camera")
    if not isinstance(records, (tuple, list)) or len(records) != len(obs.frames):
        raise ValidationError("Rig needs one anchor-to-camera pose per frame")
    transforms = []
    for record, frame in zip(records, obs.frames, strict=True):
        if (
            not isinstance(record, dict)
            or record.get("model") != "pose_3d_v1"
            or record.get("source_frame") != obs.frames[0]
            or record.get("target_frame") != frame
            or record.get("units") != obs.units
        ):
            raise ValidationError("Rig transforms must use named anchor-to-camera frames and units")
        transforms.append(rigid(record))
    if not np.allclose(transforms[0], np.eye(4), rtol=0, atol=1e-10):
        raise ValidationError("Rig anchor transform must be identity")
    return _optics(rig.get("cameras"), obs, rig["lens"]), transforms


def _project_error(objects, pixels, optics, pose, lens):
    k, d = optics
    camera_points = objects @ pose[:3, :3].T + pose[:3, 3]
    if not np.isfinite(camera_points).all() or np.any(camera_points[:, 2] <= 0):
        return None
    if lens == "pinhole":
        uv = cv2.projectPoints(camera_points, np.zeros(3), np.zeros(3), k, d)[0]
    else:
        uv = cv2.fisheye.projectPoints(
            camera_points.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), k, d
        )[0]
    error = np.linalg.norm(uv.reshape(-1, 2) - pixels, axis=1)
    return float(np.sqrt(np.mean(error**2))) if np.isfinite(error).all() else None


def _anchor_pose(objects, pixels, optics, lens):
    k, d = optics
    pixels = np.ascontiguousarray(pixels, np.float64).reshape(-1, 1, 2)
    if lens == "fisheye":
        pixels = cv2.fisheye.undistortPoints(pixels, k, d)
        k, d = np.eye(3), None
    if not np.isfinite(pixels).all():
        return None
    ok, rv, tv = cv2.solvePnP(objects, pixels, k, d, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    rv, tv = cv2.solvePnPRefineLM(objects, pixels, k, d, rv, tv)
    if not np.isfinite(rv).all() or not np.isfinite(tv).all():
        return None
    return pose_matrix(cv2.Rodrigues(rv)[0], tv)


class ValidateMulticameraRig(Tool3D):
    type_id, title, subtype = "3d.validate_multicamera", "Validate multi-camera rig", "Calibration"
    inputs = (Port("rig", "data"), Port("observations", "data"))
    outputs = (Port("report", "data"),)
    defaults = {"max_rms_px": 1.0, "max_view_rms_px": 2.0, "max_anchor_rms_px": 0.8}
    parameter_ranges = {name: NumericRange(0) for name in defaults}
    parameter_help = {
        "max_rms_px": "Maximum pooled held-out cross-camera RMS. Zero disables the limit.",
        "max_view_rms_px": "Maximum RMS for a single cross-camera prediction. Zero disables.",
        "max_anchor_rms_px": "Maximum RMS for a pose fitted in one view. Zero disables.",
    }
    input_help = {
        "rig": "A fixed rig from the calibration block.",
        "observations": "Independent captures with the original camera order and new capture IDs.",
    }
    parallel_safe = process_inputs_readonly = True

    def process(self, inputs):
        p = self.parameters
        rig = inputs["rig"]
        obs = _observations(inputs["observations"])
        optics, transforms = _rig_geometry(rig, obs)
        inv = [np.linalg.inv(t) for t in transforms]
        lens = rig["lens"]
        xyz = obs.objects.astype(np.float64)
        errors = np.zeros(obs.visible.shape, np.float64)
        counts = np.zeros(obs.visible.shape, np.uint16)
        records, reasons = [], set()
        anchor_limit = float(p["max_anchor_rms_px"])
        view_limit = float(p["max_view_rms_px"])
        global_limit = float(p["max_rms_px"])
        for f, capture in enumerate(obs.captures):
            visible = np.flatnonzero(obs.visible[:, f])
            if len(visible) < 2:
                continue
            for c in visible:
                points = obs.point_visible[c, f]
                observed_xyz, observed_pixels = xyz[points], obs.pixels[c, f, points]
                anchor = _anchor_pose(observed_xyz, observed_pixels, optics[c], lens)
                anchor_rms = (
                    None
                    if anchor is None
                    else _project_error(observed_xyz, observed_pixels, optics[c], anchor, lens)
                )
                if anchor_rms is None:
                    reasons.add("multicamera_holdout_anchor_pose")
                    continue
                if anchor_limit and anchor_rms > anchor_limit:
                    reasons.add("multicamera_holdout_anchor_rms")
                board_to_rig = inv[c] @ anchor
                for other in visible:
                    if other == c:
                        continue
                    other_points = obs.point_visible[other, f]
                    error = _project_error(
                        xyz[other_points],
                        obs.pixels[other, f, other_points],
                        optics[other],
                        transforms[other] @ board_to_rig,
                        lens,
                    )
                    records.append(
                        {
                            "capture_id": capture,
                            "fit_camera": obs.frames[c],
                            "test_camera": obs.frames[other],
                            "test_points": int(other_points.sum()),
                            "anchor_rms_px": anchor_rms,
                            "rms_px": error,
                        }
                    )
                    if error is None:
                        reasons.add("multicamera_holdout_projection")
                        continue
                    if view_limit and error > view_limit:
                        reasons.add("multicamera_holdout_view_rms")
                    errors[other, f] += error**2
                    counts[other, f] += 1
        total = int(counts.sum())
        points_per_view = obs.point_visible.sum(axis=2)
        samples = int((counts * points_per_view).sum())
        rms = float(np.sqrt((errors * points_per_view).sum() / samples)) if samples else None
        if rms is not None and global_limit and rms > global_limit:
            reasons.add("multicamera_holdout_rms")
        if not (counts.sum(axis=1) > 0).all():
            reasons.add("multicamera_holdout_camera_unverified")
        errors = np.sqrt(
            np.divide(errors, counts, out=np.full(errors.shape, np.nan), where=counts > 0)
        )
        report = {
            "model": "multicamera_holdout_v1",
            "rms_px": rms,
            "cross_projections": records,
            **_report_table(obs, errors),
        }
        return Result(
            {"report": report},
            graphics=_optional_heatmap(self, errors, view_limit or max(rms or 0, 1e-6)),
            passed=not reasons,
            message=f"{total} independent cross-camera reprojections",
            fail_reasons=sorted(reasons),
        )
