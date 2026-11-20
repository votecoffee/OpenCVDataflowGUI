"""Pinhole calibration with released target coordinates and held-out point checks."""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .calibration import CALIB_FLAGS, _lens_fit_report, camera_parameters, pinhole_fit_options
from .core import Block, Port, Result
from .geometry3d_common import image_size
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE

OBSERVATIONS_MODEL = "pinhole_target_observations_v1"
_MAX_POINTS = 128
_MAX_VIEWS = 32
_MAX_FILE_BYTES = 2 * 1024 * 1024
_FLAGS = {
    **CALIB_FLAGS,
    "fix_focal_length": cv2.CALIB_FIX_FOCAL_LENGTH,
    "fix_tangential_distortion": cv2.CALIB_FIX_TANGENT_DIST,
    "fix_thin_prism": cv2.CALIB_FIX_S1_S2_S3_S4,
    "fix_sensor_tilt": cv2.CALIB_FIX_TAUX_TAUY,
}


@dataclass(frozen=True)
class TargetObservations:
    target_id: str
    units: str
    size: tuple
    nominal: np.ndarray
    fit: tuple
    holdout: tuple


def _points(value, count, dimensions, label):
    try:
        rows = np.asarray(value, dtype=np.float32)
    except ValueError, TypeError, OverflowError:
        raise ValidationError(f"{label} must contain {count} finite coordinate rows") from None
    if (
        rows.shape != (count, dimensions)
        or not np.isfinite(rows).all()
        or len(np.unique(rows, axis=0)) != count
        or np.linalg.matrix_rank(rows.astype(np.float64) - rows[0]) < 2
    ):
        raise ValidationError(f"{label} needs {count} distinct, finite, noncollinear points")
    return rows


def target_observations(record):
    if not isinstance(record, dict) or record.get("model") != OBSERVATIONS_MODEL:
        raise ValidationError(f"Connect a {OBSERVATIONS_MODEL} record")
    target_id, units = record.get("target_id"), record.get("units")
    if not isinstance(target_id, str) or not target_id.strip() or len(target_id) > 128:
        raise ValidationError("Name one rigid calibration target")
    if units not in ("mm", "m"):
        raise ValidationError("Calibration target units must be mm or m")
    size = image_size(record.get("image_size"))
    raw = record.get("object_points")
    if not isinstance(raw, (list, tuple)) or not 10 <= len(raw) <= _MAX_POINTS:
        raise ValidationError(f"Provide 10 to {_MAX_POINTS} ordered nominal points")
    nominal = _points(raw, len(raw), 3, "Nominal target")
    if np.any(nominal[:, 2] != 0) or np.max(np.abs(nominal)) > 1e6:
        raise ValidationError("Nominal points must be on Z=0 with coordinates within ±1 million")
    views = record.get("views")
    if not isinstance(views, (list, tuple)) or not 5 <= len(views) <= _MAX_VIEWS:
        raise ValidationError(f"Provide 5 to {_MAX_VIEWS} complete target views")
    groups = {"fit": [], "holdout": []}
    names, pixels = set(), []
    for view in views:
        if not isinstance(view, dict):
            raise ValidationError("Each view must declare a capture ID, role and ordered points")
        name, role = view.get("id"), view.get("role")
        if not isinstance(name, str) or not name.strip() or len(name) > 128 or name in names:
            raise ValidationError("Each capture needs a distinct nonempty ID")
        if role not in ("fit", "holdout"):
            raise ValidationError(f"View {name} must have a fit or holdout role")
        xy = _points(view.get("image_points"), len(nominal), 2, f"View {name}")
        if np.any(xy < -0.5) or np.any(xy > np.asarray(size) - 0.5):
            raise ValidationError(f"View {name} has points outside the declared image size")
        if any(np.sqrt(np.mean((xy - previous) ** 2)) < 0.5 for previous in pixels):
            raise ValidationError(f"View {name} duplicates or nearly duplicates an earlier view")
        groups[role].append((name, xy))
        names.add(name)
        pixels.append(xy)
    if len(groups["fit"]) < 4 or not groups["holdout"]:
        raise ValidationError("Use at least four fitting views and one independent holdout view")
    return TargetObservations(
        target_id, units, size, nominal, tuple(groups["fit"]), tuple(groups["holdout"])
    )


class LoadTargetCalibrationObservations(Block):
    type_id = "input.target_calibration_observations"
    title, category = "Load calibration target observations", "Input"
    block_type, subtype = "Input", "Calibration"
    outputs = (Port("observations", "data"),)
    defaults = {"path": ""}
    project_path_parameters = ("path",)
    parameter_help = {"path": "Ordered fit and holdout observations in a JSON file."}

    def process(self, inputs):
        value = self.parameters["path"]
        if not isinstance(value, str) or not value.strip():
            raise ValidationError("Choose an observations JSON file", "path")
        path = Path(value)
        if not path.is_absolute():
            if self.project_dir is None:
                raise ValidationError("Save the project before using a relative path", "path")
            path = self.project_dir / path
        with path.open("rb") as stream:
            content = stream.read(_MAX_FILE_BYTES + 1)
        if len(content) > _MAX_FILE_BYTES:
            raise ValidationError("Target observations JSON exceeds 2 MiB", "path")
        record = json.loads(content)
        obs = target_observations(record)
        return Result(
            {"observations": record},
            message=f"{obs.target_id}: {len(obs.fit)} fit, {len(obs.holdout)} holdout views",
        )


def _scaled_nominal(points, anchor, measured_distance):
    offsets = points.astype(np.float64)[:, :2] - points[0, :2]
    last = offsets[-1]
    diagonal = float(np.linalg.norm(np.ptp(offsets, axis=0)))
    twice_area = np.abs(offsets[:, 0] * last[1] - offsets[:, 1] * last[0])
    if anchor == 0:
        anchor = 1 + int(np.argmax(twice_area[1:-1]))
    if anchor >= len(points) - 1 or twice_area[anchor] <= diagonal**2 * 1e-6:
        raise ValidationError(
            "Choose an interior point index noncollinear with the first and last points",
            "fixed_point_index",
        )
    nominal_distance = float(np.linalg.norm(offsets[anchor]))
    scale = measured_distance / nominal_distance if measured_distance else 1.0
    scaled = (points.astype(np.float64) - points[0]) * scale + points[0]
    if np.max(np.abs(scaled)) > 1e6 or len(np.unique(scaled.astype(np.float32), axis=0)) != len(
        points
    ):
        raise ValidationError(
            "Measured anchor distance gives invalid target coordinates", "measured_anchor_distance"
        )
    return scaled.astype(np.float32), anchor, nominal_distance, scale


def _holdout_residuals(holdout, target, support, measured, matrix, distortion, cancelled):
    rows = []
    for view_id, pixels in holdout:
        if cancelled():
            return None
        ok, rvec, tvec = cv2.solvePnP(
            target[support], pixels[support], matrix, distortion, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            raise ValueError(f"Could not solve pose for holdout view {view_id}")
        rotation, _ = cv2.Rodrigues(rvec)
        depths = target @ rotation[2] + tvec.ravel()[2]
        if not np.isfinite(depths).all() or np.any(depths <= 0):
            raise ValueError(f"Invalid target projection depth in holdout view {view_id}")
        projected, _ = cv2.projectPoints(target[measured], rvec, tvec, matrix, distortion)
        residuals = projected.reshape(-1, 2) - pixels[measured]
        errors = np.linalg.norm(residuals, axis=1)
        if not np.isfinite(errors).all():
            raise ValueError(f"Nonfinite reprojection error in holdout view {view_id}")
        rows.append(
            {
                "view_id": view_id,
                "rms_pixels": float(np.sqrt(np.mean(errors**2))),
                "max_error_pixels": float(errors.max()),
            }
        )
    return rows


def _target_geometry(nominal, anchor, refined, stddev):
    fixed = np.zeros(nominal.shape, dtype=bool)
    if refined is None:
        fixed[:] = True
        return nominal, fixed, None
    refined = np.asarray(refined, dtype=np.float64).reshape(-1, 3)
    stddev = np.asarray(stddev, dtype=np.float64).reshape(-1, 3)
    fixed[[0, anchor], :] = True
    fixed[-1, 2] = True
    if (
        refined.shape != nominal.shape
        or stddev.shape != nominal.shape
        or not np.isfinite(refined).all()
        or not np.isfinite(stddev).all()
        or (stddev < 0).any()
        or not np.array_equal(refined[fixed], nominal[fixed])
    ):
        raise ValueError(
            "OpenCV returned invalid target geometry, uncertainty or locked coordinates"
        )
    return refined, fixed, stddev


def _offset_plot(nominal, offsets, fixed):
    image = np.full((400, 560, 3), (31, 27, 22), np.uint8)
    lo = nominal[:, :2].min(axis=0)
    span = np.ptp(nominal[:, :2], axis=0)
    scale = min(456 / span[0], 285 / span[1])
    origin = (nominal[:, :2] - lo) * scale + [50, 60]
    largest_shift = float(np.linalg.norm(offsets[:, :2], axis=1).max())
    pixels_per_unit = min(9 * scale, 34 / largest_shift) if largest_shift else scale
    tips = origin + offsets[:, :2] * pixels_per_unit
    for point, end, height, locked in zip(origin, tips, offsets[:, 2], fixed, strict=True):
        a, b = tuple(np.rint(point).astype(int)), tuple(np.rint(end).astype(int))
        color = (65, 198, 245) if locked.all() else (215, 130, 80) if height < 0 else (85, 150, 240)
        cv2.circle(image, a, 4, (165, 165, 165), 1, cv2.LINE_AA)
        if a != b:
            cv2.arrowedLine(image, a, b, color, 1, cv2.LINE_AA, tipLength=0.3)
        cv2.circle(image, b, 2, color, -1, cv2.LINE_AA)
    cv2.putText(
        image,
        "Target XY offsets (arrows amplified)",
        (28, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (226, 226, 226),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "Gold: fixed XY   Blue: negative Z   Orange: nonnegative Z",
        (28, 378),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    return image


class RefineCalibrationTarget(Block):
    type_id = "calibration.refine_target"
    title, category = "Refine calibration target geometry", "Calibration"
    block_type, subtype = "Calibration", "Lens distortion"
    description = "Calibrate a pinhole camera with a shared target and score held-out views."
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = (Port("observations", "data"),)
    outputs = (
        Port("calibration", "data", required=False),
        Port("target", "data", required=False),
        Port("report", "data"),
        Port("holdout_rms", "number", required=False),
    )
    defaults = {
        **dict.fromkeys(_FLAGS, False),
        "fit_mode": "Release target",
        "fixed_point_index": 0,
        "measured_anchor_distance": 0.0,
        "pose_point_indices": [],
        "solver": "lu",
        "initial_camera_matrix": [],
        "initial_distortion": [],
        "max_iterations": 80,
        "epsilon": 1e-10,
        "max_fit_rms_pixels": 0.5,
        "max_holdout_rms_pixels": 0.5,
        "max_holdout_view_rms_pixels": 0.0,
        "max_target_shift": 0.0,
    }
    parameter_choices = {
        "fit_mode": ("Release target", "Fixed target"),
        "solver": ("svd", "lu", "qr"),
    }
    parameter_ranges = {
        "fixed_point_index": NumericRange(0, _MAX_POINTS - 2, whole=True),
        "measured_anchor_distance": NumericRange(0, 1e6),
        "max_iterations": NumericRange(1, 300, whole=True),
        "epsilon": NumericRange(0, 1),
        "max_fit_rms_pixels": NumericRange(1e-9),
        "max_holdout_rms_pixels": NumericRange(1e-9),
        "max_holdout_view_rms_pixels": NumericRange(0),
        "max_target_shift": NumericRange(0, 1e6),
    }
    parameter_help = {
        "fit_mode": "Fit shared target coordinates or retain the nominal geometry as a control.",
        "fixed_point_index": "Zero picks an anchor. Otherwise use a zero-based index.",
        "measured_anchor_distance": "Distance from point 0 to the chosen point, in target units.",
        "pose_point_indices": "Holdout pose-support rows. Empty uses alternating rows.",
        "max_fit_rms_pixels": "Maximum pooled fit-view reprojection RMS in pixels.",
        "max_holdout_rms_pixels": "Maximum pooled RMS on separate holdout points in pixels.",
        "max_holdout_view_rms_pixels": "Maximum RMS of any holdout view. Zero skips this check.",
        "max_target_shift": "Largest allowed point shift in target units. Zero skips the limit.",
    }
    input_help = {"observations": "A complete, ordered calibration target observation record."}
    output_help = {
        "calibration": "Standard pinhole lens profile. Use pass status to check qualification.",
        "target": "Ordered XYZ, offsets, estimated standard deviations and fixed-axis mask.",
        "report": "Fitted view metrics, holdout results, anchor distances and uncertainty summary.",
        "holdout_rms": "Pooled pixel RMS over points excluded from the holdout pose solve.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["measured_anchor_distance"] and not p["fixed_point_index"]:
            raise ValidationError(
                "Set the measured baseline's fixed point index", "fixed_point_index"
            )
        if p["fix_focal_length"] and not p["use_intrinsic_guess"]:
            raise ValidationError(
                "A fixed focal length requires an intrinsic guess", "use_intrinsic_guess"
            )
        indices = p["pose_point_indices"]
        if not isinstance(indices, list) or any(
            type(i) is not int or not 0 <= i < _MAX_POINTS for i in indices
        ):
            raise ValidationError("Use a list of zero-based point indices", "pose_point_indices")
        if len(indices) > _MAX_POINTS or len(set(indices)) != len(indices):
            raise ValidationError("Pose-support indices must be unique", "pose_point_indices")

    def process(self, inputs):
        p = self.parameters
        obs = target_observations(inputs["observations"])
        nominal, anchor, input_distance, scale = _scaled_nominal(
            obs.nominal, int(p["fixed_point_index"]), float(p["measured_anchor_distance"])
        )
        count = len(nominal)
        default_support = list(range(0, count, 2))
        if len(default_support) < 6:
            default_support.append(1)
        support = np.asarray(p["pose_point_indices"] or default_support, dtype=np.intp)
        if not 6 <= len(support) <= count - 4 or np.any(support >= count):
            raise ValidationError(
                "Use at least six pose points and leave at least four to score",
                "pose_point_indices",
            )
        measured = np.setdiff1d(np.arange(count), support)
        if np.linalg.matrix_rank(nominal[support] - nominal[support[0]]) < 2 or any(
            np.linalg.matrix_rank(xy[support] - xy[support[0]]) < 2 for _, xy in obs.holdout
        ):
            raise ValidationError("Holdout pose points must be noncollinear", "pose_point_indices")
        if self._cancelled():
            return self._cancel_result()
        matrix, distortion, flags, criteria = pinhole_fit_options(p, _FLAGS)
        release = p["fit_mode"] == "Release target"
        names, pixels = zip(*obs.fit, strict=True)
        native = cv2.calibrateCameraROExtended(
            [nominal] * len(pixels),
            list(pixels),
            obs.size,
            anchor if release else -1,
            matrix,
            distortion,
            flags=flags,
            criteria=criteria,
        )
        if self._cancelled():
            return self._cancel_result()
        (
            rms,
            matrix,
            distortion,
            rotations,
            translations,
            xyz,
            intrinsic_sd,
            pose_sd,
            xyz_sd,
            fit_errors,
        ) = native
        if not np.isfinite(rms) or rms < 0:
            raise ValueError("OpenCV returned an invalid fit error")
        if release and (xyz is None or xyz_sd is None):
            raise ValueError(
                "OpenCV did not return released target coordinates and their deviations"
            )
        refined, fixed, stddev = _target_geometry(nominal, anchor, xyz if release else None, xyz_sd)
        calibration = {
            "model": "opencv_pinhole_v1",
            "camera_matrix": matrix.tolist(),
            "distortion": distortion.reshape(-1).tolist(),
            "image_size": list(obs.size),
            "rms_pixels": float(rms),
            "views": len(pixels),
            "units": obs.units,
        }
        camera_parameters(calibration)
        fit = _lens_fit_report(
            [(name, count) for name in names],
            calibration,
            rotations,
            translations,
            intrinsic_sd,
            pose_sd,
            fit_errors,
        )
        held = _holdout_residuals(
            obs.holdout, refined, support, measured, matrix, distortion, self._cancelled
        )
        if held is None:
            return self._cancel_result()
        holdout_rms = float(np.sqrt(np.mean([row["rms_pixels"] ** 2 for row in held])))
        worst_holdout = max(held, key=lambda row: row["rms_pixels"])
        offsets = refined.astype(np.float64) - nominal
        shifts = np.linalg.norm(offsets, axis=1)
        maximum_shift = float(shifts.max())
        target = {
            "model": "refined_calibration_target_v1",
            "target_id": obs.target_id,
            "units": obs.units,
            "nominal_points": nominal.tolist(),
            "object_points": refined.tolist(),
            "offsets": offsets.tolist(),
            "fixed_axes": fixed.tolist(),
            "coordinate_stddev": np.where(fixed, None, stddev).tolist() if release else None,
        }
        report = {
            "model": "pinhole_target_fit_v1",
            "target_id": obs.target_id,
            "units": obs.units,
            "fit_mode": p["fit_mode"],
            "fit": fit,
            "holdout_views": held,
            "holdout_pose_indices": support.tolist(),
            "holdout_scored_indices": measured.tolist(),
            "holdout_rms_pixels": holdout_rms,
            "worst_holdout_view": worst_holdout["view_id"],
            "worst_holdout_rms_pixels": worst_holdout["rms_pixels"],
            "anchor_index": anchor,
            "nominal_anchor_distance": input_distance,
            "measured_anchor_distance": p["measured_anchor_distance"] or None,
            "nominal_scale": scale,
            "fitted_anchor_distance": float(np.linalg.norm(refined[anchor] - refined[0])),
            "max_target_shift": maximum_shift,
            "max_coordinate_stddev": float(stddev[~fixed].max()) if release else None,
        }
        checks = (
            ("fit_rms", float(rms), p["max_fit_rms_pixels"]),
            ("holdout_rms", holdout_rms, p["max_holdout_rms_pixels"]),
            ("holdout_view_rms", worst_holdout["rms_pixels"], p["max_holdout_view_rms_pixels"]),
            ("target_shift", maximum_shift, p["max_target_shift"]),
        )
        failed = [
            f"target_calibration_{name}" for name, value, limit in checks if limit and value > limit
        ]
        report["status"] = "failed" if failed else "ok"
        return Result(
            {
                "calibration": calibration,
                "target": target,
                "report": report,
                "holdout_rms": holdout_rms,
            },
            passed=not failed,
            fail_reasons=failed,
            message=f"Fit RMS {rms:.4g} px, holdout RMS {holdout_rms:.4g} px",
            graphics={"Target offsets": _offset_plot(nominal, offsets, fixed)}
            if self.capture_graphics
            else {},
        )

    @staticmethod
    def _cancel_result():
        return Result(
            {
                "calibration": NO_VALUE,
                "target": NO_VALUE,
                "holdout_rms": NO_VALUE,
                "report": {"model": "pinhole_target_fit_v1", "status": "cancelled"},
            },
            passed=False,
            fail_reasons=["target_calibration_cancelled"],
            message="Target calibration cancelled",
        )
