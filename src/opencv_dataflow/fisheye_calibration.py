"""Fit a reusable zero-skew fisheye camera from offline calibration targets."""

import cv2
import numpy as np

from .calibration import GRID_OPTIONS, GridCalibrationBlock
from .charuco import calibration_observations
from .core import Port, Result
from .geometry3d_common import camera, camera_record, image_size, pose_matrix, pose_record
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE

FLAGS = {
    "recompute_extrinsics": cv2.CALIB_RECOMPUTE_EXTRINSIC,
    "check_condition": cv2.CALIB_CHECK_COND,
    "fix_focal_length": cv2.CALIB_FIX_FOCAL_LENGTH,
    "fix_principal_point": cv2.CALIB_FIX_PRINCIPAL_POINT,
    **{f"fix_k{i}": getattr(cv2, f"CALIB_FIX_K{i}") for i in range(1, 5)},
}


class FisheyeCalibration(GridCalibrationBlock):
    type_id, title = "3d.fisheye_calibrate", "Calibrate fisheye camera"
    category, block_type, subtype = "3D vision", "3D vision", "Calibration"
    inputs = (
        Port("observations", "any", required=False, required_when_connected=True),
        Port("initial_camera", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("camera", "data", required=False),
        Port("rms", "number"),
        Port("fit_report", "data"),
    )
    defaults = {
        **GRID_OPTIONS,
        "image_paths": [],
        "path_base": "project",
        "columns": 9,
        "rows": 6,
        "spacing": 10.0,
        "units": "mm",
        "camera_frame": "camera",
        "target_frame": "calibration_target",
        **dict.fromkeys(FLAGS, False),
        "recompute_extrinsics": True,
        "check_condition": True,
        "max_iterations": 100,
        "epsilon": 1e-9,
        "max_rms_pixels": 1.0,
        "max_view_rms_pixels": 0.0,
        "minimum_coverage": 0.0,
    }
    parameter_choices = {
        **GridCalibrationBlock.parameter_choices,
        "path_base": ("project", "working_directory"),
        "units": ("mm", "m", "unitless"),
    }
    parameter_ranges = {
        **GridCalibrationBlock.parameter_ranges,
        "columns": NumericRange(3, 1000, whole=True),
        "rows": NumericRange(3, 1000, whole=True),
        "spacing": NumericRange(1e-9),
        "max_iterations": NumericRange(1, 1000, whole=True),
        "epsilon": NumericRange(0, 1),
        "max_rms_pixels": NumericRange(1e-9),
        "max_view_rms_pixels": NumericRange(0),
        "minimum_coverage": NumericRange(0, 1),
    }
    parameter_help = {
        **GridCalibrationBlock.parameter_help,
        "image_paths": "Three to 64 checkerboard/circle-grid images. Leave empty for Observations.",
        "path_base": "Base folder for relative calibration image paths.",
        "units": "Grid spacing units for image paths. An observation batch supplies its own.",
        "recompute_extrinsics": "Refit target-to-camera poses during intrinsic optimization.",
        "check_condition": "Have OpenCV reject ill-conditioned extrinsic updates.",
        "fix_focal_length": "Keep both focal lengths from the connected initial camera.",
        "fix_principal_point": "Keep the principal point from the connected initial camera.",
        **{f"fix_k{i}": f"Fix fisheye coefficient k{i} at zero." for i in range(1, 5)},
        "max_iterations": "Maximum iterations for the OpenCV solve.",
        "epsilon": (
            "Relative focal/principal-point change. An epsilon of zero or locking both uses "
            "the iteration limit."
        ),
        "max_rms_pixels": "Accept at most this overall reprojection RMS in source pixels.",
        "max_view_rms_pixels": "Accept at most this RMS per view. Zero disables that gate.",
        "minimum_coverage": "Minimum fraction of the sensor spanned by the joint corner hull.",
        "camera_frame": "Frame name for the fitted camera and target-to-camera pose outputs.",
        "target_frame": "Local coordinate frame of the calibration target.",
    }
    input_help = {
        "observations": "Batch or list of ChArUco observations of one board and camera grid.",
        "initial_camera": "Optional fisheye seed with four coefficients on the same image grid.",
    }
    process_inputs_readonly = parallel_safe = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        paths = p["image_paths"]
        if (
            not isinstance(paths, list)
            or (paths and not 3 <= len(paths) <= 64)
            or any(not isinstance(path, str) or not path.strip() for path in paths)
        ):
            raise ValidationError(
                "Use three to 64 grid image paths or leave the list empty", "image_paths"
            )
        for key in FLAGS:
            if type(p[key]) is not bool:
                raise ValidationError("Use a checkbox", key)
        if (
            p["fix_focal_length"]
            and p["fix_principal_point"]
            and all(p[f"fix_k{i}"] for i in range(1, 5))
        ):
            raise ValidationError("Leave at least one intrinsic parameter free", "fix_focal_length")
        for key in ("camera_frame", "target_frame"):
            if not isinstance(p[key], str) or not p[key].strip():
                raise ValidationError("Frame names must be nonempty", key)
        for key in ("columns", "rows", "max_iterations"):
            if type(p[key]) is not int:
                raise ValidationError("Use an integer", key)
        for key in ("epsilon", "max_rms_pixels", "max_view_rms_pixels", "minimum_coverage"):
            if type(p[key]) not in (int, float):
                raise ValidationError("Use a numeric value", key)

    def process(self, inputs):
        p = self.parameters
        if "observations" in inputs:
            if p["image_paths"]:
                raise ValidationError("Use image paths or ChArUco observations for one calibration")
            worlds, pixels, size, names, units = calibration_observations(inputs["observations"])
        else:
            worlds, pixels, size, names, units = self._grid_views()
        size = image_size(size)
        world_arrays = [
            np.ascontiguousarray(world.reshape(-1, 1, 3), np.float64) for world in worlds
        ]
        # The native final residual comparison expects a row of two-channel pixels.
        pixel_arrays = [np.ascontiguousarray(xy.reshape(1, -1, 2), np.float64) for xy in pixels]
        flags = cv2.CALIB_FIX_SKEW | sum(flag for key, flag in FLAGS.items() if p[key])
        initial = inputs.get("initial_camera")
        if initial is None:
            if p["fix_focal_length"] or p["fix_principal_point"]:
                raise ValidationError(
                    "Connect an initial camera to fix focal length or principal point"
                )
            k, d = np.eye(3, dtype=np.float64), np.zeros((4, 1), np.float64)
        else:
            k, d, initial_size = camera(initial)
            if initial.get("lens") != "fisheye" or initial_size != size:
                raise ValidationError("Initial camera must be fisheye on the same pixel grid")
            k, d = k.copy(), d.reshape(4, 1).copy()
            flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        # Native epsilon compares only focal lengths and principal point. With
        # both fixed it would stop after one iteration, even if distortion is moving.
        use_epsilon = p["epsilon"] > 0 and not (p["fix_focal_length"] and p["fix_principal_point"])
        criteria = (
            cv2.TERM_CRITERIA_COUNT | (cv2.TERM_CRITERIA_EPS if use_epsilon else 0),
            p["max_iterations"],
            p["epsilon"] if use_epsilon else 0.0,
        )
        rms, k, d, rvecs, tvecs = cv2.fisheye.calibrate(
            world_arrays, pixel_arrays, size, k, d, flags=flags, criteria=criteria
        )
        if not np.isfinite(rms) or rms < 0:
            raise ValidationError("Fisheye solver returned an invalid RMS")
        profile = camera_record(
            k,
            d,
            size,
            units,
            p["camera_frame"],
            "fisheye",
            rms_pixels=float(rms),
            views=len(worlds),
        )
        views = []
        for name, world, xy, rvec, tvec in zip(
            names, world_arrays, pixel_arrays, rvecs, tvecs, strict=True
        ):
            transform = pose_matrix(cv2.Rodrigues(rvec)[0], tvec)
            depth = world.reshape(-1, 3) @ transform[2, :3] + transform[2, 3]
            if np.any(depth <= 0):
                raise ValidationError("A fitted target has points outside positive camera depth")
            projected = cv2.fisheye.projectPoints(world, rvec, tvec, k, d)[0]
            errors = np.linalg.norm(projected.reshape(-1, 2) - xy.reshape(-1, 2), axis=1)
            if not np.isfinite(errors).all():
                raise ValidationError("Fisheye reprojection returned invalid residuals")
            views.append(
                {
                    "name": name,
                    "point_count": len(errors),
                    "rms_pixels": float(np.sqrt(np.mean(errors**2))),
                    "max_error_pixels": float(errors.max()),
                    "pose": pose_record(transform, p["target_frame"], p["camera_frame"], units),
                }
            )
        observed = np.concatenate(pixels).astype(np.float32)
        coverage = min(1.0, cv2.contourArea(cv2.convexHull(observed)) / (size[0] * size[1]))
        worst = max(range(len(views)), key=lambda i: views[i]["rms_pixels"])
        worst_rms = views[worst]["rms_pixels"]
        reasons = []
        if rms > p["max_rms_pixels"]:
            reasons.append(
                {
                    "code": "fisheye.overall_rms",
                    "message": f"Overall RMS {rms:.4g} px exceeds {p['max_rms_pixels']:.4g} px",
                }
            )
        if p["max_view_rms_pixels"] and worst_rms > p["max_view_rms_pixels"]:
            reasons.append(
                {
                    "code": "fisheye.view_rms",
                    "message": f"{names[worst]} RMS {worst_rms:.4g} px exceeds its limit",
                }
            )
        if coverage < p["minimum_coverage"]:
            reasons.append(
                {
                    "code": "fisheye.coverage",
                    "message": f"Corner hull {coverage:.1%} is below {p['minimum_coverage']:.1%}",
                }
            )
        return Result(
            {
                "camera": NO_VALUE if reasons else profile,
                "rms": float(rms),
                "fit_report": {
                    "model": "fisheye_fit_report_v1",
                    "image_size": list(size),
                    "rms_pixels": float(rms),
                    "coverage": coverage,
                    "worst_view_index": worst,
                    "worst_view_rms_pixels": worst_rms,
                    "views": views,
                },
            },
            passed=not reasons,
            message=f"{len(views)} views, RMS {rms:.4g} px, corner hull {coverage:.1%}",
            fail_reasons=reasons,
        )
