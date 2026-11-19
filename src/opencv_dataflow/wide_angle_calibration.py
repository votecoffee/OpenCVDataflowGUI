"""Shared calibration view ingestion and lens fit diagnostics."""

import cv2
import numpy as np

from .calibration import GRID_OPTIONS, GridCalibrationBlock
from .charuco import calibration_observations
from .core import Port, Result
from .geometry3d_common import image_size, pose_matrix, pose_record
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE


class WideAngleCalibration(GridCalibrationBlock):
    category, block_type, subtype = "3D vision", "3D vision", "Calibration"
    inputs = (Port("observations", "any", required=False, required_when_connected=True),)
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
        "max_iterations": "Maximum iterations for the OpenCV solve.",
        "max_rms_pixels": "Accept at most this overall reprojection RMS in source pixels.",
        "max_view_rms_pixels": "Accept at most this RMS per view. Zero disables that gate.",
        "minimum_coverage": "Minimum fraction of the sensor spanned by the joint corner hull.",
        "camera_frame": "Frame name for the fitted camera and target-to-camera pose outputs.",
        "target_frame": "Local coordinate frame of the calibration target.",
    }
    input_help = {
        "observations": "Batch or list of ChArUco observations of one board and camera grid.",
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
        for key in ("camera_frame", "target_frame"):
            if not isinstance(p[key], str) or not p[key].strip():
                raise ValidationError("Frame names must be nonempty", key)
        for key in ("columns", "rows", "max_iterations"):
            if type(p[key]) is not int:
                raise ValidationError("Use an integer", key)
        for key in ("epsilon", "max_rms_pixels", "max_view_rms_pixels", "minimum_coverage"):
            if type(p[key]) not in (int, float):
                raise ValidationError("Use a numeric value", key)

    def read_views(self, inputs):
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
        return world_arrays, pixel_arrays, size, names, units

    def fit_result(self, views_data, profile, rvecs, tvecs, project, *, used=None, reasons=()):
        p = self.parameters
        world_arrays, pixel_arrays, size, names, units = views_data
        lens, rms = profile["lens"], profile["rms_pixels"]
        indices = list(range(len(names))) if used is None else used
        views = []
        for index, rvec, tvec in zip(indices, rvecs, tvecs, strict=True):
            name, world, xy = names[index], world_arrays[index], pixel_arrays[index]
            transform = pose_matrix(cv2.Rodrigues(rvec)[0], tvec)
            depth = world.reshape(-1, 3) @ transform[2, :3] + transform[2, 3]
            if np.any(depth <= 0):
                raise ValidationError("A fitted target has points outside positive camera depth")
            projected = project(world, rvec, tvec)
            errors = np.linalg.norm(projected.reshape(-1, 2) - xy.reshape(-1, 2), axis=1)
            if not np.isfinite(errors).all():
                raise ValidationError("Lens reprojection returned invalid residuals")
            views.append(
                {
                    "name": name,
                    "point_count": len(errors),
                    "rms_pixels": float(np.sqrt(np.mean(errors**2))),
                    "max_error_pixels": float(errors.max()),
                    "pose": pose_record(transform, p["target_frame"], p["camera_frame"], units),
                }
            )
        observed = np.concatenate([pixel_arrays[i].reshape(-1, 2) for i in indices]).astype(
            np.float32
        )
        coverage = min(1.0, cv2.contourArea(cv2.convexHull(observed)) / (size[0] * size[1]))
        worst = max(range(len(views)), key=lambda i: views[i]["rms_pixels"])
        worst_rms = views[worst]["rms_pixels"]
        reasons = list(reasons)
        if rms > p["max_rms_pixels"]:
            reasons.append(
                {
                    "code": f"{lens}.overall_rms",
                    "message": f"Overall RMS {rms:.4g} px exceeds {p['max_rms_pixels']:.4g} px",
                }
            )
        if p["max_view_rms_pixels"] and worst_rms > p["max_view_rms_pixels"]:
            reasons.append(
                {
                    "code": f"{lens}.view_rms",
                    "message": f"{views[worst]['name']} RMS {worst_rms:.4g} px exceeds its limit",
                }
            )
        if coverage < p["minimum_coverage"]:
            reasons.append(
                {
                    "code": f"{lens}.coverage",
                    "message": f"Corner hull {coverage:.1%} is below {p['minimum_coverage']:.1%}",
                }
            )
        report = {
            "model": f"{lens}_fit_report_v1",
            "image_size": list(size),
            "rms_pixels": rms,
            "coverage": coverage,
            "worst_view_index": indices[worst],
            "worst_view_rms_pixels": worst_rms,
            "views": views,
        }
        if used is not None:
            report["used_view_indices"] = indices
            used_set = set(indices)
            report["excluded_views"] = [
                {"index": i, "name": name} for i, name in enumerate(names) if i not in used_set
            ]
        return Result(
            {"camera": NO_VALUE if reasons else profile, "rms": rms, "fit_report": report},
            passed=not reasons,
            message=f"{len(views)} views, RMS {rms:.4g} px, corner hull {coverage:.1%}",
            fail_reasons=reasons,
        )
