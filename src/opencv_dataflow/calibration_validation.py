"""Check fixed pinhole intrinsics against a separate grid observation."""

import cv2
import numpy as np

from .calibration import (
    GRID_OPTIONS,
    GridCalibrationBlock,
    GridNotFoundError,
    camera_parameters,
    find_grid,
    grid_settings,
)
from .core import Port, Result
from .geometry3d import PNP
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .result_graphics import ACCEPTED, REJECTED

_MODEL = "pinhole_validation_v1"


def _unmeasured(code, message):
    return Result(
        {
            "report": {"model": _MODEL, "status": code},
            "rms": NO_VALUE,
            "max_error": NO_VALUE,
        },
        passed=False,
        message=message,
        fail_reasons=[{"code": code, "message": message}],
    )


class ValidateLensCalibration(GridCalibrationBlock):
    type_id, title, category = (
        "calibration.validate_lens",
        "Validate lens calibration",
        "Calibration",
    )
    block_type, subtype = "Calibration", "Lens distortion"
    inputs = (Port("image"), Port("calibration", "data"))
    outputs = (Port("report", "data"), Port("rms", "number"), Port("max_error", "number"))
    defaults = {
        **GRID_OPTIONS,
        "columns": 9,
        "rows": 6,
        "spacing": 10.0,
        "units": "mm",
        "pose_method": "iterative",
        "max_rms_pixels": 1.0,
        "max_point_error_pixels": 0.0,
        "residual_scale": 12.0,
    }
    parameter_choices = {**GridCalibrationBlock.parameter_choices, "pose_method": tuple(PNP)}
    parameter_ranges = {
        **GridCalibrationBlock.parameter_ranges,
        "max_rms_pixels": NumericRange(1e-9),
        "max_point_error_pixels": NumericRange(0),
        "residual_scale": NumericRange(0, 100),
    }
    parameter_help = {
        **GridCalibrationBlock.parameter_help,
        "pose_method": "OpenCV method for fitting the board pose with fixed lens parameters.",
        "max_rms_pixels": "Maximum reprojection RMS over all detected points.",
        "max_point_error_pixels": "Maximum error of any point. Zero skips this check.",
        "residual_scale": "Display multiplier for residual vectors. Zero hides the vectors.",
    }
    process_inputs_readonly = True

    def process(self, inputs):
        p = self.parameters
        image, profile = inputs["image"], inputs["calibration"]
        camera, distortion = camera_parameters(profile)
        image_size = list(image.shape[:2][::-1])
        if profile.get("image_size") != image_size:
            raise ValidationError("Validation image dimensions must match the lens calibration")
        grid, world = grid_settings(p)
        try:
            observed, _ = find_grid(image, grid, p)
        except GridNotFoundError as exc:
            return _unmeasured("calibration_validation_grid_not_found", str(exc))
        if observed.shape != (len(world), 2) or not np.isfinite(observed).all():
            raise ValidationError("Grid detection returned invalid image coordinates")
        world = world.astype(np.float64)
        if not np.isfinite(world).all():
            raise ValidationError("Grid spacing produces nonfinite board coordinates")
        method = p["pose_method"]
        found, rotation, translation = cv2.solvePnP(
            world, observed, camera, distortion, flags=PNP[method]
        )
        if not found or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            return _unmeasured(
                "calibration_validation_pose_failed", "Could not estimate a board pose"
            )
        rotation, translation = rotation.reshape(3), translation.reshape(3)
        depth = world @ cv2.Rodrigues(rotation)[0][2] + translation[2]
        if not np.isfinite(depth).all() or np.any(depth <= 0):
            return _unmeasured(
                "calibration_validation_pose_failed",
                "Estimated grid lies outside positive camera depth",
            )
        projected = cv2.projectPoints(world, rotation, translation, camera, distortion)[0]
        projected = projected.reshape(-1, 2)
        residuals = projected - observed
        errors = np.linalg.norm(residuals, axis=1)
        if not np.isfinite(errors).all():
            raise ValidationError("Reprojection produced nonfinite point errors")
        rms = float(np.sqrt(np.mean(errors**2)))
        worst = int(np.argmax(errors))
        maximum = float(errors[worst])
        rms_limit = float(p["max_rms_pixels"])
        point_limit = float(p["max_point_error_pixels"])
        accepted = errors <= point_limit if point_limit else np.ones(len(errors), dtype=bool)
        reasons = []
        if rms > rms_limit:
            reasons.append(
                {
                    "code": "calibration_validation_rms_exceeds_limit",
                    "message": f"RMS {rms:.4g} px exceeds {rms_limit:.4g} px",
                }
            )
        if point_limit and maximum > point_limit:
            reasons.append(
                {
                    "code": "calibration_validation_point_exceeds_limit",
                    "message": f"Point {worst} error {maximum:.4g} px exceeds {point_limit:.4g} px",
                }
            )
        report = {
            "model": _MODEL,
            "status": "measured",
            "image_size": image_size,
            "pose_method": method,
            "pose_direction": "board_to_camera",
            "rotation_vector_radians": rotation.tolist(),
            "translation_vector": translation.tolist(),
            "translation_units": p["units"],
            "rms_pixels": rms,
            "max_error_pixels": maximum,
            "worst_point_index": worst,
            "point_limit_pixels": point_limit,
            "points": [
                {
                    "grid": [index % grid[0], index // grid[0]],
                    "observed": measured.tolist(),
                    "projected": predicted.tolist(),
                    "error_pixels": float(error),
                }
                for index, (measured, predicted, error) in enumerate(
                    zip(observed, projected, errors)
                )
            ],
        }
        drawings = {}
        if self.generate_graphics:
            colors = [ACCEPTED if passed else REJECTED for passed in accepted]
            drawings["Observed grid"] = GraphicDrawing(
                tuple(
                    graphic_object("marker", [point], color)
                    for point, color in zip(observed, colors)
                )
            )
            scale = float(p["residual_scale"])
            if scale:
                endpoints = observed + residuals * scale
                drawings["Reprojection residuals"] = GraphicDrawing(
                    tuple(
                        graphic_object("polyline", (start, end), color)
                        for start, end, color in zip(observed, endpoints, colors)
                    )
                )
        error_ratio = max(rms / rms_limit, maximum / point_limit if point_limit else 0)
        return Result(
            {"report": report, "rms": rms, "max_error": maximum},
            passed=not reasons,
            fail_reasons=reasons,
            score=max(0.0, 1 - error_ratio),
            message=f"{len(errors)} points · RMS {rms:.4g} px · maximum {maximum:.4g} px",
            drawings=drawings,
        )
