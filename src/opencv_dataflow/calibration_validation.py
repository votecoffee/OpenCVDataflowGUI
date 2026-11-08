"""Check fixed pinhole intrinsics against a separate grid observation."""

import cv2
import numpy as np

from .calibration import (
    GRID_OPTIONS,
    PLANAR_REGISTRATION,
    GridCalibrationBlock,
    GridNotFoundError,
    camera_parameters,
    find_grid,
    grid_settings,
    map_points,
    require_planar_grid,
)
from .charuco import observation_board, observation_points
from .core import Block, Port, Result
from .geometry3d import PNP
from .graphics_layers import GraphicDrawing, graphic_object
from .outcomes import failed_input
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, ValuePacket, require_same_frame
from .result_graphics import ACCEPTED, REJECTED
from .tickets import ticket_key

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


def _capture_key(ticket):
    return ticket_key(ticket), ticket["source"], ticket["sequence"]


class ValidateLensCalibration(GridCalibrationBlock):
    type_id, title, category = (
        "calibration.validate_lens",
        "Validate lens calibration",
        "Calibration",
    )
    block_type, subtype = "Calibration", "Lens distortion"
    inputs = (
        Port("image"),
        Port("calibration", "data"),
        Port("observation", "data", required=False, required_when_connected=True),
    )
    input_help = {"observation": "Optional identified ChArUco points from the validation image."}
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

    def _require_observation_alignment(self, image, observation):
        image_payload = self.input_payloads["image"]
        observation_payload = self.input_payloads["observation"]
        message = "ChArUco observation and validation image must share the same coordinate frame"
        require_same_frame(
            (ValuePacket(image, image_payload), ValuePacket(observation, observation_payload)),
            message=message,
        )
        image_spatial = image_payload.get("spatial")
        observation_spatial = observation_payload.get("spatial")
        if (
            image_spatial
            and observation_spatial
            and (
                image_spatial["root_size"] != observation_spatial["root_size"]
                or image_spatial["image_size"] != observation_spatial["image_size"]
                or not np.allclose(
                    image_spatial["to_root"], observation_spatial["to_root"], atol=1e-9, rtol=0
                )
            )
        ):
            raise ValidationError(message)
        image_ticket, observation_ticket = (
            image_payload.get("ticket"),
            observation_payload.get("ticket"),
        )
        if (
            image_ticket is not None
            and observation_ticket is not None
            and _capture_key(image_ticket) != _capture_key(observation_ticket)
        ):
            raise ValidationError("ChArUco observation must come from the same captured image")

    def process(self, inputs):
        p = self.parameters
        image, profile = inputs["image"], inputs["calibration"]
        camera, distortion = camera_parameters(profile)
        image_size = list(image.shape[:2][::-1])
        if profile.get("image_size") != image_size:
            raise ValidationError("Validation image dimensions must match the lens calibration")
        board = None
        if "observation" in inputs:
            observation = inputs["observation"]
            self._require_observation_alignment(image, observation)
            if failed_input(ValuePacket(observation, self.input_payloads["observation"])):
                return _unmeasured(
                    "calibration_validation_observation_failed",
                    "ChArUco detection failed for the validation image",
                )
            board, size, native_board, board_points = observation_board(observation)
            if list(size) != image_size:
                raise ValidationError(
                    "ChArUco observation dimensions must match the validation image"
                )
            ids, observed, world = observation_points(observation, native_board, board_points, size)
            corner_ids = ids.tolist()
            columns, units = int(board["columns"]) - 1, board["units"]
        else:
            grid, world = grid_settings(p)
            try:
                observed, _ = find_grid(image, grid, p)
            except GridNotFoundError as exc:
                return _unmeasured("calibration_validation_grid_not_found", str(exc))
            columns, units = grid[0], p["units"]
            corner_ids = list(range(len(observed)))
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
            label = f"Corner ID {corner_ids[worst]}" if board else f"Point {worst}"
            reasons.append(
                {
                    "code": "calibration_validation_point_exceeds_limit",
                    "message": f"{label} error {maximum:.4g} px exceeds {point_limit:.4g} px",
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
            "translation_units": units,
            "rms_pixels": rms,
            "max_error_pixels": maximum,
            "worst_point_index": worst,
            "point_limit_pixels": point_limit,
            "points": [
                {
                    "grid": [identifier % columns, identifier // columns],
                    "observed": measured.tolist(),
                    "projected": predicted.tolist(),
                    "error_pixels": float(error),
                }
                for identifier, measured, predicted, error in zip(
                    corner_ids, observed, projected, errors
                )
            ],
        }
        if board is not None:
            report["board"] = board
            report["worst_corner_id"] = corner_ids[worst]
            for point, identifier in zip(report["points"], corner_ids):
                point["corner_id"] = identifier
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


_PLANAR_REPORT = "planar_validation_v1"


def _check_points(value, name):
    try:
        points = np.asarray(value, np.float64)
    except (TypeError, ValueError, OverflowError):
        raise ValidationError("Use a list of finite [x,y] points", name) from None
    if points.shape == (0,):
        return points.reshape(0, 2)
    if (
        points.ndim != 2 or points.shape[1] != 2 or len(points) > 4096
        or not np.isfinite(points).all() or np.any(np.abs(points) > 1e9)
    ):
        raise ValidationError("Use up to 4096 finite [x,y] points within ±1 billion", name)
    return points


def _noncollinear(points):
    return (
        len(points) >= 3 and len(np.unique(points, axis=0)) == len(points)
        and np.linalg.matrix_rank(points - points[0]) == 2
    )


class ValidatePlanarCalibration(Block):
    type_id, title = "calibration.validate_planar", "Validate planar calibration"
    category = block_type = "Calibration"
    subtype = "Grid and coordinates"
    inputs = (
        Port("image"), Port("mapping", "data"),
        Port("points", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("report", "data"), Port("checked_mapping", "data", required=False),
        Port("rms", "number", required=False), Port("max_error", "number", required=False),
    )
    payload_reference = "image"
    process_inputs_readonly = True
    parallel_safe = True
    defaults = {
        "physical_points": [[0, 0], [10, 0], [0, 10]],
        "pixel_points": [], "units": "mm", "registration": "same_frame",
        "maximum_rms": 0.25, "maximum_point_error": 0.5,
    }
    parameter_choices = {"registration": PLANAR_REGISTRATION}
    parameter_ranges = {
        "maximum_rms": NumericRange(0), "maximum_point_error": NumericRange(0),
    }
    parameter_help = {
        "physical_points": "Known [x,y] positions on the calibration plane, in observation order. "
        "Use at least three distinct, non-collinear points held out from the fit.",
        "pixel_points": "Measured [x,y] image locations. A wired points list takes precedence.",
        "units": "Physical units expected in the mapping and used for the reference and limits.",
        "registration": "Fixed grid allows a saved mapping for the same camera grid.",
        "maximum_rms": "Inclusive limit for RMS physical point error.",
        "maximum_point_error": "Inclusive limit for the largest physical point error.",
    }
    input_help = {
        "image": "Current camera image for coordinate validation, graphics and ticket.",
        "mapping": "Fixed planar calibration to check without refitting.",
        "points": "Measured [x,y] fiducial locations in this image and in reference order.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        expected = _check_points(p["physical_points"], "physical_points")
        if not _noncollinear(expected):
            raise ValidationError(
                "Physical points must be distinct and non-collinear", "physical_points"
            )
        _check_points(p["pixel_points"], "pixel_points")
        units = p["units"]
        if not isinstance(units, str) or not units.strip() or len(units) > 24:
            raise ValidationError("Enter the calibration's physical units", "units")

    def process(self, inputs):
        image, mapping = inputs["image"], inputs["mapping"]
        p = self.parameters
        image_payload = self.input_payloads["image"]
        require_planar_grid(
            mapping, image.shape, image_payload["spatial"],
            self.input_payloads["mapping"].get("spatial"), p["registration"],
        )
        units = p["units"]
        if mapping["units"] != units:
            raise ValidationError("Reference units must match the planar calibration", "units")
        expected = _check_points(p["physical_points"], "physical_points")
        observed = _check_points(inputs.get("points", p["pixel_points"]), "points")
        reason = ""
        if "points" in inputs:
            points_payload = self.input_payloads["points"]
            if points_payload.get("spatial") != image_payload["spatial"]:
                raise ValidationError("Observed points must share the image coordinate frame")
            a, b = image_payload.get("ticket"), points_payload.get("ticket")
            if a is not None and b is not None and _capture_key(a) != _capture_key(b):
                raise ValidationError("Observed points must come from the current capture")
            if failed_input(ValuePacket(inputs["points"], points_payload)):
                reason = "Fiducial detection failed for the current image"
        size = np.array(image.shape[:2][::-1])
        if not reason and len(observed) != len(expected):
            reason = f"Expected {len(expected)} fiducials, observed {len(observed)}"
        if not reason and (
            not _noncollinear(observed)
            or np.any((observed < 0) | (observed > size - 1))
        ):
            reason = "Fiducials must be distinct, non-collinear and inside the image"
        report = {
            "model": _PLANAR_REPORT, "units": units,
            "expected_count": len(expected), "observed_count": len(observed),
        }
        if reason:
            code = "planar_validation_observation_unavailable"
            report.update(passed=False, status=code)
            return Result(
                {"report": report, "checked_mapping": NO_VALUE, "rms": NO_VALUE,
                 "max_error": NO_VALUE},
                passed=False, message=reason, fail_reasons=[{"code": code, "message": reason}],
            )

        measured = map_points(mapping, observed)
        residuals = measured - expected
        errors = np.hypot(residuals[:, 0], residuals[:, 1])
        if not np.isfinite(errors).all():
            raise ValidationError("Planar validation produced nonfinite residuals")
        worst = int(np.argmax(errors))
        maximum = float(errors[worst])
        rms = maximum * float(np.sqrt(np.mean((errors / maximum) ** 2))) if maximum else 0.0
        rms_limit = float(p["maximum_rms"])
        point_limit = float(p["maximum_point_error"])
        reasons = []
        if rms > rms_limit:
            reasons.append({
                "code": "planar_validation_rms_exceeds_limit",
                "message": f"RMS {rms:.4g} {units} exceeds {rms_limit:.4g} {units}",
            })
        if maximum > point_limit:
            reasons.append({
                "code": "planar_validation_point_exceeds_limit",
                "message": f"Point {worst + 1}: {maximum:.4g} {units} exceeds "
                           f"{point_limit:.4g} {units}",
            })
        passed = not reasons
        report.update(
            passed=passed, status="measured", rms=rms, max_error=maximum, worst_point_index=worst,
            points=[
                {"pixel": point.tolist(), "physical": actual.tolist(), "expected": target.tolist(),
                 "residual": residual.tolist(), "error": float(error)}
                for point, actual, target, residual, error
                in zip(observed, measured, expected, residuals, errors, strict=True)
            ],
        )
        drawings = {}
        if self.generate_graphics:
            # Always include the largest errors when the reference contains many points.
            order = np.argsort(-errors, kind="stable")[:64]
            objects = []
            for index in order:
                point = observed[index]
                color = ACCEPTED if errors[index] <= point_limit else REJECTED
                objects.append(graphic_object("marker", [point], color))
                objects.append(graphic_object(
                    "label", [point], color, text=f"{errors[index]:.3g} {units}"
                ))
            drawings["Planar validation errors"] = GraphicDrawing(tuple(objects))
        return Result(
            {"report": report, "checked_mapping": mapping if passed else NO_VALUE,
             "rms": rms, "max_error": maximum},
            passed=passed, fail_reasons=reasons, drawings=drawings,
            message=f"{len(errors)} points · RMS {rms:.4g} {units} · maximum {maximum:.4g} {units}",
        )
