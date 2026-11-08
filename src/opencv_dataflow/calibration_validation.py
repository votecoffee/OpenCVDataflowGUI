"""Fit and verify planar mappings and gauge spacing in calibrated images."""

import math

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


def _require_planar_observation(block):
    image_payload = block.input_payloads["image"]
    points_payload = block.input_payloads["points"]
    if points_payload.get("spatial") != image_payload["spatial"]:
        raise ValidationError("Observed points must share the image coordinate frame")
    a, b = image_payload.get("ticket"), points_payload.get("ticket")
    if a is not None and b is not None and _capture_key(a) != _capture_key(b):
        raise ValidationError("Observed points must come from the current capture")
    return points_payload


class InspectPlanarDistance(Block):
    type_id, title = "calibration.inspect_planar_distance", "Inspect planar point spacing"
    category = block_type = "Calibration"
    subtype = "Grid and coordinates"
    inputs = (Port("image"), Port("mapping", "data"), Port("points", "data"))
    outputs = (Port("report", "data"), Port("distance", "number", required=False))
    payload_reference = "image"
    process_inputs_readonly = True
    parallel_safe = True
    defaults = {
        "units": "mm", "registration": "same_frame", "maximum_mapping_rms": 1.0,
        "minimum_distance": 0.0, "maximum_distance": 0.0,
    }
    parameter_choices = {"registration": PLANAR_REGISTRATION}
    parameter_ranges = {
        "maximum_mapping_rms": NumericRange(0),
        "minimum_distance": NumericRange(0), "maximum_distance": NumericRange(0),
    }
    parameter_help = {
        "units": "Physical units expected in the planar calibration and used for limits.",
        "registration": "Fixed grid allows a saved mapping for the same camera grid.",
        "maximum_mapping_rms": "Inclusive limit for the calibration RMS in physical units.",
        "minimum_distance": "Inclusive minimum center-to-center distance.",
        "maximum_distance": "Inclusive maximum distance. Zero disables the upper limit.",
    }
    input_help = {
        "image": "Current camera image used for frame validation, graphics and ticket.",
        "mapping": "Planar calibration for the camera grid. Wire a checked mapping to gate on "
                   "held-out validation.",
        "points": "Exactly two [x,y] endpoints in this camera image, such as blob centroids.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        units = p["units"]
        if not isinstance(units, str) or not units.strip() or len(units) > 24:
            raise ValidationError("Enter the calibration's physical units", "units")
        maximum = float(p["maximum_distance"])
        if maximum and maximum < float(p["minimum_distance"]):
            raise ValidationError(
                "Maximum distance must be at least the minimum or zero", "maximum_distance"
            )

    def process(self, inputs):
        image, mapping = inputs["image"], inputs["mapping"]
        p = self.parameters
        image_payload = self.input_payloads["image"]
        mapping_payload = self.input_payloads["mapping"]
        require_planar_grid(
            mapping, image.shape, image_payload["spatial"],
            mapping_payload.get("spatial"), p["registration"],
        )
        image_ticket, mapping_ticket = image_payload.get("ticket"), mapping_payload.get("ticket")
        if (
            image_ticket is not None and mapping_ticket is not None
            and _capture_key(image_ticket) != _capture_key(mapping_ticket)
        ):
            raise ValidationError("Ticketed calibration must come from the current capture")
        units = p["units"]
        if mapping["units"] != units:
            raise ValidationError("Distance limits must use the planar calibration units", "units")
        if float(mapping["rms_error"]) > float(p["maximum_mapping_rms"]):
            raise ValidationError("Planar calibration exceeds the maximum RMS error")

        points_payload = _require_planar_observation(self)
        points = _check_points(inputs["points"], "points")
        report = {"units": units, "observed_count": len(points)}
        if (
            failed_input(ValuePacket(inputs["points"], points_payload))
            or len(points) != 2 or np.array_equal(points[0], points[1])
            or np.any((points < 0) | (points > np.asarray(image.shape[:2][::-1]) - 1))
        ):
            message = "Spacing needs two distinct detected points inside the current image"
            return Result(
                {"report": {**report, "status": "unmeasurable"}, "distance": NO_VALUE},
                passed=False, message=message,
                fail_reasons=[{
                    "code": "planar_distance_endpoints_unavailable", "message": message,
                }],
            )

        physical = map_points(mapping, points, require_single_side=True)
        distance = math.dist(*physical)
        if not math.isfinite(distance):
            raise ValidationError("Planar point spacing is not finite")
        low, high = float(p["minimum_distance"]), float(p["maximum_distance"])
        below = distance < low and not math.isclose(distance, low, rel_tol=1e-12)
        above = bool(high) and distance > high and not math.isclose(distance, high, rel_tol=1e-12)
        passed = not (below or above)
        label = f"{distance:.4g} {units}"
        reasons = []
        if not passed:
            limit = f"below the minimum {low:.4g}" if below else f"above the maximum {high:.4g}"
            reasons.append({
                "code": "planar_distance_out_of_range",
                "message": f"Distance {label} is {limit} {units}",
            })
        drawings = {}
        if self.generate_graphics:
            color = ACCEPTED if passed else REJECTED
            drawings["Point spacing"] = GraphicDrawing((
                graphic_object("polyline", points, color),
                *(graphic_object("marker", [point], color) for point in points),
                graphic_object("label", [points.mean(axis=0)], color, text=label),
            ))
        report.update(
            status="measured", distance=distance,
            pixels=points.tolist(), physical=physical.tolist(),
        )
        return Result(
            {"report": report, "distance": distance}, passed=passed,
            message=label, fail_reasons=reasons, drawings=drawings,
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
            points_payload = _require_planar_observation(self)
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


_PLANAR_FIT_REPORT = "planar_fit_report_v1"
_PLANAR_METHODS = {"direct": 0, "ransac": cv2.RANSAC, "lmeds": cv2.LMEDS, "rho": cv2.RHO}


def _homography_support(pixels, physical):
    if not _noncollinear(pixels) or not _noncollinear(physical) or len(pixels) < 4:
        return False
    # The normalized correspondence system must constrain all eight projective freedoms.
    pixels = pixels - pixels.mean(axis=0)
    physical = physical - physical.mean(axis=0)
    source = np.column_stack((pixels / np.max(np.abs(pixels)), np.ones(len(pixels))))
    u, v = (physical / np.max(np.abs(physical))).T
    zeros = np.zeros_like(source)
    equations = np.vstack((
        np.column_stack((source, zeros, -u[:, None] * source)),
        np.column_stack((zeros, source, -v[:, None] * source)),
    ))
    return np.linalg.matrix_rank(equations) >= 8


def _fit_unavailable(code, message, report):
    report.update(passed=False, status=code)
    return Result(
        {"report": report, "mapping": NO_VALUE, "rms": NO_VALUE, "inliers": 0},
        passed=False, message=message, fail_reasons=[{"code": code, "message": message}],
    )


class FitPlanarCalibration(Block):
    type_id, title = "calibration.fit_planar", "Fit planar calibration"
    category = block_type = "Calibration"
    subtype = "Grid and coordinates"
    inputs = (Port("image"), Port("points", "data"))
    outputs = (
        Port("mapping", "data", required=False), Port("report", "data"),
        Port("rms", "number", required=False), Port("inliers", "number"),
    )
    payload_reference = "image"
    process_inputs_readonly = True
    parallel_safe = True
    defaults = {
        "physical_points": [[0, 0], [10, 0], [10, 10], [0, 10]],
        "units": "mm", "method": "ransac", "inlier_tolerance": 0.25,
        "minimum_inliers": 4, "minimum_inlier_fraction": 0.75, "maximum_rms": 0.25,
        "max_iterations": 2000, "confidence": 0.995,
    }
    parameter_choices = {"method": tuple(_PLANAR_METHODS)}
    parameter_ranges = {
        "inlier_tolerance": NumericRange(1e-9),
        "minimum_inliers": NumericRange(4, 4096, whole=True),
        "minimum_inlier_fraction": NumericRange(0, 1), "maximum_rms": NumericRange(0),
        "max_iterations": NumericRange(1, 100_000, whole=True),
        "confidence": NumericRange(0.0001, 0.9999),
    }
    parameter_help = {
        "physical_points": "Known planar [x,y] positions in the order of the observed points. "
        "Use a separate set of points to validate the fit.",
        "units": "Physical units of the reference points and fit tolerances.",
        "inlier_tolerance": "RANSAC and RHO reprojection threshold in destination physical units.",
        "minimum_inlier_fraction": "Minimum fraction of the supplied pairs used in the fit.",
        "maximum_rms": "Maximum physical RMS error over fitting inliers.",
    }
    input_help = {
        "image": "Camera image that the measured points describe.",
        "points": "Ordered [x,y] pixel positions from this image. "
        "Detect ArUco markers can emit points in a configured ID order.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        physical = _check_points(p["physical_points"], "physical_points")
        if not _homography_support(physical, physical):
            raise ValidationError(
                "Use at least four distinct physical references that constrain a homography",
                "physical_points",
            )
        if int(p["minimum_inliers"]) > len(physical):
            raise ValidationError("Minimum inliers exceeds the reference count", "minimum_inliers")
        units = p["units"]
        if not isinstance(units, str) or not units.strip() or len(units) > 24:
            raise ValidationError("Enter the calibration's physical units", "units")

    def process(self, inputs):
        p = self.parameters
        image = inputs["image"]
        observed = _check_points(inputs["points"], "points")
        physical = _check_points(p["physical_points"], "physical_points")
        points_payload = _require_planar_observation(self)
        report = {
            "model": _PLANAR_FIT_REPORT, "units": p["units"], "method": p["method"],
            "expected_count": len(physical), "observed_count": len(observed),
        }
        if (
            failed_input(ValuePacket(inputs["points"], points_payload))
            or len(observed) != len(physical)
        ):
            return _fit_unavailable(
                "planar_fit_observation_unavailable",
                f"Expected {len(physical)} valid ordered fiducials, observed {len(observed)}",
                report,
            )
        image_size = list(image.shape[:2][::-1])
        if (
            np.any((observed < 0) | (observed > np.asarray(image_size) - 1))
            or not _homography_support(observed, physical)
        ):
            return _fit_unavailable(
                "planar_fit_geometry_invalid",
                "Fiducial geometry is degenerate or extends outside the image", report,
            )
        matrix, mask = cv2.findHomography(
            observed, physical, _PLANAR_METHODS[p["method"]], float(p["inlier_tolerance"]),
            maxIters=int(p["max_iterations"]), confidence=float(p["confidence"]),
        )
        inliers = mask.ravel().astype(bool) if mask is not None else np.zeros(len(observed), bool)
        if (
            matrix is None or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3
            or not _homography_support(observed[inliers], physical[inliers])
        ):
            return _fit_unavailable(
                "planar_fit_geometry_invalid",
                "OpenCV could not fit a supported homography", report,
            )
        origin = np.linalg.solve(matrix, np.array([0.0, 0.0, 1.0]))
        denominators = np.column_stack((observed[inliers], np.ones(inliers.sum()))) @ matrix[2]
        if abs(origin[2]) < 1e-12 or not (
            np.all(denominators > 1e-12) or np.all(denominators < -1e-12)
        ):
            return _fit_unavailable(
                "planar_fit_horizon",
                "Origin or fitting support crosses the projective horizon", report,
            )
        mapping = {
            "model": "planar_homography_v1", "pixel_to_physical": matrix.tolist(),
            "units": p["units"], "image_size": image_size,
            "origin_pixel": (origin[:2] / origin[2]).tolist(), "plane_only": True,
        }
        deltas = map_points(mapping, observed[inliers]) - physical[inliers]
        rms = float(np.sqrt(np.mean(np.sum(deltas**2, axis=1))))
        if not np.isfinite(rms):
            raise ValidationError("Planar fit produced a nonfinite residual")
        mapping["rms_error"] = rms
        count = int(inliers.sum())
        fraction = count / len(inliers)
        reasons = []
        if count < int(p["minimum_inliers"]) or fraction < float(p["minimum_inlier_fraction"]):
            reasons.append({
                "code": "planar_fit_inliers_below_minimum",
                "message": f"Only {count} of {len(inliers)} reference pairs support the mapping",
            })
        if rms > float(p["maximum_rms"]):
            reasons.append({
                "code": "planar_fit_rms_exceeds_limit",
                "message": f"Fit RMS {rms:.4g} {p['units']} exceeds the configured limit",
            })
        passed = not reasons
        report.update(passed=passed, status="measured", inlier_mask=inliers.tolist(),
                      inlier_count=count, inlier_fraction=fraction, rms=rms)
        drawings = {}
        if self.generate_graphics:
            drawings["Planar fit inliers"] = GraphicDrawing(tuple(
                graphic_object(
                    "marker", [observed[index]], ACCEPTED if inliers[index] else REJECTED
                )
                for index in np.argsort(inliers, kind="stable")[:64]
            ))
        return Result(
            {"mapping": mapping if passed else NO_VALUE, "report": report,
             "rms": rms, "inliers": count},
            passed=passed, fail_reasons=reasons, drawings=drawings,
            message=f"{count}/{len(inliers)} fit inliers · RMS {rms:.4g} {p['units']}",
        )
