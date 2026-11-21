"""Inspect a finite pinhole projection in explicit pixel and world units."""

import math
from itertools import islice

import cv2
import numpy as np

from .core import Port, Result
from .geometry3d_common import Tool3D, camera_record, matrix, points, pose_record, rigid
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, ValuePacket, require_same_frame

_MAX_POINTS = 4096
_MAX_RECOMPOSITION_ERROR = 1e-7
_MAX_COMPATIBLE_SKEW = 1e-9


def _projection_matrix(value):
    try:
        return matrix(value, (3, 4), "Projection")
    except (TypeError, ValueError, OverflowError) as error:
        raise ValidationError("Enter a finite 3×4 projection matrix", "projection") from error


def _factor_projection(projection, minimum_ratio, report):
    scale = float(np.max(np.abs(projection[:, :3])))
    if not scale:
        return None, "projection_singular"
    with np.errstate(over="ignore"):
        canonical = projection / scale
    if not np.isfinite(canonical).all():
        return None, "projection_dynamic_range"
    left = canonical[:, :3]
    singular = np.linalg.svd(left, compute_uv=False)
    ratio = float(singular[-1] / singular[0])
    report["left_reciprocal_condition"] = ratio
    if ratio < minimum_ratio:
        return None, "projection_singular"
    sign = float(np.linalg.slogdet(left)[0])
    if not sign:
        return None, "projection_singular"
    # Select the projectively equivalent matrix with a positive determinant,
    # giving positive focal lengths and a proper world-to-camera rotation.
    canonical *= sign
    k, rotation, center_h, *_ = cv2.decomposeProjectionMatrix(canonical)
    w = float(center_h[3, 0])
    if (
        not np.isfinite(k).all()
        or not np.isfinite(center_h).all()
        or np.any(k.diagonal() <= 0)
        or w == 0
    ):
        return None, "projection_decomposition"
    with np.errstate(over="ignore", invalid="ignore"):
        normalized_k = k / k[2, 2]
        center = center_h[:3, 0] / w
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = -rotation @ center
        rebuilt = k @ transform[:3]
    if not np.isfinite(normalized_k).all() or not np.isfinite(rebuilt).all():
        return None, "projection_dynamic_range"
    try:
        rigid(transform)
    except ValueError:
        return None, "projection_decomposition"
    # Normalize again for the residual so very large world offsets do not overflow the norm.
    full_scale = np.abs(canonical).max()
    reference = canonical / full_scale
    error = float(np.linalg.norm(rebuilt / full_scale - reference) / np.linalg.norm(reference))
    report["relative_recomposition_error"] = error if math.isfinite(error) else None
    if not math.isfinite(error) or error > _MAX_RECOMPOSITION_ERROR:
        return None, "projection_recomposition"
    return (normalized_k, center, transform), None


def _describe_intrinsics(k, size, sensor):
    fx, fy = (float(v) for v in k.diagonal()[:2])
    cx, cy = (float(v) for v in k[:2, 2])
    fov_x, fov_y, focal_mm, point_mm, aspect = cv2.calibrationMatrixValues(k, size, *sensor)
    spec = {
        "focal_lengths_px": [fx, fy],
        "principal_point_px": [cx, cy],
        "skew_px": float(k[0, 1]),
        "aspect_ratio_fy_over_fx": aspect,
        "fov_deg": [fov_x, fov_y],
        "fov_convention": "OpenCV axial spans across the declared pixel width and height",
        "sensor_size_mm": None,
        "pixel_pitch_mm": None,
        "focal_lengths_mm": None,
        "principal_point_mm": None,
    }
    if all(sensor):
        pitch = [extent / count for extent, count in zip(sensor, size, strict=True)]
        spec.update(
            sensor_size_mm=list(sensor),
            pixel_pitch_mm=pitch,
            focal_lengths_mm=[focal_mm, fy * pitch[1]],
            principal_point_mm=list(point_mm),
        )
    return spec


def _read_check(inputs, contexts, units):
    required = ("object_points", "image_points")
    connected = [name for name in required if name in inputs]
    if not connected:
        return None
    if len(connected) != len(required):
        raise ValidationError("Connect both ordered object and image points", *required)
    source, observed = (
        points(inputs[name], dimensions=dim) for name, dim in zip(required, (3, 2), strict=True)
    )
    if len(source) != len(observed) or len(source) > _MAX_POINTS:
        raise ValidationError(f"Connect matching sets of 1..{_MAX_POINTS} points", *required)
    declared = contexts["object_points"].get("object_point_units")
    if declared is not None and declared != units:
        raise ValidationError(
            "Object point units differ from the projection world units", "object_points"
        )
    if "image" in inputs:
        require_same_frame(
            (inputs["image"], ValuePacket(inputs["image_points"], contexts["image_points"])),
            message="Observation points must use the attached image frame",
        )
    return source, observed


def _check_projection(xyz, observed, k, transform, limit):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        camera = xyz @ transform[:3, :3].T + transform[:3, 3]
        homogeneous = camera @ k.T
        predicted = homogeneous[:, :2] / homogeneous[:, 2:3]
        residuals = np.hypot(*(predicted - observed).T)
    visible = (camera[:, 2] > 0) & np.isfinite(predicted).all(axis=1) & np.isfinite(residuals)
    usable = residuals[visible]
    maximum = float(usable.max()) if usable.size else None
    # Scale first to retain finite RMS for extreme but finite observations.
    rms = float(maximum * np.sqrt(np.mean((usable / maximum) ** 2))) if maximum else maximum
    return {
        "point_count": len(xyz),
        "valid_depth_count": len(usable),
        "invalid_depth_or_projection_count": int((~visible).sum()),
        "rms_px": rms,
        "max_error_px": maximum,
        "maximum_rms_px": limit,
        "reprojected_points_px": [
            point.tolist() if ok else None for point, ok in zip(predicted, visible, strict=True)
        ],
        "errors_px": [
            float(value) if ok else None for value, ok in zip(residuals, visible, strict=True)
        ],
    }


def _overlay(principal, size, observed, check, maximum, limit):
    objects = []

    def in_grid(point):
        return point is not None and all(0 <= v < n for v, n in zip(point, size, strict=True))

    if in_grid(principal):
        objects.append(graphic_object("marker", [principal], (245, 190, 70)))
    if check is not None:
        rows = zip(check["reprojected_points_px"], observed, check["errors_px"], strict=True)
        for found, target, error in islice(rows, maximum):
            color = (80, 225, 105) if error is not None and error <= limit else (50, 70, 245)
            if in_grid(target):
                objects.append(graphic_object("marker", [target], (60, 205, 240)))
            if in_grid(found):
                objects.append(graphic_object("marker", [found], color))
                if in_grid(target) and error > 0.5:
                    objects.append(graphic_object("polyline", [target, found], color))
    return tuple(objects)


class InspectProjectionMatrix(Tool3D):
    type_id = "3d.inspect_projection_matrix"
    title, subtype = "Inspect camera projection matrix", "Camera models"
    description = "Recover intrinsics, camera pose and scale-aware diagnostics from a 3×4 matrix."
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = (
        Port("projection", "data", required=False, required_when_connected=True),
        Port("object_points", "data", required=False, required_when_connected=True),
        Port("image_points", "data", required=False, required_when_connected=True),
        Port("image", required=False, required_when_connected=True),
    )
    outputs = (
        Port("camera", "data", required=False),
        Port("world_to_camera", "data", required=False, visual_type="pose"),
        Port("camera_to_world", "data", required=False, visual_type="pose"),
        Port("image", required=False),
        Port("report", "data"),
    )
    defaults = {
        "projection": [],
        "width": 640,
        "height": 480,
        "world_frame": "world",
        "camera_frame": "camera",
        "units": "unitless",
        "sensor_width_mm": 0.0,
        "sensor_height_mm": 0.0,
        "minimum_reciprocal_condition": 1e-9,
        "maximum_reprojection_rms_px": 1.0,
        "maximum_drawn_points": 128,
    }
    parameter_choices = {"units": ("unitless", "m", "mm")}
    parameter_ranges = {
        "width": NumericRange(1, 16384, whole=True),
        "height": NumericRange(1, 16384, whole=True),
        "sensor_width_mm": NumericRange(0, 1e6),
        "sensor_height_mm": NumericRange(0, 1e6),
        "minimum_reciprocal_condition": NumericRange(1e-14, 1),
        "maximum_reprojection_rms_px": NumericRange(0, 1e6),
        "maximum_drawn_points": NumericRange(0, 512, whole=True),
    }
    parameter_help = {
        "projection": "Inline 3×4 matrix. A connected projection takes precedence.",
        "width": "Width of the ideal image grid in pixels.",
        "height": "Height of the ideal image grid in pixels.",
        "world_frame": "Named coordinate frame of the matrix's 3D inputs.",
        "camera_frame": "Name for the recovered optical coordinate frame.",
        "units": "Length units of the input world coordinates. A matrix alone cannot assign them.",
        "sensor_width_mm": "Active sensor width in mm. Specify both sensor dimensions or zero.",
        "sensor_height_mm": "Active sensor height in mm. Specify both sensor dimensions or zero.",
        "minimum_reciprocal_condition": "Minimum singular value ratio for the left block.",
        "maximum_reprojection_rms_px": "RMS limit on independent correspondences, in pixels.",
        "maximum_drawn_points": "Limit matched-point drawings when a reference image is connected.",
    }
    input_help = {
        "projection": "A finite 3×4 world-to-image pinhole projection, up to nonzero scale.",
        "object_points": "Optional ordered world XYZ check points in the declared units.",
        "image_points": "Corresponding ideal image pixels on the declared grid.",
        "image": "Optional ideal image on which to draw the principal point and reprojections.",
    }
    output_help = {
        "camera": "Ideal pinhole model for zero-skew inputs. Distortion is not encoded in P.",
        "world_to_camera": "Rigid transform from the declared world to the optical frame.",
        "camera_to_world": "Inverse rigid transform. Its translation is the world camera center.",
        "image": "The supplied ideal image with independent result graphics.",
        "report": "Intrinsic matrix, center, FOV, sensor measures and optional point residuals.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["projection"] != []:
            _projection_matrix(p["projection"])
        names = p["world_frame"], p["camera_frame"]
        if (
            any(not isinstance(name, str) or not name.strip() for name in names)
            or names[0] == names[1]
        ):
            raise ValidationError(
                "Choose two distinct nonempty frame names", "world_frame", "camera_frame"
            )
        if bool(p["sensor_width_mm"]) != bool(p["sensor_height_mm"]):
            raise ValidationError(
                "Specify both active sensor dimensions or leave both at zero",
                "sensor_width_mm",
                "sensor_height_mm",
            )

    def process(self, inputs):
        p = self.parameters
        projection = _projection_matrix(inputs.get("projection", p["projection"]))
        image = inputs.get("image", NO_VALUE)
        size = int(p["width"]), int(p["height"])
        if image is not NO_VALUE and image.shape[1::-1] != size:
            raise ValidationError(
                "Image dimensions must match the declared projection grid",
                "image",
                "width",
                "height",
            )
        units, world, optical = p["units"], p["world_frame"], p["camera_frame"]
        checked = _read_check(inputs, self.input_payloads, units)
        report = {
            "model": "projection_matrix_inspection_v1",
            "world_frame": world,
            "camera_frame": optical,
            "world_units": units,
            "image_size": list(size),
            "status": "failed",
        }
        outputs = dict.fromkeys((port.name for port in self.outputs), NO_VALUE)
        outputs.update(image=image, report=report)
        payloads = {name: {} for name in ("camera", "world_to_camera", "camera_to_world", "report")}
        if image is not NO_VALUE:
            payloads["image"] = self.input_payloads["image"]
        factor, reason = _factor_projection(
            projection, float(p["minimum_reciprocal_condition"]), report
        )
        if reason:
            return Result(
                outputs,
                payloads=payloads,
                passed=False,
                fail_reasons=[reason],
                message="No stable finite-camera decomposition",
            )
        k, center, transform = factor
        aperture = (float(p["sensor_width_mm"]), float(p["sensor_height_mm"]))
        intrinsics = _describe_intrinsics(k, size, aperture)
        compatible = abs(intrinsics["skew_px"]) <= _MAX_COMPATIBLE_SKEW
        report.update(
            **intrinsics,
            camera_matrix=k.tolist(),
            camera_center_world=center.tolist(),
            world_to_camera_rotation=transform[:3, :3].tolist(),
            camera_model_compatible=compatible,
        )
        inverse = np.eye(4)
        inverse[:3, :3] = transform[:3, :3].T
        inverse[:3, 3] = center
        outputs.update(
            camera=camera_record(k, [], size, units, optical) if compatible else NO_VALUE,
            world_to_camera=pose_record(transform, world, optical, units),
            camera_to_world=pose_record(inverse, optical, world, units),
        )

        reasons, point_report, observed = [], None, ()
        limit = float(p["maximum_reprojection_rms_px"])
        if checked is not None:
            xyz, observed = checked
            point_report = _check_projection(xyz, observed, k, transform, limit)
            report["reprojection"] = point_report
            if point_report["invalid_depth_or_projection_count"]:
                reasons.append("projection_check_invalid_depth")
            rms = point_report["rms_px"]
            if rms is not None and rms > limit:
                reasons.append("projection_check_rms")
        report["status"] = "failed" if reasons else "ok"
        drawings = {}
        if self.generate_graphics and image is not NO_VALUE:
            objects = _overlay(
                intrinsics["principal_point_px"],
                size,
                observed,
                point_report,
                int(p["maximum_drawn_points"]),
                limit,
            )
            if objects:
                drawings["Projection checks"] = GraphicDrawing(objects, "image", from_output=True)

        horizontal, vertical = intrinsics["fov_deg"]
        message = f"Field of view {horizontal:.2f}° × {vertical:.2f}°"
        if point_report is not None and point_report["rms_px"] is not None:
            message += f", check RMS {point_report['rms_px']:.4g} px"
        return Result(
            outputs,
            payloads=payloads,
            drawings=drawings,
            passed=not reasons,
            fail_reasons=reasons,
            message=message,
        )
