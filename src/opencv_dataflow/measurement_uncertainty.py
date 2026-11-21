"""First-order uncertainty of a two-point distance on a calibrated plane."""

from statistics import NormalDist

import cv2
import numpy as np

from .calibration_revisions import REVISION, check_capture
from .core import Port, Result
from .geometry3d_common import Tool3D, require_pinhole, rigid
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, ValuePacket, require_same_frame
from .result_graphics import ACCEPTED, REJECTED

_PIXELS = ("u0", "v0", "u1", "v1")
_CAMERA = (
    "rx",
    "ry",
    "rz",
    "tx",
    "ty",
    "tz",
    "fx",
    "fy",
    "cx",
    "cy",
    "k1",
    "k2",
    "p1",
    "p2",
    "k3",
    "k4",
    "k5",
    "k6",
    "s1",
    "s2",
    "s3",
    "s4",
    "tau_x",
    "tau_y",
)
_ALL_TERMS = _PIXELS + _CAMERA
_MODEL = "planar_distance_uncertainty_v1"


def _pixel_points(value):
    try:
        xy = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("Use two [x, y] pixel pairs", "image_points") from exc
    if xy.shape != (2, 2) or xy.dtype.kind not in "uif" or not np.isfinite(xy).all():
        raise ValidationError("Use two finite [x, y] pixel pairs", "image_points")
    return np.ascontiguousarray(xy, dtype=np.float64)


def _covariance_root(value, allowed):
    if not isinstance(value, dict):
        raise ValidationError(
            "Covariance must be an object with terms and matrix", "input_covariance"
        )
    terms = value.get("terms")
    if (
        not isinstance(terms, list)
        or not 1 <= len(terms) <= len(allowed)
        or any(not isinstance(name, str) or name not in allowed for name in terms)
        or len(set(terms)) != len(terms)
    ):
        raise ValidationError(
            "Covariance terms must be distinct and present in the camera model", "input_covariance"
        )
    try:
        covariance = np.asarray(value.get("matrix"))
        if covariance.dtype.kind not in "uif":
            raise TypeError
        covariance = covariance.astype(np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("Covariance requires a numeric matrix", "input_covariance") from exc
    if covariance.shape != (len(terms), len(terms)) or not np.isfinite(covariance).all():
        raise ValidationError(
            "Covariance matrix must be finite and match the term count", "input_covariance"
        )
    variance = covariance.diagonal()
    active = variance > 0
    if np.any(variance < 0) or np.any(covariance[~active]) or np.any(covariance[:, ~active]):
        raise ValidationError(
            "Zero-variance terms require zero covariance and variances cannot be negative",
            "input_covariance",
        )
    scale = np.sqrt(variance[active])
    selected = covariance[np.ix_(active, active)]
    # Check correlation in dimensionless coordinates to avoid a tolerance tied to length units.
    correlation = selected / scale[:, None] / scale
    if not np.isfinite(correlation).all() or not np.allclose(
        correlation, correlation.T, rtol=0, atol=1e-10
    ):
        raise ValidationError(
            "Covariance must be symmetric in normalized units", "input_covariance"
        )
    values, vectors = np.linalg.eigh((correlation + correlation.T) / 2)
    if np.any(values < -1e-10):
        raise ValidationError("Covariance must be positive semidefinite", "input_covariance")
    root = np.zeros((len(terms), len(values)))
    root[active] = scale[:, None] * vectors * np.sqrt(np.maximum(0, values))
    return terms, root


def _measurement(xy, k, d, transform, minimum_cosine):
    r, t = transform[:3, :3].copy(), transform[:3, 3].copy()
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 60, 1e-13)
    uv = cv2.undistortPoints(xy[:, None], k, d, criteria=criteria).reshape(2, 2)
    rays = np.column_stack((uv, np.ones(2)))
    normal = r[:, 2]
    incidence = rays @ normal
    cosine = np.abs(incidence) / np.linalg.norm(rays, axis=1)
    if not np.isfinite(cosine).all() or (cosine < minimum_cosine).any():
        return None, "measurement_horizon", "Image rays are too close to the plane horizon"
    depth = (normal @ t) / incidence
    if not np.isfinite(depth).all() or (depth <= 0).any():
        return None, "measurement_plane_behind", "Image points lack a forward plane intersection"
    plane = (depth[:, None] * rays - t) @ r[:, :2]
    world = np.column_stack((plane, np.zeros(2)))
    rvec = cv2.Rodrigues(r)[0]
    roundtrip, native = cv2.projectPoints(world, rvec, t, k, d)
    error = np.linalg.norm(roundtrip.reshape(2, 2) - xy, axis=1)
    if not np.isfinite(error).all() or (error > 1e-4).any():
        return (
            None,
            "measurement_roundtrip",
            "The camera model did not invert at the measured pixels",
        )
    pose_jacobian = native[:, : 10 + len(d)].reshape(2, 2, -1)
    if not np.isfinite(pose_jacobian).all():
        return None, "measurement_singular", "Camera projection derivatives are not finite"
    pixel_per_plane = pose_jacobian[:, :, 3:6] @ r[:, :2]
    singular = np.linalg.svd(pixel_per_plane, compute_uv=False)
    if (singular[:, -1] <= singular[:, 0] * 1e-8).any():
        return None, "measurement_singular", "The local plane-to-image differential is singular"
    # u = project(q, model). For fixed nominal pixels, dq/dmodel = -(du/dq)^-1 du/dmodel.
    local = np.linalg.solve(pixel_per_plane, np.broadcast_to(np.eye(2), (2, 2, 2)))
    gradient = np.zeros((4, 4 + pose_jacobian.shape[2]))
    gradient[:2, :2], gradient[2:, 2:4] = local
    gradient[:, 4:] = -(local @ pose_jacobian).reshape(4, -1)
    return (plane, gradient, cosine, error), None, None


class MeasurePlanarUncertainty(Tool3D):
    type_id, title = "calibration.measure_uncertainty", "Measure planar distance uncertainty"
    category = block_type = "Calibration"
    subtype = "Lens distortion"
    inputs = (
        Port("camera", "data"),
        Port("plane_to_camera", "data", visual_type="pose"),
        Port("image_points", "data", required=False, required_when_connected=True),
        Port("covariance", "data", required=False, required_when_connected=True),
        Port("image", required=False, required_when_connected=True),
    )
    outputs = (
        Port("distance", "number", required=False),
        Port("standard_uncertainty", "number", required=False),
        Port("expanded_uncertainty", "number", required=False),
        Port("report", "data"),
        Port("image", required=False),
    )
    defaults = {
        "image_points": [[240.0, 240.0], [400.0, 240.0]],
        "pixel_stddev_px": 0.0,
        "input_covariance": {},
        "coverage_probability": 0.95,
        "minimum_ray_cosine": 0.05,
        "maximum_expanded_uncertainty": 0.0,
    }
    parameter_ranges = {
        "pixel_stddev_px": NumericRange(0, 1000),
        "coverage_probability": NumericRange(0.5, 0.9999),
        "minimum_ray_cosine": NumericRange(1e-6, 1),
        "maximum_expanded_uncertainty": NumericRange(0),
    }
    parameter_help = {
        "image_points": "Default XY endpoints in the distorted image pixel grid.",
        "pixel_stddev_px": "Independent deviation for pixels omitted from the covariance.",
        "input_covariance": "Named covariance matrix in pixels, radians and camera units.",
        "coverage_probability": "Central probability for the first-order normal interval.",
        "minimum_ray_cosine": "Minimum absolute cosine between each view ray and the plane normal.",
        "maximum_expanded_uncertainty": "Largest interval half-width in camera units. Zero skips.",
    }
    input_help = {
        "camera": "Pinhole camera on the pixel grid of both points.",
        "plane_to_camera": "Rigid pose from the measurement plane (local Z = 0) to the camera.",
        "image_points": "Two [x,y] pixel positions in the calibrated image.",
        "covariance": "Named terms and a joint matrix. Use the port or the parameter field.",
        "image": "Optional calibrated image to display the measurement line.",
    }
    payload_reference = "image"
    composes_result_image = True
    preview_safe = parallel_safe = process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        _pixel_points(p["image_points"])
        if p["input_covariance"] != {}:
            _covariance_root(p["input_covariance"], _ALL_TERMS)

    def process(self, inputs):
        p = self.parameters
        camera, pose = inputs["camera"], inputs["plane_to_camera"]
        k, d, size = require_pinhole(camera)
        units, frame = camera.get("units", "m"), camera.get("frame", "camera")
        if (
            not isinstance(pose, dict)
            or pose.get("model") != "pose_3d_v1"
            or pose.get("target_frame") != frame
            or pose.get("units") != units
            or not isinstance(pose.get("source_frame"), str)
            or not pose["source_frame"].strip()
            or pose["source_frame"] == frame
        ):
            raise ValidationError("Use a plane-to-camera pose with matching frame and length units")
        transform = rigid(pose)
        xy = _pixel_points(inputs.get("image_points", p["image_points"]))
        if np.any(xy < 0) or np.any(xy >= size):
            raise ValidationError(
                "Both points must lie inside the calibrated image", "image_points"
            )
        image = inputs.get("image")
        point_payload = self.input_payloads.get("image_points", {})
        source_payload = self.input_payloads["image"] if image is not None else point_payload
        spatial = point_payload.get("spatial")
        if spatial and spatial["image_size"] != list(size):
            raise ValidationError("Point coordinate grid does not match the camera")
        if image is not None:
            if tuple(image.shape[:2][::-1]) != size:
                raise ValidationError("Image size does not match the camera")
            require_same_frame(
                (image, ValuePacket(xy, point_payload)),
                message="Image and endpoints must use the same calibrated coordinate frame",
            )
            image_ticket, point_ticket = source_payload.get("ticket"), point_payload.get("ticket")
            if (
                image_ticket is not None
                and point_ticket is not None
                and image_ticket != point_ticket
            ):
                raise ValidationError("Image and endpoints must belong to the same capture")
        check_capture(camera.get(REVISION), source_payload, size)
        allowed = _PIXELS + _CAMERA[: 10 + len(d)]
        if "covariance" in inputs and p["input_covariance"]:
            raise ValidationError(
                "Use connected covariance or the configured matrix, not both", "input_covariance"
            )
        raw = inputs.get("covariance", p["input_covariance"])
        terms, root = (
            _covariance_root(raw, allowed)
            if raw or "covariance" in inputs
            else ([], np.zeros((0, 0)))
        )
        pixel_sigma = float(p["pixel_stddev_px"])
        independent = [name for name in _PIXELS if name not in terms and pixel_sigma]
        if not np.any(root) and not independent:
            raise ValidationError(
                "Declare nonzero pixel or model uncertainty", "pixel_stddev_px", "input_covariance"
            )

        estimate, code, message = _measurement(xy, k, d, transform, float(p["minimum_ray_cosine"]))
        probability = float(p["coverage_probability"])
        outputs = {
            "distance": NO_VALUE,
            "standard_uncertainty": NO_VALUE,
            "expanded_uncertainty": NO_VALUE,
            "image": image if image is not None else NO_VALUE,
        }
        report = {
            "model": _MODEL,
            "method": "first_order_gaussian",
            "coverage_probability": probability,
            "units": units,
            "plane_frame": pose["source_frame"],
            "camera_frame": frame,
            "image_points": xy.tolist(),
            "image_size": list(size),
            "joint_covariance_terms": terms,
            "independent_pixel_terms": independent,
            "pixel_stddev_px": pixel_sigma,
        }
        if estimate is not None:
            plane, gradient, cosine, error = estimate
            joint = gradient[:, [allowed.index(name) for name in terms]] @ root
            pixels = gradient[:, [allowed.index(name) for name in independent]] * pixel_sigma
            factor = np.column_stack((joint, pixels))
            endpoint_covariance = factor @ factor.T
            if not np.isfinite(endpoint_covariance).all():
                raise ValidationError("Propagated covariance is outside the finite numeric range")
            difference_factor = factor[2:] - factor[:2]
            distance = float(np.linalg.norm(plane[1] - plane[0]))
            coverage_factor = NormalDist().inv_cdf((1 + probability) / 2)
            # The norm is nonsmooth where the endpoints coincide. Refuse a local interval
            # when their plausible relative variation already spans the nominal separation.
            radius = float(np.linalg.svd(difference_factor, compute_uv=False)[0] * coverage_factor)
            report.update(
                plane_points_xy=plane.tolist(),
                nominal_distance=distance,
                minimum_observed_ray_cosine=float(cosine.min()),
                max_projection_roundtrip_px=float(error.max()),
                coverage_factor=coverage_factor,
                relative_endpoint_radius=radius,
            )
            if distance <= max(radius, 1e-12):
                code, message = (
                    "measurement_linearization",
                    "Endpoint separation is unresolved at this uncertainty",
                )
            else:
                direction = (plane[1] - plane[0]) / distance
                sigma = float(np.linalg.norm(direction @ difference_factor))
                width = coverage_factor * sigma
                report.update(
                    standard_uncertainty=sigma,
                    expanded_uncertainty=width,
                    interval=[distance - width, distance + width],
                    endpoint_covariance_xy=endpoint_covariance.tolist(),
                )
                outputs.update(
                    distance=distance, standard_uncertainty=sigma, expanded_uncertainty=width
                )
                limit = float(p["maximum_expanded_uncertainty"])
                if limit and width > limit:
                    code, message = (
                        "measurement_uncertainty_limit",
                        f"Uncertainty {width:.4g} {units} exceeds limit",
                    )
                else:
                    message = f"{distance:.4g} ± {width:.3g} {units} at {probability:.1%} coverage"
        report.update(status=code or "measured", passed=code is None)
        drawings = {}
        if image is not None and self.generate_graphics:
            color = ACCEPTED if code is None else REJECTED
            drawings["Measured segment"] = GraphicDrawing(
                (
                    graphic_object("polyline", xy, color),
                    *(graphic_object("marker", [point], color) for point in xy),
                )
            )
        outputs["report"] = report
        payloads = (
            {name: point_payload for name, value in outputs.items() if value is not NO_VALUE}
            if image is None
            else {}
        )
        return Result(
            outputs,
            payloads=payloads,
            passed=code is None,
            fail_reasons=[{"code": code, "message": message}] if code else [],
            message=message,
            drawings=drawings,
        )
