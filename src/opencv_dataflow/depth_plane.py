"""Separate a metric support plane and foreground on an undistorted depth grid."""

import cv2
import numpy as np

from .core import Port, Result
from .geometry3d_common import Tool3D
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .ptcloud3d import depth_intrinsics
from .selection import selection_mask
from .stereo3d import metric_depth, rgbd_function


def fit_support_plane(points, tolerance, trials, rng):
    """Select the largest three-point consensus, then refit its surface by SVD."""
    if len(points) < 3:
        return None
    best, best_score = None, (2, -np.inf)
    for _ in range(trials):
        a, b, c = points[rng.choice(len(points), size=3, replace=False)]
        ab, ac = b - a, c - a
        normal = np.cross(ab, ac)
        length = np.linalg.norm(normal)
        if length <= 1e-8 * np.linalg.norm(ab) * np.linalg.norm(ac):
            continue
        normal /= length
        errors = np.abs(np.einsum("ij,j->i", points, normal) - a @ normal)
        inlier = errors <= tolerance
        residual = errors[inlier]
        score = (len(residual), -float(residual @ residual))
        if score > best_score:
            best, best_score = inlier, score
            if score == (len(points), 0.0):
                break
    if best is None:
        return None

    for _ in range(3):
        support = points[best]
        center = support.mean(axis=0)
        _, singular, directions = np.linalg.svd(support - center, full_matrices=False)
        if singular[1] <= 1e-8 * singular[0]:
            return None
        normal = directions[-1]
        offset = float(-center @ normal)
        updated = np.abs(np.einsum("ij,j->i", points, normal) + offset) <= tolerance
        if np.array_equal(best, updated):
            break
        if np.count_nonzero(updated) < 3:
            return None
        best = updated

    # The camera origin is on the positive (foreground) side of a usable plane.
    if abs(offset) <= np.finfo(float).eps * np.linalg.norm(center):
        return None
    if offset < 0:
        return -normal, -offset
    return normal, offset


def distance_grid(depth, camera_matrix, normal, offset):
    """Evaluate n dot XYZ + d with pinhole rays, without allocating a dense XYZ image."""
    height, width = depth.shape
    k = camera_matrix
    x_term = (np.arange(width) - k[0, 2]) * (normal[0] / k[0, 0]) + normal[2]
    y_term = (np.arange(height) - k[1, 2]) * (normal[1] / k[1, 1])
    signed = np.empty(depth.shape, np.float32)
    for row in range(0, height, 128):
        band = slice(row, row + 128)
        factor = x_term + y_term[band, None]
        signed[band] = depth[band] * factor + offset
    return signed


class SeparateDepthPlane(Tool3D):
    type_id, title, subtype = (
        "3d.separate_depth_plane",
        "Separate support plane and foreground",
        "RGB-D",
    )
    parallel_safe = True
    process_inputs_readonly = True
    payload_reference = "depth"
    inputs = (
        Port("depth", "depth"),
        Port("camera", "data"),
        Port("mask", required=False),
        Port("region", "data", required=False),
    )
    outputs = (
        Port("plane", "data", required=False),
        Port("signed_distance_m", required=False),
        Port("plane_mask", required=False, visual_type="region"),
        Port("foreground_mask", required=False, visual_type="region"),
        Port("foreground_depth", "depth", required=False),
        Port("valid_mask", visual_type="region"),
        Port("report", "data"),
    )
    defaults = {
        "inlier_tolerance_m": 0.004,
        "foreground_min_m": 0.015,
        "minimum_inliers": 30,
        "minimum_inlier_fraction": 0.55,
        "maximum_fit_samples": 4096,
        "maximum_trials": 192,
        "random_seed": 0,
        "input_units": "m",
        "units_per_sample": 0.001,
    }
    parameter_choices = {"input_units": ("m", "mm")}
    parameter_ranges = {
        "inlier_tolerance_m": NumericRange(1e-6, 1000),
        "foreground_min_m": NumericRange(1e-6, 1000),
        "minimum_inliers": NumericRange(3, 100_000_000, whole=True),
        "minimum_inlier_fraction": NumericRange(0, 1),
        "maximum_fit_samples": NumericRange(3, 16_384, whole=True),
        "maximum_trials": NumericRange(1, 1024, whole=True),
        "random_seed": NumericRange(0, 2**32 - 1, whole=True),
        "units_per_sample": NumericRange(1e-9, 1000),
    }
    parameter_help = {
        "inlier_tolerance_m": "Maximum orthogonal distance to count as plane support.",
        "foreground_min_m": "Minimum signed distance toward the camera to mark foreground.",
        "minimum_inliers": "Minimum accepted plane pixels in the fitting selection.",
        "minimum_inlier_fraction": "Minimum plane fraction among valid selected depth pixels.",
        "maximum_fit_samples": "Maximum number of randomly sampled points for fitting.",
        "maximum_trials": "Maximum number of three-point hypotheses to evaluate.",
        "random_seed": "Fixed seed for reproducible sampling and model hypotheses.",
        "input_units": "Units for a depth input without a declared physical encoding.",
        "units_per_sample": "Physical units per raw uint16 count.",
    }
    input_help = {
        "depth": "Physical camera Z depth. Zero, negative, and nonfinite samples are ignored.",
        "camera": "Undistorted pinhole intrinsics for this depth grid.",
        "mask": "Limit points eligible for fitting. Classification still covers the entire frame.",
        "region": "Limit points eligible for fitting, intersecting with the mask when supplied.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if float(p["foreground_min_m"]) < float(p["inlier_tolerance_m"]):
            raise ValidationError(
                "Foreground distance must be at least the plane tolerance",
                "foreground_min_m",
                "inlier_tolerance_m",
            )

    def process(self, inputs):
        depth, model, p = inputs["depth"], inputs["camera"], self.parameters
        k, _ = depth_intrinsics(model, depth)
        values = metric_depth(depth, p["units_per_sample"], p["input_units"])
        valid = np.isfinite(values)
        selection = selection_mask(inputs, depth)
        eligible = valid if selection is None else valid & (selection != 0)
        eligible_count = int(np.count_nonzero(eligible))
        rng = np.random.default_rng(int(p["random_seed"]))
        tolerance = float(p["inlier_tolerance_m"])
        report = {
            "valid_pixels": int(np.count_nonzero(valid)),
            "fitting_pixels": eligible_count,
            "fit_samples": 0,
            "fit_inliers": 0,
            "fit_inlier_fraction": 0.0,
            "units": "m",
        }
        outputs = {
            "plane": NO_VALUE,
            "signed_distance_m": NO_VALUE,
            "plane_mask": NO_VALUE,
            "foreground_mask": NO_VALUE,
            "foreground_depth": NO_VALUE,
            "valid_mask": valid.astype(np.uint8) * 255,
            "report": report,
        }
        reasons = {
            "insufficient_fit_pixels": "Too few valid depth pixels in the fitting selection",
            "degenerate_plane": "No nondegenerate plane found in selected depth",
            "insufficient_plane_support": "Plane did not meet the selected support limits",
        }

        def failure(code):
            return Result(
                outputs,
                passed=False,
                score=report["fit_inlier_fraction"],
                fail_reasons=[{"code": code, "message": reasons[code]}],
                message=reasons[code],
            )

        if eligible_count < int(p["minimum_inliers"]):
            return failure("insufficient_fit_pixels")
        limit = int(p["maximum_fit_samples"])
        if eligible_count > limit:
            sampled = rng.choice(np.flatnonzero(eligible), size=limit, replace=False)
            sampling_mask = np.zeros(values.shape, np.uint8)
            sampling_mask.flat[sampled] = 255
        else:
            sampling_mask = eligible.astype(np.uint8) * 255
        xyz = rgbd_function("depthTo3d")(values, k, mask=sampling_mask)
        points = np.ascontiguousarray(xyz.reshape(-1, xyz.shape[-1])[:, :3], dtype=np.float64)
        if not np.isfinite(points).all():
            raise ValidationError(
                "Cannot backproject finite XYZ with these depth values and intrinsics"
            )
        report["fit_samples"] = len(points)
        plane = fit_support_plane(points, tolerance, int(p["maximum_trials"]), rng)
        if plane is None:
            return failure("degenerate_plane")

        normal, offset = plane
        signed = distance_grid(values, k, normal, offset)
        if not np.isfinite(signed[valid]).all():
            raise ValidationError(
                "Cannot represent signed residuals with these intrinsics and depth values"
            )
        inliers = np.abs(signed) <= tolerance
        fit = inliers & eligible
        count = int(np.count_nonzero(fit))
        fraction = count / eligible_count
        report.update(fit_inliers=count, fit_inlier_fraction=fraction)
        if count < int(p["minimum_inliers"]) or fraction < float(p["minimum_inlier_fraction"]):
            return failure("insufficient_plane_support")

        foreground = signed > float(p["foreground_min_m"])
        foreground_mask = foreground.astype(np.uint8) * 255
        foreground_depth = depth.copy()
        foreground_depth[~foreground] = 0 if depth.dtype == np.uint16 else np.nan
        report.update(
            plane_pixels=int(np.count_nonzero(inliers)),
            foreground_pixels=int(np.count_nonzero(foreground)),
            fit_rms_m=float(np.sqrt(np.mean(np.square(signed[fit], dtype=np.float64)))),
        )
        outputs.update(
            plane={
                "model": "camera_depth_plane_v1",
                "units": "m",
                "frame": model.get("frame", "camera"),
                "normal": normal.tolist(),
                "offset": offset,
            },
            signed_distance_m=signed,
            plane_mask=inliers.astype(np.uint8) * 255,
            foreground_mask=foreground_mask,
            foreground_depth=foreground_depth,
        )
        drawings = {}
        if self.generate_graphics and report["foreground_pixels"]:
            contours, _ = cv2.findContours(
                foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            objects = tuple(
                graphic_object(
                    "polyline",
                    cv2.approxPolyDP(contour, 0.75, True).reshape(-1, 2),
                    (0, 180, 255),
                    closed=True,
                )
                for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:128]
            )
            drawings["Foreground"] = GraphicDrawing(objects, source="depth")
        return Result(
            outputs,
            drawings=drawings,
            score=fraction,
            message=f"Plane support {fraction:.1%} · {report['foreground_pixels']} foreground",
        )
