"""Conditioned geometric circle fitting and ordered image-arc measurements."""

import copy
import math

import cv2
import numpy as np

from .collection_tools import _local_collection_payload, read_collection
from .core import Block, Port, Result
from .features import overlay_image
from .graphics_layers import GraphicDrawing, draw_object, graphic_circle, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, frame, image_with_payload
from .result_graphics import ACCEPTED, REJECTED
from .selection import selection_mask
from .tickets import ticket_key


def algebraic_circle(points):
    """Use a local algebraic fit only to initialize the geometric objective."""
    origin = points.mean(axis=0)
    local = points - origin
    scale = float(np.max(np.linalg.norm(local, axis=1)))
    if scale <= 0:
        return None
    local = local / scale
    design = np.column_stack((2 * local, np.ones(len(local))))
    spectrum = np.linalg.svd(design, compute_uv=False)
    if spectrum[-1] <= spectrum[0] * 1e-12:
        return None
    ok, solution = cv2.solve(
        design, np.sum(local * local, axis=1).reshape(-1, 1), flags=cv2.DECOMP_SVD
    )
    if not ok:
        return None
    x, y, constant = solution.ravel()
    radius_squared = x * x + y * y + constant
    if not np.isfinite(radius_squared) or radius_squared <= 0:
        return None
    return np.r_[origin + np.array([x, y]) * scale, math.sqrt(radius_squared) * scale]


def radial_residuals(points, circle):
    return np.linalg.norm(points - circle[:2], axis=1) - circle[2]


def radial_jacobian(points, circle):
    delta = circle[:2] - points
    distance = np.linalg.norm(delta, axis=1)
    direction = np.divide(
        delta, distance[:, None], out=np.zeros_like(delta), where=distance[:, None] > 0
    )
    return np.column_stack((direction, -np.ones(len(points))))


def refine_circle(points, circle, maximum_iterations, tolerance):
    """Bounded geometric Gauss-Newton with SVD steps and backtracking."""
    circle = circle.copy()
    residual = radial_residuals(points, circle)
    cost = float(residual @ residual)
    for iteration in range(maximum_iterations):
        jacobian = radial_jacobian(points, circle)
        ok, step = cv2.solve(jacobian, -residual.reshape(-1, 1), flags=cv2.DECOMP_SVD)
        if not ok or not np.isfinite(step).all():
            return circle, False, iteration + 1
        step = step.ravel()
        if np.linalg.norm(step) <= tolerance:
            return circle, True, iteration + 1
        accepted = False
        for attempt in range(16):
            candidate = circle + step * (0.5**attempt)
            if candidate[2] <= 0 or not np.isfinite(candidate).all():
                continue
            candidate_residual = radial_residuals(points, candidate)
            candidate_cost = float(candidate_residual @ candidate_residual)
            if candidate_cost < cost:
                circle, residual, cost = candidate, candidate_residual, candidate_cost
                accepted = True
                break
        if not accepted:
            gradient = np.max(np.abs(jacobian.T @ residual)) / len(points)
            return circle, bool(gradient <= tolerance), iteration + 1
    return circle, False, maximum_iterations


def estimate_circle(points, parameters):
    """Fit private normalized coordinates; return support that agrees with the final model."""
    p = parameters
    origin = np.median(points, axis=0)
    radii = np.linalg.norm(points - origin, axis=1)
    scale = float(np.median(radii))
    if scale <= 0:
        scale = float(radii.max())
    if scale <= np.finfo(float).eps * max(1.0, float(np.max(np.abs(points)))):
        return None, np.zeros(len(points), bool), "degenerate_circle", {}
    local = (points - origin) / scale
    threshold = p["inlier_tolerance"] / scale
    tolerance = p["position_tolerance"] / scale
    candidate = algebraic_circle(local) if p["method"] == "least_squares" else None
    support = np.ones(len(points), bool)
    trials = 0
    if p["method"] == "ransac":
        best = (-1, -math.inf)
        candidate = None
        rng = np.random.default_rng(p["random_seed"])
        target_trials = p["maximum_trials"]
        initial = algebraic_circle(local)
        for trial in range(p["maximum_trials"]):
            sample = (
                initial
                if trial == 0
                else algebraic_circle(local[rng.choice(len(local), 3, replace=False)])
            )
            trials = trial + 1
            if (
                sample is not None
                and p["minimum_radius"] <= sample[2] * scale <= p["maximum_radius"]
            ):
                residuals = np.abs(radial_residuals(local, sample))
                mask = residuals <= threshold
                count = int(mask.sum())
                score = (count, -float(np.sum(np.minimum(residuals / threshold, 1) ** 2)))
                if score > best and count >= 3:
                    best, candidate, support = score, sample, mask
                    chance = (count / len(points)) ** 3
                    if chance >= 1:
                        target_trials = 16
                    elif chance > 0:
                        required = math.ceil(math.log1p(-p["confidence"]) / math.log1p(-chance))
                        target_trials = max(16, min(target_trials, required))
            if trials >= target_trials:
                break
    if candidate is None or support.sum() < 3:
        return None, support, "no_circle_consensus", {"trials": trials}
    iterations = 0
    for round_index in range(5):
        candidate, converged, count = refine_circle(
            local[support], candidate, p["refinement_iterations"], tolerance
        )
        iterations += count
        if not converged:
            status = "circle_refinement_not_converged"
            break
        updated = (
            np.abs(radial_residuals(local, candidate)) <= threshold
            if p["method"] == "ransac"
            else support
        )
        if np.array_equal(updated, support):
            status = "ok"
            break
        if updated.sum() < 3:
            status = "no_circle_consensus"
            break
        support = updated
    else:
        status = "circle_consensus_not_stable"
    circle = np.r_[candidate[:2] * scale + origin, candidate[2] * scale]
    if not np.isfinite(circle).all() or np.any(np.abs(circle) > 1e8):
        return None, support, "circle_exceeds_coordinate_limits", {"trials": trials}
    return (
        circle,
        support,
        status,
        {
            "trials": trials,
            "refinement_iterations": iterations,
            "consensus_rounds": round_index + 1,
        },
    )


def circle_uncertainty(points, circle):
    """Conditional radial-IID linearization, not acquisition/calibration uncertainty."""
    jacobian = radial_jacobian(points, circle)
    _, singular, vectors = np.linalg.svd(jacobian, full_matrices=False)
    if len(singular) != 3 or singular[-1] <= singular[0] * 1e-12:
        return None, None
    condition = float(singular[0] / singular[-1])
    if len(points) <= 3:
        return condition, None
    residuals = radial_residuals(points, circle)
    variance = float(residuals @ residuals / (len(points) - 3))
    covariance = ((vectors.T / singular) @ (vectors.T / singular).T) * variance
    std = np.sqrt(np.maximum(0, np.diag(covariance)))
    return condition, {
        "model": "radial_iid_linearization_v1",
        "parameter_order": ["center_x", "center_y", "radius"],
        "covariance_px2": covariance.tolist(),
        "std_px": std.tolist(),
        "degrees_of_freedom": len(points) - 3,
        "radial_noise_std_px": math.sqrt(variance),
        "conditional_on_selected_support": True,
    }


def arc_geometry(points, circle, closed):
    theta = np.arctan2(points[:, 1] - circle[1], points[:, 0] - circle[0])
    delta = np.diff(theta)
    if closed:
        delta = np.r_[delta, theta[0] - theta[-1]]
    delta = np.arctan2(np.sin(delta), np.cos(delta))
    sweep = float(np.sum(delta))
    variation = float(np.sum(np.abs(delta)))
    backtracking = (variation - abs(sweep)) / (2 * variation) if variation > 0 else 0.0
    angles = np.sort(np.mod(theta, 2 * np.pi))
    gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
    start = float(np.degrees(theta[0]) % 360)
    extent = float(np.degrees(sweep))
    return {
        "model": "image_arc_v1",
        "center": circle[:2].tolist(),
        "radius": float(circle[2]),
        "start_angle_degrees": start,
        "end_angle_degrees": float((start + extent) % 360),
        "last_observed_angle_degrees": float(np.degrees(theta[-1]) % 360),
        "sweep_degrees": extent,
        "support_span_degrees": float(np.degrees(2 * np.pi - gaps.max())),
        "maximum_ordered_gap_degrees": float(np.degrees(np.max(np.abs(delta)))),
        "backtracking_fraction": max(0.0, backtracking),
        "closed_path": bool(closed),
        "angle_convention": "zero at +x; positive clockwise in image coordinates",
    }


def read_circle_points(block, inputs):
    value, payload, closed = inputs["points"], block.input_payloads["points"], False
    if isinstance(value, dict):
        try:
            data, payload = read_collection(block, inputs, "points")
            if data["kind"] not in ("contours", "blobs"):
                raise ValidationError("Use a contour/blob collection or ordered [x,y] points")
            index = int(block.parameters["item_index"])
            if index >= len(data["items"]):
                raise ValidationError("The selected collection item is unavailable", "item_index")
            geometry = data["items"][index]["geometry"]
            value = geometry.get("points")
            if value is None:
                contours = geometry["contours"]
                boundary = int(block.parameters["boundary_index"])
                if boundary == -1:

                    def boundary_area(index):
                        vertices = np.asarray(contours[index], np.float64)
                        if vertices.ndim != 2 or vertices.shape[1] != 2 or not len(vertices):
                            raise ValidationError("Blob boundaries need nonempty [x,y] rows")
                        local = vertices - vertices[0]
                        return abs(cv2.contourArea(local.astype(np.float32)))

                    boundary = max(
                        range(len(contours)),
                        key=boundary_area,
                    )
                if boundary >= len(contours):
                    raise ValidationError("The blob boundary is unavailable", "boundary_index")
                value = contours[boundary]
            value = np.asarray(value, np.float64) - data.get("origin", [0, 0])
            payload = _local_collection_payload(data, payload)
            closed = True
        except (ValueError, TypeError, KeyError, IndexError, cv2.error) as exc:
            raise ValidationError(str(exc)) from exc
    try:
        points = np.asarray(value, np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValidationError("Use ordered finite [x,y] point rows") from exc
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or not 3 <= len(points) <= 100_000
        or not np.isfinite(points).all()
        or np.max(np.abs(points)) > 1e7
    ):
        raise ValidationError("Use 3–100,000 ordered finite [x,y] points within ±10M pixels")
    repeated_endpoint = np.array_equal(points[0], points[-1])
    _, unique_indices = np.unique(points, axis=0, return_index=True)
    keep = np.zeros(len(points), bool)
    keep[unique_indices] = True
    closed = closed or repeated_endpoint
    if block.parameters["path"] != "auto":
        closed = block.parameters["path"] == "closed"
    return points, copy.deepcopy(payload), closed, keep


def image_grid(points, payload, image, image_payload):
    source, target = payload.get("spatial"), image_payload["spatial"]
    preserve = False
    if source and source != target:
        a, b = source.get("root_id"), target.get("root_id")
        if a is not None and b is not None:
            if a != b or source["root_size"] != target["root_size"]:
                raise ValidationError("Points and image require compatible coordinate frames")
            transform = np.linalg.solve(np.asarray(target["to_root"]), source["to_root"])
            mapped = np.column_stack((points, np.ones(len(points)))) @ transform.T
            if not (np.all(mapped[:, 2] > 1e-12) or np.all(mapped[:, 2] < -1e-12)):
                raise ValidationError("The ordered points cross the image-frame horizon")
            points = mapped[:, :2] / mapped[:, 2, None]
        elif source["image_size"] != target["image_size"]:
            raise ValidationError("Unframed points and image must use matching pixel grids")
        elif (
            a is None
            and b is None
            and not np.allclose(source["to_root"], target["to_root"], atol=1e-9, rtol=0)
        ):
            raise ValidationError("Anonymous point and image transforms must match")
        else:
            preserve = a is not None and b is None
    if (
        "ticket" in payload
        and "ticket" in image_payload
        and (ticket_key(payload["ticket"]) != ticket_key(image_payload["ticket"]))
    ):
        raise ValidationError("Points and image have different tickets")
    retained = {key: payload[key] for key in ("spatial", "pyramid") if preserve and key in payload}
    return points, {**payload, **copy.deepcopy(image_payload), **retained}


class FitCircleArc(Block):
    type_id, title = "inspect.fit_circle_arc", "Fit circle or arc"
    category = block_type = "Inspection"
    subtype = "Measurements"
    inputs = (
        Port("points", "data"),
        Port("image", required=False, required_when_connected=True),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("fit", "data", required=False),
        Port("arc", "data", required=False),
        Port("center", "data", required=False),
        Port("radius", "number", required=False),
        Port("rmse", "number", required=False),
        Port("measurements", "data"),
        Port("points", "data"),
        Port("residuals", "data"),
        Port("image", required=False),
        Port("overlay", required=False),
    )
    defaults = {
        "method": "ransac",
        "item_index": 0,
        "boundary_index": -1,
        "path": "auto",
        "inlier_tolerance": 1.0,
        "maximum_trials": 256,
        "confidence": 0.99,
        "random_seed": 0,
        "refinement_iterations": 50,
        "position_tolerance": 1e-8,
        "minimum_inliers": 6,
        "minimum_inlier_fraction": 0.5,
        "minimum_span_degrees": 30.0,
        "maximum_condition": 10_000.0,
        "maximum_gap_degrees": 120.0,
        "maximum_backtracking_fraction": 0.1,
        "minimum_radius": 0.01,
        "maximum_radius": 10_000_000.0,
        "check_rmse": True,
        "maximum_rmse": 1.0,
        "sample_limit": 2000,
    }
    parameter_choices = {"method": ("ransac", "least_squares"), "path": ("auto", "open", "closed")}
    parameter_ranges = {
        "item_index": NumericRange(0, 99_999, whole=True),
        "boundary_index": NumericRange(-1, 99_999, whole=True),
        "inlier_tolerance": NumericRange(1e-9, 1e7),
        "maximum_trials": NumericRange(16, 4096, whole=True),
        "confidence": NumericRange(0.5, 0.999999),
        "random_seed": NumericRange(0, 2**32 - 1, whole=True),
        "refinement_iterations": NumericRange(1, 200, whole=True),
        "position_tolerance": NumericRange(1e-12, 0.1),
        "minimum_inliers": NumericRange(3, 100_000, whole=True),
        "minimum_inlier_fraction": NumericRange(0, 1),
        "minimum_span_degrees": NumericRange(0, 359.99),
        "maximum_condition": NumericRange(1, 1e12),
        "maximum_gap_degrees": NumericRange(0.001, 179.99),
        "maximum_backtracking_fraction": NumericRange(0, 0.5),
        "minimum_radius": NumericRange(1e-9, 1e7),
        "maximum_radius": NumericRange(1e-9, 1e7),
        "maximum_rmse": NumericRange(0, 1e7),
        "sample_limit": NumericRange(0, 100_000, whole=True),
    }
    parameter_help = {
        "method": "RANSAC refits radial consensus; least squares uses every eligible unique point.",
        "item_index": "Current collection position. Ignored for point-list input.",
        "boundary_index": "Blob boundary; -1 chooses largest. Ignored for contours/lists.",
        "path": "Auto closes detected contours or a list with a repeated endpoint; otherwise open.",
        "inlier_tolerance": "Inclusive radial consensus distance in output pixels. RANSAC only.",
        "maximum_trials": "RANSAC candidate budget; adaptive stopping uses at least 16 trials.",
        "confidence": "Adaptive RANSAC trial target; not a measurement confidence interval.",
        "random_seed": "Repeatable local seed, 0–4294967295. Global random state is unchanged.",
        "refinement_iterations": "Geometric iterations per support refit; at most five rounds.",
        "position_tolerance": "Geometric convergence tolerance in output pixel units.",
        "minimum_inliers": "Minimum distinct fitting points. Duplicate coordinates vote once.",
        "minimum_inlier_fraction": "Minimum accepted support divided by eligible point count.",
        "minimum_span_degrees": "Smallest angular envelope, excluding the largest circular gap.",
        "maximum_condition": "Largest radial-Jacobian condition number. High means unstable.",
        "maximum_gap_degrees": "Largest consecutive support angle, including closed-path closure.",
        "maximum_backtracking_fraction": "Allowed reverse angular travel / total travel, 0 to 0.5.",
        "minimum_radius": "Smallest accepted radius, in the fitting grid's pixels.",
        "maximum_radius": "Largest accepted radius, also used to screen RANSAC candidates.",
        "check_rmse": "Judge fitting-support radial root mean square error.",
        "maximum_rmse": "Inclusive fitting-support RMSE limit in output pixels.",
        "sample_limit": "Maximum residual rows; zero skips the table without changing the fit.",
    }
    input_help = {
        "points": "Ordered [x,y] points or contour/blob collection; no edge detection.",
        "image": "Optional fitting grid and unchanged image; compatible frames map into it.",
        "mask": "Optional uint8 selection of points by nearest pixel. Others do not fit or vote.",
        "region": "Optional region intersected with the mask. Selection does not prove closure.",
    }
    parallel_safe = process_inputs_readonly = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = ["fit", "arc", "radius", "measurements"]
        self.documentation = (
            "<p>Circle fitting minimizes squared radial distances. A normalized algebraic "
            "initializer and SVD-based refinement are separate from minimum enclosing circles. "
            "RANSAC refits stable consensus support. Short or ill-conditioned arcs withhold "
            "the primary geometry; valid circle and ordered-arc states are reported separately.</p>"
            "<p>Uncertainty is a conditional radial-IID linearization, not calibration or "
            "manufacturing uncertainty. Angular support is an envelope of observed directions, "
            "not visibility or occlusion evidence.</p>"
            '<p><a href="https://arxiv.org/abs/cs/0301001">Chernov and Lesort: '
            "Least squares fitting of circles and lines</a>. The block uses a bounded CPU "
            "implementation; it does not claim a native OpenCV fitCircle binding.</p>"
        )

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["minimum_radius"] > p["maximum_radius"]:
            raise ValidationError("Minimum radius exceeds maximum radius", "minimum_radius")

    def process(self, inputs):
        p = dict(self.parameters)
        for name in (
            "item_index",
            "boundary_index",
            "maximum_trials",
            "random_seed",
            "refinement_iterations",
            "minimum_inliers",
            "sample_limit",
        ):
            p[name] = int(p[name])
        points, payload, closed, eligible = read_circle_points(self, inputs)
        unique_count = int(eligible.sum())
        image = inputs.get("image")
        if image is not None:
            if image.shape[0] * image.shape[1] > 64_000_000 or max(image.shape[:2]) > 16384:
                raise ValidationError("Reference image is limited to 64M pixels and 16384 per axis")
            points, payload = image_grid(points, payload, image, self.input_payloads["image"])
        if not np.isfinite(points).all() or np.max(np.abs(points)) > 1e7:
            raise ValidationError("Mapped points exceed the finite ±10M pixel coordinate limits")
        if "mask" in inputs or "region" in inputs:
            size = payload.get("spatial", {}).get("image_size")
            if size is None:
                if "mask" in inputs:
                    size = [inputs["mask"].shape[1], inputs["mask"].shape[0]]
                else:
                    size = inputs["region"].get("image_size")
                if not isinstance(size, list) or len(size) != 2:
                    raise ValidationError("Selection needs an image grid or image_size")
                payload["spatial"] = frame(size)
            if (
                any(type(n) is not int or not 1 <= n <= 16384 for n in size)
                or math.prod(size) > 64_000_000
            ):
                raise ValidationError("Selection grid is limited to 64M pixels and 16384 per axis")
            canvas = image_with_payload(np.broadcast_to(np.uint8(0), (size[1], size[0])), payload)
            mask = selection_mask({**inputs, "image": canvas}, canvas)
            indices = np.rint(points).astype(np.int64)
            inside = np.all((indices >= 0) & (indices < size), axis=1)
            selected = np.zeros(len(points), bool)
            selected[inside] = mask[indices[inside, 1], indices[inside, 0]] != 0
            eligible &= selected
        count = int(eligible.sum())
        if count * p["maximum_trials"] > 32_000_000 and p["method"] == "ransac":
            raise ValidationError("RANSAC exceeds 32M point trials; reduce trials or input points")
        if count * p["refinement_iterations"] * 5 > 32_000_000:
            raise ValidationError("Circle refinement exceeds 32M point iterations")
        candidate, support, status, solver = (
            None,
            np.zeros(count, bool),
            "insufficient_circle_support",
            {},
        )
        if count >= max(3, p["minimum_inliers"]):
            candidate, support, status, solver = estimate_circle(points[eligible], p)
        used = np.zeros(len(points), bool)
        used[np.flatnonzero(eligible)] = support
        reasons = [] if status == "ok" else [status]
        condition, uncertainty, arc = None, None, None
        rmse = None
        if candidate is not None:
            residual = radial_residuals(points, candidate)
            rmse = float(np.sqrt(np.mean(residual[used] ** 2))) if used.any() else None
            condition, uncertainty = circle_uncertainty(points[used], candidate)
            arc_support, arc_closed = points[used], closed
            if p["path"] == "auto" and closed and used.sum() < unique_count:
                # Trimming a cyclic contour creates an observed open arc. Start after
                # its largest unobserved gap rather than joining across that gap.
                angles = np.arctan2(
                    arc_support[:, 1] - candidate[1], arc_support[:, 0] - candidate[0]
                )
                gaps = np.diff(np.r_[angles, angles[0]])
                gaps = np.abs(np.arctan2(np.sin(gaps), np.cos(gaps)))
                arc_support = np.roll(arc_support, -(int(np.argmax(gaps)) + 1), axis=0)
                arc_closed = False
            arc = arc_geometry(arc_support, candidate, arc_closed)
            if (
                used.sum() < p["minimum_inliers"]
                or used.sum() / count < p["minimum_inlier_fraction"]
            ):
                reasons.append("insufficient_circle_support")
            if not p["minimum_radius"] <= candidate[2] <= p["maximum_radius"]:
                reasons.append("circle_radius_out_of_range")
            if condition is None or condition > p["maximum_condition"]:
                reasons.append("circle_ill_conditioned")
            if arc["support_span_degrees"] + 1e-9 < p["minimum_span_degrees"]:
                reasons.append("circle_arc_too_short")
        circle_valid = candidate is not None and not reasons
        if arc is not None:
            if (
                arc["maximum_ordered_gap_degrees"] > p["maximum_gap_degrees"] + 1e-9
                or arc["backtracking_fraction"] > p["maximum_backtracking_fraction"] + 1e-12
                or abs(arc["sweep_degrees"]) > 360 + 1e-6
            ):
                reasons.append("arc_order_unqualified")
        arc_valid = circle_valid and "arc_order_unqualified" not in reasons
        if p["check_rmse"] and rmse is not None and rmse > p["maximum_rmse"]:
            reasons.append("circle_rmse_out_of_tolerance")
        reasons = list(dict.fromkeys(reasons))
        measurements = {
            "model": "circle_arc_measurement_v1",
            "units": "pixels",
            "method": p["method"],
            "circle_valid": circle_valid,
            "arc_valid": arc_valid,
            "status": reasons or ["ok"],
            "input_count": len(points),
            "unique_count": unique_count,
            "eligible_count": count,
            "inlier_count": int(used.sum()),
            "inlier_fraction": float(used.sum() / count) if count else 0.0,
            "candidate_center": candidate[:2].tolist() if candidate is not None else None,
            "candidate_radius": float(candidate[2]) if candidate is not None else None,
            "rmse": rmse,
            "condition": condition,
            "uncertainty": uncertainty,
            "arc": arc,
            "solver": solver,
        }
        fit = (
            {
                "model": "circle_arc_fit_v1",
                "shape": "circle",
                "center": candidate[:2].tolist(),
                "radius": float(candidate[2]),
                "rmse": rmse,
                "sample_count": int(used.sum()),
                "arc_valid": arc_valid,
            }
            if circle_valid
            else NO_VALUE
        )
        rows = []
        if p["sample_limit"]:
            selected = (
                np.arange(len(points))
                if len(points) <= p["sample_limit"]
                else np.unique(np.linspace(0, len(points) - 1, p["sample_limit"], dtype=int))
            )
            for index in selected:
                rows.append(
                    {
                        "index": int(index),
                        "point": points[index].tolist(),
                        "eligible": bool(eligible[index]),
                        "used_for_fit": bool(used[index]),
                        "signed_radial_error": float(residual[index])
                        if candidate is not None
                        else None,
                    }
                )
        objects, overlay = [], NO_VALUE
        if self.generate_graphics:
            stride = max(1, math.ceil(len(points) / 1000))
            for index in range(0, len(points), stride):
                color = (
                    ACCEPTED if used[index] else REJECTED if eligible[index] else (130, 130, 130)
                )
                objects.append(graphic_object("marker", [points[index]], color, thickness=1))
            if candidate is not None:
                objects.append(graphic_circle(*candidate[:2], candidate[2], (130, 130, 130)))
                if arc is not None:
                    angles = np.radians(
                        np.linspace(
                            arc["start_angle_degrees"],
                            arc["start_angle_degrees"] + arc["sweep_degrees"],
                            128,
                        )
                    )
                    path = candidate[:2] + candidate[2] * np.column_stack(
                        (
                            np.cos(angles),
                            np.sin(angles),
                        )
                    )
                    objects.append(
                        graphic_object(
                            "polyline",
                            path,
                            ACCEPTED if arc_valid and not reasons else REJECTED,
                            closed=arc["closed_path"],
                            thickness=2,
                        )
                    )
                objects.append(
                    graphic_object(
                        "marker", [candidate[:2]], ACCEPTED if circle_valid else REJECTED
                    )
                )
            if image is not None:
                overlay = overlay_image(image)
                for obj in objects:
                    draw_object(overlay, obj, np.eye(3))
        outputs = {
            "fit": fit,
            "arc": arc if arc_valid else NO_VALUE,
            "center": candidate[:2].tolist() if circle_valid else NO_VALUE,
            "radius": float(candidate[2]) if circle_valid else NO_VALUE,
            "rmse": rmse if candidate is not None else NO_VALUE,
            "measurements": measurements,
            "points": points[used].tolist(),
            "residuals": {"model": "circle_residuals_v1", "total_count": len(points), "rows": rows},
            "image": image if image is not None else NO_VALUE,
            "overlay": overlay,
        }
        return Result(
            outputs,
            payloads={name: payload for name in outputs},
            passed=not reasons,
            score=measurements["inlier_fraction"],
            message=f"{used.sum()}/{count} fitting points · "
            + (", ".join(reasons) if reasons else f"radius {candidate[2]:.6g} px"),
            fail_reasons=reasons,
            drawings={"circle fit": GraphicDrawing(tuple(objects), "measurements", True)}
            if objects
            else {},
            graphics={"circle fit": overlay} if overlay is not NO_VALUE else {},
        )
