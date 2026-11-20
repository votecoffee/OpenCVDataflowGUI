"""Robust metric cylinder and bore measurements from partial scan surfaces."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .cloud_model_fitting import _support
from .cloud_projection import METERS
from .core import Port, Result
from .geometry3d_common import Tool3D, pose_record, rigid
from .local_surface_fit import fit_local_surfaces
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .pointclouds import PointCloud, render_cloud

_MINIMUM_ROWS = 12
_SECTOR_COUNT = 36


def _tangents(axis, preferred=None):
    u = preferred if preferred is not None else np.eye(3)[np.argmin(np.abs(axis))]
    u = u - axis * (u @ axis)
    norm = np.linalg.norm(u)
    if norm < 1e-8:
        return _tangents(axis)
    u /= norm
    return u, np.cross(axis, u)


@dataclass(frozen=True)
class Cylinder:
    center: np.ndarray
    axis: np.ndarray
    radius: float

    def radial(self, points):
        delta = points - self.center
        heights = np.einsum("ij,j->i", delta, self.axis)
        return delta - heights[:, None] * self.axis, heights

    def residuals(self, points):
        return np.linalg.norm(self.radial(points)[0], axis=1) - self.radius

    @classmethod
    def from_normals(cls, points, normals):
        a, b = normals
        axis = np.cross(a, b)
        length = np.linalg.norm(axis)
        if length < 0.2:
            return None
        axis /= length
        tangent = np.cross(axis, a)
        difference = points[1] - points[0]
        step_b = -(difference @ tangent) / (b @ tangent)
        step_a = difference @ a + step_b * (a @ b)
        radius = (abs(step_a) + abs(step_b)) / 2
        if abs(abs(step_a) - abs(step_b)) > 0.24 * radius:
            return None
        center = points[0] + step_a * a
        return cls(center - (center @ axis) * axis, axis, float(radius))

    @classmethod
    def from_axis(cls, points, axis):
        u, v = _tangents(axis)
        coordinates = points @ np.column_stack((u, v))
        edges = coordinates[1:] - coordinates[0]
        # Reject nearly collinear triples before solving their circumcircle.
        lengths = np.linalg.norm(edges, axis=1)
        if abs(np.linalg.det(edges)) <= 1e-6 * lengths.prod():
            return None
        offset = np.linalg.solve(2 * edges, np.einsum("ij,ij->i", edges, edges))
        center = coordinates[0] + offset
        return cls(center[0] * u + center[1] * v, axis, float(np.linalg.norm(offset)))


def _normal_candidates(points, normals, cancelled):
    if normals is not None:
        return normals / np.linalg.norm(normals, axis=1)[:, None], np.arange(len(points))
    tree = cKDTree(points)
    fits = fit_local_surfaces(points, tree, np.inf, 15, cancelled, minimum_neighbors=8)
    if fits is None:
        return None, np.empty(0, dtype=int)
    usable = (fits.normal_quality > 0.08) & (fits.curvature < 0.032)
    return fits.normals, np.flatnonzero(usable)


def _jacobian(cylinder, points, u, v, axis_scale):
    radial, heights = cylinder.radial(points)
    length = np.linalg.norm(radial, axis=1)
    radial /= np.where(length > 0, length, 1)[:, None]
    along_u, along_v = radial @ u, radial @ v
    return np.column_stack(
        (
            -heights * along_u / axis_scale,
            -heights * along_v / axis_scale,
            -along_u,
            -along_v,
            -np.ones(len(points)),
        )
    )


def _refine(seed, points, tolerance, cancelled):
    u, v = _tangents(seed.axis)
    roi = np.abs(seed.residuals(points)) <= 2.5 * tolerance
    support = points[roi]
    if len(support) < _MINIMUM_ROWS:
        return seed
    tangent = np.column_stack((u, v))

    def decode(params):
        direction = seed.axis + tangent @ params[:2]
        scale = np.linalg.norm(direction)
        return Cylinder(seed.center + tangent @ params[2:4], direction / scale, params[4]), scale

    def jac(params):
        candidate, scale = decode(params)
        return _jacobian(candidate, support, u, v, scale)

    solution = least_squares(
        lambda params: decode(params)[0].residuals(support),
        [0, 0, 0, 0, seed.radius],
        jac=jac,
        bounds=([-0.7, -0.7, -np.inf, -np.inf, 2 * tolerance], [0.7, 0.7, np.inf, np.inf, np.inf]),
        loss="soft_l1",
        f_scale=tolerance,
        max_nfev=24,
        callback=lambda _: _check_cancelled(cancelled),
    )
    if cancelled() or not np.isfinite(solution.x).all():
        return seed
    return decode(solution.x)[0]


def _check_cancelled(cancelled):
    if cancelled():
        raise StopIteration


def _fit(points, normals, tolerance, trials, rng, seed, cancelled, report):
    normals, candidates = _normal_candidates(points, normals, cancelled)
    report["usable_seed_normals"] = len(candidates)
    centered = points - points.mean(axis=0)
    values, vectors = np.linalg.eigh(centered.T @ centered)
    if values[1] <= 1e-12 * values[-1]:
        return None
    directions = list(vectors.T[::-1])
    if len(candidates) >= _MINIMUM_ROWS:
        selected_normals = normals[candidates]
        _, axes = np.linalg.eigh(selected_normals.T @ selected_normals)
        directions.insert(0, axes[:, 0])
    if seed is not None:
        directions.insert(0, seed.axis)
    winners = []

    def score(candidate):
        if candidate is None or candidate.radius <= 2 * tolerance:
            return
        _, rank = _support(candidate, points, tolerance)
        if rank[0] < _MINIMUM_ROWS:
            return
        winners.append((rank, candidate))
        winners.sort(key=lambda entry: entry[0], reverse=True)
        del winners[6:]

    if seed is not None:
        score(seed)
    use_normals = len(candidates) >= 2
    direction_index = 0
    for attempt in range(trials):
        if cancelled():
            return None
        if use_normals and attempt % 3 != 0:
            idx = rng.choice(candidates, 2, replace=False)
            candidate = Cylinder.from_normals(points[idx], normals[idx])
        else:
            idx = rng.choice(len(points), 3, replace=False)
            axis = directions[direction_index % len(directions)]
            direction_index += 1
            candidate = Cylinder.from_axis(points[idx], axis)
        report["hypotheses_tested"] += 1
        score(candidate)
    if not winners:
        return None

    best_rank, best = winners[0]
    for _, candidate in winners:
        if cancelled():
            return None
        for _ in range(2):
            refined = _refine(candidate, points, tolerance, cancelled)
            _, rank = _support(refined, points, tolerance)
            if rank <= _support(candidate, points, tolerance)[1]:
                break
            candidate = refined
            if rank > best_rank:
                best_rank, best = rank, candidate
    return best


def _seed_pose(record, cloud):
    if record is None:
        return None
    if (
        not isinstance(record, dict)
        or record.get("model") != "pose_3d_v1"
        or record.get("target_frame") != cloud.frame
        or record.get("units") not in METERS
        or not isinstance(record.get("source_frame"), str)
        or not record["source_frame"].strip()
    ):
        raise ValidationError(
            "Seed pose must map a named metric frame into the cloud frame", "seed_pose"
        )
    try:
        return rigid(record), METERS[record["units"]]
    except ValueError as error:
        raise ValidationError(str(error), "seed_pose") from error


def _measure_support(radial, heights, u, v, scale):
    angles = np.mod(np.arctan2(radial @ v, radial @ u), 2 * np.pi)
    bins = (angles * (_SECTOR_COUNT / (2 * np.pi))).astype(int).clip(0, _SECTOR_COUNT - 1)
    counts = np.bincount(bins, minlength=_SECTOR_COUNT)
    occupied = counts >= max(1, int(np.ceil(len(angles) * 0.005)))
    covered = np.sort(angles[occupied[bins]])
    gaps = np.diff(covered, append=covered[0] + 2 * np.pi)
    low, median, high = np.quantile(heights, [0.02, 0.5, 0.98])
    return {
        "angular_span_deg": float(np.degrees(2 * np.pi - gaps.max())),
        "occupied_angular_sectors": int(occupied.sum()),
        "angular_sector_count": _SECTOR_COUNT,
        "axial_support_m": float((high - low) * scale),
        "observed_axial_span_m": float(np.ptp(heights) * scale),
        "axial_limits_m": ((np.array([low, high]) - median) * scale).tolist(),
    }, float(median)


def _preview(cloud, keep, center, axis, limits):
    # Keep selection identical to the renderer and avoid recoloring full-resolution clouds.
    index = np.linspace(0, len(keep) - 1, min(len(keep), 20_000), dtype=int)
    colors = np.array([[235, 86, 74], [99, 215, 135]], np.uint8)[keep[index].astype(int)]
    preview = PointCloud(cloud.points[index], cloud.units, cloud.frame, colors=colors)
    axis_line = center + np.asarray(limits)[:, None] * axis
    return render_cloud(preview, segments=axis_line[None] / METERS[cloud.units])


class MeasureCloudCylinder(Tool3D):
    type_id = "3d.cloud_cylinder"
    title, subtype = "Measure point-cloud cylinder or bore", "Point clouds"
    description = "Fit a cylindrical wall through clutter and measure diameter and supported axis."
    payload_reference = "cloud"
    parallel_safe = preview_safe = process_inputs_readonly = True
    inputs = (
        Port("cloud", "cloud"),
        Port("seed_pose", "data", required=False, required_when_connected=True, visual_type="pose"),
    )
    outputs = (
        Port("model", "data", required=False),
        Port("axis_pose", "data", required=False, visual_type="pose"),
        Port("diameter_m", "number", required=False),
        Port("inliers", "cloud", required=False),
        Port("outliers", "cloud", required=False),
        Port("inlier_indices", "data", required=False),
        Port("outlier_indices", "data", required=False),
        Port("signed_residuals_m", "data", required=False),
        Port("report", "data"),
    )
    defaults = {
        "surface": "outer",
        "inlier_tolerance_m": 0.001,
        "minimum_inliers": 24,
        "minimum_inlier_fraction": 0.55,
        "minimum_angular_span_deg": 60.0,
        "minimum_axial_span_m": 0.005,
        "minimum_diameter_m": 0.0,
        "maximum_diameter_m": 0.0,
        "maximum_fit_samples": 4096,
        "maximum_trials": 256,
        "random_seed": 0,
        "max_input_points": 250_000,
    }
    parameter_choices = {"surface": ("outer", "bore")}
    parameter_ranges = {
        "inlier_tolerance_m": NumericRange(1e-9, 1e12),
        "minimum_inliers": NumericRange(_MINIMUM_ROWS, 1_000_000, whole=True),
        "minimum_inlier_fraction": NumericRange(0, 1),
        "minimum_angular_span_deg": NumericRange(30, 359),
        "minimum_axial_span_m": NumericRange(0, 1e12),
        "minimum_diameter_m": NumericRange(0, 1e12),
        "maximum_diameter_m": NumericRange(0, 1e12),
        "maximum_fit_samples": NumericRange(_MINIMUM_ROWS, 16_384, whole=True),
        "maximum_trials": NumericRange(1, 1024, whole=True),
        "random_seed": NumericRange(0, 2**32 - 1, whole=True),
        "max_input_points": NumericRange(_MINIMUM_ROWS, 1_000_000, whole=True),
    }
    parameter_help = {
        "surface": "Bore mode signs inward radial error as positive. Outer mode signs outward.",
        "inlier_tolerance_m": "Inclusive radial distance from the fitted wall.",
        "minimum_inliers": "Required supporting rows after full-cloud classification.",
        "minimum_inlier_fraction": "Required supporting fraction of all input rows.",
        "minimum_angular_span_deg": "Required observed angle, excluding the largest angular gap.",
        "minimum_axial_span_m": "Required 2–98% axial span. At least four tolerances are needed.",
        "minimum_diameter_m": "Lower diameter acceptance limit. Zero disables it.",
        "maximum_diameter_m": "Upper diameter acceptance limit. Zero disables it.",
        "maximum_fit_samples": "Maximum representative rows used to find and refine cylinder fits.",
        "maximum_trials": "Maximum reproducible sample-consensus hypotheses.",
        "random_seed": "Seed for sample selection and hypothesis order.",
        "max_input_points": "Maximum supported rows for full-cloud classification.",
    }
    input_help = {
        "cloud": "Metric XYZ. Optional unit normals can seed the fit regardless of their sign.",
        "seed_pose": "Optional fixture-to-cloud metric pose. Its Z axis and origin seed the fit.",
    }
    output_help = {
        "model": "Metric fitted cylinder, supported axial range and observed geometry.",
        "axis_pose": "Pose at the median observed axial location with Z along the fitted axis.",
        "diameter_m": "Least-squares wall diameter in meters.",
        "inliers": "Supporting scan rows in input order with their original attributes.",
        "outliers": "Excluded scan rows in input order with their original attributes.",
        "inlier_indices": "Zero-based input row indices of the inliers.",
        "outlier_indices": "Zero-based input row indices of the outliers.",
        "signed_residuals_m": "One signed radial error per input row in original order.",
        "report": "Metric diameter, inlier and coverage statistics, and acceptance status.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["maximum_diameter_m"] and p["minimum_diameter_m"] > p["maximum_diameter_m"]:
            raise ValidationError(
                "Minimum diameter must not exceed the maximum",
                "minimum_diameter_m",
                "maximum_diameter_m",
            )
        if p["minimum_inliers"] > p["max_input_points"]:
            raise ValidationError(
                "Minimum inliers must fit within the point budget",
                "minimum_inliers",
                "max_input_points",
            )

    def process(self, inputs):
        cloud, p = inputs["cloud"], self.parameters
        if cloud.units not in METERS:
            raise ValidationError("Cylinder measurements require a cloud in m or mm", "cloud")
        n = len(cloud.points)
        if n > int(p["max_input_points"]):
            raise ValidationError("Cloud exceeds the point budget", "max_input_points")
        seed_record = _seed_pose(inputs.get("seed_pose"), cloud)
        tolerance_m = float(p["inlier_tolerance_m"])
        report = {
            "model": "cloud_cylinder_report_v1",
            "surface": p["surface"],
            "frame": cloud.frame,
            "cloud_units": cloud.units,
            "inlier_tolerance_m": tolerance_m,
            "input_points": n,
            "fit_samples": 0,
            "hypotheses_tested": 0,
            "classified_points": 0,
            "inlier_count": 0,
            "inlier_fraction": 0.0,
            "status": "failed",
        }
        outputs = {key.name: NO_VALUE for key in self.outputs}
        outputs["report"] = report

        def fail(code, message):
            return Result(
                outputs,
                passed=False,
                score=report["inlier_fraction"],
                fail_reasons=[f"cloud_cylinder_{code}"],
                message=message,
            )

        if n < int(p["minimum_inliers"]):
            return fail("insufficient_points", "Cloud has fewer rows than the required support")
        if self._cancelled():
            return fail("cancelled", "Cylinder measurement cancelled")
        rng = np.random.default_rng(int(p["random_seed"]))
        count = min(n, int(p["maximum_fit_samples"]))
        ids = np.arange(n) if count == n else np.sort(rng.choice(n, count, replace=False))
        sample = cloud.points[ids].astype(np.float64) * METERS[cloud.units]
        origin = np.median(sample, axis=0)
        span = np.quantile(sample, [0.05, 0.95], axis=0)
        scale = float(np.linalg.norm(span[1] - span[0]))
        if scale <= 4 * tolerance_m:
            return fail("degenerate_scale", "Scan extent is too small for the requested tolerance")
        tolerance = tolerance_m / scale
        sample = (sample - origin) / scale
        seed, preferred_rotation = None, None
        if seed_record is not None:
            transform, to_m = seed_record
            preferred_rotation = transform[:3, :3]
            axis = preferred_rotation[:, 2].copy()
            axis /= np.linalg.norm(axis)
            center = (transform[:3, 3] * to_m - origin) / scale
            rough = Cylinder(center, axis, 0.0)
            seed = Cylinder(center, axis, float(np.median(rough.residuals(sample))))
        report["fit_samples"] = count
        normals = None if cloud.normals is None else cloud.normals[ids].astype(np.float64)
        report["seed_normals"] = "estimated" if normals is None else "supplied"
        candidate = _fit(
            sample,
            normals,
            tolerance,
            int(p["maximum_trials"]),
            rng,
            seed,
            self._cancelled,
            report,
        )
        if self._cancelled():
            return fail("cancelled", "Cylinder measurement cancelled")
        if candidate is None:
            return fail("no_fit", "No cylinder reached the minimum sample consensus")

        axis = candidate.axis.copy()
        reference = (
            preferred_rotation[:, 2]
            if preferred_rotation is not None
            else np.eye(3)[np.argmax(np.abs(axis))]
        )
        if axis @ reference < 0:
            axis *= -1
        model = Cylinder(candidate.center, axis, candidate.radius)
        points = cloud.points.astype(np.float64) * METERS[cloud.units]
        radial, heights = model.radial((points - origin) / scale)
        errors = (np.linalg.norm(radial, axis=1) - model.radius) * scale
        keep = np.abs(errors) <= tolerance_m
        inliers = int(keep.sum())
        fraction = inliers / n
        report.update(classified_points=n, inlier_count=inliers, inlier_fraction=fraction)
        if inliers < int(p["minimum_inliers"]) or fraction < float(p["minimum_inlier_fraction"]):
            return fail("insufficient_support", "Cylinder does not meet the inlier limits")

        u, v = _tangents(axis, None if preferred_rotation is None else preferred_rotation[:, 0])
        coverage, median = _measure_support(radial[keep], heights[keep], u, v, scale)
        diameter = 2 * model.radius * scale
        report.update(
            **coverage,
            candidate_diameter_m=diameter,
            inlier_rms_m=float(np.sqrt(np.mean(errors[keep] ** 2))),
            maximum_inlier_error_m=float(np.abs(errors[keep]).max()),
        )
        if coverage["angular_span_deg"] < float(p["minimum_angular_span_deg"]):
            return fail(
                "angular_support", "The observed wall spans too little of the circumference"
            )
        if coverage["axial_support_m"] < max(float(p["minimum_axial_span_m"]), 4 * tolerance_m):
            return fail("axial_support", "The observed wall has too little axial extent")
        midpoint = model.center + median * axis
        centered_model = Cylinder(midpoint, axis, model.radius)
        fitted_sample = sample[np.abs(model.residuals(sample)) <= tolerance]
        jac = _jacobian(centered_model, fitted_sample, u, v, 1)
        singular = np.linalg.svd(jac, compute_uv=False)
        condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else np.inf
        report["jacobian_condition"] = condition if np.isfinite(condition) else None
        if condition > 1e4:
            return fail("ill_conditioned", "The sampled wall does not constrain a unique fit")
        if self._cancelled():
            return fail("cancelled", "Cylinder measurement cancelled")

        center = origin + midpoint * scale
        transform = np.eye(4)
        transform[:3, :3] = np.column_stack((u, v, axis))
        transform[:3, 3] = center
        outputs.update(
            model={
                "schema": "point_cloud_model_v1",
                "kind": "cylinder",
                "surface": p["surface"],
                "frame": cloud.frame,
                "units": "m",
                "point_on_axis_m": center.tolist(),
                "axis_direction": axis.tolist(),
                "radius_m": diameter / 2,
                "diameter_m": diameter,
                "axial_limits_m": coverage["axial_limits_m"],
                "coefficients": [*center.tolist(), *axis.tolist(), diameter / 2],
            },
            axis_pose=pose_record(transform, f"{self.id}:axis", cloud.frame, "m"),
            diameter_m=diameter,
        )
        for name, mask in (("inliers", keep), ("outliers", ~keep)):
            if name in self.exposed_outputs:
                outputs[name] = cloud.subset(mask)
            index_port = f"{name[:-1]}_indices"
            if index_port in self.exposed_outputs:
                outputs[index_port] = np.flatnonzero(mask).tolist()
        if "signed_residuals_m" in self.exposed_outputs:
            outputs["signed_residuals_m"] = (
                errors if p["surface"] == "outer" else -errors
            ).tolist()

        limits = float(p["minimum_diameter_m"]), float(p["maximum_diameter_m"]) or np.inf
        accepted = limits[0] <= diameter <= limits[1]
        report.update(status="ok" if accepted else "failed", diameter_m=diameter)
        payload = self.input_payloads["cloud"]
        return Result(
            outputs,
            payloads={name: payload for name in ("inliers", "outliers")},
            graphics={
                "cylinder fit": _preview(cloud, keep, center, axis, coverage["axial_limits_m"])
            }
            if self.capture_graphics
            else {},
            passed=accepted,
            score=fraction,
            fail_reasons=[] if accepted else ["cloud_cylinder_diameter_limit"],
            message=f"Diameter {diameter * 1000:.5g} mm from {inliers}/{n} rows"
            + ("" if accepted else " outside diameter limits"),
        )
