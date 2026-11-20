"""Bounded sample-consensus fits on unorganized metric clouds."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .cloud_projection import METERS
from .core import Port, Result
from .geometry3d_common import Tool3D, fit_plane_normal, plane_sample_normal
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .pointclouds import PointCloud, render_cloud


@dataclass(frozen=True)
class Plane:
    normal: np.ndarray
    origin: np.ndarray

    @classmethod
    def from_sample(cls, points):
        normal = plane_sample_normal(*points)
        return None if normal is None else cls(normal, points[0])

    def refined(self, points):
        fitted = fit_plane_normal(points)
        return None if fitted is None else Plane(*fitted)

    def residuals(self, points):
        return np.einsum("ij,j->i", points - self.origin, self.normal)

    def canonical(self):
        if self.normal[np.argmax(np.abs(self.normal))] < 0:
            return Plane(-self.normal, self.origin)
        return self

    def record(self):
        normal = self.normal.tolist()
        offset = float(-self.origin @ self.normal)
        return {"normal": normal, "offset_m": offset, "coefficients": [*normal, offset]}


@dataclass(frozen=True)
class Sphere:
    center: np.ndarray
    radius: float

    @classmethod
    def from_sample(cls, points):
        edges = points[1:] - points[0]
        singular = np.linalg.svd(edges, compute_uv=False)
        if singular[-1] <= 1e-8 * singular[0]:
            return None
        scale = singular[0]
        scaled = edges / scale
        center = np.linalg.solve(2 * scaled, np.sum(scaled**2, axis=1)) * scale
        return cls(points[0] + center, float(np.linalg.norm(center)))

    def refined(self, points):
        origin = points.mean(axis=0)
        centered = points - origin
        singular = np.linalg.svd(centered, compute_uv=False)
        if singular[-1] <= 1e-8 * singular[0]:
            return None
        scale = np.linalg.norm(np.ptp(centered, axis=0))
        scaled = centered / scale

        def distances(params):
            return np.linalg.norm(params[:3] - scaled, axis=1) - params[3]

        def jacobian(params):
            offsets = params[:3] - scaled
            lengths = np.linalg.norm(offsets, axis=1)
            center_derivative = offsets / np.where(lengths > 0, lengths, 1)[:, None]
            return np.column_stack((center_derivative, -np.ones(len(points))))

        guess = np.r_[(self.center - origin) / scale, self.radius / scale]
        fitted = least_squares(distances, guess, jac=jacobian, max_nfev=20).x
        if not np.isfinite(fitted).all() or fitted[3] <= 0:
            return None
        return Sphere(origin + fitted[:3] * scale, float(fitted[3] * scale))

    def residuals(self, points):
        return np.linalg.norm(points - self.center, axis=1) - self.radius

    def canonical(self):
        return self

    def record(self):
        center = self.center.tolist()
        return {"center_m": center, "radius_m": self.radius, "coefficients": [*center, self.radius]}


def _support(model, points, tolerance):
    distances = model.residuals(points)
    keep = np.abs(distances) <= tolerance
    errors = distances[keep]
    return keep, (len(errors), -float(errors @ errors))


def _fit_sample(points, model_class, sample_size, tolerance, trials, rng, accepts, cancelled):
    best = None
    best_score = (sample_size - 1, -np.inf)
    tested = valid = 0
    for _ in range(trials):
        if cancelled():
            return None, tested, valid
        tested += 1
        sample = points[rng.choice(len(points), size=sample_size, replace=False)]
        candidate = model_class.from_sample(sample)
        if candidate is None or not accepts(candidate):
            continue
        valid += 1
        keep, score = _support(candidate, points, tolerance)
        if score > best_score:
            best, best_keep, best_score = candidate, keep, score
            if score == (len(points), 0.0):
                break
    if best is None:
        return None, tested, valid

    # Bound the refit and never exchange a stronger sample consensus for a weaker one.
    for _ in range(2):
        if cancelled():
            return None, tested, valid
        refined = best.refined(points[best_keep])
        if refined is None or not accepts(refined):
            break
        keep, score = _support(refined, points, tolerance)
        if score < best_score:
            break
        converged = np.array_equal(keep, best_keep)
        best, best_keep, best_score = refined, keep, score
        if converged:
            break
    return best.canonical(), tested, valid


class FitCloudModel(Tool3D):
    type_id = "3d.cloud_model_fit"
    title, subtype = "Fit point-cloud plane or sphere", "Point clouds"
    description = "Fit one metric plane or sphere robustly and separate supporting scan rows."
    payload_reference = "cloud"
    parallel_safe = preview_safe = process_inputs_readonly = True
    inputs = (Port("cloud", "cloud"),)
    outputs = (
        Port("model", "data", required=False),
        Port("inliers", "cloud", required=False),
        Port("outliers", "cloud", required=False),
        Port("inlier_indices", "data", required=False),
        Port("outlier_indices", "data", required=False),
        Port("signed_residuals_m", "data", required=False),
        Port("report", "data"),
    )
    defaults = {
        "model": "plane",
        "inlier_tolerance_m": 0.003,
        "minimum_inliers": 20,
        "minimum_inlier_fraction": 0.55,
        "minimum_sphere_radius_m": 0.0,
        "maximum_sphere_radius_m": 0.0,
        "maximum_fit_samples": 4096,
        "maximum_trials": 192,
        "random_seed": 0,
        "max_input_points": 250_000,
    }
    parameter_choices = {"model": ("plane", "sphere")}
    parameter_ranges = {
        "inlier_tolerance_m": NumericRange(1e-9, 1e12),
        "minimum_inliers": NumericRange(3, 1_000_000, whole=True),
        "minimum_inlier_fraction": NumericRange(0, 1),
        "minimum_sphere_radius_m": NumericRange(0, 1e12),
        "maximum_sphere_radius_m": NumericRange(0, 1e12),
        "maximum_fit_samples": NumericRange(3, 16_384, whole=True),
        "maximum_trials": NumericRange(1, 1024, whole=True),
        "random_seed": NumericRange(0, 2**32 - 1, whole=True),
        "max_input_points": NumericRange(3, 1_000_000, whole=True),
    }
    parameter_help = {
        "model": "Choose a plane or a sphere model.",
        "inlier_tolerance_m": "Inclusive perpendicular or radial distance to the surface.",
        "minimum_inliers": "Smallest accepted support count across all input rows.",
        "minimum_inlier_fraction": "Smallest accepted support fraction across all input rows.",
        "minimum_sphere_radius_m": "Smallest allowed sphere radius in meters.",
        "maximum_sphere_radius_m": "Largest allowed sphere radius in meters. Zero disables it.",
        "maximum_fit_samples": "Limit rows used for sample consensus and refitting.",
        "maximum_trials": "Maximum random model hypotheses to try.",
        "random_seed": "Seed for reproducible sample selection and hypothesis order.",
        "max_input_points": "Maximum rows admitted for full-cloud classification.",
    }
    input_help = {"cloud": "Unorganized XYZ in meters or millimeters in a single named frame."}
    output_help = {
        "model": "Plane [nx, ny, nz, d] or sphere [cx, cy, cz, r] with metric offsets.",
        "inliers": "Supporting rows in input order with original scan attributes.",
        "outliers": "Other rows in input order with original scan attributes.",
        "inlier_indices": "Original zero-based row indices of supporting points.",
        "outlier_indices": "Original zero-based row indices of non-supporting points.",
        "signed_residuals_m": "Per-source-row signed distance. Positive is outside a sphere.",
        "report": "Metric tolerance, sampled trials, full-cloud support and inlier errors.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["model"] != "sphere":
            return
        for field in ("minimum_inliers", "maximum_fit_samples", "max_input_points"):
            if p[field] < 4:
                raise ValidationError("Sphere fitting requires at least four points", field)
        lower = float(p["minimum_sphere_radius_m"])
        upper = float(p["maximum_sphere_radius_m"])
        if upper and lower > upper:
            raise ValidationError(
                "Minimum sphere radius must not exceed the maximum",
                "minimum_sphere_radius_m",
                "maximum_sphere_radius_m",
            )

    def process(self, inputs):
        cloud, p = inputs["cloud"], self.parameters
        if cloud.units not in METERS:
            raise ValidationError("Model fitting requires a cloud in m or mm", "cloud")
        n = len(cloud.points)
        if n > p["max_input_points"]:
            raise ValidationError("Cloud exceeds the point budget", "max_input_points")
        kind = p["model"]
        model_class, sample_size = (Plane, 3) if kind == "plane" else (Sphere, 4)
        tolerance = float(p["inlier_tolerance_m"])
        lower = float(p["minimum_sphere_radius_m"])
        upper = float(p["maximum_sphere_radius_m"]) or np.inf
        report = {
            "model": kind,
            "frame": cloud.frame,
            "cloud_units": cloud.units,
            "input_points": n,
            "inlier_tolerance_m": tolerance,
            "fit_samples": 0,
            "hypotheses_tested": 0,
            "valid_hypotheses": 0,
            "classified_points": 0,
            "inlier_count": 0,
            "inlier_fraction": 0.0,
        }
        outputs = {port.name: NO_VALUE for port in self.outputs}
        outputs["report"] = report

        def failure(reason, message):
            return Result(
                outputs,
                passed=False,
                score=report["inlier_fraction"],
                fail_reasons=[f"cloud_model_{reason}"],
                message=message,
            )

        if self._cancelled():
            return failure("cancelled", "Point-cloud model fitting cancelled")
        if n < sample_size:
            return failure(
                "insufficient_points", f"The {kind} requires at least {sample_size} rows"
            )

        points = cloud.points.astype(np.float64) * METERS[cloud.units]
        rng = np.random.default_rng(int(p["random_seed"]))
        limit = int(p["maximum_fit_samples"])
        sample = points if n <= limit else points[np.sort(rng.choice(n, limit, replace=False))]
        report["fit_samples"] = len(sample)

        def accepts(candidate):
            return kind == "plane" or 0 < candidate.radius and lower <= candidate.radius <= upper

        fitted, tested, valid = _fit_sample(
            sample,
            model_class,
            sample_size,
            tolerance,
            int(p["maximum_trials"]),
            rng,
            accepts,
            self._cancelled,
        )
        report.update(hypotheses_tested=tested, valid_hypotheses=valid)
        if self._cancelled():
            return failure("cancelled", "Point-cloud model fitting cancelled")
        if fitted is None:
            return failure("no_valid_fit", f"No valid {kind} fit within the hypothesis limits")

        residuals = fitted.residuals(points)
        keep = np.abs(residuals) <= tolerance
        errors = residuals[keep]
        count = len(errors)
        fraction = count / n
        if self._cancelled():
            return failure("cancelled", "Point-cloud model fitting cancelled")
        report.update(
            classified_points=n,
            inlier_count=count,
            inlier_fraction=fraction,
            inlier_rms_m=float(np.sqrt(np.mean(errors**2))) if count else None,
            maximum_inlier_error_m=float(np.abs(errors).max()) if count else None,
        )
        if count < p["minimum_inliers"] or fraction < float(p["minimum_inlier_fraction"]):
            return failure("insufficient_support", f"The {kind} did not meet the support limits")

        outputs.update(
            model={
                "schema": "point_cloud_model_v1",
                "kind": kind,
                "units": "m",
                "frame": cloud.frame,
                **fitted.record(),
            },
            inliers=cloud.subset(keep),
            outliers=cloud.subset(~keep),
        )
        if "inlier_indices" in self.exposed_outputs:
            outputs["inlier_indices"] = np.flatnonzero(keep).tolist()
        if "outlier_indices" in self.exposed_outputs:
            outputs["outlier_indices"] = np.flatnonzero(~keep).tolist()
        if "signed_residuals_m" in self.exposed_outputs:
            outputs["signed_residuals_m"] = residuals.tolist()
        graphics = {}
        if self.capture_graphics:
            colors = np.empty((n, 3), np.uint8)
            colors[:] = [235, 86, 74]
            colors[keep] = [99, 215, 135]
            preview = PointCloud(cloud.points, cloud.units, cloud.frame, colors=colors)
            graphics["fit support"] = render_cloud(preview)
        payload = self.input_payloads["cloud"]
        return Result(
            outputs,
            graphics=graphics,
            payloads={"inliers": payload, "outliers": payload},
            passed=True,
            score=fraction,
            message=f"Fitted {kind} with {count} of {n} rows",
        )
