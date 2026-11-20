"""Estimate point-cloud normals and expose local PCA fit quality and provenance."""

import math

import numpy as np
from scipy.spatial import cKDTree

from .cloud_projection import METERS
from .core import Port, Result
from .geometry3d_common import Tool3D
from .local_surface_fit import fit_local_surfaces
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .pointclouds import PointCloud, render_cloud


def _viewpoint(values):
    try:
        point = np.asarray(values["viewpoint_m"], dtype=np.float64)
    except ValueError, TypeError, OverflowError:
        raise ValidationError("Viewpoint must be finite XYZ in meters", "viewpoint_m") from None
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValidationError("Viewpoint must be finite XYZ in meters", "viewpoint_m")
    return point


def _oriented_normals(normals, points, mode, viewpoint):
    # Resolve the sign deterministically before any viewpoint-based flipping.
    dominant_axis = np.argmax(np.abs(normals), axis=1)
    canonical = (
        normals * np.where(normals[np.arange(len(normals)), dominant_axis] < 0, -1, 1)[:, None]
    )
    if mode == "canonical":
        return canonical
    rays = viewpoint - points
    scale = np.max(np.abs(rays), axis=1)
    rays /= np.where(scale > 0, scale, 1)[:, None]
    facing = np.einsum("ij,ij->i", canonical, rays)
    if mode == "away_from_viewpoint":
        facing = -facing
    return canonical * np.where(facing < -1e-14, -1, 1)[:, None]


def _nullable(values):
    return [value if math.isfinite(value) else None for value in values.tolist()]


class MeasureCloudSurfaceQuality(Tool3D):
    type_id = "3d.cloud_surface_quality"
    title, subtype = "Measure point-cloud normals and curvature", "Point clouds"
    description = "Estimate oriented unit normals and identify unreliable local surface fits."
    payload_reference = "cloud"
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = (Port("cloud", "cloud"),)
    outputs = (
        Port("cloud", "cloud", required=False),
        Port("rejected_cloud", "cloud", required=False),
        Port("curvature", "data", required=False),
        Port("valid_indices", "data", required=False),
        Port("source_diagnostics", "data", required=False),
        Port("report", "data"),
    )
    defaults = {
        "radius_m": 0.015,
        "fit_neighbors": 24,
        "minimum_neighbors": 3,
        "minimum_normal_quality": 0.0,
        "orientation": "toward_viewpoint",
        "viewpoint_m": [0.0, 0.0, 0.0],
        "max_input_points": 250_000,
    }
    parameter_choices = {"orientation": ("toward_viewpoint", "away_from_viewpoint", "canonical")}
    parameter_ranges = {
        "radius_m": NumericRange(1e-9, 1e12),
        "fit_neighbors": NumericRange(3, 128, whole=True),
        "minimum_neighbors": NumericRange(3, 128, whole=True),
        "minimum_normal_quality": NumericRange(0, 1),
        "max_input_points": NumericRange(1, 1_000_000, whole=True),
    }
    parameter_help = {
        "radius_m": "Inclusive distance in meters for neighboring scan rows.",
        "fit_neighbors": "Maximum nearest rows used for each local fit, including the center.",
        "minimum_neighbors": "Minimum rows needed to attempt a local fit, including the center.",
        "minimum_normal_quality": "Minimum normalized eigenvalue gap of an accepted normal.",
        "orientation": "Choose viewpoint-facing, viewpoint-away or canonical normal signs.",
        "viewpoint_m": "XYZ viewpoint in meters in the input cloud's frame.",
        "max_input_points": "Maximum number of rows admitted to the spatial index.",
    }
    input_help = {
        "cloud": "Unorganized XYZ in meters or millimeters with a named coordinate frame."
    }
    output_help = {
        "cloud": "Accepted original rows with fitted unit normals, in scan order.",
        "rejected_cloud": "Rows with sparse, ambiguous or low-quality fits, in scan order.",
        "curvature": "Dimensionless PCA curvature for each row of the accepted cloud.",
        "valid_indices": "Original zero-based row IDs of the accepted cloud.",
        "source_diagnostics": "Neighbor counts, curvature and normal quality for source rows.",
        "report": "Thresholds, accepted and rejected point counts, and accepted curvature range.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        _viewpoint(p)
        if p["minimum_neighbors"] > p["fit_neighbors"]:
            raise ValidationError(
                "Minimum neighbors must not exceed the fit neighbor limit",
                "minimum_neighbors",
                "fit_neighbors",
            )

    def process(self, inputs):
        cloud, p = inputs["cloud"], self.parameters
        if cloud.units not in METERS:
            raise ValidationError("Surface quality requires a cloud in m or mm", "cloud")
        n = len(cloud.points)
        if n > p["max_input_points"]:
            raise ValidationError("Cloud exceeds the point budget", "max_input_points")
        radius = float(p["radius_m"])
        minimum_neighbors = int(p["minimum_neighbors"])
        minimum_quality = float(p["minimum_normal_quality"])
        orientation = p["orientation"]
        report = {
            "frame": cloud.frame,
            "cloud_units": cloud.units,
            "input_points": n,
            "classified_points": 0,
            "radius_m": radius,
            "fit_neighbors": int(p["fit_neighbors"]),
            "minimum_neighbors": minimum_neighbors,
            "minimum_normal_quality": minimum_quality,
            "orientation": orientation,
        }
        outputs = {port.name: NO_VALUE for port in self.outputs}
        outputs["report"] = report

        def failure(reason, message):
            return Result(
                outputs, passed=False, fail_reasons=[f"cloud_surface_{reason}"], message=message
            )

        if self._cancelled():
            return failure("cancelled", "Point-cloud surface measurement cancelled")
        if not n:
            return failure("empty", "Cloud contains no points")

        points = cloud.points.astype(np.float64) * METERS[cloud.units]
        fits = fit_local_surfaces(
            points,
            cKDTree(points),
            radius,
            int(p["fit_neighbors"]),
            self._cancelled,
            minimum_neighbors=minimum_neighbors,
        )
        if fits is None or self._cancelled():
            return failure("cancelled", "Point-cloud surface measurement cancelled")

        enough = fits.neighbor_counts >= minimum_neighbors
        unique = np.isfinite(fits.curvature)
        keep = unique & (fits.normal_quality >= minimum_quality)
        accepted = cloud.subset(keep)
        normals = _oriented_normals(fits.normals[keep], points[keep], orientation, _viewpoint(p))
        accepted.normals = np.ascontiguousarray(normals, np.float32)
        curvature = fits.curvature[keep]
        count = len(curvature)
        report.update(
            classified_points=n,
            accepted_points=count,
            rejected_points=n - count,
            insufficient_neighbors=int(np.count_nonzero(~enough)),
            ambiguous_normals=int(np.count_nonzero(enough & ~unique)),
            below_minimum_quality=int(np.count_nonzero(unique & ~keep)),
            minimum_curvature=float(curvature.min()) if count else None,
            maximum_curvature=float(curvature.max()) if count else None,
        )
        outputs.update(cloud=accepted, rejected_cloud=cloud.subset(~keep))
        if "curvature" in self.exposed_outputs:
            outputs["curvature"] = curvature.tolist()
        if "valid_indices" in self.exposed_outputs:
            outputs["valid_indices"] = np.flatnonzero(keep).tolist()
        if "source_diagnostics" in self.exposed_outputs:
            outputs["source_diagnostics"] = {
                "neighbor_counts": fits.neighbor_counts.tolist(),
                "curvature": _nullable(fits.curvature),
                "normal_quality": _nullable(fits.normal_quality),
            }
        graphics = {}
        if self.capture_graphics:
            colors = np.empty((n, 3), np.uint8)
            colors[:] = [225, 75, 65]
            if count:
                span = float(curvature.max()) or 1.0
                colors[keep] = np.rint(
                    np.array([45, 130, 240]) + (curvature / span)[:, None] * [200, 75, -180]
                ).astype(np.uint8)
            preview = PointCloud(cloud.points, cloud.units, cloud.frame, colors=colors)
            graphics["surface quality"] = render_cloud(preview)
        payload = self.input_payloads["cloud"]
        return Result(
            outputs,
            graphics=graphics,
            payloads={"cloud": payload, "rejected_cloud": payload},
            passed=bool(count),
            score=count / n,
            fail_reasons=[] if count else ["cloud_surface_no_valid_fits"],
            message=f"Accepted {count} of {n} local surface fits",
        )
