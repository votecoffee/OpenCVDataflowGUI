"""Radius-density and statistical outlier rejection for metric point clouds."""

import numpy as np
from scipy.spatial import cKDTree

from .cloud_projection import METERS
from .core import Port, Result
from .geometry3d_common import Tool3D, cloud_graphics
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE

_RADIUS_BATCH = 8192
_NEIGHBOR_ELEMENTS = 262_144


def _radius_counts(tree, points, radius, cancelled):
    counts = np.empty(len(points), np.int32)
    for start in range(0, len(points), _RADIUS_BATCH):
        if cancelled():
            return None
        band = slice(start, start + _RADIUS_BATCH)
        counts[band] = tree.query_ball_point(points[band], radius, return_length=True) - 1
    return None if cancelled() else counts


def _mean_distances(tree, points, neighbors, cancelled):
    means = np.empty(len(points), np.float64)
    k = neighbors + 1
    block = _NEIGHBOR_ELEMENTS // k
    for start in range(0, len(points), block):
        if cancelled():
            return None
        band = slice(start, start + block)
        distances, _ = tree.query(points[band], k=k)
        # Drop one zero-distance entry for the query row. Other coincident rows count.
        means[band] = distances[:, 1:].mean(axis=1)
    return None if cancelled() else means


class RejectCloudOutliers(Tool3D):
    type_id = "3d.cloud_outlier"
    title, subtype = "Reject point-cloud outliers", "Point clouds"
    description = "Split a metric cloud by neighborhood density while retaining original rows."
    payload_reference = "cloud"
    parallel_safe = preview_safe = process_inputs_readonly = True
    inputs = (Port("cloud", "cloud"),)
    outputs = (
        Port("cloud", "cloud"),
        Port("removed_cloud", "cloud"),
        Port("kept_indices", "data", required=False),
        Port("removed_indices", "data", required=False),
        Port("report", "data"),
    )
    defaults = {
        "method": "radius",
        "radius_m": 0.01,
        "minimum_neighbors": 3,
        "statistical_neighbors": 12,
        "standard_deviations": 2.0,
        "max_input_points": 250_000,
    }
    parameter_choices = {"method": ("radius", "statistical")}
    parameter_ranges = {
        "radius_m": NumericRange(1e-9, 1e12),
        "minimum_neighbors": NumericRange(1, 1000000, whole=True),
        "statistical_neighbors": NumericRange(1, 256, whole=True),
        "standard_deviations": NumericRange(0, 100),
        "max_input_points": NumericRange(1, 1000000, whole=True),
    }
    parameter_help = {
        "method": "Filter by neighborhood count or mean distance to nearest neighbors.",
        "radius_m": "Inclusive radius in meters, independent of the cloud's stored units.",
        "minimum_neighbors": "Radius count required to retain a row. The query row is excluded.",
        "statistical_neighbors": "Requested nearest other rows, capped by the available points.",
        "standard_deviations": "Retain local means up to the global mean plus this multiple of SD.",
        "max_input_points": "Maximum cloud size admitted to the index.",
    }
    input_help = {
        "cloud": "XYZ in meters or millimeters. Coincident rows retain separate identities."
    }

    def process(self, inputs):
        cloud, p = inputs["cloud"], self.parameters
        if cloud.units not in METERS:
            raise ValidationError("Outlier rejection requires a cloud in m or mm")
        n = len(cloud.points)
        if n > int(p["max_input_points"]):
            raise ValidationError("Cloud exceeds the configured point budget", "max_input_points")
        method = p["method"]
        radius = float(p["radius_m"])
        minimum = int(p["minimum_neighbors"])
        requested = int(p["statistical_neighbors"])
        effective = min(requested, max(0, n - 1))
        deviations = float(p["standard_deviations"])
        report = {
            "method": method,
            "frame": cloud.frame,
            "cloud_units": cloud.units,
            "input_points": n,
        }
        if method == "radius":
            report.update(radius_m=radius, minimum_neighbors=minimum)
        else:
            report.update(
                requested_neighbors=requested,
                effective_neighbors=effective,
                standard_deviations=deviations,
            )

        keep = np.zeros(n, bool)
        classified, reason = True, ""
        if self._cancelled():
            classified, reason = False, "cloud_outlier_cancelled"
        elif not n:
            reason = "cloud_outlier_empty"
        elif method == "statistical" and not effective:
            classified, reason = False, "cloud_outlier_insufficient_neighbors"
        else:
            points = cloud.points.astype(np.float64) * METERS[cloud.units]
            tree = cKDTree(points)
            if method == "radius":
                counts = _radius_counts(tree, points, radius, self._cancelled)
                if counts is not None:
                    keep = counts >= minimum
                    report.update(
                        minimum_observed_neighbors=int(counts.min()),
                        maximum_observed_neighbors=int(counts.max()),
                    )
                classified = counts is not None
            else:
                means = _mean_distances(tree, points, effective, self._cancelled)
                if means is not None:
                    center = float(means.mean())
                    spread = float(means.std())
                    cutoff = center + deviations * spread
                    keep = means <= cutoff
                    report.update(
                        global_mean_m=center, population_sd_m=spread, maximum_local_mean_m=cutoff
                    )
                classified = means is not None
            if not classified:
                reason = "cloud_outlier_cancelled"

        kept = np.flatnonzero(keep)
        dropped = np.flatnonzero(~keep) if classified else np.empty(0, np.intp)
        filtered = cloud.subset(kept)
        removed = cloud.subset(dropped)
        report.update(
            classified_points=n if classified else 0,
            kept_points=len(kept),
            removed_points=len(dropped),
        )
        reason = reason or ("" if len(kept) else "cloud_outlier_no_inliers")
        payload = self.input_payloads["cloud"]
        return Result(
            {
                "cloud": filtered,
                "removed_cloud": removed,
                "kept_indices": kept.tolist()
                if "kept_indices" in self.exposed_outputs
                else NO_VALUE,
                "removed_indices": (
                    dropped.tolist() if "removed_indices" in self.exposed_outputs else NO_VALUE
                ),
                "report": report,
            },
            graphics=cloud_graphics(self, filtered),
            payloads={"cloud": payload, "removed_cloud": payload},
            passed=not reason,
            score=len(kept) / n if n else 0.0,
            fail_reasons=[reason] if reason else [],
            message=f"Retained {len(kept)} of {n} cloud points",
        )
