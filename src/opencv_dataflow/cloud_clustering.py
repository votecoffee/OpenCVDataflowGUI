"""Euclidean clustering of point clouds with scan-row provenance."""

import numpy as np
from scipy.spatial import cKDTree

from .cloud_neighborhoods import count_radius_pairs, labeled_cloud_preview, pair_components
from .cloud_projection import METERS
from .core import Port, Result
from .geometry3d_common import Tool3D
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE


def _measure_components(points, components, count):
    sizes = np.bincount(components, minlength=count)
    lower = np.full((count, 3), np.inf)
    upper = np.full((count, 3), -np.inf)
    first_rows = np.full(count, len(points), np.intp)
    np.minimum.at(lower, components, points)
    np.maximum.at(upper, components, points)
    np.minimum.at(first_rows, components, np.arange(len(points)))
    return sizes, lower, upper, first_rows


def _cluster_records(selected, labels, kept, sizes, lower, upper, extents):
    # Group rows once and keep the original order within each cluster.
    ordered_rows = kept[np.argsort(labels[kept], kind="stable")]
    boundaries = np.cumsum(sizes[selected])[:-1]
    return [
        {
            "label": label,
            "point_count": int(sizes[component]),
            "source_indices": rows.tolist(),
            "minimum_xyz_m": lower[component].tolist(),
            "maximum_xyz_m": upper[component].tolist(),
            "extent_xyz_m": extents[component].tolist(),
            "longest_extent_m": float(extents[component].max()),
        }
        for label, (component, rows) in enumerate(
            zip(selected, np.split(ordered_rows, boundaries), strict=True), start=1
        )
    ]


class ClusterPointCloud(Tool3D):
    type_id = "3d.cloud_cluster"
    title, subtype = "Cluster point cloud by distance", "Point clouds"
    description = "Group nearby scan rows into parts and filter groups by count and physical size."
    payload_reference = "cloud"
    parallel_safe = preview_safe = process_inputs_readonly = True
    inputs = (Port("cloud", "cloud"),)
    outputs = (
        Port("cloud", "cloud"),
        Port("rejected_cloud", "cloud"),
        Port("clusters", "data"),
        Port("labels", "data", required=False),
        Port("count", "integer"),
        Port("report", "data"),
    )
    defaults = {
        "radius_m": 0.01,
        "minimum_points": 3,
        "maximum_points": 0,
        "minimum_extent_m": 0.0,
        "maximum_extent_m": 0.0,
        "max_input_points": 250_000,
        "max_neighbor_pairs": 2_000_000,
        "max_clusters": 256,
    }
    parameter_ranges = {
        "radius_m": NumericRange(1e-9, 1e12),
        "minimum_points": NumericRange(1, 1_000_000, whole=True),
        "maximum_points": NumericRange(0, 1_000_000, whole=True),
        "minimum_extent_m": NumericRange(0, 1e12),
        "maximum_extent_m": NumericRange(0, 1e12),
        "max_input_points": NumericRange(1, 1_000_000, whole=True),
        "max_neighbor_pairs": NumericRange(1, 6_000_000, whole=True),
        "max_clusters": NumericRange(1, 4096, whole=True),
    }
    parameter_help = {
        "radius_m": "Inclusive XYZ distance in meters. Neighbor chains form one cluster.",
        "minimum_points": "Smallest accepted point count per cluster.",
        "maximum_points": "Largest accepted point count per cluster. Zero disables this limit.",
        "minimum_extent_m": "Smallest accepted longest XYZ axis extent in meters.",
        "maximum_extent_m": "Largest accepted longest XYZ axis extent. Zero disables this limit.",
        "max_input_points": "Maximum cloud size admitted to the spatial index.",
        "max_neighbor_pairs": "Maximum unique unordered pairs admitted to the radius graph.",
        "max_clusters": "Maximum accepted clusters per result. Exceeding this limit fails the run.",
    }
    input_help = {"cloud": "Metric XYZ in meters or millimeters. Original scan rows retain IDs."}
    output_help = {
        "cloud": "Accepted rows in source order, with original per-row attributes.",
        "rejected_cloud": "Rows rejected by the cluster filters, in source order.",
        "clusters": "Accepted clusters with bounds in meters and original zero-based row indices.",
        "labels": "One label per source row. Accepted labels start at one. Zero means rejected.",
        "count": "Number of accepted clusters.",
        "report": "Search radius, pair count, rejected-cluster counts and classified-point count.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        values = self.parameters if parameters is None else parameters
        for minimum, maximum in (
            ("minimum_points", "maximum_points"),
            ("minimum_extent_m", "maximum_extent_m"),
        ):
            limit = float(values[maximum])
            if limit and float(values[minimum]) > limit:
                raise ValidationError(
                    "Cluster minimum must not exceed its maximum", minimum, maximum
                )

    def process(self, inputs):
        cloud, p = inputs["cloud"], self.parameters
        if cloud.units not in METERS:
            raise ValidationError("Cloud clustering requires a cloud in m or mm")
        n = len(cloud.points)
        if n > p["max_input_points"]:
            raise ValidationError("Cloud exceeds the configured point budget", "max_input_points")
        radius = float(p["radius_m"])
        report = {
            "input_points": n,
            "frame": cloud.frame,
            "cloud_units": cloud.units,
            "radius_m": radius,
            "max_neighbor_pairs": p["max_neighbor_pairs"],
            "max_clusters": p["max_clusters"],
        }

        def failed(reason, message):
            return self._result(cloud, report, reason=reason, message=message)

        if self._cancelled():
            return failed("cloud_cluster_cancelled", "Cloud clustering cancelled")
        if not n:
            return self._result(
                cloud,
                report,
                labels=np.empty(0, np.int32),
                reason="cloud_cluster_empty",
                message="Cloud contains no points",
            )

        points = cloud.points.astype(np.float64) * METERS[cloud.units]
        tree = cKDTree(points)
        pair_count = count_radius_pairs(tree, radius)
        report["neighbor_pairs"] = pair_count
        if self._cancelled():
            return failed("cloud_cluster_cancelled", "Cloud clustering cancelled")
        if pair_count > p["max_neighbor_pairs"]:
            return failed("cloud_cluster_pair_budget", "Cloud exceeds the neighbor pair budget")

        pairs = tree.query_pairs(radius, output_type="ndarray")
        report["neighbor_pairs"] = len(pairs)
        if len(pairs) > p["max_neighbor_pairs"]:
            return failed("cloud_cluster_pair_budget", "Cloud exceeds the neighbor pair budget")
        if self._cancelled():
            return failed("cloud_cluster_cancelled", "Cloud clustering cancelled")

        count, components = pair_components(pairs, n)
        if self._cancelled():
            return failed("cloud_cluster_cancelled", "Cloud clustering cancelled")
        sizes, lower, upper, first_rows = _measure_components(points, components, count)
        extents = upper - lower
        longest = extents.max(axis=1)
        count_ok = (sizes >= p["minimum_points"]) & (sizes <= (p["maximum_points"] or n))
        minimum_extent, maximum_extent = (
            float(p[key]) for key in ("minimum_extent_m", "maximum_extent_m")
        )
        extent_ok = (longest >= minimum_extent) & (longest <= (maximum_extent or np.inf))
        selected = np.flatnonzero(count_ok & extent_ok)
        report.update(
            candidate_clusters=count,
            count_rejected_clusters=int(np.count_nonzero(~count_ok)),
            extent_rejected_clusters=int(np.count_nonzero(count_ok & ~extent_ok)),
            matching_clusters=len(selected),
        )
        if len(selected) > p["max_clusters"]:
            return failed(
                "cloud_cluster_result_budget", "Cloud exceeds the accepted cluster budget"
            )
        if self._cancelled():
            return failed("cloud_cluster_cancelled", "Cloud clustering cancelled")

        selected = selected[np.argsort(first_rows[selected])]
        label_by_component = np.zeros(count, np.int32)
        label_by_component[selected] = np.arange(1, len(selected) + 1, dtype=np.int32)
        labels = label_by_component[components]
        records = []
        if len(selected) and "clusters" in self.exposed_outputs:
            records = _cluster_records(
                selected, labels, np.flatnonzero(labels), sizes, lower, upper, extents
            )
        return self._result(
            cloud,
            report,
            labels=labels,
            clusters=records,
            count=len(selected),
            reason="" if len(selected) else "cloud_cluster_no_matches",
            message=f"Accepted {len(selected)} of {count} clusters",
        )

    def _result(self, cloud, report, *, labels=None, clusters=(), count=0, reason, message):
        kept = np.flatnonzero(labels) if labels is not None else np.empty(0, np.intp)
        rejected = np.flatnonzero(labels == 0) if labels is not None else np.empty(0, np.intp)
        accepted_cloud = cloud.subset(kept)
        graphics = {}
        if count and self.capture_graphics:
            graphics["clusters"] = labeled_cloud_preview(accepted_cloud, labels[kept], count)
        payload = self.input_payloads["cloud"]
        report.update(
            classified_points=len(labels) if labels is not None else 0,
            accepted_points=len(kept),
            rejected_points=len(rejected),
            accepted_clusters=count,
        )
        return Result(
            {
                "cloud": accepted_cloud,
                "rejected_cloud": cloud.subset(rejected),
                "clusters": list(clusters) if "clusters" in self.exposed_outputs else NO_VALUE,
                "labels": (
                    labels.tolist()
                    if labels is not None and "labels" in self.exposed_outputs
                    else NO_VALUE
                ),
                "count": count,
                "report": report,
            },
            graphics=graphics,
            payloads={"cloud": payload, "rejected_cloud": payload},
            passed=not reason,
            score=len(kept) / len(cloud.points) if len(cloud.points) else 0.0,
            fail_reasons=[reason] if reason else [],
            message=message,
        )
