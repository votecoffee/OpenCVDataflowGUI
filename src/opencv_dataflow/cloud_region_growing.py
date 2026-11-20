"""Split neighboring cloud surfaces by normal direction and local curvature."""

import numpy as np
from scipy.spatial import cKDTree

from .cloud_neighborhoods import count_radius_pairs, labeled_cloud_preview, pair_components
from .cloud_projection import METERS
from .core import Port, Result
from .geometry3d_common import Tool3D
from .local_surface_fit import fit_local_surfaces
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE


def _read_curvatures(value, size):
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValidationError("Curvature must provide one value per source row", "curvature")
    try:
        values = np.asarray(value, np.float64)
    except ValueError, TypeError, OverflowError:
        raise ValidationError(
            "Curvature must contain values from 0 to 1/3 or null", "curvature"
        ) from None
    if values.shape != (size,):
        raise ValidationError("Curvature must provide one value per source row", "curvature")
    # NaN is the array representation of a missing (null) curvature.
    if np.any(np.isinf(values) | (values < 0) | (values > 1 / 3)):
        raise ValidationError("Curvature must contain values from 0 to 1/3 or null", "curvature")
    return values


def _smooth_pairs(pairs, normals, eligible, cosine, cancelled):
    keep = np.empty(len(pairs), bool)
    for start in range(0, len(pairs), 65_536):
        if cancelled():
            return None
        band = slice(start, start + 65_536)
        left, right = pairs[band].T
        dot = np.einsum("ij,ij->i", normals[left], normals[right])
        # Unsigned comparisons tolerate opposite normal signs on the same surface.
        keep[band] = eligible[left] & eligible[right] & (np.abs(dot) >= cosine - 1e-12)
    return pairs[keep]


class GrowCloudRegions(Tool3D):
    type_id = "3d.cloud_region_grow"
    title, subtype = "Grow point-cloud surface regions", "Point clouds"
    description = "Segment touching surfaces using local normal angles and curvature."
    payload_reference = "cloud"
    parallel_safe = preview_safe = process_inputs_readonly = True
    inputs = (Port("cloud", "cloud"), Port("curvature", "data", required=False))
    outputs = (
        Port("cloud", "cloud"),
        Port("rejected_cloud", "cloud"),
        Port("regions", "data"),
        Port("labels", "data", required=False),
        Port("curvature", "data", required=False),
        Port("count", "integer"),
        Port("report", "data"),
    )
    defaults = {
        "radius_m": 0.01,
        "normal_angle_deg": 20.0,
        "maximum_curvature": 0.05,
        "fit_neighbors": 24,
        "minimum_points": 3,
        "max_input_points": 250_000,
        "max_neighbor_pairs": 2_000_000,
        "max_regions": 256,
    }
    parameter_ranges = {
        "radius_m": NumericRange(1e-9, 1e12),
        "normal_angle_deg": NumericRange(0, 90),
        "maximum_curvature": NumericRange(0, 1 / 3),
        "fit_neighbors": NumericRange(3, 128, whole=True),
        "minimum_points": NumericRange(1, 1_000_000, whole=True),
        "max_input_points": NumericRange(1, 1_000_000, whole=True),
        "max_neighbor_pairs": NumericRange(1, 6_000_000, whole=True),
        "max_regions": NumericRange(1, 4096, whole=True),
    }
    parameter_help = {
        "radius_m": "Maximum XYZ distance in meters for surface fits and region links.",
        "normal_angle_deg": "Maximum unsigned angle between linked normals, in degrees.",
        "maximum_curvature": "Maximum smallest-to-total variance ratio for a region member.",
        "fit_neighbors": "Maximum nearest rows per local fit, including the center.",
        "minimum_points": "Smallest accepted region by row count.",
        "max_input_points": "Maximum cloud size admitted to the spatial index.",
        "max_neighbor_pairs": "Maximum unique unordered pairs admitted to the radius graph.",
        "max_regions": "Maximum accepted regions per result. Exceeding this limit fails the run.",
    }
    input_help = {
        "cloud": "Metric XYZ. Uses stored normals when present or estimates them from neighbors.",
        "curvature": "Optional source-order ratios from 0 to 1/3. Null marks an unusable row.",
    }
    output_help = {
        "cloud": "Accepted source rows with their original attributes and attached unit normals.",
        "rejected_cloud": "Unassigned rows with original attributes.",
        "regions": "Accepted region records with original zero-based source row indices.",
        "labels": "One label per source row. Zero means unassigned.",
        "curvature": "Source-order supplied or estimated curvature. Null denotes an invalid fit.",
        "count": "Number of accepted regions.",
        "report": "Geometry sources, pair and row counts, and size filter results.",
    }

    def process(self, inputs):
        cloud, p = inputs["cloud"], self.parameters
        if cloud.units not in METERS:
            raise ValidationError("Surface region growing requires a cloud in m or mm", "cloud")
        n = len(cloud.points)
        if n > p["max_input_points"]:
            raise ValidationError("Cloud exceeds the configured point budget", "max_input_points")
        radius = float(p["radius_m"])
        angle = float(p["normal_angle_deg"])
        maximum_curvature = float(p["maximum_curvature"])
        given_curvature = inputs.get("curvature", NO_VALUE)
        supplied_curvature = given_curvature is not NO_VALUE and given_curvature is not None
        report = {
            "frame": cloud.frame,
            "cloud_units": cloud.units,
            "input_points": n,
            "radius_m": radius,
            "normal_angle_deg": angle,
            "maximum_curvature": maximum_curvature,
            "normal_source": "cloud" if cloud.normals is not None else "local_fit",
            "curvature_source": "input" if supplied_curvature else "local_fit",
        }

        def failed(code, message):
            return self._result(cloud, report, reason=f"cloud_region_{code}", message=message)

        if self._cancelled():
            return failed("cancelled", "Surface region growing cancelled")
        if not n:
            return failed("empty", "Cloud contains no points")

        points = cloud.points.astype(np.float64) * METERS[cloud.units]
        tree = cKDTree(points)
        pair_count = count_radius_pairs(tree, radius)
        report["neighbor_pairs"] = pair_count
        if self._cancelled():
            return failed("cancelled", "Surface region growing cancelled")
        if pair_count > p["max_neighbor_pairs"]:
            return failed("pair_budget", "Cloud exceeds the neighbor pair budget")
        pairs = tree.query_pairs(radius, output_type="ndarray")
        report["neighbor_pairs"] = len(pairs)
        if len(pairs) > p["max_neighbor_pairs"]:
            return failed("pair_budget", "Cloud exceeds the neighbor pair budget")
        if self._cancelled():
            return failed("cancelled", "Surface region growing cancelled")

        curvature = _read_curvatures(given_curvature, n) if supplied_curvature else None
        normals = cloud.normals
        if curvature is None or normals is None:
            fitted = fit_local_surfaces(
                points, tree, radius, int(p["fit_neighbors"]), self._cancelled
            )
            if fitted is None or self._cancelled():
                return failed("cancelled", "Surface region growing cancelled")
            if normals is None:
                normals = fitted.normals
            if curvature is None:
                curvature = fitted.curvature
        lengths = np.linalg.norm(normals.astype(np.float64), axis=1)
        valid = np.isfinite(curvature) & (lengths > 0)
        normals = normals / np.where(lengths > 0, lengths, 1)[:, None]
        eligible = valid & (curvature <= maximum_curvature)
        report.update(
            invalid_geometry_points=n - int(np.count_nonzero(valid)),
            high_curvature_points=int(np.count_nonzero(valid & ~eligible)),
        )
        links = _smooth_pairs(pairs, normals, eligible, np.cos(np.deg2rad(angle)), self._cancelled)
        if links is None or self._cancelled():
            return failed("cancelled", "Surface region growing cancelled")

        count, components = pair_components(links, n)
        sizes = np.bincount(components[eligible], minlength=count)
        selected = np.flatnonzero(sizes >= p["minimum_points"])
        report.update(
            smooth_pairs=len(links),
            candidate_regions=int(np.count_nonzero(sizes)),
            matching_regions=len(selected),
        )
        if self._cancelled():
            return failed("cancelled", "Surface region growing cancelled")
        if len(selected) > p["max_regions"]:
            return failed("result_budget", "Cloud exceeds the accepted region budget")

        first = np.full(count, n, np.intp)
        np.minimum.at(first, components[eligible], np.flatnonzero(eligible))
        selected = selected[np.argsort(first[selected])]
        label_by_component = np.zeros(count, np.int32)
        label_by_component[selected] = np.arange(1, len(selected) + 1, dtype=np.int32)
        labels = label_by_component[components]
        return self._result(
            cloud,
            report,
            labels=labels,
            normals=normals,
            curvature=curvature,
            sizes=sizes[selected],
            reason="" if len(selected) else "cloud_region_no_matches",
            message=f"Accepted {len(selected)} surface regions",
        )

    def _result(
        self, cloud, report, *, labels=None, normals=None, curvature=None, sizes=(), reason, message
    ):
        empty = np.empty(0, np.intp)
        kept = np.flatnonzero(labels) if labels is not None else empty
        unassigned = np.flatnonzero(labels == 0) if labels is not None else empty
        accepted = cloud.subset(kept)
        if normals is not None and accepted.normals is None:
            accepted.normals = np.ascontiguousarray(normals[kept], np.float32)
        count = len(sizes)
        graphics = {}
        if count and self.capture_graphics:
            graphics["regions"] = labeled_cloud_preview(accepted, labels[kept], count)
        records = []
        if count and "regions" in self.exposed_outputs:
            ordered = kept[np.argsort(labels[kept], kind="stable")]
            groups = np.split(ordered, np.cumsum(sizes)[:-1])
            records = [
                {"label": label, "point_count": len(rows), "source_indices": rows.tolist()}
                for label, rows in enumerate(groups, start=1)
            ]
        curvatures = NO_VALUE
        if curvature is not None and "curvature" in self.exposed_outputs:
            curvatures = [value if np.isfinite(value) else None for value in curvature.tolist()]
        report.update(
            classified_points=len(labels) if labels is not None else 0,
            accepted_points=len(kept),
            unassigned_points=len(unassigned),
            accepted_regions=count,
        )
        payload = self.input_payloads["cloud"]
        return Result(
            {
                "cloud": accepted,
                "rejected_cloud": cloud.subset(unassigned),
                "regions": records if "regions" in self.exposed_outputs else NO_VALUE,
                "labels": (
                    labels.tolist()
                    if labels is not None and "labels" in self.exposed_outputs
                    else NO_VALUE
                ),
                "curvature": curvatures,
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
