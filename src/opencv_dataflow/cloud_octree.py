"""Bounded sparse octree queries with original scan indices."""

from heapq import heappush, heapreplace

import numpy as np

from .cloud_projection import METERS
from .core import Port, Result
from .geometry3d_common import Tool3D, cloud_graphics
from .parameter_validation import NumericRange, ValidationError

MAX_LEVELS = 20
MAX_QUERIES = 128
LEAF_POINTS = 32
LEAF_BATCH = 4096
CELL_EDGES = np.arange(1, 8, dtype=np.int64)
COORDINATE_LIMIT_M = 1e12


def _squared_box_distance(query, origin, side, padding):
    return sum(
        max(low - padding - value, value - (low + side + padding), 0.0) ** 2
        for value, low in zip(query, origin)
    )


class SparseOctree:
    """Sorted three-bit Morton prefixes describe occupied cells without allocating a dense grid."""

    def __init__(self, points_m, indices, origin, side, levels):
        self.origin = tuple(float(value) for value in origin)
        self.side = float(side)
        self.levels = levels
        cells = np.floor((points_m - origin) / (side / (1 << levels))).astype(np.int64)
        np.clip(cells, 0, (1 << levels) - 1, out=cells)
        codes = np.zeros(len(points_m), np.int64)
        for bit in range(levels):
            for axis in range(3):
                codes |= ((cells[:, axis] >> bit) & 1) << (3 * bit + axis)
        order = np.argsort(codes, kind="stable")
        self.codes = codes[order]
        self.indices = indices[order]
        self.points = points_m[order]
        extent = max(abs(value) + self.side for value in self.origin)
        self.padding = float(np.spacing(extent)) * (levels + 4)

    @property
    def nbytes(self):
        return self.codes.nbytes + self.indices.nbytes + self.points.nbytes

    def _children(self, start, end, prefix, level, origin, side, query):
        stride = 1 << (3 * (level - 1))
        splits = np.searchsorted(self.codes[start:end], prefix + CELL_EDGES * stride)
        edges = (start, *(start + splits).tolist(), end)
        half = side / 2
        children = []
        for octant, (first, last) in enumerate(zip(edges, edges[1:])):
            if first == last:
                continue
            lower = tuple(origin[axis] + ((octant >> axis) & 1) * half for axis in range(3))
            bound = _squared_box_distance(query, lower, half, self.padding)
            children.append((bound, first, last, prefix + octant * stride, level - 1, lower, half))
        children.sort(key=lambda child: child[0], reverse=True)
        return children

    def _consider(self, start, end, query, radius_sq, limit, best):
        for first in range(start, end, LEAF_BATCH):
            batch = slice(first, min(first + LEAF_BATCH, end))
            ids = self.indices[batch]
            delta = self.points[batch] - query
            squares = np.einsum("ij,ij->i", delta, delta)
            cutoff = min(radius_sq, -best[0][0]) if len(best) == limit else radius_sq
            close = np.flatnonzero(squares <= cutoff)
            if len(close) > limit:
                order = np.lexsort((ids[close], squares[close]))[:limit]
                close = close[order]
            for i in close:
                entry = (-float(squares[i]), -int(ids[i]))
                if len(best) < limit:
                    heappush(best, entry)
                elif entry > best[0]:
                    heapreplace(best, entry)

    def query(self, point_m, limit, radius_m=None):
        if not len(self.codes):
            return [], False
        keep = min(len(self.codes), limit + (radius_m is not None))
        radius_sq = float("inf") if radius_m is None else radius_m**2
        query = tuple(float(value) for value in point_m)
        best = []
        root_bound = _squared_box_distance(query, self.origin, self.side, self.padding)
        stack = [(root_bound, 0, len(self.codes), 0, self.levels, self.origin, self.side)]
        while stack:
            bound, start, end, prefix, level, origin, side = stack.pop()
            cutoff = min(radius_sq, -best[0][0]) if len(best) == keep else radius_sq
            if bound > cutoff:
                continue
            if level == 0 or end - start <= LEAF_POINTS:
                self._consider(start, end, query, radius_sq, keep, best)
            else:
                stack.extend(self._children(start, end, prefix, level, origin, side, query))
        ranked = sorted((-square, -index) for square, index in best)
        return [(index, square**0.5) for square, index in ranked[:limit]], len(ranked) > limit


def _triples(value, limit, field, *, single=False):
    if not isinstance(value, (list, tuple)) or len(value) > (3 if single else limit):
        raise ValidationError(f"Provide up to {limit} finite numeric XYZ coordinates", field)
    try:
        rows = np.asarray(value)
        if single and rows.shape == (3,):
            rows = rows.reshape(1, 3)
        if rows.shape == (0,):
            return np.empty((0, 3), np.float64)
        if (
            rows.ndim == 2
            and rows.shape[1] == 3
            and len(rows) <= limit
            and rows.dtype.kind in "iuf"
        ):
            rows = rows.astype(np.float64)
            if np.isfinite(rows).all():
                return rows
    except ValueError, TypeError, OverflowError:
        pass
    raise ValidationError(f"Provide up to {limit} finite numeric XYZ coordinates", field)


def _fit_cube(points, empty_bounds, minimum_side):
    if not len(points):
        return empty_bounds
    origin = points.min(axis=0)
    side = max(float(np.ptp(points, axis=0).max()), minimum_side)
    return origin, side


class QueryCloudOctree(Tool3D):
    type_id = "3d.cloud_octree_query"
    title, subtype = "Query point-cloud octree", "Point clouds"
    description = "Find exact nearest points in a metric cube while retaining source row indices."
    payload_reference = "cloud"
    parallel_safe = preview_safe = process_inputs_readonly = True
    inputs = (
        Port("cloud", "cloud"),
        Port("queries", "data", required=False, required_when_connected=True),
    )
    outputs = (Port("neighbors", "cloud"), Port("count", "integer"), Port("report", "data"))
    defaults = {
        "query_points": [[0, 0, 0]],
        "query_units": "cloud",
        "search": "knn",
        "max_neighbors": 8,
        "radius_m": 0.1,
        "bounds": "fit",
        "minimum_corner_m": [0, 0, 0],
        "cube_side_m": 1.0,
        "subdivision": "depth",
        "max_depth": 10,
        "resolution_m": 0.01,
        "max_input_points": 250000,
    }
    parameter_choices = {
        "query_units": ("cloud", "m", "mm"),
        "search": ("knn", "radius"),
        "bounds": ("fit", "explicit"),
        "subdivision": ("depth", "resolution"),
    }
    parameter_ranges = {
        "max_neighbors": NumericRange(1, 256, whole=True),
        "radius_m": NumericRange(1e-9, 1e12),
        "cube_side_m": NumericRange(1e-9, 2e12),
        "max_depth": NumericRange(1, MAX_LEVELS, whole=True),
        "resolution_m": NumericRange(1e-9, 2e12),
        "max_input_points": NumericRange(1, 1000000, whole=True),
    }
    parameter_help = {
        "query_points": "XYZ requests in the cloud coordinate frame when queries is unwired.",
        "query_units": "Use the cloud units or explicitly interpret XYZ in meters or millimeters.",
        "search": "KNN or nearest matches inside an inclusive physical radius.",
        "max_neighbors": "Maximum per query. Radius results indicate truncation.",
        "radius_m": "Inclusive Euclidean distance in meters for radius queries.",
        "bounds": "Fit to the cloud or index only points inside the explicit cube, faces included.",
        "minimum_corner_m": "Minimum XYZ corner of the explicit cube in meters.",
        "cube_side_m": "Length of each side of the explicit metric cube.",
        "subdivision": "Stop subdivision at a configured depth or physical cell size.",
        "max_depth": "Deepest cell level in depth mode. The root cube is level zero.",
        "resolution_m": "Maximum cell side at deepest level in resolution mode.",
        "max_input_points": "Reject larger input clouds before building temporary index arrays.",
    }
    input_help = {
        "cloud": "Metric cloud. Neighbors keep their original row indices, including duplicates.",
        "queries": "List of XYZ rows in the cloud frame, or record with points, frame and units.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        _triples(p["query_points"], MAX_QUERIES, "query_points")
        corner = _triples(p["minimum_corner_m"], 1, "minimum_corner_m", single=True)
        if corner.shape != (1, 3) or np.any(np.abs(corner) > COORDINATE_LIMIT_M):
            raise ValidationError("Specify one metric XYZ minimum corner", "minimum_corner_m")

    def _queries(self, inputs, cloud):
        p = self.parameters
        raw = inputs.get("queries", p["query_points"])
        payload = self.input_payloads.get("queries", {})
        declared = payload.get("coordinate_frame", payload.get("optical_frame", cloud.frame))
        if declared != cloud.frame:
            raise ValidationError("Query points belong to a different coordinate frame")
        units = cloud.units if p["query_units"] == "cloud" else p["query_units"]
        if isinstance(raw, dict):
            if raw.get("frame") != cloud.frame or raw.get("units") not in METERS:
                raise ValidationError("Query record must name the cloud frame and metric units")
            units, raw = raw["units"], raw.get("points")
        queries = _triples(raw, MAX_QUERIES, "query_points") * METERS[units]
        if np.any(np.abs(queries) > COORDINATE_LIMIT_M):
            raise ValidationError("Query coordinates exceed the ±1e12 m limit")
        return queries

    def process(self, inputs):
        cloud, p = inputs["cloud"], self.parameters
        if cloud.units not in METERS:
            raise ValidationError("Octree queries require cloud units of m or mm")
        if len(cloud.points) > int(p["max_input_points"]):
            raise ValidationError("Cloud exceeds the configured point budget", "max_input_points")
        queries = self._queries(inputs, cloud)
        points = cloud.points.astype(np.float64) * METERS[cloud.units]
        if np.any(np.abs(points) > COORDINATE_LIMIT_M):
            raise ValidationError("Cloud coordinates exceed the ±1e12 m limit")
        bounds = np.asarray(p["minimum_corner_m"], np.float64).reshape(3), float(p["cube_side_m"])
        resolution = float(p["resolution_m"])
        if p["bounds"] == "fit":
            bounds = _fit_cube(
                points, bounds, resolution if p["subdivision"] == "resolution" else 1e-9
            )
        origin, side = bounds
        levels = int(p["max_depth"])
        if p["subdivision"] == "resolution":
            levels = max(0, int(np.ceil(np.log2(side / resolution))))
            if levels > MAX_LEVELS:
                raise ValidationError(
                    "Resolution requires more than 20 subdivision levels", "resolution_m"
                )
        indices = np.arange(len(points), dtype=np.int32)
        if p["bounds"] == "explicit":
            inside = np.all((points >= origin) & (points <= origin + side), axis=1)
            indices, points = indices[inside], points[inside]
        index = SparseOctree(points, indices, origin, side, levels)
        limit = int(p["max_neighbors"])
        radius = float(p["radius_m"]) if p["search"] == "radius" else None
        records = []
        union = set()
        for query in queries:
            matches, truncated = index.query(query, limit, radius)
            ids = [row for row, _ in matches]
            records.append(
                {
                    "xyz_m": query.tolist(),
                    "indices": ids,
                    "distances_m": [dist for _, dist in matches],
                    "truncated": truncated,
                }
            )
            union.update(ids)
        chosen = np.array(sorted(union), dtype=np.int32)
        neighbors = cloud.subset(chosen)
        count = len(chosen)
        report = {
            "frame": cloud.frame,
            "cloud_units": cloud.units,
            "search": p["search"],
            "radius_m": radius,
            "indexed_points": int(len(points)),
            "excluded_points": len(cloud.points) - int(len(points)),
            "minimum_corner_m": origin.tolist(),
            "cube_side_m": side,
            "cell_side_m": side / (1 << levels),
            "levels": levels,
            "index_bytes": int(index.nbytes),
            "neighbor_cloud_source_indices": chosen.tolist(),
            "queries": records,
        }
        return Result(
            {"neighbors": neighbors, "count": count, "report": report},
            graphics=cloud_graphics(self, neighbors),
            passed=bool(count),
            fail_reasons=[] if count else ["octree_no_neighbors"],
            message=f"{count} unique neighbors across {len(queries)} queries",
        )
