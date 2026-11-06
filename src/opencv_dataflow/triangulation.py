"""Build Delaunay connectivity from 2D points or measured object centers."""

import math
from itertools import combinations

import cv2
import numpy as np

from .collection_tools import read_collection
from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import ValidationError
from .result_graphics import ACCEPTED


def _triangulate(sites):
    low, high = sites.min(axis=0), sites.max(axis=0)
    extent = float(np.max(high - low))
    if extent < 1e-6:
        raise ValidationError("Sites must span at least 0.000001 pixels")
    origin = (low + high) / 2
    scale = math.ldexp(1.0, 10 - math.frexp(extent)[1])
    local = ((sites - origin) * scale).astype(np.float32)
    lookup = {tuple(point): index for index, point in enumerate(local)}
    if len(lookup) != len(sites):
        raise ValidationError("Distinct sites coincide at OpenCV coordinate precision")
    hull_area = cv2.contourArea(cv2.convexHull(local))
    if hull_area <= 0:
        raise ValidationError("Provide at least three non-collinear sites")

    # Keep virtual vertices far from the local sites (within ±512) so they
    # do not replace faces with shallow angles near the hull.
    radius = 1 << 25
    subdivision = cv2.Subdiv2D((-radius, -radius, 2 * radius, 2 * radius))
    vertex_ids = {subdivision.insert(tuple(map(float, point))) for point in local}
    if len(vertex_ids) != len(sites):
        raise ValidationError("OpenCV could not distinguish all sites")
    faces = np.asarray(subdivision.getTriangleList()).reshape(-1, 3, 2)
    triangles = sorted({tuple(sorted(lookup[tuple(point)] for point in face)) for face in faces})
    edges = sorted({edge for triangle in triangles for edge in combinations(triangle, 2)})

    # Subdiv2D can leave faces attached to virtual vertices for nearly collinear inputs.
    if (
        len({index for triangle in triangles for index in triangle}) != len(sites)
        or len(sites) - len(edges) + len(triangles) != 1
    ):
        raise ValidationError("OpenCV did not produce a complete triangulation")
    corners = local[np.asarray(triangles)].astype(np.float64)
    ab, ac = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    areas = np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]) / 2
    if np.any(areas <= 0) or not math.isclose(float(areas.sum()), hull_area, rel_tol=1e-9):
        raise ValidationError("OpenCV did not produce a complete triangulation")
    return triangles, edges


class DelaunayTriangulation(Block):
    type_id, title = "inspect.delaunay_triangulation", "Delaunay triangulation"
    category, block_type, subtype = "Inspection", "Inspection", "Measurements"
    inputs = (
        Port("points", "data", required=False, required_when_connected=True),
        Port("collection", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("points", "data"),
        Port("source_indices", "data"),
        Port("triangles", "data"),
        Port("edges", "data"),
        Port("edge_lengths", "data"),
        Port("triangle_count", "number"),
    )
    process_inputs_readonly = True

    def process(self, inputs):
        if len(inputs) != 1:
            raise ValidationError("Connect either points or a detected collection")
        source = next(iter(inputs))
        if source == "collection":
            data, payload = read_collection(self, inputs)
            points = [
                [item["metrics"].get("center_x"), item["metrics"].get("center_y")]
                for item in data["items"]
            ]
        else:
            points = inputs["points"]
            payload = self.input_payloads.get("points", {})
        try:
            points = np.asarray(points, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError("Sites must be numeric [x, y] pairs") from exc
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or not 3 <= len(points) <= 10_000
            or not np.isfinite(points).all()
            or np.max(np.abs(points)) > 10_000_000
        ):
            raise ValidationError("Provide 3 to 10,000 finite [x, y] sites within ±10 million")
        _, first = np.unique(points, axis=0, return_index=True)
        first.sort()
        sites = points[first]
        if len(sites) < 3:
            raise ValidationError("Provide at least three distinct sites")

        triangles, edges = _triangulate(sites)
        ends = sites[np.asarray(edges)]
        outputs = {
            "points": sites.tolist(),
            "source_indices": first.tolist(),
            "triangles": [list(triangle) for triangle in triangles],
            "edges": [list(edge) for edge in edges],
            "edge_lengths": np.linalg.norm(ends[:, 1] - ends[:, 0], axis=1).tolist(),
            "triangle_count": len(triangles),
        }
        drawings = {}
        if self.generate_graphics:
            objects = tuple(graphic_object("polyline", edge, ACCEPTED) for edge in ends)
            drawings["Delaunay edges"] = GraphicDrawing(objects, source)
        return Result(
            outputs,
            payloads=dict.fromkeys(outputs, payload),
            drawings=drawings,
            message=f"{len(sites)} sites, {len(triangles)} triangles, {len(edges)} edges",
        )
