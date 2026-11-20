"""Exact closest-point queries on triangle surfaces with a bounded AABB traversal."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

_LEAF_SIZE = 8
_QUERY_BATCH = 512
_BARY_EPS = 1e-10
_EDGE_CORNERS = ((1, 2), (2, 0), (0, 1))


def _normals(vectors):
    length = np.linalg.norm(vectors, axis=1)
    return np.divide(
        vectors, length[:, None], out=np.zeros_like(vectors), where=length[:, None] > 0
    )


def _triangle_geometry(vertices, faces):
    triangles = vertices[faces]
    a, b, c = triangles.transpose(1, 0, 2)
    ab, ac, bc = b - a, c - a, c - b
    maximum_side_squared = np.max(np.sum(np.stack((ab, ac, bc)) ** 2, axis=2), axis=0)
    vectors = np.cross(ab, ac)
    area = np.linalg.norm(vectors, axis=1)
    usable = area > maximum_side_squared * 1e-12
    return usable, triangles[usable], vectors[usable] / area[usable, None], area[usable]


@dataclass
class SurfaceFeatures:
    edge_open: np.ndarray
    edge_normal: np.ndarray
    vertex_open: np.ndarray
    vertex_normal: np.ndarray


def _features(vertex_ids, normals, cosine):
    vertex_count = int(vertex_ids.max()) + 1
    oriented = vertex_ids[:, _EDGE_CORNERS].reshape(-1, 2)
    edges, group, counts = np.unique(
        np.sort(oriented, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    rows = np.arange(len(oriented))
    first = np.full(len(edges), len(rows), np.intp)
    last = np.full(len(edges), -1, np.intp)
    np.minimum.at(first, group, rows)
    np.maximum.at(last, group, rows)
    normal_a, normal_b = normals[first // 3], normals[last // 3]
    open_edge = counts == 1
    orientation = oriented[:, 0] < oriented[:, 1]
    combined = _normals(normal_a + normal_b)
    reliable = (
        (counts == 2)
        & (orientation[first] != orientation[last])
        & (np.einsum("ij,ij->i", normal_a, normal_b) >= cosine)
        & np.any(combined, axis=1)
    )
    combined[~reliable] = 0

    # At a vertex require compatible normals and a consistently oriented, closed one-ring.
    # A flat triangulation vertex satisfies this even when its vertex ID was split at a seam.
    open_vertex = np.zeros(vertex_count, bool)
    open_vertex[edges[open_edge].ravel()] = True
    unreliable_vertex = np.zeros(vertex_count, bool)
    unreliable_vertex[edges[~reliable].ravel()] = True
    vertex_sum = np.zeros((vertex_count, 3), np.float64)
    np.add.at(vertex_sum, vertex_ids.ravel(), np.repeat(normals, 3, axis=0))
    vertex_normal = _normals(vertex_sum)
    compatible = np.einsum("fij,fj->fi", vertex_normal[vertex_ids], normals) >= cosine
    unreliable_vertex[vertex_ids[~compatible]] = True
    vertex_normal[unreliable_vertex] = 0
    return SurfaceFeatures(
        open_edge[group].reshape(-1, 3),
        combined[group].reshape(-1, 3, 3),
        open_vertex[vertex_ids],
        vertex_normal[vertex_ids],
    )


@dataclass
class BoxNode:
    low: np.ndarray
    high: np.ndarray
    rows: np.ndarray | None = None
    children: tuple = ()


def _build_tree(low, high, centers, rows):
    lower, upper = low[rows].min(axis=0), high[rows].max(axis=0)
    if len(rows) <= _LEAF_SIZE:
        return BoxNode(lower, upper, rows)
    axis = np.argmax(np.ptp(centers[rows], axis=0))
    middle = len(rows) // 2
    split = rows[np.argpartition(centers[rows, axis], middle)]
    return BoxNode(
        lower,
        upper,
        children=tuple(
            _build_tree(low, high, centers, part) for part in (split[:middle], split[middle:])
        ),
    )


def _closest_rows(points, triangles, normals, area):
    """Compute exact face/edge/vertex projections for every point/triangle pair."""
    a, b, c = triangles.transpose(1, 0, 2)
    ab, ac = b - a, c - a
    relative = points[:, None, :] - a
    signed = np.einsum("pfi,fi->pf", relative, normals)
    planar = relative - signed[:, :, None] * normals
    beta = np.einsum("pfi,fi->pf", np.cross(planar, ac), normals) / area
    gamma = np.einsum("pfi,fi->pf", np.cross(ab, planar), normals) / area
    bary = np.stack((1 - beta - gamma, beta, gamma), axis=2)
    inside = np.all(bary >= 0, axis=2)
    zeros = np.count_nonzero(bary <= _BARY_EPS, axis=2)
    feature = np.where(zeros == 0, -1, np.where(zeros == 1, bary.argmin(2), 3 + bary.argmax(2)))
    closest = planar + a
    squared = np.where(inside, signed**2, np.inf)
    for edge, (first, last) in enumerate(_EDGE_CORNERS):
        start = triangles[:, first]
        direction = triangles[:, last] - start
        offset = points[:, None, :] - start
        proportion = np.clip(
            np.einsum("pfi,fi->pf", offset, direction) / np.sum(direction**2, axis=1), 0, 1
        )
        delta = offset - proportion[:, :, None] * direction
        d2 = np.sum(delta**2, axis=2)
        take = ~inside & (d2 < squared)
        closest[take] = (start + proportion[:, :, None] * direction)[take]
        squared[take] = d2[take]
        edge_feature = np.where(
            proportion <= _BARY_EPS,
            3 + first,
            np.where(proportion >= 1 - _BARY_EPS, 3 + last, edge),
        )
        feature[take] = edge_feature[take]
    return squared, closest, feature


@dataclass
class SurfaceMatches:
    face: np.ndarray
    distance: np.ndarray
    closest: np.ndarray
    boundary: np.ndarray
    signed: np.ndarray
    ambiguous: np.ndarray


def _reserve(work, limit, field, count):
    if work["box_tests"] + work["triangle_tests"] + count > limit:
        work["budget_exceeded"] = True
        return False
    work[field] += count
    return True


class TriangleSurface:
    def __init__(self, vertices, faces, maximum_angle_deg):
        usable, self.triangles, self.normals, self.area = _triangle_geometry(vertices, faces)
        self.face_ids = np.flatnonzero(usable)
        self.root = None
        self.cosine = np.cos(np.deg2rad(maximum_angle_deg)) - 1e-12
        if len(self.face_ids):
            usable_faces = faces[usable]
            # Weld exact XYZ for incidence so OBJ attribute seams share physical edges.
            coordinates, ids = np.unique(
                vertices[usable_faces].reshape(-1, 3), axis=0, return_inverse=True
            )
            self.features = _features(ids.reshape(-1, 3), self.normals, self.cosine)
            low, high = self.triangles.min(axis=1), self.triangles.max(axis=1)
            self.root = _build_tree(low, high, (low + high) / 2, np.arange(len(usable_faces)))
            self.vertices = cKDTree(coordinates)

    def _process_leaf(self, batch, rows, node, radius2, best, face, location, feature, ambiguous):
        squared, points, corners = _closest_rows(
            batch[rows], self.triangles[node.rows], self.normals[node.rows], self.area[node.rows]
        )
        for column, candidate in enumerate(node.rows):
            d2 = squared[:, column]
            inside = d2 <= radius2
            exists = face[rows] >= 0
            tied = inside & exists & np.isclose(d2, best[rows], rtol=1e-12, atol=1e-24)
            if np.any(tied):
                local = np.flatnonzero(tied)
                same = (
                    np.linalg.norm(points[local, column] - location[rows[local]], axis=1) <= 1e-10
                )
                both_interior = (feature[rows[local]] < 0) & (corners[local, column] < 0)
                incompatible = (
                    np.einsum("ij,j->i", self.normals[face[rows[local]]], self.normals[candidate])
                    < self.cosine
                )
                ambiguous[rows[local[~same | (both_interior & incompatible)]]] = True
            winner = inside & (
                ~exists | (d2 < best[rows]) | ((d2 == best[rows]) & (candidate < face[rows]))
            )
            selected = rows[winner]
            ambiguous[selected] &= tied[winner]
            best[selected], face[selected] = d2[winner], candidate
            location[selected], feature[selected] = points[winner, column], corners[winner, column]

    def _nearest(self, batch, radius, budget, work, cancelled):
        n = len(batch)
        face = np.full(n, -1, np.int32)
        closest = np.full((n, 3), np.nan)
        feature = np.full(n, -1, np.int8)
        ambiguous = np.zeros(n, bool)
        if self.root is None:
            return face, closest, feature, ambiguous, np.full(n, np.nan)

        radius2 = radius**2
        nearest_vertex, _ = self.vertices.query(
            batch, distance_upper_bound=np.nextafter(radius, np.inf)
        )
        best = np.minimum(nearest_vertex**2, radius2)
        todo = [(self.root, np.arange(n))]
        while todo:
            if cancelled():
                return None
            node, rows = todo.pop()
            if not _reserve(work, budget, "box_tests", len(rows)):
                return None
            xyz = batch[rows]
            offset = np.maximum(np.maximum(node.low - xyz, xyz - node.high), 0)
            box_distance = np.sum(offset**2, axis=1)
            bound = best[rows]
            rows = rows[box_distance <= bound * (1 + 1e-12) + 1e-24]
            if not len(rows):
                continue
            if node.rows is None:
                todo.extend((child, rows) for child in reversed(node.children))
                continue
            if not _reserve(work, budget, "triangle_tests", len(rows) * len(node.rows)):
                return None
            self._process_leaf(batch, rows, node, radius2, best, face, closest, feature, ambiguous)
        return face, closest, feature, ambiguous, np.where(face >= 0, np.sqrt(best), np.nan)

    def _annotate(self, batch, face, closest, feature, ambiguous):
        boundary = np.zeros(len(batch), bool)
        signed = np.full(len(batch), np.nan)
        hit = np.flatnonzero(face >= 0)
        if not len(hit):
            return boundary, signed
        sides, triangles = feature[hit], face[hit]
        normals = self.normals[triangles]
        edge = (sides >= 0) & (sides < 3)
        boundary[hit[edge]] = self.features.edge_open[triangles[edge], sides[edge]]
        normals[edge] = self.features.edge_normal[triangles[edge], sides[edge]]
        vertex = sides >= 3
        boundary[hit[vertex]] = self.features.vertex_open[triangles[vertex], sides[vertex] - 3]
        normals[vertex] = self.features.vertex_normal[triangles[vertex], sides[vertex] - 3]
        valid = ~boundary[hit] & ~ambiguous[hit] & np.any(normals, axis=1)
        usable = hit[valid]
        signed[usable] = np.einsum("ij,ij->i", batch[usable] - closest[usable], normals[valid])
        return boundary, signed

    def query(self, points, radius, budget, cancelled):
        n = len(points)
        result = SurfaceMatches(
            np.full(n, -1, np.int32),
            np.full(n, np.nan),
            np.full((n, 3), np.nan),
            np.zeros(n, bool),
            np.full(n, np.nan),
            np.zeros(n, bool),
        )
        work = {"box_tests": 0, "triangle_tests": 0, "budget_exceeded": False}
        for start in range(0, n, _QUERY_BATCH):
            if cancelled():
                return None, work
            band = slice(start, start + _QUERY_BATCH)
            batch = points[band]
            nearest = self._nearest(batch, radius, budget, work, cancelled)
            if nearest is None:
                return None, work
            face, closest, feature, ambiguous, distance = nearest
            boundary, signed = self._annotate(batch, face, closest, feature, ambiguous)
            hit = face >= 0
            result.face[start + np.flatnonzero(hit)] = self.face_ids[face[hit]]
            result.distance[band], result.closest[band] = distance, closest
            result.boundary[band], result.signed[band], result.ambiguous[band] = (
                boundary,
                signed,
                ambiguous,
            )
        return result, work
