"""Inspect indexed triangle topology and certify the volume of a convex shell."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, QhullError

from .core import Port, Result
from .geometry3d_common import Tool3D
from .meshes import MAX_FACES, render_wireframe
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE

_SUPPORT_EPS = 1e-9
_EXAMPLES = 24


def _components(size, first, second):
    graph = coo_matrix((np.ones(len(first), bool), (first, second)), shape=(size, size)).tocsr()
    return connected_components(graph, directed=False)


def _edge_topology(faces, vertex_count):
    """Group undirected edges while retaining each face's directed corners."""
    start = faces.reshape(-1)
    stop = faces[:, (1, 2, 0)].reshape(-1)
    low = np.minimum(start, stop)
    high = np.maximum(start, stop)
    key = low.astype(np.int64) * vertex_count + high
    order = np.argsort(key, kind="stable")
    same = key[order[:-1]] == key[order[1:]]
    groups = np.r_[0, np.flatnonzero(~same) + 1]
    count = np.diff(groups, append=len(order))
    representative = order[groups]
    boundary = representative[count == 1]
    nonmanifold = representative[count > 2]
    double = groups[count == 2]
    conflict = order[double][
        (start[order[double]] < stop[order[double]])
        == (start[order[double + 1]] < stop[order[double + 1]])
    ]

    a, b = order[:-1][same], order[1:][same]
    face_components, _ = _components(len(faces), a // 3, b // 3)
    corners = np.arange(len(start))
    next_corner = (corners // 3) * 3 + (corners + 1) % 3
    forward = start < stop
    lower_corner = np.where(forward, corners, next_corner)
    upper_corner = np.where(forward, next_corner, corners)
    _, fan = _components(
        len(start),
        np.r_[lower_corner[a], upper_corner[a]],
        np.r_[lower_corner[b], upper_corner[b]],
    )
    first = np.full(vertex_count, len(start), dtype=np.intp)
    last = np.full(vertex_count, -1, dtype=np.intp)
    np.minimum.at(first, start, fan)
    np.maximum.at(last, start, fan)
    split_vertices = np.flatnonzero((last >= 0) & (first != last))

    def endpoints(rows):
        return np.column_stack((low[rows], high[rows]))

    boundary_edges = endpoints(boundary)
    loops, other = _boundary_components(boundary_edges)
    return {
        "unique_edges": len(count),
        "boundary_edges": len(boundary),
        "boundary_loops": loops,
        "branched_or_open_boundaries": other,
        "nonmanifold_edges": len(nonmanifold),
        "inconsistent_winding_edges": len(conflict),
        "split_vertex_fans": len(split_vertices),
        "face_components": face_components,
        "unused_vertices": int(np.count_nonzero(last < 0)),
        "boundary_edge_examples": boundary_edges[:_EXAMPLES].tolist(),
        "nonmanifold_edge_examples": endpoints(nonmanifold[:_EXAMPLES]).tolist(),
        "winding_edge_examples": endpoints(conflict[:_EXAMPLES]).tolist(),
        "split_vertex_examples": split_vertices[:_EXAMPLES].tolist(),
    }


def _boundary_components(edges):
    if not len(edges):
        return 0, 0
    vertices, positions = np.unique(edges, return_inverse=True)
    degree = np.bincount(positions.ravel())
    first, second = positions.reshape(-1, 2).T
    count, label = _components(len(vertices), first, second)
    other = len(np.unique(label[degree != 2]))
    return count - other, other


def _volume(vertices, faces, used, work_limit, cancelled):
    """Qualify one geometrically convex, oriented shell before publishing volume."""
    if len(faces) * len(used) > work_limit:
        return None, "support_budget"
    points = vertices[used]
    if len(np.unique(points, axis=0)) != len(points):
        return None, "coincident_vertices"
    extent = np.ptp(points, axis=0)
    span = float(extent.max())
    anchor = points.min(axis=0) + extent / 2
    normalized = (points - anchor) / span
    # Remap to the sorted referenced vertices without involving unused vertices.
    triangles = normalized[np.searchsorted(used, faces)]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    signed = float(np.einsum("ij,ij->i", triangles[:, 0], normals).sum() / 6)
    if not signed:
        return None, "zero_signed_volume"
    unit_normals = np.sign(signed) * normals / np.linalg.norm(normals, axis=1)[:, None]
    offsets = np.einsum("ij,ij->i", unit_normals, triangles[:, 0])
    block = max(1, min(128, 262_144 // len(points)))
    for start in range(0, len(faces), block):
        if cancelled():
            return None, "cancelled"
        band = slice(start, start + block)
        if np.any(unit_normals[band] @ normalized.T - offsets[band, None] > _SUPPORT_EPS):
            return None, "not_convex"
    try:
        hull = ConvexHull(normalized)
    except QhullError:
        return None, "unresolved_hull"
    if cancelled():
        return None, "cancelled"
    area = float(np.linalg.norm(normals, axis=1).sum() / 2)
    if not np.isclose(area, hull.area, atol=0, rtol=1e-8) or not np.isclose(
        abs(signed),
        hull.volume,
        atol=0,
        rtol=1e-8,
    ):
        return None, "hull_disagreement"
    return signed * span**3, "certified_convex_shell"


class InspectTriangleMesh(Tool3D):
    type_id = "3d.mesh_inspect"
    title, subtype = "Inspect triangle mesh topology", "Mesh geometry"
    description = "Measure area, check mesh topology, and qualify convex-shell volume."
    preview_safe = parallel_safe = process_inputs_readonly = True
    payload_reference = "mesh"
    inputs = (Port("mesh", "mesh"),)
    outputs = (
        Port("area", "number", required=False),
        Port("volume", "number", required=False),
        Port("report", "data"),
    )
    defaults = {
        "minimum_triangle_area": 0.0,
        "require_closed": False,
        "require_volume": False,
        "max_faces": 250_000,
        "max_support_tests": 10_000_000,
    }
    parameter_ranges = {
        "minimum_triangle_area": NumericRange(0, 1e30),
        "max_faces": NumericRange(1, MAX_FACES, whole=True),
        "max_support_tests": NumericRange(1, 250_000_000, whole=True),
    }
    parameter_help = {
        "minimum_triangle_area": "Flag triangles at or below this area in squared input units.",
        "require_closed": "Fail when any edge has only one incident face.",
        "require_volume": "Fail unless a single convex shell qualifies for volume.",
        "max_faces": "Maximum number of triangles admitted to the topology check.",
        "max_support_tests": "Limit on face-plane tests against referenced vertices for volume.",
    }
    input_help = {"mesh": "Indexed triangle mesh with a declared frame and length unit."}
    output_help = {
        "area": "Sum of input triangle areas in squared mesh units when indices are valid.",
        "volume": "Unsigned enclosed volume in cubed mesh units for a qualified convex shell.",
        "report": "Topology, diagnostic index samples, area, and volume qualification.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for field in ("require_closed", "require_volume"):
            if type(p[field]) is not bool:
                raise ValidationError("Use a checkbox value", field)

    def process(self, inputs):
        mesh, p = inputs["mesh"], self.parameters
        faces = mesh.faces
        if len(faces) > p["max_faces"]:
            raise ValidationError("Mesh exceeds the face budget", "max_faces")
        report = {
            "frame": mesh.frame,
            "length_units": mesh.units,
            "vertices": len(mesh.vertices),
            "faces": len(faces),
            "minimum_triangle_area": float(p["minimum_triangle_area"]),
        }
        outputs = {"area": NO_VALUE, "volume": NO_VALUE, "report": report}

        def result(reasons, graphics=None):
            report["status"] = "failed" if reasons else "ok"
            return Result(
                outputs,
                passed=not reasons,
                fail_reasons=[f"mesh_inspect_{name}" for name in reasons],
                graphics=graphics or {},
                message=", ".join(reason.replace("_", " ") for reason in reasons)
                if reasons
                else f"{len(faces):,} triangles · {report['volume_status'].replace('_', ' ')}",
            )

        if self._cancelled():
            report["volume_status"] = "cancelled"
            return result(["cancelled"])
        faces = faces.astype(np.intp, copy=False)
        vertices = mesh.vertices.astype(np.float64)
        triangles = vertices[faces]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        areas = np.linalg.norm(cross, axis=1) / 2
        degenerate = np.flatnonzero(areas <= p["minimum_triangle_area"])
        del triangles, cross
        duplicate = len(faces) - len(np.unique(np.sort(faces, axis=1), axis=0))
        topology = _edge_topology(faces, len(vertices))
        if self._cancelled():
            report["volume_status"] = "cancelled"
            return result(["cancelled"])
        area = float(areas.sum())
        outputs["area"] = area
        report.update(
            topology,
            area=area,
            degenerate_faces=len(degenerate),
            duplicate_faces=duplicate,
            degenerate_face_examples=degenerate[:_EXAMPLES].tolist(),
        )
        reasons = []
        for problem in (
            "degenerate_faces",
            "duplicate_faces",
            "nonmanifold_edges",
            "split_vertex_fans",
            "inconsistent_winding_edges",
        ):
            if report[problem]:
                reasons.append(problem)
        closed = topology["boundary_edges"] == 0
        report["closed"] = closed
        if reasons:
            signed, status = None, "topology_defects"
        elif not closed:
            signed, status = None, "open_boundary"
        elif topology["face_components"] != 1:
            signed, status = None, "multiple_shells"
        else:
            used = np.unique(faces)
            signed, status = _volume(
                vertices, faces, used, int(p["max_support_tests"]), self._cancelled
            )
        report.update(volume_status=status, volume=None, winding=None)
        if p["require_closed"] and not closed:
            reasons.append("boundary_edges")
        if signed is not None:
            outputs["volume"] = report["volume"] = abs(signed)
            report["winding"] = "outward" if signed > 0 else "inward"
        elif status == "cancelled":
            reasons.append("cancelled")
        elif p["require_volume"]:
            reasons.append("volume_unavailable")
        graphics = {}
        if self.capture_graphics:
            edges = (
                (topology["boundary_edge_examples"], (57, 147, 255)),
                (topology["winding_edge_examples"], (225, 95, 220)),
                (topology["nonmanifold_edge_examples"], (73, 75, 250)),
            )
            graphics["Topology"] = render_wireframe(mesh, highlight_edges=edges)
        return result(reasons, graphics)
