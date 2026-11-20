"""Bounded endpoint edge contraction for indexed review meshes."""

from collections import defaultdict

import numpy as np

from .core import Port, Result
from .geometry3d_common import Tool3D
from .mesh_inspection import _edge_topology
from .meshes import TriangleMesh, render_wireframe
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE

_MAX_VERTEX_FACES = 128
_TOPOLOGY_DEFECTS = (
    "nonmanifold_edges",
    "inconsistent_winding_edges",
    "split_vertex_fans",
)


def _indexed_edges(faces, vertex_count):
    other = faces[:, (1, 2, 0)]
    lo = np.minimum(faces, other).astype(np.int64)
    hi = np.maximum(faces, other)
    return np.unique(lo * vertex_count + hi, return_counts=True)


def _face_normals(points, faces):
    triangles = points[faces]
    return np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])


class _EdgeContractions:
    def __init__(self, points, faces, boundary, normals, deviation_limit, cosine_limit):
        self.points, self.faces = points, faces.copy()
        self.boundary = boundary
        self.normals = normals
        self.limit, self.cosine = deviation_limit, cosine_limit
        self.active = np.ones(len(faces), dtype=bool)
        self.count = len(faces)
        self.parents = np.arange(len(points), dtype=np.intp)
        self.radii = np.zeros(len(points))
        self.incident = defaultdict(set)
        for row, face in enumerate(faces.tolist()):
            for vertex in face:
                self.incident[vertex].add(row)

    def candidates(self):
        size = len(self.points)
        keys, counts = _indexed_edges(self.faces[self.active], size)
        keys = keys[counts == 2]
        left, right = keys // size, keys % size
        allowed = ~(self.boundary[left] & self.boundary[right])
        left, right = left[allowed], right[allowed]
        distances = np.linalg.norm(self.points[left] - self.points[right], axis=1)
        order = np.argsort(distances, kind="stable")
        return zip(left[order].tolist(), right[order].tolist(), distances[order].tolist())

    def try_contract(self, left, right, distance):
        first, second = self.incident.get(left), self.incident.get(right)
        if not first or not second or max(len(first), len(second)) > _MAX_VERTEX_FACES:
            return False
        removed = first & second
        if len(removed) != 2:
            return False

        # The links may meet only at the two vertices opposite the collapsing edge.
        opposites = set(self.faces[sorted(removed)].ravel()) - {left, right}
        first_link = set(self.faces[sorted(first)].ravel()) - {left, right}
        second_link = set(self.faces[sorted(second)].ravel()) - {left, right}
        if first_link & second_link != opposites:
            return False

        first_survivors, second_survivors = sorted(first - removed), sorted(second - removed)
        directions = (
            (left, right, second_survivors, first_survivors),
            (right, left, first_survivors, second_survivors),
        )
        for keep, discard, edited, fixed in sorted(
            directions, key=lambda path: self.radii[path[1]]
        ):
            reach = self.radii[discard] + distance
            if self.boundary[discard] or reach > self.limit:
                continue
            updated = self.faces[edited].copy()
            updated[updated == discard] = keep
            # In a tetrahedron the vertex-link test alone allows duplicate faces.
            sorted_fixed = {tuple(sorted(face)) for face in self.faces[fixed].tolist()}
            if any(tuple(sorted(face)) in sorted_fixed for face in updated.tolist()):
                continue
            if not self._keeps_winding(edited, updated):
                continue
            self._commit(keep, discard, removed, edited, updated, reach)
            return True
        return False

    def _keeps_winding(self, edited, updated):
        normals = _face_normals(self.points, updated)
        lengths = np.linalg.norm(normals, axis=1)
        original = self.normals[edited]
        signed = np.einsum("ij,ij->i", normals, original)
        return bool(np.all((lengths > 0) & (signed >= lengths * (self.cosine - 1e-12))))

    def _commit(self, keep, discard, removed, edited, updated, reach):
        for row in sorted(removed):
            for vertex in self.faces[row]:
                self.incident[vertex].remove(row)
            self.active[row] = False
        self.faces[edited] = updated
        self.incident[keep].update(edited)
        del self.incident[discard]
        self.count -= 2
        self.parents[discard] = keep
        self.radii[keep] = max(self.radii[keep], reach)

    def run(self, target, check_limit, cancelled):
        checks = 0
        while self.count > target and self.count > 2:
            before = self.count
            if cancelled():
                return "cancelled", checks
            for left, right, distance in self.candidates():
                if checks >= check_limit:
                    return "edge_check_limit", checks
                if checks % 128 == 0 and cancelled():
                    return "cancelled", checks
                checks += 1
                if self.try_contract(left, right, distance) and self.count <= target:
                    return "target_reached", checks
                if self.count <= 2:
                    break
            if self.count == before:
                break
        return ("target_reached" if self.count <= target else "constraints"), checks

    def qualify(self, original_faces, used, boundary_edges, original_topology):
        faces = self.faces[self.active]
        parents = self.parents
        while not np.array_equal(parents[parents], parents):
            parents = parents[parents]
        retained = np.unique(faces)
        mapped = parents[original_faces]
        collapsed = mapped[~self.active]
        keys, counts = _indexed_edges(faces, len(parents))
        mapped_edges, _ = _indexed_edges(mapped, len(parents))
        mapped_edges = mapped_edges[mapped_edges // len(parents) != mapped_edges % len(parents)]
        topology = _edge_topology(faces, len(parents))
        euler_before = len(used) - original_topology["unique_edges"] + len(original_faces)
        euler_after = len(retained) - topology["unique_edges"] + len(faces)
        if (
            not np.array_equal(mapped[self.active], faces)
            or not np.any(collapsed == collapsed[:, (1, 2, 0)], axis=1).all()
            or not np.isin(parents[used], retained).all()
            or len(np.setdiff1d(mapped_edges, keys, assume_unique=True))
            or not np.array_equal(keys[counts == 1], boundary_edges)
            or any(topology[defect] for defect in _TOPOLOGY_DEFECTS)
            or topology["face_components"] != original_topology["face_components"]
            or euler_before != euler_after
        ):
            raise RuntimeError("Mesh simplification failed its topology and correspondence checks")
        displacements = np.linalg.norm(self.points[used] - self.points[parents[used]], axis=1)
        # Barycentric interpolation of the vertex map bounds displacement over each face.
        # Every final triangle also has a surviving source face as its preimage.
        maximum = float(displacements.max())
        if maximum > self.limit * (1 + 1e-12):
            raise RuntimeError("Mesh simplification exceeded the configured deviation")
        return faces, retained, maximum, float(np.sqrt(np.mean(displacements**2)))


class SimplifyTriangleMesh(Tool3D):
    type_id = "3d.mesh_simplify"
    title, subtype = "Simplify triangle mesh for review", "Mesh geometry"
    description = "Reduce face count within a geometry budget while keeping indexed boundaries."
    preview_safe = parallel_safe = process_inputs_readonly = True
    payload_reference = "mesh"
    inputs = (Port("mesh", "mesh"),)
    outputs = (Port("mesh", "mesh", required=False), Port("report", "data"))
    defaults = {
        "target_faces": 2000,
        "deviation_mode": "Bounds fraction",
        "max_deviation": 0.05,
        "max_normal_change_deg": 45.0,
        "max_input_faces": 50_000,
        "max_edge_checks": 200_000,
    }
    parameter_choices = {"deviation_mode": ("Bounds fraction", "Mesh units")}
    parameter_ranges = {
        "target_faces": NumericRange(1, 100_000, whole=True),
        "max_deviation": NumericRange(0, 1e30),
        "max_normal_change_deg": NumericRange(0, 89),
        "max_input_faces": NumericRange(1, 100_000, whole=True),
        "max_edge_checks": NumericRange(1, 1_000_000, whole=True),
    }
    parameter_help = {
        "target_faces": "Maximum requested output face count.",
        "deviation_mode": "Use a fraction of the referenced bounds diagonal or mesh length units.",
        "max_deviation": "Largest allowed two-way surface displacement bound.",
        "max_normal_change_deg": "Maximum change from a retained face's input normal.",
        "max_input_faces": "Reject meshes exceeding this input face count before simplifying.",
        "max_edge_checks": "Stop after this many candidate edge checks.",
    }
    input_help = {"mesh": "A consistently wound manifold triangle mesh with optional boundaries."}
    output_help = {
        "mesh": "Review mesh with retained vertex attributes. Pass means the target was reached.",
        "report": "Face counts, boundary count, geometry bound and termination reason.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["deviation_mode"] == "Bounds fraction" and p["max_deviation"] > 1:
            raise ValidationError("Bounds fraction must be between 0 and 1", "max_deviation")

    def process(self, inputs):
        source, p = inputs["mesh"], self.parameters
        faces = source.faces
        n = len(faces)
        if n > p["max_input_faces"]:
            raise ValidationError("Mesh exceeds the input face limit", "max_input_faces")
        report = {
            "frame": source.frame,
            "units": source.units,
            "input_faces": n,
            "target_faces": int(p["target_faces"]),
        }
        if self._cancelled():
            return self._result(report, "cancelled")
        points = source.vertices.astype(np.float64)
        normals = _face_normals(points, faces)
        lengths = np.linalg.norm(normals, axis=1)
        topology = _edge_topology(faces, len(points))
        report.update({name: topology[name] for name in _TOPOLOGY_DEFECTS})
        report["degenerate_faces"] = int(np.count_nonzero(lengths == 0))
        report["duplicate_faces"] = n - len(np.unique(np.sort(faces, axis=1), axis=0))
        defects = (*_TOPOLOGY_DEFECTS, "degenerate_faces", "duplicate_faces")
        if any(report[name] for name in defects):
            return self._result(report, "invalid_topology")

        used = np.unique(faces)
        diagonal = float(np.linalg.norm(np.ptp(points[used], axis=0)))
        limit = float(p["max_deviation"])
        if p["deviation_mode"] == "Bounds fraction":
            limit *= diagonal
        keys, counts = _indexed_edges(faces, len(points))
        boundaries = keys[counts == 1]
        boundary = np.zeros(len(points), dtype=bool)
        boundary[boundaries // len(points)] = boundary[boundaries % len(points)] = True
        cosine = float(np.cos(np.deg2rad(p["max_normal_change_deg"])))
        solver = _EdgeContractions(
            points, faces, boundary, normals / lengths[:, None], limit, cosine
        )
        status, checks = solver.run(
            int(p["target_faces"]), int(p["max_edge_checks"]), self._cancelled
        )
        report.update(edge_checks=checks, boundary_edges=len(boundaries), deviation_limit=limit)
        if status == "cancelled" or self._cancelled():
            return self._result(report, "cancelled")
        triangles, keep, maximum, rms = solver.qualify(faces, used, boundaries, topology)
        if self._cancelled():
            return self._result(report, "cancelled")
        attributes = {
            name: getattr(source, name)[keep] if getattr(source, name) is not None else None
            for name in ("normals", "colors", "texcoords")
        }
        mesh = TriangleMesh(
            source.vertices[keep],
            np.searchsorted(keep, triangles),
            source.units,
            source.frame,
            **attributes,
        )
        report.update(
            output_faces=len(mesh.faces),
            output_vertices=len(keep),
            surface_deviation_bound=maximum,
            vertex_displacement_rms=rms,
        )
        return self._result(report, status, mesh)

    def _result(self, report, status, mesh=NO_VALUE):
        report["status"] = status
        passed = status == "target_reached"
        return Result(
            {"mesh": mesh, "report": report},
            passed=passed,
            fail_reasons=[] if passed else [f"mesh_simplify_{status}"],
            message=status.replace("_", " ")
            if mesh is NO_VALUE
            else f"{report['input_faces']:,} to {report['output_faces']:,} triangles "
            f"({status.replace('_', ' ')})",
            graphics={"Simplified mesh": render_wireframe(mesh)}
            if self.capture_graphics and mesh is not NO_VALUE
            else {},
        )
