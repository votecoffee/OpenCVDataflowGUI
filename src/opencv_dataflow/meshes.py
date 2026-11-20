"""Typed triangle geometry, preview and bounded PLY export."""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .geometry_fileio import geometry_output

MAX_VERTICES = 1_000_000
MAX_FACES = 1_000_000


@dataclass
class TriangleMesh:
    vertices: np.ndarray
    faces: np.ndarray
    units: str = "m"
    frame: str = "world"
    normals: np.ndarray | None = None
    colors: np.ndarray | None = None
    texcoords: np.ndarray | None = None

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices)
        self.faces = np.asarray(self.faces)
        for name in ("normals", "colors", "texcoords"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, np.asarray(value))
        self.validate()
        self.vertices = np.ascontiguousarray(self.vertices, np.float32)
        self.faces = np.ascontiguousarray(self.faces, np.int32)
        for name, dtype in (
            ("normals", np.float32),
            ("colors", np.uint8),
            ("texcoords", np.float32),
        ):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, np.ascontiguousarray(value, dtype))

    def validate(self):
        vertices, faces = self.vertices, self.faces
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or vertices.dtype.kind not in "fiu"
            or not 1 <= len(vertices) <= MAX_VERTICES
            or not np.isfinite(vertices).all()
            or np.any(np.abs(vertices.astype(np.float64)) > np.finfo(np.float32).max)
        ):
            raise ValueError("Mesh requires finite Nx3 vertices within the float32 range")
        if (
            faces.ndim != 2
            or faces.shape[1] != 3
            or faces.dtype.kind not in "iu"
            or not 1 <= len(faces) <= MAX_FACES
            or np.any((faces < 0) | (faces >= len(vertices)))
            or np.any(faces[:, 0] == faces[:, 1])
            or np.any(faces[:, 1] == faces[:, 2])
            or np.any(faces[:, 2] == faces[:, 0])
        ):
            raise ValueError("Mesh requires bounded triangles with three distinct vertex indices")
        if (
            self.units not in ("m", "mm", "unitless")
            or not isinstance(self.frame, str)
            or not self.frame
        ):
            raise ValueError("Mesh requires explicit units and a named coordinate frame")

        for name, widths in (("normals", (3,)), ("colors", (3,)), ("texcoords", (2, 3))):
            value = getattr(self, name)
            if value is None:
                continue
            if (
                value.ndim != 2
                or len(value) != len(vertices)
                or value.shape[1] not in widths
                or value.dtype.kind not in "fiu"
                or not np.isfinite(value).all()
                or np.any(np.abs(value.astype(np.float64)) > np.finfo(np.float32).max)
            ):
                raise ValueError(f"Mesh {name} must contain one finite vector per vertex")
            if name == "colors" and (
                np.any((value < 0) | (value > 255)) or np.any(value != np.rint(value))
            ):
                raise ValueError("Mesh colors must be RGB bytes")

    @property
    def nbytes(self):
        return sum(
            value.nbytes
            for value in (self.vertices, self.faces, self.normals, self.colors, self.texcoords)
            if value is not None
        )

    def describe(self):
        return {
            "model": "triangle_mesh_v1",
            "vertex_count": len(self.vertices),
            "face_count": len(self.faces),
            "normals": self.normals is not None,
            "colors": self.colors is not None,
            "texture_dimensions": 0 if self.texcoords is None else self.texcoords.shape[1],
            "units": self.units,
            "frame": self.frame,
            "minimum": self.vertices.min(axis=0).tolist(),
            "maximum": self.vertices.max(axis=0).tolist(),
            "bytes": self.nbytes,
        }


def render_wireframe(mesh):
    image = np.full((400, 560, 3), (31, 27, 22), np.uint8)
    rotation = cv2.Rodrigues(np.array([-0.38, 0.54, 0.06]))[0]
    vertices = mesh.vertices @ rotation.T
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    span = np.maximum(high[:2] - low[:2], 1e-9)
    scale = min(510 / span[0], 350 / span[1])
    xy = np.rint((vertices[:, :2] - (low[:2] + high[:2]) / 2) * scale + [280, 200])
    choice = np.linspace(0, len(mesh.faces) - 1, min(len(mesh.faces), 2500), dtype=int)
    polylines = xy.astype(np.int32)[mesh.faces[choice]]
    cv2.polylines(image, list(polylines), True, (65, 198, 245), 1, cv2.LINE_AA)
    return image


def write_mesh_ply(path, mesh, maximum_faces=MAX_FACES):
    """Publish a complete binary PLY, preserving any existing destination."""
    path = Path(path)
    if path.suffix.lower() != ".ply":
        raise ValueError("Mesh export requires a .ply destination")
    mesh.validate()
    if len(mesh.faces) > maximum_faces:
        raise ValueError("Mesh exceeds the configured export face limit")
    suffix = json.dumps({"units": mesh.units, "frame": mesh.frame}, ensure_ascii=True)
    if len(f"comment dataflow_metadata {suffix}\n") > 8192:
        raise ValueError("PLY metadata comment exceeds the line limit")
    fields = [("x y z", "float", "<f4", mesh.vertices)]
    if mesh.normals is not None:
        fields.append(("nx ny nz", "float", "<f4", mesh.normals))
    if mesh.colors is not None:
        fields.append(("red green blue", "uchar", "u1", mesh.colors))
    if mesh.texcoords is not None:
        names = (
            "texture_u texture_v"
            if mesh.texcoords.shape[1] == 2
            else "texture_u texture_v texture_w"
        )
        fields.append((names, "float", "<f4", mesh.texcoords))
    columns = [
        (name, kind, dtype, values[:, i])
        for names, kind, dtype, values in fields
        for i, name in enumerate(names.split())
    ]
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"comment dataflow_metadata {suffix}",
        f"element vertex {len(mesh.vertices)}",
        *(f"property {kind} {name}" for name, kind, _, _ in columns),
        f"element face {len(mesh.faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    vertices = np.empty(len(mesh.vertices), dtype=[(name, dtype) for name, _, dtype, _ in columns])
    for name, _, _, values in columns:
        vertices[name] = values
    faces = np.empty(len(mesh.faces), dtype=[("count", "u1"), ("indices", "<i4", 3)])
    faces["count"], faces["indices"] = 3, mesh.faces
    with geometry_output(path, ".mesh-") as stream:
        stream.write(("\n".join(header) + "\n").encode("ascii"))
        stream.write(vertices)
        stream.write(faces)
