"""Typed triangle geometry and bounded PLY output."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np

MAX_VERTICES = 1_000_000
MAX_FACES = 1_000_000


@dataclass
class TriangleMesh:
    vertices: np.ndarray
    faces: np.ndarray
    units: str = "m"
    frame: str = "world"

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices)
        self.faces = np.asarray(self.faces)
        self.validate()
        self.vertices = np.ascontiguousarray(self.vertices, np.float32)
        self.faces = np.ascontiguousarray(self.faces, np.int32)

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

    @property
    def nbytes(self):
        return self.vertices.nbytes + self.faces.nbytes

    def describe(self):
        return {
            "model": "triangle_mesh_v1",
            "vertex_count": len(self.vertices),
            "face_count": len(self.faces),
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
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"comment dataflow_metadata {suffix}\n"
        f"element vertex {len(mesh.vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(mesh.faces)}\n"
        "property list uchar int vertex_indices\nend_header\n"
    ).encode("ascii")
    rows = np.empty(len(mesh.faces), dtype=[("count", "u1"), ("indices", "<i4", (3,))])
    rows["count"], rows["indices"] = 3, mesh.faces
    stream = NamedTemporaryFile(prefix=".mesh-", dir=path.parent, delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            stream.write(header)
            stream.write(mesh.vertices.astype("<f4", copy=False))
            stream.write(rows)
        if os.name == "nt":
            temporary.rename(path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
