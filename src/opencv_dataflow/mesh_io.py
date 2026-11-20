"""Bounded PLY/OBJ triangle mesh interchange and graph file tools."""

import json
from pathlib import Path

import numpy as np

from .cloud_io import PLY_TYPES, ascii_vertices, cloud_path
from .core import Block, Port, Result
from .geometry_fileio import MAX_GEOMETRY_BYTES
from .meshes import MAX_FACES, MAX_VERTICES, TriangleMesh, render_wireframe, write_mesh_ply
from .obj_mesh import read_obj_mesh, write_obj_mesh
from .parameter_validation import NumericRange, ValidationError

_VERTEX_FIELDS = {
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "red",
    "green",
    "blue",
    "texture_u",
    "texture_v",
    "texture_w",
    "u",
    "v",
    "w",
}
_FORMATS = {"ascii": "", "binary_little_endian": "<", "binary_big_endian": ">"}
_METADATA = "comment dataflow_metadata "


def _ply_header(stream, max_vertices, max_faces):
    if stream.readline(8193) not in (b"ply\n", b"ply\r\n"):
        raise ValueError("Invalid PLY signature")
    raw_format = stream.readline(8193)
    if len(raw_format) > 8192:
        raise ValueError("Oversized PLY format line")
    format_line = raw_format.decode("ascii").split()
    if (
        len(format_line) != 3
        or format_line[0] != "format"
        or format_line[1] not in _FORMATS
        or format_line[2] != "1.0"
    ):
        raise ValueError("Unsupported mesh PLY format")
    vertices = faces = 0
    columns = {}
    face_types = None
    metadata = None
    for _ in range(254):
        data = stream.readline(8193)
        if not data or len(data) > 8192:
            raise ValueError("Truncated or oversized mesh PLY header")
        line = data.decode("ascii").strip()
        fields = line.split()
        if line == "end_header":
            break
        if line.startswith(_METADATA):
            if metadata is not None:
                raise ValueError("Duplicate PLY dataflow metadata comments")
            metadata = json.loads(line[len(_METADATA) :])
            if not isinstance(metadata, dict):
                raise ValueError("Invalid PLY dataflow metadata comment")
        elif not fields or fields[0] in ("comment", "obj_info"):
            continue
        elif fields[0] == "element" and len(fields) == 3:
            count = int(fields[2])
            if fields[1] == "vertex" and not vertices and not faces and 1 <= count <= max_vertices:
                vertices = count
            elif fields[1] == "face" and vertices and not faces and 1 <= count <= max_faces:
                faces = count
            else:
                raise ValueError("PLY requires bounded vertex and face elements in that order")
        elif fields[0] == "property" and vertices and not faces and len(fields) == 3:
            kind, name = fields[1:]
            if name not in _VERTEX_FIELDS or name in columns or kind not in PLY_TYPES:
                raise ValueError("Unsupported or duplicate PLY vertex property")
            columns[name] = PLY_TYPES[kind]
        elif fields[0] == "property" and faces and face_types is None and len(fields) == 5:
            label, count_type, index_type, name = fields[1:]
            if (
                label != "list"
                or name not in ("vertex_indices", "vertex_index")
                or count_type not in PLY_TYPES
                or index_type not in PLY_TYPES
            ):
                raise ValueError("PLY mesh requires a face vertex-index list")
            face_types = np.dtype(PLY_TYPES[count_type]), np.dtype(PLY_TYPES[index_type])
            if any(dtype.kind not in "iu" for dtype in face_types):
                raise ValueError("PLY face lists require integer counts and indices")
        else:
            raise ValueError("Unsupported mesh PLY header layout")
    else:
        raise ValueError("Mesh PLY header exceeds 256 lines")
    if not vertices or not faces or face_types is None or not {"x", "y", "z"} <= columns.keys():
        raise ValueError("PLY requires XYZ vertices and triangle faces")
    return format_line[1], vertices, faces, columns, face_types, metadata or {}


def _optional_columns(columns, names, label):
    available = set(columns).intersection(names)
    if available and available != set(names):
        raise ValueError(f"Incomplete PLY {label} fields")
    return np.column_stack([columns[name] for name in names]) if available else None


def _texture_columns(columns):
    result = None
    for u, v, w in (("texture_u", "texture_v", "texture_w"), ("u", "v", "w")):
        names = (u, v, w) if w in columns else (u, v)
        values = _optional_columns(columns, names, "texture coordinate")
        if values is not None:
            if result is not None:
                raise ValueError("Ambiguous PLY texture coordinate fields")
            result = values
    return result


def _binary_records(stream, count, dtype):
    data = stream.read(count * dtype.itemsize)
    if len(data) != count * dtype.itemsize:
        raise ValueError("Truncated binary PLY mesh data")
    return np.frombuffer(data, dtype=dtype)


def _ascii_faces(stream, count, vertex_count):
    faces = np.empty((count, 3), np.int32)
    for row in range(count):
        data = stream.readline(8193)
        if len(data) > 8192:
            raise ValueError("Oversized ASCII PLY face row")
        fields = data.split()
        if len(fields) != 4 or int(fields[0]) != 3:
            raise ValueError("PLY mesh faces must contain exactly three indices")
        if any(b"_" in value for value in fields[1:]):
            raise ValueError("Invalid PLY face index")
        values = [int(value) for value in fields[1:]]
        if any(not 0 <= value < vertex_count for value in values):
            raise ValueError("PLY face index is outside the vertex array")
        faces[row] = values
    return faces


def read_mesh_ply(stream, units, frame, max_vertices, max_faces):
    encoding, n, m, fields, face_types, info = _ply_header(stream, max_vertices, max_faces)
    if encoding == "ascii":
        values = ascii_vertices(stream, n, len(fields))
        columns = {name: values[:, i] for i, name in enumerate(fields)}
        faces = _ascii_faces(stream, m, n)
        while tail := stream.read(8192):
            if stream.tell() > MAX_GEOMETRY_BYTES:
                raise ValueError("Geometry files are limited to 256 MiB")
            if tail.strip():
                raise ValueError("Unexpected data after PLY triangles")
    else:
        byteorder = _FORMATS[encoding]
        dtype = np.dtype(list(fields.items())).newbyteorder(byteorder)
        values = _binary_records(stream, n, dtype)
        columns = {name: values[name] for name in fields}
        count_type, index_type = face_types
        dtype = np.dtype([("count", count_type), ("indices", index_type, 3)]).newbyteorder(
            byteorder
        )
        records = _binary_records(stream, m, dtype)
        if np.any(records["count"] != 3):
            raise ValueError("PLY mesh faces must contain exactly three indices")
        if stream.read(1):
            raise ValueError("Unexpected data after PLY triangles")
        faces = records["indices"]
    xyz = np.column_stack([columns[name] for name in ("x", "y", "z")])
    normals = _optional_columns(columns, ("nx", "ny", "nz"), "normal")
    colors = _optional_columns(columns, ("red", "green", "blue"), "RGB")
    return TriangleMesh(
        xyz,
        faces,
        info.get("units", units),
        info.get("frame", frame),
        normals=normals,
        colors=colors,
        texcoords=_texture_columns(columns),
    )


def read_mesh(path, units="m", frame="file", max_vertices=MAX_VERTICES, max_faces=MAX_FACES):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in (".obj", ".ply"):
        raise ValueError("Choose a triangle mesh in OBJ or PLY format")
    if path.stat().st_size > MAX_GEOMETRY_BYTES:
        raise ValueError("Geometry files are limited to 256 MiB")
    with path.open("rb") as stream:
        read = read_obj_mesh if suffix == ".obj" else read_mesh_ply
        return read(stream, units, frame, max_vertices, max_faces)


def write_mesh(path, mesh, maximum_faces=MAX_FACES):
    path = Path(path)
    if path.suffix.lower() == ".obj":
        mesh.validate()
        if len(mesh.faces) > maximum_faces:
            raise ValueError("Mesh exceeds the configured export face limit")
        write_obj_mesh(path, mesh)
    else:
        write_mesh_ply(path, mesh, maximum_faces)


class LoadMesh(Block):
    project_path_parameters = ("path",)
    type_id, title, category = "input.mesh", "Load triangle mesh (OBJ / PLY)", "Input"
    block_type, subtype = "Input", "3D geometry"
    outputs = (Port("mesh", "mesh"), Port("report", "data"))
    defaults = {
        "path": "",
        "units": "m",
        "frame": "file",
        "max_vertices": 250_000,
        "max_faces": 250_000,
    }
    parameter_choices = {"units": ("m", "mm", "unitless")}
    parameter_ranges = {
        "max_vertices": NumericRange(3, MAX_VERTICES, whole=True),
        "max_faces": NumericRange(1, MAX_FACES, whole=True),
    }
    parameter_help = {
        "path": "Choose a triangulated OBJ or PLY file. Relative paths use the project directory.",
        "units": "Vertex coordinate units when file metadata does not specify them.",
        "frame": "Vertex coordinate frame when file metadata does not specify it.",
        "max_vertices": "Limit vertex rows, including any added by OBJ attribute seams.",
        "max_faces": "Limit imported triangle faces.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not isinstance(p["frame"], str) or not p["frame"].strip():
            raise ValidationError("Mesh frame must be named", "frame")

    def process(self, inputs):
        p = self.parameters
        mesh = read_mesh(
            cloud_path(self),
            p["units"],
            p["frame"],
            int(p["max_vertices"]),
            int(p["max_faces"]),
        )
        return Result(
            {"mesh": mesh, "report": mesh.describe()},
            graphics={"Mesh wireframe": render_wireframe(mesh)} if self.capture_graphics else {},
            message=f"Loaded {len(mesh.faces)} triangles",
        )
