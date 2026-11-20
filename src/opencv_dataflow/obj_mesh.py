"""Wavefront triangle meshes with point, normal and texture index validation."""

import json
from array import array

import numpy as np

from .geometry_fileio import geometry_output
from .meshes import TriangleMesh
from .obj_cloud import (
    OBJ_METADATA_PREFIX,
    obj_header,
    obj_lines,
    obj_values,
    obj_vertex,
    write_obj_vertices,
)

_IGNORED = {b"o", b"g", b"s", b"mtllib", b"usemtl"}


def _index(token, size):
    digits = token[1:] if token.startswith((b"+", b"-")) else token
    if not digits.isdigit():
        raise ValueError("Invalid OBJ face index")
    value = int(token)
    index = value - 1 if value > 0 else size + value
    if not value or not 0 <= index < size:
        raise ValueError("OBJ face index must refer to an existing row")
    return index


def _corner(token, counts):
    fields = token.split(b"/")
    if not fields[0] or len(fields) > 3 or (len(fields) > 1 and not fields[-1]):
        raise ValueError("Invalid OBJ face corner")
    indices = tuple(_index(value, counts[i]) if value else -1 for i, value in enumerate(fields))
    return indices + (-1,) * (3 - len(indices))


def _aligned_mesh(vertices, corners, normals, colors, texcoords, max_vertices):
    vertex_indices = corners[:, 0]
    streams = [texcoords, normals]
    separate = False
    for i, values in enumerate(streams, start=1):
        refs = corners[:, i]
        present = refs >= 0
        if present.any():
            if not present.all():
                raise ValueError("OBJ face corner attributes must be complete")
            separate |= len(values) != len(vertices) or not np.array_equal(refs, vertex_indices)
        elif values is not None:
            if len(values) != len(vertices):
                raise ValueError("OBJ unindexed vertex attributes must be row-aligned")
            # Native OpenCV also emits parallel attribute streams without explicit references.
            corners[:, i] = vertex_indices
    if not separate:
        return vertices, vertex_indices.reshape(-1, 3), normals, colors, texcoords

    # Split vertices at UV or normal seams. Faces keep their original order and winding.
    unique, inverse = np.unique(corners, axis=0, return_inverse=True)
    if len(unique) > max_vertices:
        raise ValueError("OBJ mesh exceeds the vertex limit after splitting attribute seams")
    rows, texture_indices, normal_indices = unique.T
    if len(np.unique(rows)) != len(vertices):
        raise ValueError("OBJ separate attribute indices require every vertex to be referenced")
    return (
        vertices[rows],
        inverse.reshape(-1, 3),
        normals[normal_indices] if normals is not None else None,
        colors[rows] if colors is not None else None,
        texcoords[texture_indices] if texcoords is not None else None,
    )


def read_obj_mesh(stream, units, frame, max_vertices, max_faces):
    vertices, normals, textures, corners = array("f"), array("f"), array("f"), array("i")
    colors = bytearray()
    metadata = None
    color_present = None
    texture_width = 0
    for row, line in obj_lines(stream):
        try:
            if line.startswith(OBJ_METADATA_PREFIX):
                if metadata is not None:
                    raise ValueError("Duplicate OBJ dataflow metadata comments")
                metadata = json.loads(line[len(OBJ_METADATA_PREFIX) :])
                if not isinstance(metadata, dict):
                    raise ValueError("Invalid OBJ dataflow metadata comment")
                continue
            fields = line.partition(b"#")[0].split()
            if not fields:
                continue
            kind, values = fields[0], fields[1:]
            if kind == b"v":
                if len(vertices) // 3 >= max_vertices:
                    raise ValueError("OBJ mesh exceeds the vertex limit")
                xyz, rgb = obj_vertex(values)
                if color_present is not None and color_present != bool(rgb):
                    raise ValueError("OBJ vertex RGB must be supplied for every vertex or omitted")
                color_present = bool(rgb)
                vertices.extend(xyz)
                colors.extend(rgb)
            elif kind == b"vn":
                if len(values) != 3 or len(normals) // 3 >= max_vertices:
                    raise ValueError("OBJ mesh requires bounded three-component normals")
                normals.extend(obj_values(values))
            elif kind == b"vt":
                if len(values) not in (2, 3) or texture_width not in (0, len(values)):
                    raise ValueError(
                        "OBJ texture coordinates require a consistent width of two or three"
                    )
                texture_width = len(values)
                if len(textures) // texture_width >= max_vertices:
                    raise ValueError("OBJ mesh exceeds the texture coordinate limit")
                textures.extend(obj_values(values))
            elif kind == b"f":
                if len(values) != 3:
                    raise ValueError("OBJ mesh faces must be triangles")
                if len(corners) // 9 >= max_faces:
                    raise ValueError("OBJ mesh exceeds the face limit")
                counts = (
                    len(vertices) // 3,
                    len(textures) // (texture_width or 1),
                    len(normals) // 3,
                )
                face = [_corner(corner, counts) for corner in values]
                if len({corner[0] for corner in face}) != 3:
                    raise ValueError("OBJ triangles require three distinct vertex indices")
                for corner in face:
                    corners.extend(corner)
            elif kind not in _IGNORED:
                raise ValueError(f"Unsupported OBJ mesh directive {kind.decode('ascii')!r}")
        except (ValueError, UnicodeError, OverflowError) as exc:
            raise ValueError(f"OBJ line {row}: {exc}") from exc
    mapping = np.frombuffer(corners, dtype=np.int32).reshape(-1, 3)
    if not len(mapping):
        raise ValueError("OBJ mesh requires triangle faces")
    vertices = np.frombuffer(vertices, dtype=np.float32).reshape(-1, 3)
    normals = np.frombuffer(normals, dtype=np.float32).reshape(-1, 3) if normals else None
    colors = np.frombuffer(colors, dtype=np.uint8).reshape(-1, 3) if colors else None
    texcoords = (
        np.frombuffer(textures, dtype=np.float32).reshape(-1, texture_width) if textures else None
    )
    # Validate every declared row before dropping or reordering attributes at seams.
    for values in (normals, texcoords):
        if values is not None and not np.isfinite(values).all():
            raise ValueError("OBJ attributes exceed the supported float32 range")
    xyz, faces, normals, colors, texcoords = _aligned_mesh(
        vertices, mapping, normals, colors, texcoords, max_vertices
    )
    info = metadata or {}
    return TriangleMesh(
        xyz,
        faces,
        info.get("units", units),
        info.get("frame", frame),
        normals=normals,
        colors=colors,
        texcoords=texcoords,
    )


def write_obj_mesh(path, mesh):
    header = obj_header(mesh.units, mesh.frame)
    with geometry_output(path, ".mesh-obj-") as stream:
        stream.write(header)
        write_obj_vertices(stream, mesh.vertices, mesh.colors, mesh.normals)
        if mesh.texcoords is not None:
            np.savetxt(
                stream, mesh.texcoords, fmt="vt " + " ".join(["%.9g"] * mesh.texcoords.shape[1])
            )
        has_normal, has_texture = mesh.normals is not None, mesh.texcoords is not None
        if has_texture:
            index_format, columns = ("%d/%d/%d", 3) if has_normal else ("%d/%d", 2)
        else:
            index_format, columns = ("%d//%d", 2) if has_normal else ("%d", 1)
        for first in range(0, len(mesh.faces), 4096):
            indices = mesh.faces[first : first + 4096] + 1
            np.savetxt(
                stream, np.repeat(indices, columns, axis=1), fmt="f " + " ".join([index_format] * 3)
            )
