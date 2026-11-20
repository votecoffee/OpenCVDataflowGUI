"""Bounded Wavefront OBJ interchange for ordered, attributed point rows."""

import json
import math
from array import array

import numpy as np

from .geometry_fileio import MAX_GEOMETRY_BYTES, geometry_output
from .pointclouds import PointCloud

MAX_CLOUD_BYTES = MAX_GEOMETRY_BYTES
_MAX_LINE_BYTES = 8192
# Midpoint above the largest finite float32, where rounding first overflows.
_FLOAT32_OVERFLOW = math.ldexp(1.0, 128) - math.ldexp(1.0, 103)
OBJ_METADATA_PREFIX = b"# dataflow_metadata "
_IGNORED_DIRECTIVES = {b"o", b"g", b"s", b"p", b"vt", b"mtllib", b"usemtl"}


def obj_values(fields):
    if any(b"_" in field for field in fields):
        raise ValueError("Invalid OBJ number")
    try:
        values = [float(field) for field in fields]
    except ValueError, OverflowError:
        raise ValueError("Invalid OBJ number") from None
    if not all(math.isfinite(value) for value in values):
        raise ValueError("OBJ numbers must be finite")
    return values


def obj_vertex(fields):
    if len(fields) not in (3, 4, 6):
        raise ValueError("OBJ vertex requires XYZ, optional unit weight, or XYZ and RGB")
    values = obj_values(fields)
    xyz = values[:3]
    if any(abs(value) >= _FLOAT32_OVERFLOW for value in xyz):
        raise ValueError("OBJ coordinates exceed the supported float32 range")
    if len(values) == 4 and values[3] != 1:
        raise ValueError("OBJ point-cloud weight must be one")
    rgb = values[3:] if len(values) == 6 else []
    if any(not 0 <= value <= 1 for value in rgb):
        raise ValueError("OBJ vertex RGB values must be normalized from zero to one")
    return xyz, [round(value * 255) for value in rgb]


def _normal(fields):
    if len(fields) != 3:
        raise ValueError("OBJ normal requires three components")
    vector = obj_values(fields)
    scale = max(map(abs, vector))
    if scale == 0:
        raise ValueError("OBJ normals must have nonzero length")
    vector = [value / scale for value in vector]
    length = math.hypot(*vector)
    return [value / length for value in vector]


def obj_lines(stream):
    total_bytes = 0
    row_number = 0
    while line := stream.readline(_MAX_LINE_BYTES + 1):
        row_number += 1
        total_bytes += len(line)
        if len(line) > _MAX_LINE_BYTES:
            raise ValueError(f"OBJ line {row_number} exceeds {_MAX_LINE_BYTES} bytes")
        if total_bytes > MAX_CLOUD_BYTES:
            raise ValueError("Point-cloud files are limited to 256 MiB")
        if row_number == 1:
            line = line.removeprefix(b"\xef\xbb\xbf")
        yield row_number, line.lstrip()


def read_obj_cloud(stream, units, frame, max_points):
    """Read vertex order, inline RGB and an optional row-aligned normal stream."""
    vertices, normals, colors = array("f"), array("f"), bytearray()
    metadata = None
    colored = None
    for row_number, line in obj_lines(stream):
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
            directive = fields[0]
            if directive == b"v":
                if len(vertices) // 3 >= max_points:
                    raise ValueError("OBJ vertex count exceeds the point limit")
                xyz, rgb = obj_vertex(fields[1:])
                if colored is not None and colored != bool(rgb):
                    raise ValueError("OBJ vertex RGB must be supplied for every vertex or omitted")
                colored = bool(rgb)
                vertices.extend(xyz)
                colors.extend(rgb)
            elif directive == b"vn":
                if len(normals) // 3 >= max_points:
                    raise ValueError("OBJ normal count exceeds the point limit")
                normals.extend(_normal(fields[1:]))
            elif directive not in _IGNORED_DIRECTIVES:
                kind = directive.decode("ascii", errors="replace")
                raise ValueError(f"Unsupported OBJ point-cloud directive {kind!r}")
        except (UnicodeError, ValueError) as exc:
            raise ValueError(f"OBJ line {row_number}: {exc}") from exc
    if normals and len(normals) != len(vertices):
        raise ValueError("OBJ normals must contain one ordered normal per vertex")
    attributes = metadata or {}
    return PointCloud(
        np.frombuffer(vertices, dtype=np.float32).reshape(-1, 3),
        attributes.get("units", units),
        attributes.get("frame", frame),
        colors=np.frombuffer(colors, dtype=np.uint8).reshape(-1, 3) if colored else None,
        normals=np.frombuffer(normals, dtype=np.float32).reshape(-1, 3) if normals else None,
    )


def obj_header(units, frame):
    metadata = json.dumps({"units": units, "frame": frame}, ensure_ascii=True)
    header = OBJ_METADATA_PREFIX + metadata.encode("ascii") + b"\n"
    if len(header) > _MAX_LINE_BYTES:
        raise ValueError("OBJ metadata comment exceeds the line limit")
    return header


def write_obj_vertices(stream, points, colors, normals):
    for first in range(0, len(points), 4096):
        rows = slice(first, first + 4096)
        values = points[rows]
        if colors is not None:
            values = np.column_stack((values, colors[rows].astype(np.float64) / 255))
        fmt = "v " + " ".join(["%.9g"] * values.shape[1])
        np.savetxt(stream, values, fmt=fmt)
    if normals is not None:
        np.savetxt(stream, normals, fmt="vn %.9g %.9g %.9g")


def write_obj_cloud(path, cloud):
    """Publish a complete ASCII OBJ without replacing an existing destination."""
    header = obj_header(cloud.units, cloud.frame)
    with geometry_output(path, ".cloud-obj-") as stream:
        stream.write(header)
        write_obj_vertices(stream, cloud.points, cloud.colors, cloud.normals)
