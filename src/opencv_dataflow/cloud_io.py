"""Bounded point-cloud interchange with explicit frame and unit metadata."""

import json
from pathlib import Path

import numpy as np

from .core import Block, Port, Result
from .geometry3d_common import cloud_graphics
from .geometry_fileio import geometry_output
from .obj_cloud import MAX_CLOUD_BYTES, read_obj_cloud, write_obj_cloud
from .parameter_validation import NumericRange
from .pcd_cloud import read_pcd_vertices, write_pcd_cloud
from .point_attributes import POINT_ATTRIBUTE_DTYPES
from .pointclouds import PointCloud
from .processing import integer

PLY_TYPES = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int16": "<i2",
    "uint16": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "int32": "<i4",
    "uint32": "<u4",
    "float": "<f4",
    "double": "<f8",
    "float32": "<f4",
    "float64": "<f8",
}


def cloud_path(block):
    value = block.parameters["path"]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Select a point-cloud file")
    path = Path(value)
    return path if path.is_absolute() or block.project_dir is None else block.project_dir / path


def make_cloud(
    columns, metadata, units, frame, *, source_size=None, viewpoint=None, track_source_rows=False
):
    if not all(k in columns for k in ("x", "y", "z")):
        raise ValueError("Cloud file must contain XYZ fields")
    with np.errstate(over="ignore"):
        xyz = np.column_stack([columns[k] for k in ("x", "y", "z")]).astype(np.float32)
    keep = np.isfinite(xyz).all(axis=1)
    normals = None
    if all(k in columns for k in ("nx", "ny", "nz")):
        normals = np.column_stack([columns[k] for k in ("nx", "ny", "nz")])
    elif all(k in columns for k in ("normal_x", "normal_y", "normal_z")):
        normals = np.column_stack([columns[k] for k in ("normal_x", "normal_y", "normal_z")])
    if normals is not None:
        normals = normals.astype(np.float64, copy=False)
        with np.errstate(over="ignore", invalid="ignore"):
            lengths = np.linalg.norm(normals, axis=1)
        keep &= np.isfinite(normals).all(axis=1) & np.isfinite(lengths) & (lengths > 1e-8)
        normals = normals[keep] / lengths[keep, None]
    colors = None
    if all(k in columns for k in ("red", "green", "blue")):
        colors = np.column_stack([columns[k] for k in ("red", "green", "blue")])[keep]
    elif all(k in columns for k in ("r", "g", "b")):
        colors = np.column_stack([columns[k] for k in ("r", "g", "b")])[keep]
    else:
        packed = columns.get("rgb", columns.get("rgba"))
        if packed is not None:
            if np.issubdtype(packed.dtype, np.floating):
                packed = packed.astype("<f4", copy=False).view("<u4")
            packed = packed[keep].astype(np.uint32, copy=False)
            colors = np.c_[(packed >> 16) & 255, (packed >> 8) & 255, packed & 255]
    pixels = None
    if source_size is not None:
        width, _ = source_size
        indices = np.flatnonzero(keep)
        pixels = np.column_stack([indices % width, indices // width])
    units, frame = metadata.get("units", units), metadata.get("frame", frame)
    cloud_metadata = (
        {"pcd_viewpoint": {"frame": frame, "units": units, "values": viewpoint}}
        if viewpoint is not None
        else {}
    )
    attributes = {name: columns[name][keep] for name in POINT_ATTRIBUTE_DTYPES if name in columns}
    if track_source_rows and "source_index" not in attributes:
        attributes["source_index"] = np.flatnonzero(keep)
    return PointCloud(
        xyz[keep],
        units,
        frame,
        colors,
        normals,
        pixels,
        source_size,
        cloud_metadata,
        attributes=attributes,
    )


def ascii_vertices(stream, count, fields):
    if count * fields > 24_000_000:
        raise ValueError("ASCII cloud exceeds 24 million scalar fields")
    values = np.empty((count, fields), np.float64)
    for first in range(0, count, 4096):
        last = min(first + 4096, count)
        rows = [stream.readline(8192).decode("ascii").split() for _ in range(first, last)]
        if any(len(row) != fields for row in rows):
            raise ValueError("Truncated or malformed cloud vertices")
        values[first:last] = np.asarray(rows, np.float64)
    return values


def cloud_headers(stream):
    headers, metadata = [], {}
    for _ in range(256):
        line = stream.readline(8193)
        if not line:
            raise ValueError("Truncated cloud header")
        if len(line) > 8192:
            raise ValueError("Oversized cloud header line")
        text = line.decode("ascii").strip()
        headers.append(text)
        marker = "dataflow_metadata "
        if marker in text:
            metadata = json.loads(text.split(marker, 1)[1])
            if not isinstance(metadata, dict):
                raise ValueError("Invalid cloud metadata comment")
        if text == "end_header" or text.split(maxsplit=1)[:1] == ["DATA"]:
            break
    else:
        raise ValueError("Cloud header exceeds 256 lines")
    return headers, metadata


def ply_vertex_schema(headers, max_points):
    format_line = next((h for h in headers if h.startswith("format ")), "")
    count = None
    properties = []
    in_vertices = False
    for h in headers:
        if h.startswith("element "):
            _, name, number = h.split()
            in_vertices = name == "vertex"
            if in_vertices:
                count = integer(int(number), "Vertex count", 0, max_points)
        elif h.startswith("property ") and in_vertices:
            pieces = h.split()
            if len(pieces) != 3 or pieces[1] not in PLY_TYPES:
                raise ValueError("PLY vertex list/unknown properties are unsupported")
            properties.append((pieces[2], PLY_TYPES[pieces[1]]))
    if count is None or not properties or len(properties) > 64:
        raise ValueError("PLY must declare vertices with bounded scalar properties")
    names = [p[0] for p in properties]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate PLY vertex properties")
    return format_line, count, properties


def read_cloud(path, units="m", frame="file", max_points=2000000, *, track_source_rows=False):
    file_size = path.stat().st_size
    if file_size > MAX_CLOUD_BYTES:
        raise ValueError("Point-cloud files are limited to 256 MiB")
    with path.open("rb") as stream:
        if path.suffix.lower() == ".obj":
            cloud = read_obj_cloud(stream, units, frame, max_points)
            if track_source_rows:
                cloud.attributes["source_index"] = np.arange(len(cloud.points), dtype=np.int32)
            return cloud
        headers, metadata = cloud_headers(stream)
        if headers[0] == "ply":
            format_line, count, properties = ply_vertex_schema(headers, max_points)
            names = [name for name, _ in properties]
            if format_line == "format binary_little_endian 1.0":
                dtype = np.dtype(properties)
                expected = count * dtype.itemsize
                if expected > file_size - stream.tell():
                    raise ValueError("Truncated binary PLY vertices")
                data = stream.read(expected)
                if len(data) != expected:
                    raise ValueError("Truncated binary PLY vertices")
                values = np.frombuffer(data, dtype=dtype)
                columns = {name: values[name] for name in names}
            elif format_line == "format ascii 1.0":
                values = ascii_vertices(stream, count, len(names))
                columns = {name: values[:, i] for i, name in enumerate(names)}
            else:
                raise ValueError("PLY must be ASCII or binary little endian 1.0")
        else:
            columns, size, view = read_pcd_vertices(stream, headers, max_points)
            return make_cloud(
                columns,
                metadata,
                units,
                frame,
                source_size=size,
                viewpoint=view,
                track_source_rows=track_source_rows,
            )
    return make_cloud(columns, metadata, units, frame, track_source_rows=track_source_rows)


def write_cloud(
    path, cloud, binary=False, *, pcd_organization="unorganized", write_inspection_attributes=True
):
    suffix = path.suffix.lower()
    if suffix not in (".ply", ".pcd", ".obj") or (binary and suffix == ".obj"):
        raise ValueError("Choose .ply or .pcd (ASCII/binary), or .obj (ASCII)")
    attributes = cloud.attributes if write_inspection_attributes else {}
    if suffix == ".obj":
        if attributes:
            raise ValueError(
                "Use PLY or PCD to export inspection attributes, or turn off their export"
            )
        write_obj_cloud(path, cloud)
        return
    if suffix == ".pcd":
        write_pcd_cloud(path, cloud, binary, pcd_organization, attributes)
        return
    names, types, columns = ["x", "y", "z"], ["float"] * 3, [cloud.points[:, i] for i in range(3)]
    if cloud.normals is not None:
        names += ["nx", "ny", "nz"]
        types += ["float"] * 3
        columns += [cloud.normals[:, i] for i in range(3)]
    if cloud.colors is not None:
        names += ["red", "green", "blue"]
        types += ["uchar"] * 3
        columns += [cloud.colors[:, i] for i in range(3)]
    for name, values in attributes.items():
        names.append(name)
        types.append("double" if values.dtype.kind == "f" else "int")
        columns.append(values)
    metadata = json.dumps({"units": cloud.units, "frame": cloud.frame}, ensure_ascii=True)
    metadata_line = f"comment dataflow_metadata {metadata}"
    if len(metadata_line) >= 8192:
        raise ValueError("PLY frame metadata exceeds the header line limit")
    n = len(cloud.points)
    header = [
        "ply",
        f"format {'binary_little_endian' if binary else 'ascii'} 1.0",
        metadata_line,
        f"element vertex {n}",
    ]
    header += [f"property {t} {name}" for t, name in zip(types, names)]
    header += ["end_header"]
    with geometry_output(path, ".ply-") as stream:
        stream.write(("\n".join(header) + "\n").encode("ascii"))
        if binary:
            values = np.empty(n, dtype=[(name, PLY_TYPES[t]) for name, t in zip(names, types)])
            for name, column in zip(names, columns):
                values[name] = column
            stream.write(values.tobytes())
        else:
            values = np.column_stack(columns)
            np.savetxt(
                stream,
                values,
                fmt=[{"double": "%.17g", "float": "%.9g"}.get(t, "%d") for t in types],
            )


class LoadCloud(Block):
    project_path_parameters = ("path",)
    type_id, title, category = "input.cloud", "Load PLY / PCD point cloud", "Input"
    block_type, subtype = "Input", "Point clouds"
    outputs = (Port("cloud", "cloud"), Port("count", "number"))
    defaults = {
        "path": "",
        "units": "m",
        "frame": "file",
        "max_points": 1000000,
        "track_source_rows": False,
    }
    parameter_choices = {"units": ("m", "mm", "unitless")}
    parameter_ranges = {"max_points": NumericRange(1, 2_000_000, whole=True)}
    parameter_help = {
        "path": "Choose an ASCII or binary PLY/PCD point cloud, or an ASCII OBJ.",
        "max_points": "Limit on declared point rows, including empty pixels in an organized PCD.",
        "track_source_rows": "Record original zero-based file rows if source_index is absent.",
    }

    def process(self, inputs):
        p = self.parameters
        cloud = read_cloud(
            cloud_path(self),
            p["units"],
            p["frame"],
            integer(p["max_points"], "Max points", 1, 2000000),
            track_source_rows=p["track_source_rows"],
        )
        return Result(
            {"cloud": cloud, "count": len(cloud.points)},
            graphics=cloud_graphics(self, cloud),
            passed=bool(len(cloud.points)),
        )


class SaveCloud(Block):
    type_id, title, category = "output.cloud", "Save PLY / PCD point cloud", "Output"
    block_type, subtype = "Output", "Point clouds"
    inputs = (Port("cloud", "cloud"),)
    outputs = (Port("path", "data"),)
    defaults = {
        "path": "cloud.ply",
        "binary_ply": True,
        "pcd_encoding": "ascii",
        "pcd_organization": "unorganized",
        "write_inspection_attributes": True,
    }
    parameter_choices = {
        "pcd_encoding": ("ascii", "binary"),
        "pcd_organization": ("unorganized", "source_pixels"),
    }
    parameter_help = {
        "path": "Choose a new .ply, .pcd or .obj file.",
        "binary_ply": "Use binary vertices for .ply files.",
        "pcd_encoding": "Use ASCII rows or uncompressed little-endian binary records for .pcd.",
        "pcd_organization": (
            "Use a flat list or restore the source pixel grid with NaN at missing pixels."
        ),
        "write_inspection_attributes": "Write scalar inspection columns with PLY or PCD.",
    }

    def process(self, inputs):
        path = cloud_path(self)
        p = self.parameters
        suffix = path.suffix.lower()
        binary = (suffix == ".ply" and p["binary_ply"]) or (
            suffix == ".pcd" and p["pcd_encoding"] == "binary"
        )
        write_cloud(
            path,
            inputs["cloud"],
            binary,
            pcd_organization=p["pcd_organization"],
            write_inspection_attributes=p["write_inspection_attributes"],
        )
        attributes = inputs["cloud"].attributes if p["write_inspection_attributes"] else {}
        message = "Saved point cloud"
        if attributes:
            message += f" with {', '.join(attributes)}"
        return Result({"path": str(path)}, message=message)


CLOUD_IO_BLOCKS = (LoadCloud, SaveCloud)
