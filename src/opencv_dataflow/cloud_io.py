"""Bounded PLY/PCD point-cloud interchange, preserving explicit frame and unit metadata."""

import json
from pathlib import Path

import numpy as np

from .core import Block, Port, Result
from .geometry3d_common import cloud_graphics
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


def make_cloud(columns, metadata, units, frame):
    if not all(k in columns for k in ("x", "y", "z")):
        raise ValueError("Cloud file must contain XYZ fields")
    xyz = np.column_stack([columns[k] for k in ("x", "y", "z")])
    keep = np.isfinite(xyz).all(axis=1)
    normals = None
    if all(k in columns for k in ("nx", "ny", "nz")):
        normals = np.column_stack([columns[k] for k in ("nx", "ny", "nz")])
    elif all(k in columns for k in ("normal_x", "normal_y", "normal_z")):
        normals = np.column_stack([columns[k] for k in ("normal_x", "normal_y", "normal_z")])
    if normals is not None:
        lengths = np.linalg.norm(normals, axis=1)
        keep &= np.isfinite(normals).all(axis=1) & (lengths > 1e-8)
        normals = normals[keep] / lengths[keep, None]
    colors = None
    if all(k in columns for k in ("red", "green", "blue")):
        colors = np.column_stack([columns[k] for k in ("red", "green", "blue")])[keep]
    elif "rgb" in columns:
        packed = columns["rgb"]
        packed = (
            packed.astype(np.float32).view(np.uint32)
            if np.issubdtype(packed.dtype, np.floating)
            else packed.astype(np.uint32)
        )
        colors = np.c_[(packed >> 16) & 255, (packed >> 8) & 255, packed & 255][keep]
    return PointCloud(
        xyz[keep], metadata.get("units", units), metadata.get("frame", frame), colors, normals
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


def read_cloud(path, units="m", frame="file", max_points=2000000):
    if path.stat().st_size > 256 * 1024 * 1024:
        raise ValueError("Point-cloud files are limited to 256 MiB")
    metadata, headers = {}, []
    with path.open("rb") as stream:
        for _ in range(256):
            line = stream.readline(8192)
            if not line:
                raise ValueError("Truncated cloud header")
            text = line.decode("ascii").strip()
            headers.append(text)
            marker = "dataflow_metadata "
            if marker in text:
                metadata = json.loads(text.split(marker, 1)[1])
                if not isinstance(metadata, dict):
                    raise ValueError("Invalid cloud metadata comment")
            if text == "end_header" or text.startswith("DATA "):
                break
        else:
            raise ValueError("Cloud header exceeds 256 lines")
        if headers[0] == "ply":
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
            if format_line == "format binary_little_endian 1.0":
                dtype = np.dtype(properties)
                data = stream.read(count * dtype.itemsize)
                if len(data) != count * dtype.itemsize:
                    raise ValueError("Truncated binary PLY vertices")
                values = np.frombuffer(data, dtype=dtype)
                columns = {name: values[name] for name in names}
            elif format_line == "format ascii 1.0":
                values = ascii_vertices(stream, count, len(names))
                columns = {name: values[:, i] for i, name in enumerate(names)}
            else:
                raise ValueError("PLY must be ASCII or binary little endian 1.0")
        else:
            config = {h.split()[0]: h.split()[1:] for h in headers if h and not h.startswith("#")}
            if config.get("DATA") != ["ascii"]:
                raise ValueError("PCD currently supports DATA ascii only")
            names = config.get("FIELDS", [])
            if not names or len(set(names)) != len(names) or len(names) > 64:
                raise ValueError("Invalid PCD fields")
            count = integer(int(config["POINTS"][0]), "Point count", 0, max_points)
            if config.get("COUNT", ["1"] * len(names)) != ["1"] * len(names):
                raise ValueError("PCD fields must have COUNT 1")
            values = ascii_vertices(stream, count, len(names))
            columns = {name: values[:, i] for i, name in enumerate(names)}
            if "rgb" in columns:
                field_type = config.get("TYPE", ["F"] * len(names))[names.index("rgb")]
                columns["rgb"] = columns["rgb"].astype(
                    np.float32 if field_type == "F" else np.uint32
                )
    return make_cloud(columns, metadata, units, frame)


def write_cloud(path, cloud, binary=False):
    suffix = path.suffix.lower()
    if suffix not in (".ply", ".pcd") or (suffix == ".pcd" and binary):
        raise ValueError("Choose .ply (ASCII/binary) or .pcd (ASCII)")
    names, types, columns = ["x", "y", "z"], ["float"] * 3, [cloud.points[:, i] for i in range(3)]
    if cloud.normals is not None:
        names += ["nx", "ny", "nz"] if suffix == ".ply" else ["normal_x", "normal_y", "normal_z"]
        types += ["float"] * 3
        columns += [cloud.normals[:, i] for i in range(3)]
    if cloud.colors is not None:
        if suffix == ".ply":
            names += ["red", "green", "blue"]
            types += ["uchar"] * 3
            columns += [cloud.colors[:, i] for i in range(3)]
        else:
            rgb = cloud.colors.astype(np.uint32)
            names += ["rgb"]
            types += ["uint"]
            columns += [(rgb[:, 0] << 16) + (rgb[:, 1] << 8) + rgb[:, 2]]
    metadata = json.dumps({"units": cloud.units, "frame": cloud.frame}, ensure_ascii=True)
    n = len(cloud.points)
    if suffix == ".ply":
        header = [
            "ply",
            f"format {'binary_little_endian' if binary else 'ascii'} 1.0",
            f"comment dataflow_metadata {metadata}",
            f"element vertex {n}",
        ]
        header += [f"property {t} {name}" for t, name in zip(types, names)]
        header += ["end_header"]
    else:
        header = [
            "# .PCD v0.7",
            f"# dataflow_metadata {metadata}",
            "VERSION 0.7",
            "FIELDS " + " ".join(names),
            "SIZE " + " ".join("4" for _ in names),
            "TYPE " + " ".join("U" if t == "uint" else "F" for t in types),
            "COUNT " + " ".join("1" for _ in names),
            f"WIDTH {n}",
            "HEIGHT 1",
            "VIEWPOINT 0 0 0 1 0 0 0",
            f"POINTS {n}",
            "DATA ascii",
        ]
    with path.open("xb") as stream:
        stream.write(("\n".join(header) + "\n").encode("ascii"))
        if binary:
            values = np.empty(n, dtype=[(name, PLY_TYPES[t]) for name, t in zip(names, types)])
            for name, column in zip(names, columns):
                values[name] = column
            stream.write(values.tobytes())
        else:
            values = np.column_stack(columns)
            np.savetxt(
                stream, values, fmt=["%d" if t in ("uint", "uchar") else "%.9g" for t in types]
            )


class LoadCloud(Block):
    type_id, title, category = "input.cloud", "Load PLY / PCD point cloud", "Input"
    block_type, subtype = "Input", "Point clouds"
    outputs = (Port("cloud", "cloud"), Port("count", "number"))
    defaults = {"path": "", "units": "m", "frame": "file", "max_points": 1000000}
    parameter_choices = {"units": ("m", "mm", "unitless")}

    def process(self, inputs):
        p = self.parameters
        cloud = read_cloud(
            cloud_path(self),
            p["units"],
            p["frame"],
            integer(p["max_points"], "Max points", 1, 2000000),
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
    defaults = {"path": "cloud.ply", "binary_ply": True}

    def process(self, inputs):
        path = cloud_path(self)
        write_cloud(
            path,
            inputs["cloud"],
            self.parameters["binary_ply"] if path.suffix.lower() == ".ply" else False,
        )
        return Result(
            {"path": str(path)}, message="Saved point-cloud XYZ/color/normals/units/frame"
        )


CLOUD_IO_BLOCKS = (LoadCloud, SaveCloud)
