"""Bounded PCD headers, typed point rows and organized grid output."""

import json

import numpy as np

from .geometry_fileio import MAX_GEOMETRY_BYTES, geometry_output
from .processing import integer

_XYZ = ("x", "y", "z")
_NORMALS = ("normal_x", "normal_y", "normal_z")
_KNOWN = {*_XYZ, *_NORMALS, "nx", "ny", "nz", "rgb", "rgba", "red", "green", "blue", "r", "g", "b"}
_TYPES = {
    ("F", 4): "<f4",
    ("F", 8): "<f8",
    **{(kind, size): f"<{kind.lower()}{size}" for kind in ("I", "U") for size in (1, 2, 4, 8)},
}
_IDENTITY_VIEW = [0, 0, 0, 1, 0, 0, 0]
_MAX_POINTS = 2_000_000


def _dimension(config, key):
    values = config.get(key)
    if values is None:
        return None
    if len(values) != 1:
        raise ValueError(f"PCD {key} requires one integer")
    return integer(int(values[0]), f"PCD {key}", 1 if key == "HEIGHT" else 0, _MAX_POINTS)


def _viewpoint(values):
    numbers = np.asarray(values, np.float64)
    if numbers.shape != (7,) or not np.isfinite(numbers).all() or not np.any(numbers[3:]):
        raise ValueError("PCD VIEWPOINT requires finite XYZ and a nonzero WXYZ quaternion")
    return numbers.tolist()


def _layout(config, encoding):
    names = config.get("FIELDS", config.get("COLUMNS", []))
    if not names or len(names) > 64:
        raise ValueError("PCD must declare 1–64 fields")
    sizes = config.get("SIZE", ["4"] * len(names) if encoding == "ascii" else [])
    types = config.get("TYPE", ["F"] * len(names) if encoding == "ascii" else [])
    counts = config.get("COUNT", ["1"] * len(names))
    if not len(names) == len(sizes) == len(types) == len(counts):
        raise ValueError("PCD FIELDS, SIZE, TYPE and COUNT must describe the same fields")
    known = {}
    offset = scalar = 0
    used = set()
    for name, size, kind, count in zip(names, sizes, types, counts, strict=True):
        size, count = int(size), int(count)
        dtype = _TYPES.get((kind, size))
        if dtype is None or not 1 <= count <= 4096:
            raise ValueError(f"Unsupported PCD type, size or count for {name}")
        if name != "_" and name in used:
            raise ValueError(f"Duplicate PCD field {name}")
        used.add(name)
        if name in _KNOWN:
            if count != 1 or (name in ("rgb", "rgba") and size != 4):
                raise ValueError(f"PCD field {name} requires a scalar of the supported size")
            known[name] = (dtype, offset, scalar)
        offset += size * count
        scalar += count
    if any(name not in known for name in _XYZ):
        raise ValueError("PCD must contain scalar XYZ fields")
    return known, offset, scalar


def _ascii_columns(stream, count, fields, scalars):
    if count * scalars > 24_000_000:
        raise ValueError("ASCII PCD exceeds 24 million scalar fields")
    columns = {name: np.empty(count, dtype=dtype) for name, (dtype, _, _) in fields.items()}
    chunk = max(1, min(4096, 32768 // scalars))
    for start in range(0, count, chunk):
        end = min(start + chunk, count)
        rows = []
        while len(rows) < end - start:
            line = stream.readline(8193)
            if not line or len(line) > 8192:
                raise ValueError("Truncated or oversized ASCII PCD row")
            row = line.split()
            if row and not row[0].startswith(b"#"):
                if len(row) != scalars:
                    raise ValueError("Malformed ASCII PCD row")
                rows.append(row)
        for name, (dtype, _, index) in fields.items():
            try:
                with np.errstate(over="ignore"):
                    columns[name][start:end] = np.asarray([row[index] for row in rows], dtype=dtype)
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"Invalid ASCII PCD value in {name}") from exc
    for line in iter(lambda: stream.readline(8193), b""):
        tail = line.strip()
        if len(line) > 8192 or (tail and not tail.startswith(b"#")):
            raise ValueError("ASCII PCD contains data beyond its declared point count")
    return columns


def read_pcd_vertices(stream, headers, max_points):
    config = {}
    for line in headers:
        if line and not line.startswith("#"):
            key, *values = line.split()
            if key in config:
                raise ValueError(f"Duplicate PCD header entry {key}")
            config[key] = values
    encoding = config.get("DATA")
    if encoding not in (["ascii"], ["binary"]):
        raise ValueError("PCD requires DATA ascii or uncompressed DATA binary")
    mode = encoding[0]
    width, height, count = (_dimension(config, key) for key in ("WIDTH", "HEIGHT", "POINTS"))
    height = height if height is not None else 1
    if width is None and mode == "ascii" and height == 1:
        width = count
    if width is None or (height > 1 and width == 0):
        raise ValueError("PCD requires valid WIDTH and HEIGHT dimensions")
    if count is not None and count != width * height:
        raise ValueError("PCD POINTS does not match WIDTH × HEIGHT")
    count = integer(width * height, "PCD point count (including holes)", 0, max_points)
    viewpoint = _viewpoint(config["VIEWPOINT"]) if "VIEWPOINT" in config else None
    fields, stride, scalars = _layout(config, mode)
    if mode == "binary":
        offset = stream.tell()
        expected = count * stride
        if expected > MAX_GEOMETRY_BYTES or stream.seek(0, 2) - offset != expected:
            raise ValueError("Binary PCD body length does not match its point layout")
        stream.seek(offset)
        dtype = np.dtype(
            {
                "names": list(fields),
                "formats": [field[0] for field in fields.values()],
                "offsets": [field[1] for field in fields.values()],
                "itemsize": stride,
            }
        )
        body = stream.read(expected)
        if len(body) != expected or stream.read(1):
            raise ValueError("Binary PCD body changed while reading")
        records = np.frombuffer(body, dtype=dtype)
        columns = {name: records[name] for name in fields}
    else:
        columns = _ascii_columns(stream, count, fields, scalars)
    return columns, (width, height) if height > 1 else None, viewpoint


def write_pcd_cloud(path, cloud, binary, organization):
    width, height = len(cloud.points), 1
    indices = slice(None)
    if organization == "source_pixels":
        if cloud.pixels is None or cloud.source_size is None or cloud.source_size[1] <= 1:
            raise ValueError("Organized PCD requires a source pixel grid with at least two rows")
        width, height = cloud.source_size
        if width * height > _MAX_POINTS:
            raise ValueError("Organized PCD is limited to 2 million pixel locations")
        indices = cloud.pixels[:, 1].astype(np.int64) * width + cloud.pixels[:, 0]
        if len(np.unique(indices)) != len(indices):
            raise ValueError("Organized PCD cannot write several points to the same source pixel")
    elif organization != "unorganized":
        raise ValueError("Choose unorganized or source_pixels for PCD organization")
    vectors = list(zip(_XYZ, cloud.points.T, strict=True))
    if cloud.normals is not None:
        vectors.extend(zip(_NORMALS, cloud.normals.T, strict=True))
    dtype = [(name, "<f4") for name, _ in vectors]
    if cloud.colors is not None:
        dtype.append(("rgb", "<f4" if binary else "<u4"))
    count = width * height
    records = np.zeros(count, dtype=dtype)
    for name, values in vectors:
        records[name].fill(np.nan)
        records[name][indices] = values
    if cloud.colors is not None:
        r, g, b = cloud.colors.astype(np.uint32).T
        packed = ((r << 16) | (g << 8) | b).astype("<u4", copy=False)
        records["rgb"][indices] = packed.view("<f4") if binary else packed
    viewpoint = cloud.metadata.get("pcd_viewpoint")
    if (
        isinstance(viewpoint, dict)
        and viewpoint.get("frame") == cloud.frame
        and viewpoint.get("units") == cloud.units
    ):
        view = _viewpoint(viewpoint.get("values"))
    else:
        view = _IDENTITY_VIEW
    names = list(records.dtype.names)
    metadata = json.dumps({"units": cloud.units, "frame": cloud.frame}, ensure_ascii=True)
    metadata_line = f"# dataflow_metadata {metadata}"
    if len(metadata_line) >= 8192:
        raise ValueError("PCD frame metadata exceeds the header line limit")
    header = [
        "# .PCD v0.7",
        metadata_line,
        "VERSION 0.7",
        "FIELDS " + " ".join(names),
        "SIZE " + " ".join(["4"] * len(names)),
        "TYPE " + " ".join(t[1].upper() for _, t in dtype),
        "COUNT " + " ".join(["1"] * len(names)),
        f"WIDTH {width}",
        f"HEIGHT {height}",
        "VIEWPOINT " + " ".join(format(value, ".17g") for value in view),
        f"POINTS {count}",
        f"DATA {'binary' if binary else 'ascii'}",
    ]
    with geometry_output(path, ".pcd-") as stream:
        stream.write(("\n".join(header) + "\n").encode("ascii"))
        if binary:
            stream.write(memoryview(records).cast("B"))
        else:
            formats = ["%u" if t == "<u4" else "%.9g" for _, t in dtype]
            np.savetxt(stream, records, fmt=formats)
