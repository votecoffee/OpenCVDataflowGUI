"""Bounded radius filtering of binary PLY scans using a disk-backed spatial index."""

import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.spatial import cKDTree

from .cloud_io import PLY_TYPES, cloud_headers, cloud_path, ply_vertex_schema
from .cloud_outliers import _radius_counts
from .cloud_projection import METERS
from .core import Block, Port, Result
from .geometry_fileio import MAX_GEOMETRY_BYTES, geometry_output
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE

_BATCH_ROWS = 8192
_INDEX_CACHE_KIB = 8192
_MAX_SOURCE_ROWS = 16_000_000
_MAX_WINDOW_POINTS = 250_000
_WINDOW_DTYPE = np.dtype([("row", "<i8"), ("xyz", "<f8", 3), ("core", "?")])
_SQL_WINDOW = """
SELECT id, x, y, z, t0 = ? AND t1 = ? AND t2 = ?
FROM points INDEXED BY points_by_tile
WHERE t0 IN (?, ?, ?) AND t1 IN (?, ?, ?) AND t2 BETWEEN ? AND ?
AND x BETWEEN ? AND ? AND y BETWEEN ? AND ? AND z BETWEEN ? AND ?
LIMIT ?
"""


class ScanInterrupted(Exception):
    """A configured scan limit or cancellation was reached."""


def _read_scan_header(stream, settings):
    stat = os.fstat(stream.fileno())
    if stat.st_size > MAX_GEOMETRY_BYTES:
        raise ValidationError("Large clouds are limited to 256 MiB per PLY file", "path")
    header, metadata = cloud_headers(stream)
    element_names = [
        line.split()[1] for line in header if line.startswith("element ") and len(line.split()) == 3
    ]
    if header[0] != "ply" or header[-1] != "end_header" or element_names != ["vertex"]:
        raise ValidationError("Use a vertex-only binary PLY scan", "path")
    format_line, count, properties = ply_vertex_schema(header, int(settings["max_source_rows"]))
    if format_line != "format binary_little_endian 1.0":
        raise ValidationError("Use binary little-endian PLY", "path")
    dtype = np.dtype(properties)
    if not {"x", "y", "z"} <= set(dtype.names):
        raise ValidationError("PLY scan requires x, y and z properties", "path")
    body_start = stream.tell()
    if body_start + dtype.itemsize * count != stat.st_size:
        raise ValidationError("Binary PLY size differs from its declared vertex rows", "path")
    units = metadata.get("units", settings["units"])
    frame = metadata.get("frame", settings["frame"])
    if units not in ("m", "mm") or not isinstance(frame, str) or not frame.strip():
        raise ValidationError("Use metric units and a named frame", "units", "frame")
    return dtype, count, body_start, units, frame, stat


def _bands(stream, dtype, count, cancelled):
    for first in range(0, count, _BATCH_ROWS):
        if cancelled():
            raise ScanInterrupted("cancelled")
        size = min(count - first, _BATCH_ROWS)
        raw = stream.read(size * dtype.itemsize)
        if len(raw) != size * dtype.itemsize:
            raise ValidationError("Binary PLY changed while it was being read", "path")
        yield first, np.frombuffer(raw, dtype, count=size)


def _open_index(db, settings, export_rows):
    db.execute("PRAGMA page_size = 4096")
    db.execute("PRAGMA journal_mode = OFF")
    db.execute(f"PRAGMA cache_size = -{_INDEX_CACHE_KIB}")
    db.execute("PRAGMA temp_store = FILE")
    db.execute(f"PRAGMA max_page_count = {int(settings['max_index_mib']) * 256}")
    db.execute("CREATE TABLE points (id INTEGER PRIMARY KEY, t0, t1, t2, x, y, z)")
    db.execute("CREATE INDEX points_by_tile ON points (t0, t1, t2)")
    if export_rows:
        db.execute("CREATE TABLE retained (id INTEGER PRIMARY KEY)")


def _stage_points(db, stream, dtype, count, tile_width, scale, cancelled):
    valid_count = 0
    for first, rows in _bands(stream, dtype, count, cancelled):
        # Match the XYZ precision of the ordinary PLY reader.
        with np.errstate(over="ignore", invalid="ignore"):
            xyz = np.column_stack([rows[axis] for axis in ("x", "y", "z")]).astype(np.float32)
        ok = np.isfinite(xyz).all(axis=1)
        points = xyz[ok].astype(np.float64) * scale
        cells = np.floor(points / tile_width)
        if np.any(np.abs(cells) >= 2**40):
            raise ValidationError(
                "Tile size is too small for the scan coordinate range", "tile_size_m"
            )
        keys = cells.astype(np.int64)
        row_ids = np.flatnonzero(ok) + first
        db.executemany(
            "INSERT INTO points VALUES (?, ?, ?, ?, ?, ?, ?)",
            zip(
                row_ids.tolist(),
                *map(np.ndarray.tolist, keys.T),
                *map(np.ndarray.tolist, points.T),
                strict=True,
            ),
        )
        valid_count += len(row_ids)
    db.commit()
    return valid_count


def _classify_tiles(db, settings, export_rows, report, cancelled):
    radius = float(settings["radius_m"])
    minimum = int(settings["minimum_neighbors"])
    width = float(settings["tile_size_m"])
    limit = int(settings["max_window_points"])
    tiles = db.execute("SELECT DISTINCT t0, t1, t2 FROM points ORDER BY t0, t1, t2")
    for index, key in enumerate(tiles, 1):
        if cancelled():
            raise ScanInterrupted("cancelled")
        if index > settings["max_tiles"]:
            raise ScanInterrupted("tile_budget")
        a, b, c = key
        # Radius is at most half a tile width. A radius neighbor can only
        # lie in the core tile or one of its 26 immediate neighbors.
        # Clip the neighboring cells to the radius-expanded tile. The margin
        # covers division and multiplication rounding when reconstructing tile edges.
        bounds = []
        for axis in key:
            low, high = axis * width - radius, (axis + 1) * width + radius
            margin = 8 * np.finfo(np.float64).eps * max(abs(low), abs(high), width)
            bounds.extend((low - margin, high + margin))
        window = db.execute(
            _SQL_WINDOW,
            (*key, a - 1, a, a + 1, b - 1, b, b + 1, c - 1, c + 1, *bounds, limit + 1),
        )
        # Read bounded records without building a list of Python tuples.
        rows = np.fromiter(
            ((row, (x, y, z), owner) for row, x, y, z, owner in window),
            dtype=_WINDOW_DTYPE,
        )
        if len(rows) > limit:
            raise ScanInterrupted("window_budget")
        xyz = np.ascontiguousarray(rows["xyz"])
        core = rows["core"]
        counts = _radius_counts(cKDTree(xyz), xyz[core], radius, cancelled)
        if counts is None:
            raise ScanInterrupted("cancelled")
        accepted = counts >= minimum
        if export_rows:
            ids = rows["row"][core][accepted]
            db.executemany("INSERT INTO retained VALUES (?)", ((int(row),) for row in ids))
        low, high = int(counts.min()), int(counts.max())
        previous_low = report["minimum_observed_neighbors"]
        report.update(
            tiles=index,
            candidate_visits=report["candidate_visits"] + len(rows),
            peak_window_points=max(report["peak_window_points"], len(rows)),
            peak_core_points=max(report["peak_core_points"], len(counts)),
            kept_points=report["kept_points"] + int(accepted.sum()),
            classified_points=report["classified_points"] + len(counts),
            neighbor_count_sum=report["neighbor_count_sum"] + int(counts.sum(dtype=np.int64)),
            minimum_observed_neighbors=low if previous_low is None else min(low, previous_low),
            maximum_observed_neighbors=max(high, report["maximum_observed_neighbors"]),
        )
    db.commit()


def _require_unchanged(stream, expected):
    actual = os.fstat(stream.fileno())
    if (actual.st_size, actual.st_mtime_ns) != (expected.st_size, expected.st_mtime_ns):
        raise ValidationError("Binary PLY changed while it was being processed", "path")


def _write_retained(
    db, stream, destination, dtype, count, units, frame, report, cancelled, original
):
    add_id = "source_index" not in dtype.names
    if add_id and len(dtype.names) >= 64:
        raise ValidationError(
            "There is no free PLY property for the generated source_index", "path"
        )
    out_dtype = np.dtype(dtype.descr + [("source_index", "<i4")]) if add_id else dtype
    aliases = {np.dtype(value).str: name for name, value in PLY_TYPES.items()}
    identity = json.dumps({"units": units, "frame": frame}, ensure_ascii=True)
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"comment dataflow_metadata {identity}",
        f"element vertex {report['kept_points']}",
        *[f"property {aliases[encoding]} {name}" for name, encoding in out_dtype.descr],
        "end_header",
    ]
    contents = ("\n".join(header) + "\n").encode("ascii")
    if (
        len(header[2]) >= 8192
        or len(contents) + out_dtype.itemsize * report["kept_points"] > MAX_GEOMETRY_BYTES
    ):
        raise ScanInterrupted("output_budget")
    with geometry_output(destination, ".cloud-tile-") as target:
        target.write(contents)
        for first, data in _bands(stream, dtype, count, cancelled):
            ids = np.fromiter(
                (
                    r[0] - first
                    for r in db.execute(
                        "SELECT id FROM retained WHERE id >= ? AND id < ? ORDER BY id",
                        (first, first + len(data)),
                    )
                ),
                dtype=np.intp,
            )
            selected = data[ids]
            if add_id:
                packed = np.empty(len(selected), out_dtype)
                for name in dtype.names:
                    packed[name] = selected[name]
                packed["source_index"] = ids + first
                selected = packed
            target.write(selected.tobytes())
        if cancelled():
            raise ScanInterrupted("cancelled")
        _require_unchanged(stream, original)


def _parameter_path(block, name):
    value = block.parameters[name]
    if not isinstance(value, str):
        raise ValidationError("Enter a file or folder path", name)
    if not value.strip():
        return None
    path = Path(value)
    return block.project_dir / path if block.project_dir and not path.is_absolute() else path


class TiledCloudRadius(Block):
    type_id, title, category = (
        "input.cloud_tiled_radius",
        "Filter a large PLY cloud by tiles",
        "Input",
    )
    block_type, subtype = "Input", "Point clouds"
    description = (
        "Stream a binary PLY scan, count metric neighbors, and optionally write kept rows."
    )
    project_path_parameters = ("path", "output_path", "scratch_directory")
    outputs = (Port("path", "text", required=False), Port("report", "data"))
    defaults = {
        "path": "",
        "output_path": "",
        "scratch_directory": "",
        "units": "m",
        "frame": "file",
        "radius_m": 0.01,
        "minimum_neighbors": 3,
        "tile_size_m": 0.08,
        "max_source_rows": 4_000_000,
        "max_window_points": 100_000,
        "max_tiles": 250_000,
        "max_index_mib": 512,
    }
    parameter_choices = {"units": ("m", "mm")}
    parameter_ranges = {
        "radius_m": NumericRange(1e-9, 1e12),
        "minimum_neighbors": NumericRange(1, _MAX_WINDOW_POINTS - 1, whole=True),
        "tile_size_m": NumericRange(2e-9, 1e16),
        "max_source_rows": NumericRange(1, _MAX_SOURCE_ROWS, whole=True),
        "max_window_points": NumericRange(1, _MAX_WINDOW_POINTS, whole=True),
        "max_tiles": NumericRange(1, 1_000_000, whole=True),
        "max_index_mib": NumericRange(1, 4096, whole=True),
    }
    parameter_help = {
        "path": "Vertex-only binary little-endian PLY, up to 256 MiB.",
        "output_path": "Optional new PLY file. Leave blank to report counts only.",
        "scratch_directory": "Temporary index folder. Defaults to the system temp folder.",
        "units": "Units for files without Dataflow metadata.",
        "frame": "Coordinate frame for files without Dataflow metadata.",
        "radius_m": "Inclusive Euclidean radius in meters.",
        "minimum_neighbors": "Required number of other rows. Coincident rows count separately.",
        "tile_size_m": "Cubic tile edge in meters. Must be at least twice the radius.",
        "max_source_rows": "Limit on source rows, including nonfinite XYZ.",
        "max_window_points": "Maximum points in the core tile and its radius halo.",
        "max_tiles": "Maximum occupied core tiles to process.",
        "max_index_mib": "Maximum size of the temporary SQLite index, in MiB.",
    }
    output_help = {
        "path": "New PLY path after complete, exclusive publication.",
        "report": "Frame, counts, neighborhood reduction and observed window sizes.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["tile_size_m"] < 2 * p["radius_m"]:
            raise ValidationError(
                "Tile size must be at least twice the radius", "tile_size_m", "radius_m"
            )

    def process(self, inputs):
        p = self.parameters
        source = cloud_path(self)
        destination = _parameter_path(self, "output_path")
        scratch = _parameter_path(self, "scratch_directory")
        if source.suffix.lower() != ".ply":
            raise ValidationError("Choose a binary PLY source file", "path")
        if destination is not None and (
            destination.suffix.lower() != ".ply" or os.path.lexists(destination)
        ):
            raise ValidationError("Choose a new PLY output path", "output_path")
        report = {
            "radius_m": p["radius_m"],
            "minimum_neighbors": p["minimum_neighbors"],
            "tile_size_m": p["tile_size_m"],
            "batch_rows": _BATCH_ROWS,
            "sqlite_cache_kib": _INDEX_CACHE_KIB,
            "tiles": 0,
            "candidate_visits": 0,
            "peak_window_points": 0,
            "peak_core_points": 0,
            "classified_points": 0,
            "kept_points": 0,
            "neighbor_count_sum": 0,
            "minimum_observed_neighbors": None,
            "maximum_observed_neighbors": 0,
        }
        output = NO_VALUE
        status = "ok"
        with source.open("rb") as stream:
            dtype, count, start, units, frame, original = _read_scan_header(stream, p)
            report.update(source_rows=count, cloud_units=units, frame=frame)
            with TemporaryDirectory(prefix="dataflow-cloud-", dir=scratch) as temporary:
                database = Path(temporary) / "spatial.sqlite"
                try:
                    with closing(sqlite3.connect(database)) as db:
                        _open_index(db, p, destination is not None)
                        valid = _stage_points(
                            db,
                            stream,
                            dtype,
                            count,
                            p["tile_size_m"],
                            METERS[units],
                            self._cancelled,
                        )
                        report.update(valid_points=valid, invalid_xyz_rows=count - valid)
                        _classify_tiles(db, p, destination is not None, report, self._cancelled)
                        report["index_bytes"] = db.execute("PRAGMA page_count").fetchone()[0] * 4096
                        report["removed_points"] = valid - report["kept_points"]
                        report["mean_observed_neighbors"] = (
                            report["neighbor_count_sum"] / valid if valid else None
                        )
                        _require_unchanged(stream, original)
                        if self._cancelled():
                            raise ScanInterrupted("cancelled")
                        if not report["kept_points"]:
                            status = "no_inliers"
                        elif destination is not None:
                            stream.seek(start)
                            _write_retained(
                                db,
                                stream,
                                destination,
                                dtype,
                                count,
                                units,
                                frame,
                                report,
                                self._cancelled,
                                original,
                            )
                            output = str(destination)
                except ScanInterrupted as exc:
                    status = str(exc)
                except sqlite3.OperationalError as exc:
                    if getattr(exc, "sqlite_errorcode", None) != sqlite3.SQLITE_FULL:
                        raise
                    status = "index_budget"
        report["status"] = status
        return Result(
            {"path": output, "report": report},
            passed=status == "ok",
            score=report["kept_points"] / count if count else 0.0,
            fail_reasons=[] if status == "ok" else [f"cloud_tiles_{status}"],
            message=f"{report['kept_points']:,}/{count:,} rows kept · {status}",
        )
