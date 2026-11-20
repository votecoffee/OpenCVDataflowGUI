"""Project XYZ into a calibrated angular scanner grid with return provenance."""

import hashlib
import json

import cv2
import numpy as np

from .cloud_projection import METERS
from .core import Port, Result
from .geometry3d_common import Tool3D, rigid
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, SPHERICAL_RANGE_GRID, frame

_MAX_ROWS = 1024
_MAX_CELLS = 2_097_152
_MAX_POINTS = 1_000_000


def _elevation_edges(parameters):
    raw = parameters["elevation_edges_deg"]
    if not isinstance(raw, list) or len(raw) > _MAX_ROWS + 1:
        raise ValidationError("Enter a list of at most 1,025 angular edges", "elevation_edges_deg")
    if raw:
        if len(raw) < 2 or any(type(value) not in (int, float) for value in raw):
            raise ValidationError("Enter at least two numeric angular edges", "elevation_edges_deg")
        bound = NumericRange(-90, 90)
        edges = np.array(
            [bound.check(value, "Elevation edge", "elevation_edges_deg") for value in raw]
        )
        if np.any(np.diff(edges) <= 0):
            raise ValidationError(
                "Elevation edges must be strictly increasing", "elevation_edges_deg"
            )
        return edges
    lo, hi = float(parameters["elevation_min_deg"]), float(parameters["elevation_max_deg"])
    if lo >= hi:
        raise ValidationError(
            "Maximum elevation must exceed minimum elevation",
            "elevation_min_deg",
            "elevation_max_deg",
        )
    edges = np.linspace(lo, hi, int(parameters["rows"]) + 1)
    if np.any(np.diff(edges) <= 0):
        raise ValidationError("Elevation span is too small for this row count", "rows")
    return edges


def _scanner_xyz(cloud, scanner_frame, pose):
    if cloud.units not in METERS:
        raise ValidationError("Scanner projection requires meters or millimeters", "cloud")
    xyz = cloud.points.astype(np.float64) * METERS[cloud.units]
    if pose is None:
        if cloud.frame != scanner_frame:
            raise ValidationError(
                "Connect a pose from the cloud frame to the scanner frame", "transform"
            )
        return xyz
    if (
        not isinstance(pose, dict)
        or pose.get("source_frame") != cloud.frame
        or pose.get("target_frame") != scanner_frame
        or pose.get("units") not in METERS
    ):
        raise ValidationError(
            "Pose must map the cloud frame to the scanner in metric units", "transform"
        )
    motion = rigid(pose)
    return xyz @ motion[:3, :3].T + motion[:3, 3] * METERS[pose["units"]]


def _return_owners(rows, cells, distance, intensity, policy):
    if policy == "strongest":
        signal = intensity[rows]
        power = np.where(np.isfinite(signal), -signal, np.inf)
        order = np.lexsort((rows, distance, power, cells))
    else:
        rank = distance if policy == "nearest" else -distance
        order = np.lexsort((rows, rank, cells))
    sorted_cells = cells[order]
    start = np.r_[0, 1 + np.flatnonzero(sorted_cells[1:] != sorted_cells[:-1])]
    return sorted_cells[start], rows[order[start]], np.diff(start, append=len(order))


def _raster(size, fill, dtype, cells, values):
    image = np.full((size[1], size[0]), fill, dtype)
    image.flat[cells] = values
    return image


def _range_preview(ranges, cells, values):
    raster = np.zeros(ranges.shape, np.uint8)
    if len(values):
        low, high = float(values.min()), float(values.max())
        raster.flat[cells] = 128 if high == low else np.rint(255 * (values - low) / (high - low))
    color = cv2.applyColorMap(raster, cv2.COLORMAP_TURBO)
    color[~np.isfinite(ranges)] = 0
    return color


class ScannerRangeImage(Tool3D):
    type_id = "3d.cloud_range_image"
    title, subtype = "Organize scanner point cloud", "Point clouds"
    description = "Rasterize metric scanner returns into calibrated azimuth and elevation bins."
    payload_reference = "cloud"
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = (
        Port("cloud", "cloud"),
        Port("transform", "data", required=False, required_when_connected=True, visual_type="pose"),
    )
    outputs = (
        Port("range", "image", required=False),
        Port("intensity", "image", required=False),
        Port("valid_mask", "image", required=False, visual_type="region"),
        Port("source_rows", "image", required=False),
        Port("source_indices", "image", required=False),
        Port("return_count", "image", required=False),
        Port("report", "data"),
    )
    defaults = {
        "scanner_frame": "scanner",
        "columns": 720,
        "rows": 32,
        "azimuth_start_deg": -180.0,
        "azimuth_span_deg": 360.0,
        "elevation_min_deg": -25.0,
        "elevation_max_deg": 15.0,
        "elevation_edges_deg": [],
        "minimum_range_m": 0.01,
        "maximum_range_m": 150.0,
        "return_policy": "nearest",
    }
    parameter_choices = {"return_policy": ("nearest", "farthest", "strongest")}
    parameter_ranges = {
        "columns": NumericRange(1, 8192, whole=True),
        "rows": NumericRange(1, _MAX_ROWS, whole=True),
        "azimuth_start_deg": NumericRange(-180, 180),
        "azimuth_span_deg": NumericRange(1e-5, 360),
        "elevation_min_deg": NumericRange(-90, 90),
        "elevation_max_deg": NumericRange(-90, 90),
        "minimum_range_m": NumericRange(1e-9, 1e6),
        "maximum_range_m": NumericRange(1e-9, 1e6),
    }
    parameter_help = {
        "scanner_frame": "Scanner axes are X forward, Y left and Z up.",
        "columns": "Number of equally spaced azimuth bins.",
        "rows": "Number of equally spaced elevation bins when custom edges are empty.",
        "azimuth_start_deg": "Azimuth edge at column zero, measured from +X toward +Y.",
        "azimuth_span_deg": "Angular width of the grid. Use 360 degrees for a full revolution.",
        "elevation_min_deg": "Lower edge for the uniform elevation grid.",
        "elevation_max_deg": "Upper exclusive edge for the uniform elevation grid.",
        "elevation_edges_deg": "Optional calibrated edges in ascending order, from low to high.",
        "minimum_range_m": "Inclusive minimum radial distance to accept.",
        "maximum_range_m": "Inclusive maximum radial distance to accept.",
        "return_policy": "Select closest, farthest, or strongest measured intensity in each bin.",
    }
    input_help = {
        "cloud": "XYZ measured from the scanner origin or linked to it by a rigid pose.",
        "transform": "Optional metric pose mapping the cloud frame into the named scanner frame.",
    }
    output_help = {
        "range": "Float32 radial meters. Empty bins contain NaN.",
        "intensity": "Float64 intensity of the selected return. Missing values contain NaN.",
        "valid_mask": "255 in bins with a selected return.",
        "source_rows": "Exact int32 input row number. Empty bins contain -1.",
        "source_indices": "Optional int32 source_index attribute. Unknown values contain -1.",
        "return_count": "Admitted returns in each bin before the return policy.",
        "report": "Angular grid, frame, occupancy, clipping and collision counts.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        name = p["scanner_frame"]
        if not isinstance(name, str) or not name.strip() or len(name) > 128:
            raise ValidationError(
                "Use a named scanner frame of up to 128 characters", "scanner_frame"
            )
        height = len(_elevation_edges(p)) - 1
        if height * int(p["columns"]) > _MAX_CELLS:
            raise ValidationError("Angular grid exceeds 2,097,152 bins", "columns", "rows")
        if float(p["minimum_range_m"]) >= float(p["maximum_range_m"]):
            raise ValidationError(
                "Maximum range must exceed minimum range", "minimum_range_m", "maximum_range_m"
            )

    def process(self, inputs):
        cloud, p = inputs["cloud"], self.parameters
        if len(cloud.points) > _MAX_POINTS:
            raise ValidationError("Admit up to 1,000,000 point-cloud rows", "cloud")
        policy = p["return_policy"]
        intensity = cloud.attributes.get("intensity")
        if policy == "strongest" and intensity is None:
            raise ValidationError(
                "Strongest return requires the intensity attribute", "return_policy"
            )
        if self._cancelled():
            return self._cancelled_result()

        edges = _elevation_edges(p)
        columns, height = int(p["columns"]), len(edges) - 1
        size = (columns, height)
        start, span = float(p["azimuth_start_deg"]), float(p["azimuth_span_deg"])
        lower, upper = float(p["minimum_range_m"]), float(p["maximum_range_m"])
        with np.errstate(over="ignore", invalid="ignore"):
            xyz = _scanner_xyz(cloud, p["scanner_frame"], inputs.get("transform"))
            horizontal = np.hypot(xyz[:, 0], xyz[:, 1])
            radii = np.hypot(horizontal, xyz[:, 2])
        if not np.isfinite(radii).all():
            raise ValidationError(
                "Transformed scanner coordinates exceed finite range", "transform"
            )
        metric = (radii >= lower) & (radii <= upper)
        poles = metric & (horizontal == 0)
        rows = np.flatnonzero(metric & ~poles)
        az = np.degrees(np.arctan2(xyz[rows, 1], xyz[rows, 0]))
        el = np.degrees(np.arctan2(xyz[rows, 2], horizontal[rows]))
        offsets = np.remainder(az - start, 360.0)
        offsets[offsets == 360] = 0
        inside = (offsets < span) & (el >= edges[0]) & (el < edges[-1])
        outside = int(np.count_nonzero(~inside))
        col = np.minimum((offsets[inside] * columns / span).astype(np.intp), columns - 1)
        row = height - np.searchsorted(edges, el[inside], side="right")
        rows = rows[inside]
        if rows.size:
            cells, owners, counts = _return_owners(
                rows, row * columns + col, radii[rows], intensity, policy
            )
        else:
            cells = owners = counts = np.empty(0, dtype=np.intp)
        if self._cancelled():
            return self._cancelled_result()

        grid = {
            "model": "spherical_range_grid_v1",
            "scanner_frame": p["scanner_frame"],
            "image_size": list(size),
            "azimuth_start_deg": start,
            "azimuth_span_deg": span,
            "elevation_edges_deg": edges.tolist(),
            "row_order": "high_to_low",
            "axis_convention": "X forward, Y left, Z up",
        }
        digest = hashlib.sha256(json.dumps(grid, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        target = {"spatial": frame(size, f"{self.id}:angular:{digest}"), SPHERICAL_RANGE_GRID: grid}
        source_indices = cloud.attributes.get("source_index")
        values = radii[owners]
        output = {
            "range": _raster(size, np.nan, np.float32, cells, values),
            "valid_mask": _raster(size, 0, np.uint8, cells, 255),
            "source_rows": _raster(size, -1, np.int32, cells, owners),
            "return_count": _raster(size, 0, np.int32, cells, counts),
            "intensity": NO_VALUE
            if intensity is None
            else _raster(size, np.nan, np.float64, cells, intensity[owners]),
            "source_indices": NO_VALUE
            if source_indices is None
            else _raster(size, -1, np.int32, cells, source_indices[owners]),
            "report": {
                "grid": grid,
                "input_frame": cloud.frame,
                "input_units": cloud.units,
                "units": "m",
                "status": "ok" if len(cells) else "no_returns",
                "return_policy": policy,
                "minimum_range_m": lower,
                "maximum_range_m": upper,
                "input_points": len(xyz),
                "range_clipped_points": int(np.count_nonzero(~metric)),
                "undefined_azimuth_points": int(np.count_nonzero(poles)),
                "outside_grid_points": outside,
                "accepted_returns": len(rows),
                "occupied_bins": len(cells),
                "empty_bins": columns * height - len(cells),
                "multiple_return_bins": int(np.count_nonzero(counts > 1)),
                "discarded_returns": len(rows) - len(cells),
                "strongest_fallback_bins": int(np.isnan(intensity[owners]).sum())
                if policy == "strongest"
                else 0,
            },
        }
        graphics = (
            {"Radial range": _range_preview(output["range"], cells, values)}
            if self.capture_graphics
            else {}
        )
        return Result(
            output,
            graphics=graphics,
            payloads={
                name: target for name, value in output.items() if isinstance(value, np.ndarray)
            },
            passed=bool(len(cells)),
            fail_reasons=[] if len(cells) else ["scanner_range_image_no_returns"],
            message=f"{len(cells):,} occupied bins from {len(rows):,} returns",
        )

    def _cancelled_result(self):
        return Result(
            {
                **dict.fromkeys((port.name for port in self.outputs), NO_VALUE),
                "report": {"status": "cancelled"},
            },
            passed=False,
            fail_reasons=["scanner_range_image_cancelled"],
            message="Scanner projection cancelled",
        )
