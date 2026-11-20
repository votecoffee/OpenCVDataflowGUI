"""Measure signed heights and surface coverage in a fixed metric datum grid."""

import cv2
import numpy as np

from .cloud_comparison import deviation_statistics
from .cloud_projection import METERS
from .core import Port, Result
from .geometry3d_common import Tool3D, matrix, rigid
from .images import ImageArray
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, frame

_MAX_POINTS = 2_000_000
_POSITION = NumericRange(-1e12, 1e12)


def _datum_axes(datum, cloud_frame):
    if not isinstance(datum, dict) or datum.get("units") not in ("m", "mm"):
        raise ValidationError("Datum needs a metric plane or fixture pose", "datum")
    scale = METERS[datum["units"]]
    model = datum.get("model")
    try:
        if model == "pose_3d_v1":
            reference = datum.get("source_frame")
            if (
                datum.get("target_frame") != cloud_frame
                or not isinstance(reference, str)
                or not reference.strip()
            ):
                raise ValueError("Fixture pose must map a named datum into the cloud frame")
            transform = rigid(datum)
            # Normalize the small drift allowed by the rigid pose validator.
            left, _, right = np.linalg.svd(transform[:3, :3])
            return transform[:3, 3] * scale, left @ right, reference, "fixture_pose"

        if datum.get("frame") != cloud_frame:
            raise ValueError("Datum plane frame must match the cloud frame")
        if model == "camera_depth_plane_v1":
            normal = matrix(datum.get("normal"), (3,), "Datum normal")
            offset = float(datum["offset"])
        elif datum.get("schema") == "point_cloud_model_v1" and datum.get("kind") == "plane":
            coefficients = matrix(datum.get("coefficients"), (4,), "Datum plane coefficients")
            normal, offset = coefficients[:3], coefficients[3]
        else:
            raise ValueError("Use a fitted cloud plane, camera depth plane or fixture pose")
        magnitude = np.linalg.norm(normal)
        if not np.isfinite(magnitude) or magnitude <= 1e-12 or not np.isfinite(offset):
            raise ValueError("Datum plane requires a finite, nonzero normal and offset")
        normal = normal / magnitude
        anchor = -float(offset / magnitude) * scale * normal
        # Project the cloud X axis onto the plane unless it is almost parallel to the normal.
        tangent_axis = int(abs(normal[0]) >= 0.9)
        u = np.eye(3)[tangent_axis] - normal * normal[tangent_axis]
        u /= np.linalg.norm(u)
        return anchor, np.column_stack((u, np.cross(normal, u), normal)), cloud_frame, "plane"
    except (ValueError, KeyError, TypeError) as exc:
        raise ValidationError(str(exc), "datum") from exc


def _grid_edges(p, axis, count):
    low, high = p[f"{axis}_min_m"], p[f"{axis}_max_m"]
    edges = np.linspace(low, high, count + 1)
    if np.any(np.diff(edges) <= 0):
        raise ValidationError(
            f"{axis.upper()} limits must define distinct grid edges",
            f"{axis}_min_m",
            f"{axis}_max_m",
            "grid_width",
            "grid_height",
        )
    return edges


def _peak_map(cell, delta, support):
    """Pick the greatest absolute residual, breaking ties in original scan order."""
    peak = np.zeros(len(support), np.float64)
    absolute = np.abs(delta)
    np.maximum.at(peak, cell, absolute)
    maxima = np.flatnonzero(absolute == peak[cell])
    selected = np.full(len(support), len(delta), dtype=np.intp)
    np.minimum.at(selected, cell[maxima], maxima)
    result = np.full(len(support), np.nan, np.float64)
    result[support] = delta[selected[support]]
    return result


def _preview(deviation, coverage, defects, negative, positive):
    scale = max(negative, positive, 1e-12)
    valid = coverage != 0
    display = np.zeros(deviation.shape, np.uint8)
    display[valid] = np.rint(127.5 * (np.clip(deviation[valid] / scale, -1, 1) + 1)).astype(
        np.uint8
    )
    color = cv2.applyColorMap(display, cv2.COLORMAP_TURBO)
    color[~valid] = [32, 28, 25]
    color[~valid & (defects != 0)] = [40, 168, 242]
    return color


class MeasureCloudDatum(Tool3D):
    type_id = "3d.cloud_datum"
    title, subtype = "Measure surface against a datum", "Point clouds"
    description = "Measure signed heights, datum-parallel flatness and observed grid support."
    preview_safe = parallel_safe = process_inputs_readonly = True
    payload_reference = "cloud"
    inputs = (Port("cloud", "cloud"), Port("datum", "data"))
    outputs = (
        Port("deviation", required=False),
        Port("coverage_mask", required=False, visual_type="region"),
        Port("defect_mask", required=False, visual_type="region"),
        Port("defects", "cloud", required=False),
        Port("report", "data"),
    )
    defaults = {
        "u_min_m": -0.1,
        "u_max_m": 0.1,
        "v_min_m": -0.1,
        "v_max_m": 0.1,
        "grid_width": 32,
        "grid_height": 32,
        "minimum_points_per_cell": 1,
        "minimum_coverage": 0.0,
        "nominal_height_m": 0.0,
        "negative_tolerance_m": 0.002,
        "positive_tolerance_m": 0.002,
        "minimum_in_tolerance_fraction": 1.0,
        "check_peak_to_valley": False,
        "maximum_peak_to_valley_m": 0.004,
        "max_input_points": 250_000,
    }
    parameter_ranges = {
        "u_min_m": _POSITION,
        "u_max_m": _POSITION,
        "v_min_m": _POSITION,
        "v_max_m": _POSITION,
        "grid_width": NumericRange(1, 1024, whole=True),
        "grid_height": NumericRange(1, 1024, whole=True),
        "minimum_points_per_cell": NumericRange(1, _MAX_POINTS, whole=True),
        "minimum_coverage": NumericRange(0, 1),
        "nominal_height_m": _POSITION,
        "negative_tolerance_m": NumericRange(0, 1e12),
        "positive_tolerance_m": NumericRange(0, 1e12),
        "minimum_in_tolerance_fraction": NumericRange(0, 1),
        "maximum_peak_to_valley_m": NumericRange(0, 1e12),
        "max_input_points": NumericRange(1, _MAX_POINTS, whole=True),
    }
    parameter_help = {
        "u_min_m": "Inclusive lower U edge of the measured footprint in meters.",
        "u_max_m": "Inclusive upper U edge of the measured footprint in meters.",
        "v_min_m": "Inclusive lower V edge of the measured footprint in meters.",
        "v_max_m": "Inclusive upper V edge of the measured footprint in meters.",
        "grid_width": "Number of equal cells along U.",
        "grid_height": "Number of equal cells along V. Image rows go from high V to low V.",
        "minimum_points_per_cell": "Required samples to publish a cell value in the deviation map.",
        "minimum_coverage": "Required fraction of footprint cells with sufficient samples.",
        "nominal_height_m": "Expected signed height from the datum along its positive normal.",
        "negative_tolerance_m": "Inclusive allowed distance below the nominal height.",
        "positive_tolerance_m": "Inclusive allowed distance above the nominal height.",
        "minimum_in_tolerance_fraction": "Required fraction of in-window points inside the band.",
        "check_peak_to_valley": "Apply a limit to overall datum-parallel height spread.",
        "maximum_peak_to_valley_m": "Maximum observed highest minus lowest datum height.",
        "max_input_points": "Maximum number of cloud rows admitted to measurement.",
    }
    input_help = {
        "cloud": "Measured points in meters or millimeters.",
        "datum": "Fitted plane or pose mapping datum-local XY and positive Z into the cloud.",
    }
    output_help = {
        "deviation": "Signed deviation from nominal in meters. NaN marks unsupported cells.",
        "coverage_mask": "Cells with the required number of measured points.",
        "defect_mask": "Cells with an observed height violation, including sparse cells.",
        "defects": "Original points outside the signed height band, in input order.",
        "report": "Grid geometry, signed heights, datum-parallel spread and tolerance counts.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        _grid_edges(p, "u", int(p["grid_width"]))
        _grid_edges(p, "v", int(p["grid_height"]))
        if type(p["check_peak_to_valley"]) is not bool:
            raise ValidationError(
                "Peak-to-valley checking must be a checkbox", "check_peak_to_valley"
            )

    def process(self, inputs):
        cloud, p = inputs["cloud"], self.parameters
        if cloud.units not in METERS:
            raise ValidationError("Surface measurement requires a cloud in m or mm", "cloud")
        if len(cloud.points) > p["max_input_points"]:
            raise ValidationError("Cloud exceeds the datum point budget", "max_input_points")
        origin, basis, reference, datum_kind = _datum_axes(inputs["datum"], cloud.frame)
        if not np.isfinite(origin).all() or np.abs(origin).max() > np.finfo(np.float32).max:
            raise ValidationError("Datum origin exceeds the cloud coordinate range", "datum")

        width, height = int(p["grid_width"]), int(p["grid_height"])
        u_edges, v_edges = _grid_edges(p, "u", width), _grid_edges(p, "v", height)
        nominal = float(p["nominal_height_m"])
        negative, positive = float(p["negative_tolerance_m"]), float(p["positive_tolerance_m"])
        geometry = {
            "model": "cloud_datum_grid_v1",
            "cloud_frame": cloud.frame,
            "reference_frame": reference,
            "datum_kind": datum_kind,
            "units": "m",
            "origin_cloud_m": origin.tolist(),
            "u_axis_cloud": basis[:, 0].tolist(),
            "v_axis_cloud": basis[:, 1].tolist(),
            "normal_cloud": basis[:, 2].tolist(),
            "uv_window_m": [float(v) for v in (u_edges[0], v_edges[0], u_edges[-1], v_edges[-1])],
            "cell_pitch_m": [
                (u_edges[-1] - u_edges[0]) / width,
                (v_edges[-1] - v_edges[0]) / height,
            ],
            "grid_size": [width, height],
            "row_order": "v_descending",
            "nominal_height_m": nominal,
        }
        report = {
            **geometry,
            "input_points": len(cloud.points),
            "band_m": [-negative, positive],
            "minimum_points_per_cell": int(p["minimum_points_per_cell"]),
            "minimum_coverage": float(p["minimum_coverage"]),
            "minimum_in_tolerance_fraction": float(p["minimum_in_tolerance_fraction"]),
            "peak_to_valley_limit_m": float(p["maximum_peak_to_valley_m"])
            if p["check_peak_to_valley"]
            else None,
        }
        outputs = dict.fromkeys((port.name for port in self.outputs), NO_VALUE)
        outputs["report"] = report

        def cancelled():
            report["status"] = "cancelled"
            return Result(outputs, passed=False, fail_reasons=["cloud_datum_cancelled"])

        if self._cancelled():
            return cancelled()
        local = (cloud.points.astype(np.float64) * METERS[cloud.units] - origin) @ basis
        u, v, _ = local.T
        rows = np.flatnonzero(
            (u >= u_edges[0]) & (u <= u_edges[-1]) & (v >= v_edges[0]) & (v <= v_edges[-1])
        )
        observed = local[rows, 2]
        delta = observed - nominal
        outside = (delta < -negative) | (delta > positive)
        col = np.minimum(np.searchsorted(u_edges, u[rows], side="right") - 1, width - 1)
        row = (
            height - 1 - np.minimum(np.searchsorted(v_edges, v[rows], side="right") - 1, height - 1)
        )
        cell = row * width + col
        counts = np.bincount(cell, minlength=width * height)
        valid = counts >= p["minimum_points_per_cell"]
        coverage = valid.sum() / len(valid)
        if self._cancelled():
            return cancelled()

        shape = (height, width)
        deviation = _peak_map(cell, delta, valid).reshape(shape)
        mask = valid.reshape(shape).astype(np.uint8) * 255
        defects = np.zeros(width * height, np.uint8)
        defects[cell[outside]] = 255
        defects = defects.reshape(shape)
        bad_rows = rows[outside]
        bad_cloud = cloud.subset(bad_rows)
        if "source_index" not in bad_cloud.attributes:
            bad_cloud.attributes["source_index"] = bad_rows.astype(np.int32)
        bad_cloud.metadata["cloud_datum"] = geometry

        flatness = float(np.ptp(observed)) if len(observed) else None
        in_band = (len(rows) - len(bad_rows)) / len(rows) if len(rows) else None
        report.update(
            selected_points=len(rows),
            supported_cells=int(valid.sum()),
            coverage=float(coverage),
            in_tolerance_fraction=in_band,
            out_of_tolerance_points=len(bad_rows),
            defect_cells=int(np.count_nonzero(defects)),
            signed_height_m=deviation_statistics(observed),
            nominal_deviation_m=deviation_statistics(delta),
            datum_parallel_peak_to_valley_m=flatness,
        )
        reasons = []
        if not len(rows):
            reasons.append("cloud_datum_empty")
        if not valid.any() or coverage < p["minimum_coverage"]:
            reasons.append("cloud_datum_insufficient_coverage")
        if in_band is not None and in_band < p["minimum_in_tolerance_fraction"]:
            reasons.append("cloud_datum_height_tolerance")
        if (
            p["check_peak_to_valley"]
            and flatness is not None
            and flatness > p["maximum_peak_to_valley_m"]
        ):
            reasons.append("cloud_datum_peak_to_valley")
        report["status"] = "failed" if reasons else "ok"
        outputs.update(
            deviation=ImageArray(deviation, "gray"),
            coverage_mask=ImageArray(mask, "gray"),
            defect_mask=ImageArray(defects, "gray"),
            defects=bad_cloud,
        )

        payload = {**self.input_payloads["cloud"], "cloud_datum": geometry}
        raster = {k: v for k, v in payload.items() if k not in ("spatial", "pyramid")}
        raster["spatial"] = frame([width, height], f"{self.id}:datum")
        return Result(
            outputs,
            payloads={
                **dict.fromkeys(("deviation", "coverage_mask", "defect_mask"), raster),
                "defects": payload,
                "report": payload,
            },
            graphics={"Datum deviation": _preview(deviation, mask, defects, negative, positive)}
            if self.capture_graphics
            else {},
            passed=not reasons,
            score=float(coverage * (in_band or 0)),
            fail_reasons=reasons,
            message=f"{coverage:.1%} coverage · {len(bad_rows):,} height violations",
        )
