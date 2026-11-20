"""Measure visible part dimensions relative to a calibrated support plane."""

import cv2
import numpy as np

from .core import Port, Result
from .depth_plane import distance_grid
from .geometry3d_common import Tool3D
from .graphics_layers import GraphicDrawing, graphic_object
from .label_measurement import MAX_LABEL_PIXELS
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .ptcloud3d import depth_intrinsics
from .result_graphics import ACCEPTED
from .selection import selection_mask
from .stereo3d import metric_depth
from .tickets import ticket_key

_MAX_BOUNDARY_PIXELS = 262_144
_CELL_CORNERS = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
_DIMENSIONS = ("length_m", "width_m", "thickness_m")


def _read_plane(plane, camera_frame):
    if (
        not isinstance(plane, dict)
        or plane.get("model") != "camera_depth_plane_v1"
        or plane.get("frame") != camera_frame
        or plane.get("units") not in ("m", "mm")
    ):
        raise ValidationError("Support plane must use the depth camera frame and metric units")
    try:
        normal = np.asarray(plane.get("normal"), dtype=np.float64)
        offset = float(plane.get("offset", 0))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("Support plane normal and offset must be numeric") from exc
    scale = np.linalg.norm(normal)
    if (
        normal.shape != (3,)
        or not np.isfinite(normal).all()
        or not np.isclose(scale, 1, rtol=0, atol=1e-6)
        or not np.isfinite(offset)
        or offset <= 0
    ):
        raise ValidationError("Support plane needs a unit normal and positive camera-side offset")
    return normal / scale, (offset / scale) * (0.001 if plane["units"] == "mm" else 1)


def _basis(normal):
    seed = np.array([1.0, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1.0, 0])
    first = seed - (seed @ normal) * normal
    first /= np.linalg.norm(first)
    return np.column_stack((first, np.cross(first, normal)))


def _lift_boundary(height, mask, k, normal, offset, basis):
    y, x = np.nonzero(mask)
    projected = []
    used = np.zeros(mask.shape, bool)
    for start in range(0, len(x), 8192):
        band = slice(start, start + 8192)
        rows, cols = y[band], x[band]
        uv = np.column_stack((cols, rows))[:, None, :] + _CELL_CORNERS
        uv = (uv - k[:2, 2]) / [k[0, 0], k[1, 1]]
        rays = np.concatenate((uv, np.ones((*uv.shape[:2], 1))), axis=2)
        denominators = rays @ normal
        accepted = np.isfinite(denominators).all(axis=1) & (denominators < -1e-9).all(axis=1)
        distances = (height[rows[accepted], cols[accepted], None] - offset) / denominators[accepted]
        points = (rays[accepted] @ basis) * distances[..., None]
        finite = np.isfinite(points).all(axis=(1, 2)) & (distances > 0).all(axis=1)
        kept = np.flatnonzero(accepted)[finite]
        used[rows[kept], cols[kept]] = True
        projected.append(points[finite].reshape(-1, 2))
    return np.concatenate(projected) if projected else np.empty((0, 2)), used


def _enclosing_box(points):
    low, high = points.min(axis=0), points.max(axis=0)
    span = float(np.max(high - low))
    if not np.isfinite(span) or span <= 0:
        return None
    local = ((points - low) / span).astype(np.float32)
    hull = cv2.convexHull(local)
    box = cv2.minAreaRect(hull)
    short, long = sorted(side * span for side in box[1])
    if short <= 0:
        return None
    return long, short, cv2.boxPoints(box).astype(np.float64) * span + low


def _drawing(k, top_corners, report, missing):
    objects = []
    if top_corners is not None and (top_corners[:, 2] > 0).all():
        uv = (top_corners @ k.T)[:, :2] / top_corners[:, 2, None]
        if np.isfinite(uv).all():
            objects.append(graphic_object("polyline", uv, ACCEPTED, closed=True))
            label = " × ".join(f"{1000 * report[name]:.2f}" for name in _DIMENSIONS) + " mm"
            objects.append(
                graphic_object("label", (uv[np.argmin(uv[:, 1])],), ACCEPTED, text=label)
            )
    y, x = np.nonzero(missing)
    ids = np.linspace(0, len(x) - 1, min(96, len(x)), dtype=np.intp)
    for idx in ids:
        point = (x[idx], y[idx])
        objects.append(graphic_object("marker", (point,), (0, 150, 255), thickness=1))
    return GraphicDrawing(tuple(objects), source="depth")


class MeasureDepthSilhouette(Tool3D):
    type_id, title, subtype = "3d.depth_silhouette", "Measure calibrated depth silhouette", "Depth"
    inputs = (
        Port("depth", "depth"),
        Port("camera", "data"),
        Port("plane", "data"),
        Port("mask", visual_type="region"),
        Port("region", "data", required=False, required_when_connected=True, visual_type="region"),
    )
    outputs = (
        *(Port(name, "number", required=False) for name in _DIMENSIONS),
        Port("boundary_mask", visual_type="region"),
        Port("invalid_boundary_mask", visual_type="region"),
        Port("report", "data"),
    )
    defaults = {
        "minimum_height_m": 0.001,
        "minimum_boundary_coverage": 0.9,
        "minimum_face_coverage": 0.7,
        "minimum_face_pixels": 16,
        "reject_image_border": True,
        "input_units": "m",
        "units_per_sample": 0.001,
    }
    parameter_choices = {"input_units": ("m", "mm")}
    parameter_ranges = {
        "minimum_height_m": NumericRange(0, 1000),
        "minimum_boundary_coverage": NumericRange(0, 1),
        "minimum_face_coverage": NumericRange(0, 1),
        "minimum_face_pixels": NumericRange(3, MAX_LABEL_PIXELS, whole=True),
        "units_per_sample": NumericRange(1e-9, 1000),
    }
    parameter_help = {
        "minimum_height_m": "Reject samples at or below this distance in front of the datum.",
        "minimum_boundary_coverage": "Required fraction of boundary pixels usable in the box.",
        "minimum_face_coverage": "Required fraction of selected face pixels above the datum.",
        "minimum_face_pixels": "Minimum retained face pixels used for the median thickness.",
        "reject_image_border": "Withhold dimensions when the silhouette touches the image edge.",
        "input_units": "Physical units for untagged depth.",
        "units_per_sample": "Physical units per raw uint16 count.",
    }
    input_help = {
        "depth": "Positive camera Z on an undistorted pinhole grid.",
        "camera": "Intrinsics for this depth grid and a named metric camera frame.",
        "plane": "Metric support datum oriented toward the camera.",
        "mask": "One complete visible top silhouette, including cells with missing depth samples.",
        "region": "Optional intersection with the silhouette mask.",
    }
    payload_reference = "depth"
    flow_primary = _DIMENSIONS
    parallel_safe = preview_safe = process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if type(p["reject_image_border"]) is not bool:
            raise ValidationError("Reject image border must be a boolean", "reject_image_border")

    def process(self, inputs):
        p = self.parameters
        depth, camera = inputs["depth"], inputs["camera"]
        if depth.size > MAX_LABEL_PIXELS:
            raise ValidationError(f"Depth silhouette exceeds the {MAX_LABEL_PIXELS:,} pixel limit")
        k, _ = depth_intrinsics(camera, depth)
        if camera.get("units") not in ("m", "mm"):
            raise ValidationError("Camera must declare metric units")
        camera_frame = camera.get("frame", "camera")
        if self.input_payloads["depth"].get("optical_frame", camera_frame) != camera_frame:
            raise ValidationError("Depth optical frame differs from the camera model")
        normal, offset = _read_plane(inputs["plane"], camera_frame)
        tickets = {ticket_key(v["ticket"]) for v in self.input_payloads.values() if "ticket" in v}
        if len(tickets) > 1:
            raise ValidationError("Silhouette measurement inputs must refer to the same ticket")

        selected = selection_mask(inputs, depth)
        component_count = (
            cv2.connectedComponentsWithAlgorithm(selected, 8, cv2.CV_32S, cv2.CCL_SAUF)[0] - 1
        )
        inner = cv2.erode(
            selected, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0
        )
        region, boundary = selected != 0, (selected != 0) & (inner == 0)
        touches = bool(
            region[0].any() or region[-1].any() or region[:, 0].any() or region[:, -1].any()
        )
        boundary_count = int(boundary.sum())
        if boundary_count > _MAX_BOUNDARY_PIXELS:
            raise ValidationError("Silhouette exceeds the 262,144 boundary pixel processing limit")
        depth_m = metric_depth(depth, p["units_per_sample"], p["input_units"])
        height = distance_grid(depth_m, k, normal, offset)
        has_depth = np.isfinite(depth_m)
        face = region & has_depth & (height > float(p["minimum_height_m"])) & (height < offset)
        basis = _basis(normal)
        projected, used = _lift_boundary(height, boundary & face, k, normal, offset, basis)
        missing = boundary & ~used
        face_count, pixel_count, used_count = (int(a.sum()) for a in (face, region, used))
        face_fraction = face_count / pixel_count if pixel_count else 0.0
        boundary_fraction = used_count / boundary_count if boundary_count else 0.0
        report = {
            "model": "calibrated_depth_silhouette_v1",
            "units": "m",
            "camera_frame": camera_frame,
            "selected_pixels": pixel_count,
            "components": component_count,
            "touches_image_border": touches,
            "face_pixels": face_count,
            "face_coverage": face_fraction,
            "boundary_pixels": boundary_count,
            "usable_boundary_pixels": used_count,
            "boundary_coverage": boundary_fraction,
            "invalid_boundary_pixels": boundary_count - used_count,
            "missing_depth_boundary_pixels": int((boundary & ~has_depth).sum()),
            "rejected_height_boundary_pixels": int((boundary & has_depth & ~face).sum()),
            "unprojectable_boundary_pixels": int((boundary & face & ~used).sum()),
            **dict.fromkeys((*_DIMENSIONS, "height_p05_m", "height_p95_m", "box_on_datum_xyz_m")),
        }
        fail = []
        if component_count != 1:
            fail.append("silhouette_requires_one_component")
        if touches and p["reject_image_border"]:
            fail.append("silhouette_touches_image_border")
        if face_count < int(p["minimum_face_pixels"]) or face_fraction < float(
            p["minimum_face_coverage"]
        ):
            fail.append("silhouette_face_support_insufficient")
        if used_count < 3 or boundary_fraction < float(p["minimum_boundary_coverage"]):
            fail.append("silhouette_boundary_support_insufficient")
        top = None
        if not fail:
            box = _enclosing_box(projected)
            if box is None:
                fail.append("silhouette_degenerate_metric_extent")
            else:
                length, width, corners = box
                p05, thickness, p95 = np.percentile(height[face], [5, 50, 95])
                datum = corners @ basis.T - offset * normal
                report.update(
                    length_m=length,
                    width_m=width,
                    thickness_m=float(thickness),
                    height_p05_m=float(p05),
                    height_p95_m=float(p95),
                    box_on_datum_xyz_m=datum.tolist(),
                )
                top = datum + thickness * normal
        drawings = {}
        if self.generate_graphics and (top is not None or missing.any()):
            drawings["Silhouette measurements"] = _drawing(k, top, report, missing)
        message = f"{used_count}/{boundary_count} usable boundary pixels"
        if fail:
            detail = fail[0].removeprefix("silhouette_").replace("_", " ")
            message = f"Dimensions withheld. {detail.capitalize()}. {message}"
        return Result(
            {
                **{name: report[name] if not fail else NO_VALUE for name in _DIMENSIONS},
                "boundary_mask": boundary.astype(np.uint8) * 255,
                "invalid_boundary_mask": missing.astype(np.uint8) * 255,
                "report": report,
            },
            drawings=drawings,
            passed=not fail,
            score=boundary_fraction,
            fail_reasons=fail,
            message=message,
        )
