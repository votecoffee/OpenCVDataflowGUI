"""Measure the continuous intersection of two convex image-plane polygons."""

import math

import cv2
import numpy as np

from .contour_analysis import read_ordered_points
from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import ValuePacket, require_same_frame
from .result_graphics import ACCEPTED, REJECTED

_MAX_VERTICES = 16_384


def _points(value, field):
    try:
        points = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValidationError("Provide a table of [x, y] vertices", field) from exc
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or not 3 <= len(points) <= _MAX_VERTICES
        or not np.isfinite(points).all()
        or np.max(np.abs(points)) > 1e6
    ):
        raise ValidationError("Provide 3 to 16,384 finite vertices within ±1 million", field)
    return points


def _polygon(points, hull_mode, field):
    if hull_mode:
        boundary = cv2.convexHull(points)
    else:
        boundary = cv2.approxPolyDP(points, 0.0, True)
        count = len(boundary)
        if count < 3:
            raise ValidationError("Polygon must enclose a nonzero area", field)
        indices = cv2.convexHull(boundary, returnPoints=False).ravel()
        steps = np.diff(indices, append=indices[0]) % count
        if len(indices) != count or not (np.all(steps == 1) or np.all(steps == count - 1)):
            raise ValidationError(
                "Vertices must follow a convex boundary. Use convex hull mode for other points.",
                field,
            )
    area = cv2.contourArea(boundary)
    if area <= 0:
        raise ValidationError("Polygon must enclose a nonzero area", field)
    return boundary, area


def _intersection(a, b, include_containment):
    area, points = cv2.intersectConvexConvex(a, b, handleNested=include_containment)
    if not math.isfinite(area) or area < 0:
        raise ValidationError("OpenCV could not compute a stable polygon intersection")
    if points is None:
        return np.empty((0, 2)), 0.0
    points = points.reshape(-1, 2)
    # OpenCV can include a float-max marker in zero-area results.
    points = points[~np.all(points == np.finfo(np.float32).max, axis=1)]
    if not len(points):
        return np.empty((0, 2)), 0.0
    if not np.isfinite(points).all() or np.max(np.abs(points)) > 2_048:
        raise ValidationError("OpenCV returned invalid intersection vertices")
    boundary = cv2.convexHull(points).reshape(-1, 2)
    return boundary, cv2.contourArea(boundary)


class ConvexPolygonOverlap(Block):
    type_id, title = "inspect.convex_polygon_overlap", "Convex polygon overlap"
    category, block_type, subtype = "Inspection", "Inspection", "Measurements"
    inputs = (
        Port("points_a", "data", required=False, required_when_connected=True),
        Port("points_b", "data", required=False, required_when_connected=True),
        Port("image", "image", required=False),
    )
    outputs = (
        Port("measurements", "data"),
        Port("points", "data"),
        Port("area", "number"),
        Port("iou", "number"),
        Port("classification", "text"),
    )
    process_inputs_readonly = True
    defaults = {
        "points_a": [[10, 10], [50, 10], [50, 50], [10, 50]],
        "points_b": [[30, 20], [60, 20], [60, 60], [30, 60]],
        "a_item_index": 0,
        "b_item_index": 0,
        "input_mode": "ordered",
        "include_containment": True,
        "check_iou": False,
        "minimum_iou": 0.5,
    }
    parameter_choices = {"input_mode": ("ordered", "convex_hull")}
    parameter_ranges = {
        "a_item_index": NumericRange(0, whole=True),
        "b_item_index": NumericRange(0, whole=True),
        "minimum_iou": NumericRange(0, 1),
    }
    parameter_help = {
        "points_a": "Vertices for polygon A. Wired points or a contour collection take precedence.",
        "points_b": "Vertices for polygon B. Wired points or a contour collection take precedence.",
        "a_item_index": "Zero-based contour collection item for A.",
        "b_item_index": "Zero-based contour collection item for B.",
        "input_mode": "Use ordered convex boundaries or form a convex hull from each point set.",
        "include_containment": "Include strictly nested polygons in the intersection.",
        "minimum_iou": "Minimum intersection-over-union ratio for a passing result.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for name in ("points_a", "points_b"):
            _points(p[name], name)

    def _read(self, inputs, side):
        port = f"points_{side}"
        if port not in inputs:
            return _points(self.parameters[port], port), {}
        value, payload = read_ordered_points(
            self, inputs, port, self.parameters[f"{side}_item_index"]
        )
        return _points(value, port), payload

    def process(self, inputs):
        p = self.parameters
        a, payload_a = self._read(inputs, "a")
        b, payload_b = self._read(inputs, "b")
        framed = [("points_a", a, payload_a), ("points_b", b, payload_b)]
        if "image" in inputs:
            framed.append(("image", inputs["image"], self.input_payloads.get("image", {})))
        require_same_frame(
            [ValuePacket(value, payload) for _, value, payload in framed],
            message="Polygons and preview image must use the same coordinate frame",
        )
        fallback = framed[-1] if "image" in inputs else framed[0]
        source, _, payload = next((item for item in framed if item[2].get("spatial")), fallback)
        origin = (
            np.maximum(a.min(axis=0), b.min(axis=0)) + np.minimum(a.max(axis=0), b.max(axis=0))
        ) / 2
        span = max(float(np.max(np.abs(a - origin))), float(np.max(np.abs(b - origin))))
        if span < 1e-6:
            raise ValidationError("Polygon extent must be at least 0.000001 pixels")
        factor = math.ldexp(1.0, 10 - math.frexp(span)[1])
        hull_mode = p["input_mode"] == "convex_hull"
        native_a, area_a = _polygon(
            ((a - origin) * factor).astype(np.float32), hull_mode, "points_a"
        )
        native_b, area_b = _polygon(
            ((b - origin) * factor).astype(np.float32), hull_mode, "points_b"
        )
        local, area = _intersection(native_a, native_b, p["include_containment"])
        smaller = min(area_a, area_b)
        if area > smaller * (1 + 1e-5):
            raise ValidationError("Intersection area exceeded the smaller polygon")
        area = min(area, smaller)
        if math.isclose(area, smaller, rel_tol=1e-6):
            area, classification = smaller, "contained"
        elif area > 0:
            classification = "partial"
        else:
            classification = "touching" if len(local) else "none"
        iou = area / (area_a + area_b - area)
        points = (local / factor + origin).tolist()
        area_scale = factor**2
        minimum = float(p["minimum_iou"])
        passed = not p["check_iou"] or iou >= minimum
        outputs = {
            "measurements": {
                "area_a": area_a / area_scale,
                "area_b": area_b / area_scale,
                "area": area / area_scale,
                "iou": iou,
                "fraction_a": area / area_a,
                "fraction_b": area / area_b,
                "classification": classification,
            },
            "points": points,
            "area": area / area_scale,
            "iou": iou,
            "classification": classification,
        }
        drawings = {}
        if self.generate_graphics:
            drawing_source = "image" if "image" in inputs else source
            outlines = tuple(
                graphic_object(
                    "polyline", native.reshape(-1, 2) / factor + origin, color, closed=True
                )
                for native, color in ((native_a, (255, 180, 0)), (native_b, (0, 210, 255)))
            )
            drawings["polygons"] = GraphicDrawing(outlines, drawing_source)
            if points:
                intersection = graphic_object(
                    "marker" if len(points) == 1 else "polyline",
                    points,
                    ACCEPTED if passed else REJECTED,
                    closed=len(points) > 2,
                )
                drawings["intersection"] = GraphicDrawing((intersection,), drawing_source)
        return Result(
            outputs,
            payloads=dict.fromkeys(outputs, payload),
            passed=passed,
            score=iou,
            message=f"{classification.capitalize()}, IoU {iou:.3f}",
            fail_reasons=[] if passed else ["polygon_overlap_below_minimum"],
            drawings=drawings,
        )
