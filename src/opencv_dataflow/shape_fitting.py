"""Fit 2D measurement geometry to image points or a detected contour."""

import math

import cv2
import numpy as np

from .collection_tools import read_collection
from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import ValuePacket, require_same_frame
from .processing import finite, integer
from .result_graphics import ACCEPTED, REJECTED

LINE_DISTANCES = {
    "l2": cv2.DIST_L2,
    "l1": cv2.DIST_L1,
    "l12": cv2.DIST_L12,
    "fair": cv2.DIST_FAIR,
    "welsch": cv2.DIST_WELSCH,
    "huber": cv2.DIST_HUBER,
}
ELLIPSE_FITS = {
    "standard": cv2.fitEllipse,
    "ams": cv2.fitEllipseAMS,
    "direct": cv2.fitEllipseDirect,
}


def shape_points(block, inputs):
    if len(inputs) != 1:
        raise ValueError("Connect either points or a detected collection")
    if "collection" in inputs:
        data, payload = read_collection(block, inputs)
        index = integer(block.parameters["item_index"], "Item index", 0, len(data["items"]) - 1)
        geometry = data["items"][index]["geometry"]
        points = geometry.get("points")
        if points is None:
            contours = geometry["contours"]
            points = max(contours, key=lambda c: cv2.contourArea(np.asarray(c, np.float32)))
    else:
        points = inputs["points"]
        payload = block.input_payloads.get("points", {})
    points = np.asarray(points, np.float64)
    return points, payload, "collection" if "collection" in inputs else "points"


class FitShape(Block):
    type_id, title = "inspect.fit_shape", "Fit line, ellipse or enclosing circle"
    category, block_type, subtype = "Inspection", "Inspection", "Measurements"
    inputs = (Port("points", "data", required=False), Port("collection", "data", required=False))
    outputs = (Port("fit", "data"), Port("points", "data"), Port("rmse", "number"))
    defaults = {
        "shape": "line",
        "item_index": 0,
        "line_distance": "l2",
        "line_distance_parameter": 0.0,
        "line_radius_epsilon": 0.01,
        "line_angle_epsilon": 0.01,
        "ellipse_method": "standard",
        "check_rmse": False,
        "maximum_rmse": 1.0,
    }
    parameter_choices = {
        "shape": ("line", "ellipse", "enclosing_circle"),
        "line_distance": tuple(LINE_DISTANCES),
        "ellipse_method": tuple(ELLIPSE_FITS),
    }
    parameter_help = {
        "item_index": "Zero-based position in the collection, after filtering or sorting.",
        "line_distance_parameter": "Robust loss constant. Zero selects the OpenCV default.",
        "line_angle_epsilon": "Angle convergence tolerance in radians.",
        "line_radius_epsilon": "Position convergence tolerance in pixels.",
        "maximum_rmse": "Maximum root mean squared distance to the fitted boundary, in pixels.",
    }

    def process(self, inputs):
        points, payload, _ = shape_points(self, inputs)
        shape = self.parameters["shape"]
        minimum = {"line": 2, "ellipse": 5, "enclosing_circle": 2}[shape]
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or not minimum <= len(points) <= 100_000
            or not np.isfinite(points).all()
            or np.max(np.abs(points)) > 1e7
            or len(np.unique(points, axis=0)) < minimum
        ):
            raise ValueError(f"Provide {minimum} or more distinct finite [x,y] points")
        shift = points.mean(axis=0)
        centered = points - shift
        native = centered.astype(np.float32)
        if shape == "line":
            fit, anchors, residual = self.fit_line(points, native, shift)
        elif shape == "ellipse":
            if np.linalg.matrix_rank(centered) < 2:
                raise ValueError("Ellipse points must span two dimensions")
            ellipse = ELLIPSE_FITS[self.parameters["ellipse_method"]](native)
            center, axes, angle = ellipse
            if not np.isfinite([*center, *axes, angle]).all() or min(axes) <= 0:
                raise ValueError("OpenCV returned a degenerate ellipse")
            nearest = cv2.getClosestEllipsePoints(ellipse, native).reshape(-1, 2)
            residual = np.linalg.norm(centered - nearest, axis=1)
            center = (np.asarray(center) + shift).tolist()
            anchors = [list(center)]
            fit = {"center": list(center), "axis_lengths": list(axes), "angle_degrees": angle}
        else:
            center, radius = cv2.minEnclosingCircle(native)
            center = (np.asarray(center) + shift).tolist()
            residual = np.abs(np.linalg.norm(points - center, axis=1) - radius)
            anchors = [list(center)]
            fit = {"center": list(center), "radius": radius}
        rmse = float(np.sqrt(np.mean(np.square(residual))))
        if not np.isfinite(rmse):
            raise ValueError("OpenCV returned a nonfinite fitting residual")
        passed = not self.parameters["check_rmse"] or rmse <= finite(
            self.parameters["maximum_rmse"], "Maximum RMSE", 0
        )
        outputs = {
            "fit": {"shape": shape, **fit, "rmse": rmse, "sample_count": len(points)},
            "points": anchors,
            "rmse": rmse,
        }
        return Result(
            outputs,
            payloads={name: payload for name in outputs},
            passed=passed,
            message=f"{shape.replace('_', ' ')} fit RMSE {rmse:.4g} px",
            fail_reasons=[] if passed else ["shape_fit_out_of_tolerance"],
        )

    def fit_line(self, points, native, shift):
        loss = LINE_DISTANCES[self.parameters["line_distance"]]
        loss_param = finite(self.parameters["line_distance_parameter"], "Distance parameter", 0)
        radius_eps = finite(self.parameters["line_radius_epsilon"], "Radius tolerance", 0)
        angle_eps = finite(self.parameters["line_angle_epsilon"], "Angle tolerance", 0)
        fitted = cv2.fitLine(native, loss, loss_param, radius_eps, angle_eps).ravel().astype(float)
        direction, origin = fitted[:2], fitted[2:] + shift
        direction /= np.linalg.norm(direction)
        if direction[np.argmax(np.abs(direction))] < 0:
            direction = -direction
        centered = points - origin
        projection = centered @ direction
        ends = origin + np.array([projection.min(), projection.max()])[:, None] * direction
        residual = centered @ np.array([-direction[1], direction[0]])
        fit = {
            "origin": origin.tolist(),
            "direction": direction.tolist(),
            "endpoints": ends.tolist(),
        }
        return fit, ends.tolist(), residual


_COORDINATE = NumericRange(-1e7, 1e7)
_RECT_SIZE = NumericRange(0.001, 3e7)
_ANGLE = NumericRange()


def _rectangle(center, size, angle, label):
    if isinstance(center, (str, dict)) or isinstance(size, (str, dict)):
        raise ValidationError(f"{label} needs [x, y] center and [width, height] size")
    try:
        x, y = center
        width, height = size
    except (TypeError, ValueError):
        raise ValidationError(f"{label} needs [x, y] center and [width, height] size") from None
    return (
        (_COORDINATE.check(x, f"{label} center X"), _COORDINATE.check(y, f"{label} center Y")),
        (_RECT_SIZE.check(width, f"{label} width"), _RECT_SIZE.check(height, f"{label} height")),
        _ANGLE.check(angle, f"{label} angle") % 360,
    )


class BoundingRotatedRectangle(Block):
    type_id, title = "inspect.bounding_rotated_rectangle", "Minimum-area rotated rectangle"
    category, block_type, subtype = "Inspection", "Inspection", "Measurements"
    inputs = FitShape.inputs
    outputs = (Port("rectangle", "data"), Port("points", "data"), Port("area", "number"))
    defaults = {"item_index": 0}
    parameter_ranges = {"item_index": NumericRange(0, whole=True)}
    parameter_help = {"item_index": FitShape.parameter_help["item_index"]}

    def process(self, inputs):
        points, payload, source = shape_points(self, inputs)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or not 3 <= len(points) <= 100_000
            or not np.isfinite(points).all()
            or np.max(np.abs(points)) > _COORDINATE.maximum
        ):
            raise ValidationError("Provide 3 to 100,000 finite [x, y] points within ±10 million")
        origin = points.mean(axis=0)
        center, size, angle = cv2.minAreaRect((points - origin).astype(np.float32))
        if min(size) < _RECT_SIZE.minimum:
            raise ValidationError(
                "Points must enclose a rectangle at least 0.001 pixels wide and high"
            )
        corners = cv2.boxPoints((center, size, angle)).astype(np.float64) + origin
        rectangle = {
            "center": (origin + center).tolist(),
            "size": list(size),
            "angle_degrees": angle,
        }
        outputs = {"rectangle": rectangle, "points": corners.tolist(), "area": size[0] * size[1]}
        drawings = {}
        if self.generate_graphics:
            outline = graphic_object("polyline", corners, ACCEPTED, closed=True)
            drawings["rectangle"] = GraphicDrawing((outline,), source)
        return Result(outputs, payloads=dict.fromkeys(outputs, payload), drawings=drawings)


def _checked_enclosure(hull, hull_area, vertices, side_limit):
    if vertices is None or not np.isfinite(vertices).all():
        return None
    boundary = cv2.convexHull(vertices).reshape(-1, 2)
    if not 3 <= len(boundary) <= side_limit:
        return None
    area = cv2.contourArea(boundary)
    if area <= 0 or area < hull_area * (1 - 1e-5):
        return None
    polygon = boundary.astype(np.float64)
    edges = np.roll(polygon, -1, axis=0) - polygon
    normals = np.column_stack((-edges[:, 1], edges[:, 0]))
    clearance = hull.reshape(-1, 2) @ normals.T - np.sum(polygon * normals, axis=1)
    tolerance = 32 * np.finfo(np.float32).eps * max(1.0, float(np.max(np.abs(polygon))))
    if np.any(clearance < -tolerance * np.linalg.norm(normals, axis=1)):
        return None
    return polygon, area


def _fit_enclosure(hull, hull_area, side_limit):
    if len(hull) <= side_limit:
        return hull.reshape(-1, 2).astype(np.float64), hull_area, "hull"
    if side_limit == 3:
        _, vertices = cv2.minEnclosingTriangle(hull)
        triangle = _checked_enclosure(hull, hull_area, vertices, side_limit)
        if triangle is None:
            raise ValidationError("OpenCV returned an invalid enclosing triangle")
        return *triangle, "triangle"
    if len(hull) > 256:
        raise ValidationError("Polygon search is limited to 256 hull vertices")
    rectangle = _checked_enclosure(
        hull, hull_area, cv2.boxPoints(cv2.minAreaRect(hull)), side_limit
    )
    try:
        _, vertices = cv2.minEnclosingConvexPolygon(hull, side_limit)
    except cv2.error:
        vertices = None
    polygon = _checked_enclosure(hull, hull_area, vertices, side_limit)
    if polygon is not None and (rectangle is None or polygon[1] <= rectangle[1]):
        return *polygon, "convex_polygon"
    if rectangle is not None:
        return *rectangle, "rectangle"
    raise ValidationError("OpenCV could not produce a valid enclosing polygon")


class BoundingConvexPolygon(Block):
    type_id, title = "inspect.bounding_convex_polygon", "Enclosing convex polygon"
    category, block_type, subtype = "Inspection", "Inspection", "Measurements"
    inputs = FitShape.inputs
    outputs = (
        Port("measurements", "data"),
        Port("points", "data"),
        Port("area", "number"),
        Port("fill_ratio", "number"),
    )
    defaults = {"item_index": 0, "maximum_sides": 3}
    parameter_ranges = {
        "item_index": NumericRange(0, whole=True),
        "maximum_sides": NumericRange(3, 16, whole=True),
    }
    parameter_help = {
        "item_index": FitShape.parameter_help["item_index"],
        "maximum_sides": "Three selects the triangle solver. Larger limits allow polygon search.",
    }

    def process(self, inputs):
        points, payload, source = shape_points(self, inputs)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or not 3 <= len(points) <= 100_000
            or not np.isfinite(points).all()
            or np.max(np.abs(points)) > _COORDINATE.maximum
        ):
            raise ValidationError("Provide 3 to 100,000 finite [x, y] points within ±10 million")
        low, high = points.min(axis=0), points.max(axis=0)
        origin = (low + high) / 2
        extent = float(np.max(high - low))
        if extent < 1e-6:
            raise ValidationError("Points must span at least 0.000001 pixels")
        factor = math.ldexp(1.0, 10 - math.frexp(extent)[1])
        hull = cv2.convexHull(((points - origin) * factor).astype(np.float32))
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            raise ValidationError("Points must enclose a nonzero area")
        side_limit = int(self.parameters["maximum_sides"])
        polygon, area, method = _fit_enclosure(hull, hull_area, side_limit)
        world = polygon / factor + origin
        fill = min(1.0, hull_area / area)
        outputs = {
            "measurements": {
                "method": method,
                "side_count": len(polygon),
                "area": area / factor**2,
                "hull_area": hull_area / factor**2,
                "fill_ratio": fill,
            },
            "points": world.tolist(),
            "area": area / factor**2,
            "fill_ratio": fill,
        }
        drawings = {}
        if self.generate_graphics:
            outline = graphic_object("polyline", world, ACCEPTED, closed=True)
            drawings["enclosing polygon"] = GraphicDrawing((outline,), source)
        return Result(
            outputs,
            payloads=dict.fromkeys(outputs, payload),
            drawings=drawings,
            message=f"{len(polygon)} sides, {fill:.1%} hull fill, {method.replace('_', ' ')}",
        )


class RotatedRectangleOverlap(Block):
    type_id, title = "inspect.rotated_rectangle_overlap", "Rotated rectangle overlap"
    category, block_type, subtype = "Inspection", "Inspection", "Measurements"
    inputs = (
        Port("rectangle_a", "data", required=False),
        Port("rectangle_b", "data", required=False),
        Port("image", "image", required=False),
    )
    outputs = (
        Port("measurements", "data"),
        Port("points", "data"),
        Port("area", "number"),
        Port("iou", "number"),
        Port("classification", "text"),
    )
    defaults = {
        "a_center_x": 50.0,
        "a_center_y": 50.0,
        "a_width": 40.0,
        "a_height": 24.0,
        "a_angle_degrees": 0.0,
        "b_center_x": 60.0,
        "b_center_y": 50.0,
        "b_width": 40.0,
        "b_height": 24.0,
        "b_angle_degrees": 0.0,
        "check_iou": False,
        "minimum_iou": 0.5,
    }
    parameter_ranges = {
        **{
            f"{side}_{field}": limits
            for side in ("a", "b")
            for field, limits in (
                ("center_x", _COORDINATE),
                ("center_y", _COORDINATE),
                ("width", _RECT_SIZE),
                ("height", _RECT_SIZE),
                ("angle_degrees", _ANGLE),
            )
        },
        "minimum_iou": NumericRange(0, 1),
    }
    parameter_help = {
        "a_angle_degrees": "Width-axis angle measured clockwise in image coordinates.",
        "b_angle_degrees": "Width-axis angle measured clockwise in image coordinates.",
        "check_iou": "Judge overlap using intersection area divided by union area.",
        "minimum_iou": "Minimum intersection-over-union ratio for a passing result.",
    }

    def read_rectangle(self, inputs, side):
        port = f"rectangle_{side}"
        label = f"Rectangle {side.upper()}"
        if port in inputs:
            value = inputs[port]
            if (
                not isinstance(value, dict)
                or not {"center", "size", "angle_degrees"} <= value.keys()
            ):
                raise ValidationError(f"{label} needs center, size and angle_degrees fields")
            return _rectangle(value["center"], value["size"], value["angle_degrees"], label)
        p = self.parameters
        return _rectangle(
            (p[f"{side}_center_x"], p[f"{side}_center_y"]),
            (p[f"{side}_width"], p[f"{side}_height"]),
            p[f"{side}_angle_degrees"],
            label,
        )

    def process(self, inputs):
        connected = [name for name in ("image", "rectangle_a", "rectangle_b") if name in inputs]
        source = next(
            (name for name in connected if self.input_payloads.get(name, {}).get("spatial")),
            connected[0] if connected else "image",
        )
        require_same_frame(
            [
                ValuePacket(value, self.input_payloads.get(name, {}))
                for name, value in inputs.items()
            ],
            message="Rectangles and preview image must use the same coordinate frame",
        )
        payload = self.input_payloads.get(source, {})
        a, b = (self.read_rectangle(inputs, side) for side in ("a", "b"))
        origin = (np.asarray(a[0]) + b[0]) / 2
        local_a, local_b = (
            (tuple(np.asarray(center) - origin), size, angle) for center, size, angle in (a, b)
        )
        kind, polygon = cv2.rotatedRectangleIntersection(local_a, local_b)
        area_a, area_b = a[1][0] * a[1][1], b[1][0] * b[1][1]
        if kind == cv2.INTERSECT_NONE:
            points, area, classification = [], 0.0, "disjoint"
        else:
            polygon = cv2.convexHull(polygon)
            points = (polygon.reshape(-1, 2).astype(np.float64) + origin).tolist()
            smaller_area = min(area_a, area_b)
            area = min(cv2.contourArea(polygon), smaller_area)
            # OpenCV may mark containment along a shared edge as partial.
            if kind == cv2.INTERSECT_FULL or math.isclose(area, smaller_area, rel_tol=1e-6):
                area, classification = smaller_area, "contained"
            else:
                classification = "partial" if area > 0 else "touching"
        iou = area / (area_a + area_b - area)
        minimum_iou = float(self.parameters["minimum_iou"])
        passed = not self.parameters["check_iou"] or iou >= minimum_iou
        outputs = {
            "measurements": {
                "area_a": area_a,
                "area_b": area_b,
                "area": area,
                "iou": iou,
                "fraction_a": area / area_a,
                "fraction_b": area / area_b,
                "classification": classification,
            },
            "points": points,
            "area": area,
            "iou": iou,
            "classification": classification,
        }
        drawings = {}
        if self.generate_graphics:
            outlines = tuple(
                graphic_object(
                    "polyline", cv2.boxPoints(rect).astype(np.float64) + origin, color, closed=True
                )
                for rect, color in ((local_a, (255, 180, 0)), (local_b, (0, 210, 255)))
            )
            drawings["rectangles"] = GraphicDrawing(outlines, source)
            if points:
                intersection = graphic_object(
                    "marker" if len(points) == 1 else "polyline",
                    points,
                    ACCEPTED if passed else REJECTED,
                    closed=len(points) > 2,
                )
                drawings["intersection"] = GraphicDrawing((intersection,), source)
        return Result(
            outputs,
            payloads=dict.fromkeys(outputs, payload),
            passed=passed,
            score=iou,
            message=f"{classification.capitalize()}, IoU {iou:.3f}",
            fail_reasons=[]
            if passed
            else [
                {
                    "code": "rectangle_overlap_below_minimum",
                    "message": f"IoU {iou:.6g} is below the minimum {minimum_iou:.6g}",
                }
            ],
            drawings=drawings,
        )
