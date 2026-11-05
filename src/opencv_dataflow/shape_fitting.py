"""Fit 2D measurement geometry to image points or a detected contour."""

import cv2
import numpy as np

from .collection_tools import read_collection
from .core import Block, Port, Result
from .processing import finite, integer

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
        if len(inputs) != 1:
            raise ValueError("Connect either points or a detected collection")
        if "collection" in inputs:
            data, payload = read_collection(self, inputs)
            index = integer(self.parameters["item_index"], "Item index", 0, len(data["items"]) - 1)
            geometry = data["items"][index]["geometry"]
            points = geometry.get("points")
            if points is None:
                contours = geometry["contours"]
                points = max(contours, key=lambda c: cv2.contourArea(np.asarray(c, np.float32)))
        else:
            points = inputs["points"]
            payload = self.input_payloads.get("points", {})
        points = np.asarray(points, np.float64)
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
