"""Hough line detection for sparse points in an image coordinate frame."""

import math

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .objects import collection, line_item
from .parameter_validation import NumericRange, ValidationError
from .payloads import ValuePacket, require_same_frame
from .result_graphics import ACCEPTED
from .sample_tables import finite_features
from .selection import SELECTION_INPUTS, selection_mask

_MAX_ACCUMULATOR_CELLS = 8_388_608
_MAX_POINT_OPERATIONS = 64_000_000


class HoughPointSetLines(Block):
    type_id, title = "inspect.hough_point_lines", "Hough lines from points"
    category, block_type, subtype = "Inspection", "Inspection", "Features"
    inputs = (*SELECTION_INPUTS, Port("points", "data"))
    outputs = (
        Port("collection", "data"),
        Port("polar_lines", "data"),
        Port("count", "number"),
        Port("point_count", "number"),
    )
    payload_reference = "image"
    process_inputs_readonly = True
    defaults = {
        "minimum_votes": 12,
        "maximum_lines": 32,
        "minimum_lines": 1,
        "rho_mode": "auto",
        "rho_min": -512.0,
        "rho_max": 512.0,
        "rho_step": 1.0,
        "theta_min_degrees": 0.0,
        "theta_max_degrees": 180.0,
        "theta_step_degrees": 1.0,
        "support_distance": 0.0,
    }
    parameter_choices = {"rho_mode": ("auto", "custom")}
    parameter_ranges = {
        "minimum_votes": NumericRange(1, 100_000, whole=True),
        "maximum_lines": NumericRange(1, 512, whole=True),
        "minimum_lines": NumericRange(0, 512, whole=True),
        "rho_min": NumericRange(-32768, 32768),
        "rho_max": NumericRange(-32768, 32768),
        "rho_step": NumericRange(0.01, 32768),
        "theta_min_degrees": NumericRange(0, 180),
        "theta_max_degrees": NumericRange(0, 180),
        "theta_step_degrees": NumericRange(0.01, 180),
        "support_distance": NumericRange(0, 32768),
    }
    parameter_help = {
        "minimum_votes": "Fewest point votes required in a Hough bin, inclusive.",
        "maximum_lines": "Maximum native peaks returned in vote order.",
        "minimum_lines": "Fewest finite segments required to pass. Zero permits an empty result.",
        "rho_mode": "Fit a symmetric distance search to the points, or set the native range.",
        "rho_min": "Lower signed normal-distance limit in Custom mode, in pixels.",
        "rho_max": "Upper signed normal-distance limit in Custom mode, in pixels.",
        "rho_step": "Spacing between normal-distance bins, in pixels.",
        "theta_min_degrees": "First line-normal angle. Zero searches vertical lines.",
        "theta_max_degrees": "Upper angle bound for the line normal, in degrees.",
        "theta_step_degrees": "Angle bin spacing. A 90-degree normal describes a horizontal line.",
        "support_distance": "Distance to the peak line for endpoint support. Zero uses half a bin.",
    }
    input_help = {
        "image": "Coordinate frame and overlay background for the sparse points.",
        "points": "Finite [x, y] rows in input image coordinates. Image pixels do not vote.",
        "mask": "Keep voting points whose nearest pixel is inside this mask.",
        "region": "Keep voting points whose nearest pixel is inside this region.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if int(p["minimum_lines"]) > int(p["maximum_lines"]):
            raise ValidationError("Minimum lines exceeds maximum lines", "minimum_lines")
        if p["rho_mode"] == "custom" and float(p["rho_min"]) >= float(p["rho_max"]):
            raise ValidationError("Rho maximum must exceed its minimum", "rho_min", "rho_max")
        if float(p["theta_min_degrees"]) >= float(p["theta_max_degrees"]):
            raise ValidationError(
                "Theta maximum must exceed its minimum", "theta_min_degrees", "theta_max_degrees"
            )

    def _search(self, points):
        p = self.parameters
        rho_step = float(p["rho_step"])
        if p["rho_mode"] == "auto":
            limit = max(1, math.ceil(np.linalg.norm(points, axis=1).max() / rho_step)) * rho_step
            low, high = -limit, limit
        else:
            low, high = float(p["rho_min"]), float(p["rho_max"])
        theta_min, theta_max, theta_step = (
            math.radians(float(p[key]))
            for key in ("theta_min_degrees", "theta_max_degrees", "theta_step_degrees")
        )
        rho_bins = round((high - low + 1) / rho_step)
        angle_bins = math.ceil((theta_max - theta_min) / theta_step) + 1
        if rho_bins < 1 or (rho_bins + 2) * (angle_bins + 2) > _MAX_ACCUMULATOR_CELLS:
            raise ValidationError("Hough accumulator needs 1 to 8,388,608 cells. Adjust the steps.")
        if len(points) * (angle_bins + int(p["maximum_lines"])) > _MAX_POINT_OPERATIONS:
            raise ValidationError("Too many Hough votes. Reduce the points, angles, or line limit.")
        found = cv2.HoughLinesPointSet(
            points.astype(np.float32).reshape(-1, 1, 2),
            int(p["maximum_lines"]),
            int(p["minimum_votes"]) - 1,
            low,
            high,
            rho_step,
            theta_min,
            theta_max,
            theta_step,
        )
        return [] if found is None else found.reshape(-1, 3)

    def process(self, inputs):
        image = inputs["image"]
        width, height = image.shape[1::-1]
        if max(width, height) > 16384 or width * height > 64_000_000:
            raise ValidationError("Line collections are limited to 16,384 per axis and 64M pixels")
        raw = inputs["points"]
        require_same_frame(
            (image, ValuePacket(raw, self.input_payloads["points"])),
            message="Voting points must use the input image coordinate frame",
        )
        points = np.empty((0, 2)) if raw == [] else finite_features(raw, max_columns=2)
        if points.shape[1] != 2 or np.any(points < 0) or np.any(points > [width - 1, height - 1]):
            raise ValidationError("Voting points must be [x, y] rows within the input image")
        mask = selection_mask(inputs, image)
        if mask is not None:
            x, y = np.rint(points).astype(np.intp).T
            points = points[mask[y, x] != 0]

        p = self.parameters
        found = self._search(points) if len(points) >= int(p["minimum_votes"]) else []
        support_distance = float(p["support_distance"]) or float(p["rho_step"]) / 2
        # The native polar coordinates are float32. Allow a small rounding margin at the bin edge.
        tolerance = support_distance + 4 * np.finfo(np.float32).eps * max(width, height)
        items, polar_lines = [], []
        for index, (votes, rho, theta) in enumerate(found):
            nx, ny = math.cos(theta), math.sin(theta)
            supported = points[np.abs(points[:, 0] * nx + points[:, 1] * ny - rho) <= tolerance]
            polar = {
                "votes": int(votes),
                "rho_pixels": float(rho),
                "normal_angle_degrees": math.degrees(theta),
                "support_count": len(supported),
            }
            polar_lines.append(polar)
            if not len(supported):
                continue
            tangent = (-ny, nx)
            positions = supported @ tangent
            endpoints = rho * np.array([nx, ny]) + np.outer(
                [positions.min(), positions.max()], tangent
            )
            item = line_item(endpoints, index)
            item["metrics"]["strength"] = int(votes)
            item["geometry"]["polar"] = polar
            items.append(item)

        passed = len(items) >= int(p["minimum_lines"])
        result = Result(
            {
                "collection": collection("lines", items, image, self.input_payloads["image"]),
                "polar_lines": polar_lines,
                "count": len(items),
                "point_count": len(points),
            },
            passed=passed,
            fail_reasons=[] if passed else ["hough_point_lines_below_minimum"],
            message=f"{len(items)} supported lines from {len(points)} voting points",
        )
        if self.generate_graphics and items:
            lines = tuple(
                graphic_object("polyline", item["geometry"]["points"], ACCEPTED, thickness=1)
                for item in items
            )
            result.drawings["point-set Hough lines"] = GraphicDrawing(lines)
        return result
