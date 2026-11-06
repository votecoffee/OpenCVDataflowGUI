"""Standard and weighted Hough peaks from a monochrome edge image."""

import math

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .hough_geometry import supported_lines
from .images import DepthArray
from .objects import collection
from .parameter_validation import NumericRange, ValidationError
from .result_graphics import ACCEPTED
from .selection import SELECTION_INPUTS, selection_mask

_MAX_EDGE_PIXELS = 1_000_000
_MAX_ACCUMULATOR_CELLS = 8_388_608
_MAX_POINT_OPERATIONS = 64_000_000


class HoughPolarLines(Block):
    type_id, title = "inspect.hough_polar_lines", "Hough polar lines"
    category, block_type, subtype = "Inspection", "Inspection", "Features"
    inputs = SELECTION_INPUTS
    outputs = (
        Port("collection", "data"),
        Port("polar_lines", "data"),
        Port("count", "number"),
        Port("edge_pixel_count", "number"),
    )
    payload_reference = "image"
    process_inputs_readonly = True
    defaults = {
        "vote_mode": "binary",
        "minimum_votes": 50,
        "maximum_lines": 32,
        "minimum_lines": 1,
        "rho_step": 1.0,
        "theta_min_degrees": 0.0,
        "theta_max_degrees": 180.0,
        "theta_step_degrees": 1.0,
        "support_distance": 0.0,
    }
    parameter_choices = {"vote_mode": ("binary", "weighted")}
    parameter_ranges = {
        "minimum_votes": NumericRange(1, 255 * _MAX_EDGE_PIXELS, whole=True),
        "maximum_lines": NumericRange(1, 512, whole=True),
        "minimum_lines": NumericRange(0, 512, whole=True),
        "rho_step": NumericRange(0.01, 32768),
        "theta_min_degrees": NumericRange(0, 180),
        "theta_max_degrees": NumericRange(0, 180),
        "theta_step_degrees": NumericRange(0.01, 180),
        "support_distance": NumericRange(0, 32768),
    }
    parameter_help = {
        "vote_mode": "Binary gives each nonzero pixel one vote. Weighted uses its intensity.",
        "minimum_votes": "Inclusive accumulator minimum. Weighted mode uses intensity sums.",
        "maximum_lines": "Maximum native peaks reported, ordered by accumulator value.",
        "minimum_lines": "Fewest finite segments required to pass. Zero permits an empty result.",
        "rho_step": "Spacing between normal-distance bins, in pixels.",
        "theta_min_degrees": "First line-normal angle. Zero searches vertical lines.",
        "theta_max_degrees": "Upper angle bound for the line normal, in degrees.",
        "theta_step_degrees": "Angle bin spacing. A 90-degree normal describes a horizontal line.",
        "support_distance": "Distance to the peak line for endpoint support. Zero uses half a bin.",
    }
    input_help = {
        "image": "8-bit monochrome edge map. Zero-valued pixels never vote.",
        "mask": "Only selected edge pixels contribute to the accumulator.",
        "region": "Only edge pixels inside the region contribute to the accumulator.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if int(p["minimum_lines"]) > int(p["maximum_lines"]):
            raise ValidationError("Minimum lines exceeds maximum lines", "minimum_lines")
        if float(p["theta_min_degrees"]) >= float(p["theta_max_degrees"]):
            raise ValidationError(
                "Theta maximum must exceed its minimum", "theta_min_degrees", "theta_max_degrees"
            )

    def _peaks(self, work, edge_count):
        p = self.parameters
        rho = float(p["rho_step"])
        minimum, maximum, step = (
            math.radians(float(p[key]))
            for key in ("theta_min_degrees", "theta_max_degrees", "theta_step_degrees")
        )
        # OpenCV narrows the bin steps to float32. Budget before allocating the accumulator.
        rho_bins = round(float(np.float32(2 * sum(work.shape) + 1) / np.float32(rho)))
        angle_bins = 1 + math.floor((maximum - minimum) / float(np.float32(step)))
        if rho_bins < 1 or (rho_bins + 2) * (angle_bins + 2) > _MAX_ACCUMULATOR_CELLS:
            raise ValidationError("Hough accumulator needs 1 to 8,388,608 cells. Adjust the steps.")
        limit = int(p["maximum_lines"])
        if edge_count * (angle_bins + limit) > _MAX_POINT_OPERATIONS:
            raise ValidationError(
                "Too many Hough votes. Reduce edge pixels, angles, or line limit."
            )
        weighted = p["vote_mode"] == "weighted"
        threshold = int(p["minimum_votes"])
        if edge_count * (255 if weighted else 1) < threshold:
            return []
        found = cv2.HoughLinesWithAccumulator(
            work,
            rho,
            step,
            threshold - 1,
            min_theta=minimum,
            max_theta=maximum,
            use_edgeval=weighted,
        )
        # Native columns are rho, theta, votes. Shared support geometry takes votes first.
        return [] if found is None else found.reshape(-1, 3)[:limit, (2, 0, 1)]

    def process(self, inputs):
        image = inputs["image"]
        if isinstance(image, DepthArray) or image.ndim != 2 or image.dtype != np.uint8:
            raise ValidationError("Hough polar lines requires an 8-bit monochrome edge map")
        height, width = image.shape
        if not image.size or max(width, height) > 16384 or image.size > 64_000_000:
            raise ValidationError("Use a nonempty edge map up to 16,384 per axis and 64M pixels")
        mask = selection_mask(inputs, image)
        work = np.asarray(image) if mask is None else cv2.bitwise_and(image, image, mask=mask)
        edge_count = cv2.countNonZero(work)
        if edge_count > _MAX_EDGE_PIXELS:
            raise ValidationError("Use a sparse edge map with at most one million voting pixels")
        peaks = self._peaks(work, edge_count)
        items, polar_lines = [], []
        if len(peaks):
            points = cv2.findNonZero(work).reshape(-1, 2)
            distance = float(self.parameters["support_distance"])
            distance = distance or float(self.parameters["rho_step"]) / 2
            items, polar_lines = supported_lines(peaks, points, distance, (width, height))
        passed = len(items) >= int(self.parameters["minimum_lines"])
        result = Result(
            {
                "collection": collection("lines", items, image, self.input_payloads["image"]),
                "polar_lines": polar_lines,
                "count": len(items),
                "edge_pixel_count": edge_count,
            },
            passed=passed,
            fail_reasons=[] if passed else ["hough_polar_lines_below_minimum"],
            message=f"{len(items)} supported lines from {edge_count} edge pixels",
        )
        if self.generate_graphics and items:
            lines = tuple(
                graphic_object("polyline", item["geometry"]["points"], ACCEPTED, thickness=1)
                for item in items
            )
            result.drawings["polar Hough lines"] = GraphicDrawing(lines)
        return result
