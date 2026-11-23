"""Fast Hough accumulator and peak lines in the source image frame."""

import math

import cv2
import numpy as np

from .core import Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .images import DepthArray
from .objects import collection, line_item
from .parameter_validation import NumericRange, ValidationError
from .payloads import frame
from .processing import ImageTool
from .result_graphics import ACCEPTED
from .selection import SELECTION_INPUTS, selection_mask

_MAX_CELLS = 8_388_608
# Native angle enum, followed by accumulator row coefficients for height, width, and bias.
_ANGLES = {
    "full": ("ARO_315_135", 2, 2, -3),
    "vertical": ("ARO_315_45", 2, 0, -1),
    "horizontal": ("ARO_45_135", 0, 2, -1),
    "near_vertical": ("ARO_CTR_VER", 1, 0, 0),
    "near_horizontal": ("ARO_CTR_HOR", 0, 1, 0),
    "vertical_down_right": ("ARO_315_0", 1, 0, 0),
    "vertical_down_left": ("ARO_0_45", 1, 0, 0),
    "horizontal_up_right": ("ARO_45_90", 0, 1, 0),
    "horizontal_down_right": ("ARO_90_135", 0, 1, 0),
}
_OPERATIONS = {
    "sum": "FHT_ADD",
    "average": "FHT_AVE",
    "minimum": "FHT_MIN",
    "maximum": "FHT_MAX",
}
_DEPTHS = {"float32": cv2.CV_32F, "float64": cv2.CV_64F, "int32": cv2.CV_32S}


class FastHoughLines(ImageTool):
    type_id, title, subtype = "inspect.fast_hough_lines", "Fast Hough lines", "Features"
    category = block_type = "Inspection"
    description = "Transform an edge map into a Fast Hough accumulator and recover its peak lines."
    inputs = SELECTION_INPUTS
    outputs = (
        Port("image"),
        Port("accumulator"),
        Port("collection", "data"),
        Port("peaks", "data"),
        Port("count", "number"),
    )
    payload_reference = "image"
    preview_safe = parallel_safe = process_inputs_readonly = True
    defaults = {
        "angle_range": "full",
        "operation": "sum",
        "layout": "deskewed",
        "accumulator_depth": "float32",
        "minimum_response": 0.0,
        "minimum_peak_fraction": 0.6,
        "suppression_bins": 7,
        "maximum_lines": 24,
        "minimum_lines": 1,
    }
    parameter_choices = {
        "angle_range": tuple(_ANGLES),
        "operation": tuple(_OPERATIONS),
        "layout": ("deskewed", "raw"),
        "accumulator_depth": tuple(_DEPTHS),
    }
    parameter_ranges = {
        "minimum_response": NumericRange(0, 255 * 4096),
        "minimum_peak_fraction": NumericRange(0, 1),
        "suppression_bins": NumericRange(0, 128, whole=True),
        "maximum_lines": NumericRange(1, 64, whole=True),
        "minimum_lines": NumericRange(0, 64, whole=True),
    }
    parameter_help = {
        "angle_range": "Search all slopes, one direction family, one quadrant, or within "
        "about 26.6 degrees of an axis. Direction names use image coordinates.",
        "operation": "Native intensity reduction along digital lines. Average combines pairs "
        "recursively, and integer depth rounds the intermediate results.",
        "layout": "Deskewed shifts rows to reduce wraparound. Raw keeps the cyclic grid.",
        "accumulator_depth": "Native accumulator precision. Float32 represents 8-bit sums "
        "exactly within this block's image limits.",
        "minimum_response": "Inclusive minimum in native accumulator units. Zero still "
        "requires a positive response.",
        "minimum_peak_fraction": "Require this fraction of the strongest accumulator response.",
        "suppression_bins": "Ignore this many bins in each direction around each examined peak.",
        "maximum_lines": "Keep at most this many visible lines, strongest first.",
        "minimum_lines": "Fewest retained lines needed to pass. Zero allows no detections.",
    }
    input_help = {
        "image": "8-bit monochrome edge map. Higher pixel values have more weight in Sum mode.",
        "mask": "Zero votes outside this mask before the transform.",
        "region": "Zero votes outside this region and any wired mask before the transform.",
    }
    output_help = {
        "image": "Original edge map in its input coordinate frame.",
        "accumulator": "Native response map in a separate frame. Column and row are bin indices.",
        "collection": "Line hypotheses clipped to the source image, with response strength.",
        "peaks": "Bin indices, native response, endpoints and signed normal offset for each line.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if int(p["minimum_lines"]) > int(p["maximum_lines"]):
            raise ValidationError(
                "Minimum lines must be at most maximum lines", "minimum_lines", "maximum_lines"
            )

    def process(self, inputs):
        source = inputs["image"]
        if isinstance(source, DepthArray) or source.ndim != 2 or source.dtype != np.uint8:
            raise ValidationError("Fast Hough lines requires an 8-bit monochrome edge map")
        height, width = source.shape
        p = self.parameters
        angle, h_factor, w_factor, bias = _ANGLES[p["angle_range"]]
        rows = h_factor * height + w_factor * width + bias
        columns = width + height
        if not (2 <= min(height, width) and max(height, width) <= 4096):
            raise ValidationError("Fast Hough lines accepts 2 to 4096 pixels per image axis")
        if rows * columns > _MAX_CELLS:
            raise ValidationError(
                "Fast Hough accumulator exceeds 8,388,608 cells. Crop, downsample, "
                "or select a narrower angle range.",
                "angle_range",
            )
        api = getattr(cv2, "ximgproc", None)
        transform = getattr(api, "FastHoughTransform", None)
        to_line = getattr(api, "HoughPoint2Line", None)
        if transform is None or to_line is None:
            raise ValidationError("Fast Hough lines requires OpenCV contrib ximgproc")
        angle_range = getattr(api, angle)
        layout = api.HDO_DESKEW if p["layout"] == "deskewed" else api.HDO_RAW
        selection = selection_mask(inputs, source)
        work = np.asarray(source)
        if selection is not None:
            work = cv2.bitwise_and(work, work, mask=selection)
        accumulator = transform(
            work,
            _DEPTHS[p["accumulator_depth"]],
            angleRange=angle_range,
            op=getattr(api, _OPERATIONS[p["operation"]]),
            makeSkew=layout,
        )
        items, peaks = [], []
        search = np.ones(accumulator.shape, np.uint8)
        strongest = cv2.minMaxLoc(accumulator)[1]
        minimum = max(float(p["minimum_response"]), strongest * float(p["minimum_peak_fraction"]))
        radius = int(p["suppression_bins"])
        limit = int(p["maximum_lines"])
        # Invalid or cyclic bins can fail to intersect the source. Bound recovery work.
        for _ in range(2 * limit):
            _, response, _, bin_xy = cv2.minMaxLoc(accumulator, mask=search)
            if response <= 0 or response < minimum:
                break
            x, y = bin_xy
            search[max(0, y - radius) : y + radius + 1, max(0, x - radius) : x + radius + 1] = 0
            line = to_line(bin_xy, work, angleRange=angle_range, makeSkew=layout)
            visible, start, end = cv2.clipLine((0, 0, width, height), line[:2], line[2:])
            if not visible or start == end:
                continue
            dx, dy = line[2] - line[0], line[3] - line[1]
            phi = math.atan2(-dx, dy) % math.pi
            peak = {
                "bin": [x, y],
                "response": response,
                "points": [list(start), list(end)],
                "normal_angle_degrees": math.degrees(phi),
                "rho_pixels": line[0] * math.cos(phi) + line[1] * math.sin(phi),
            }
            item = line_item((*start, *end), len(items))
            item["metrics"]["strength"] = response
            item["geometry"]["fast_hough"] = peak
            items.append(item)
            peaks.append(peak)
            if len(peaks) == limit:
                break

        passed = len(peaks) >= int(p["minimum_lines"])
        result = Result(
            {
                "image": source,
                "accumulator": accumulator,
                "collection": collection("lines", items, source, self.input_payloads["image"]),
                "peaks": peaks,
                "count": len(peaks),
            },
            payloads={
                "accumulator": {
                    "spatial": frame([columns, rows], f"{self.id}:fast-hough"),
                    "fast_hough": {
                        "source_size": [width, height],
                        "angle_range": p["angle_range"],
                        "layout": p["layout"],
                        "operation": p["operation"],
                    },
                },
            },
            passed=passed,
            fail_reasons=[] if passed else ["fast_hough_lines_below_minimum"],
            message=f"{len(peaks)} lines from a {columns} × {rows} Fast Hough accumulator",
        )
        if self.generate_graphics and peaks:
            result.drawings["Fast Hough lines"] = GraphicDrawing(
                tuple(
                    graphic_object("polyline", peak["points"], ACCEPTED, thickness=1)
                    for peak in peaks
                )
            )
        return result
