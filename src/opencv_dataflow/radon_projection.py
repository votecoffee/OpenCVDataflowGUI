"""Native Radon projections and source-grid line geometry."""

import math

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .objects import collection, line_item
from .parameter_validation import NumericRange, ValidationError
from .processing import gray_image, image_value
from .result_graphics import ACCEPTED
from .selection import selection_mask

MAX_SIDE = 4096
MAX_ROTATED_PIXELS = 240_000_000
DEPTHS = (np.uint8, np.uint16, np.int16, np.float32, np.float64)


def sampled_angles(parameters):
    start = float(parameters["start_angle"])
    span = float(parameters["end_angle"]) - start
    step = math.copysign(float(parameters["angle_step"]), span)
    # The native angle count uses cvRound, including ties to even.
    count = round(span / step)
    if not 1 <= count <= 1440:
        raise ValidationError(
            "Angle range and step must give 1 to 1,440 projection bins",
            "start_angle", "end_angle", "angle_step",
        )
    return start + step * np.arange(count)


def projection_geometry(width, height, crop):
    if crop:
        return min(width, height), (width // 2, height // 2)
    side = math.ceil(math.hypot(width, height))
    center = side // 2
    return side, (center - (side - width) // 2, center - (side - height) // 2)


def circle_domain(width, height, side, origin, selection):
    domain = np.zeros((height, width), np.uint8)
    r = side // 2
    left, top = origin[0] - r, origin[1] - r
    cv2.circle(domain[top:top + side, left:left + side], (r, r), r, 255, -1)
    return domain if selection is None else cv2.bitwise_and(domain, selection)


def native_input(source, domain):
    if source.ndim == 2:
        pixels = image_value(source, DEPTHS)
    else:
        pixels = gray_image(source, (np.uint8, np.uint16, np.float32, np.float64), auto_depth=True)
    dtype = np.float64 if pixels.dtype in (np.uint16, np.int16) else pixels.dtype
    # The native transpose mutates the input buffer when it is square.
    pixels = np.array(pixels, dtype=dtype, order="C", copy=True)
    if domain is not None:
        pixels[domain == 0] = 0
    return pixels


def longest_segment(size, origin, normal, rho, domain):
    """Find the longest run of source pixels crossed by this line inside the selection."""
    w, h = size
    nx, ny = normal
    ox, oy = origin
    if abs(nx) >= abs(ny):
        y = np.arange(h)
        x = ox + (rho - ny * (y - oy)) / nx
    else:
        x = np.arange(w)
        y = oy + (rho - nx * (x - ox)) / ny
    valid = (x >= -1e-9) & (x <= w - 1 + 1e-9) & (y >= -1e-9) & (y <= h - 1 + 1e-9)
    if domain is not None:
        xi = np.rint(x).astype(int).clip(0, w - 1)
        yi = np.rint(y).astype(int).clip(0, h - 1)
        valid &= domain[yi, xi] != 0
    transitions = np.flatnonzero(np.diff(valid, prepend=False, append=False))
    start, end = max(zip(transitions[::2], transitions[1::2], strict=True),
                     key=lambda span: span[1] - span[0], default=(0, 0))
    if end - start < 2:
        return None
    positions = [start, end - 1]
    return np.column_stack((x[positions], y[positions])).clip(0, [w - 1, h - 1])


def projection_peaks(raw, angles, origin, size, domain, parameters):
    maximum = int(parameters["maximum_peaks"])
    if maximum == 0:
        return [], []
    # Independent scoring data keeps the signed projection intact.
    scores = raw.astype(np.float64)
    polarity = parameters["peak_polarity"]
    if polarity == "negative":
        scores *= -1
    elif polarity == "absolute":
        np.abs(scores, out=scores)
    center_row = len(scores) // 2
    minimum = float(parameters["minimum_strength"])
    distance = int(parameters["distance_separation"])
    separation = float(parameters["angle_separation"])
    peaks, lines = [], []
    for index in range(maximum):
        _, strength, _, (column, row) = cv2.minMaxLoc(scores)
        if strength <= 0 or strength < minimum:
            break
        angle, rho = float(angles[column]), row - center_row
        radians = math.radians(angle)
        normal = (math.cos(radians), -math.sin(radians))
        segment = longest_segment(size, origin, normal, rho, domain)
        peak = {
            "angle_index": column,
            "distance_index": row,
            "normal_angle_degrees": angle,
            "normal_xy": list(normal),
            "line_angle_degrees": (90.0 - angle) % 180.0,
            "distance_pixels": rho,
            "projection": float(raw[row, column]),
            "strength": strength,
            "segment": None if segment is None else segment.tolist(),
        }
        if segment is not None:
            item = line_item(segment, index)
            item["metrics"]["strength"] = strength
            lines.append(item)
        peaks.append(peak)

        delta = angles - angle
        nearby = np.flatnonzero(np.abs((delta + 90) % 180 - 90) <= separation)
        nearby_rhos = np.where(np.cos(np.deg2rad(delta[nearby])) < 0, -rho, rho)
        for other_col, other_rho in zip(nearby, nearby_rhos, strict=True):
            other_row = center_row + int(other_rho)
            lo, hi = max(0, other_row - distance), min(len(scores), other_row + distance + 1)
            scores[lo:hi, other_col] = -np.inf
    return peaks, lines


class RadonProjection(Block):
    type_id, title = "inspect.radon_projection", "Radon angular projection"
    category = block_type = "Inspection"
    subtype = "Measurements"
    description = "Sum intensity along angled lines and map strong projection peaks to the source image."
    inputs = (
        Port("image"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("projection"),
        Port("preview"),
        Port("axes", "data"),
        Port("peaks", "data"),
        Port("collection", "data"),
    )
    payload_reference = "image"
    parallel_safe = preview_safe = process_inputs_readonly = True
    defaults = {
        "start_angle": 0.0,
        "end_angle": 180.0,
        "angle_step": 1.0,
        "circle_crop": False,
        "peak_polarity": "positive",
        "maximum_peaks": 1,
        "minimum_strength": 0.0,
        "angle_separation": 5.0,
        "distance_separation": 4,
    }
    parameter_choices = {"peak_polarity": ("positive", "negative", "absolute")}
    parameter_ranges = {
        "start_angle": NumericRange(-360, 360),
        "end_angle": NumericRange(-360, 360),
        "angle_step": NumericRange(0.01, 360),
        "maximum_peaks": NumericRange(0, 64, whole=True),
        "minimum_strength": NumericRange(0),
        "angle_separation": NumericRange(0, 90),
        "distance_separation": NumericRange(0, MAX_SIDE, whole=True),
    }
    parameter_help = {
        "start_angle": "First line-normal angle. Zero gives vertical lines.",
        "end_angle": "Sweep boundary. OpenCV rounds the span divided by the angle step to get the bin count.",
        "angle_step": "Angle spacing in degrees. A descending interval sweeps in reverse.",
        "circle_crop": "Use the native centered circle cut from the largest inscribed square.",
        "peak_polarity": "Rank positive sums, negative sums by magnitude, or absolute sums.",
        "maximum_peaks": "Maximum reported peaks after suppression. Zero publishes profiles only.",
        "minimum_strength": "Minimum absolute or polarity-selected projection sum for a reported peak.",
        "angle_separation": "Suppress nearby line directions within this many degrees of a selected peak.",
        "distance_separation": "Suppress nearby peak distances in pixels within the angle window.",
    }
    input_help = {
        "image": "Intensity image or color converted to grayscale. Higher depths retain signed or fractional sums.",
        "mask": "Only selected source pixels contribute to the projections.",
        "region": "Intersect this source-grid region with the optional mask.",
    }
    output_help = {
        "image": "Source image with peak line spans available as result graphics.",
        "projection": "Unnormalized projection sums. Columns are angles and rows are signed distances.",
        "preview": "Contrast-stretched uint8 view of the projection.",
        "axes": "Angles, distance sampling, and the native rotation origin in source pixels.",
        "peaks": "Ranked signed sums with source-grid line geometry.",
        "collection": "Longest contiguous selected segment of each peak line, in source pixels.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.preview_output = "image"

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if type(p["circle_crop"]) is not bool:
            raise ValidationError("Circle crop must be boolean", "circle_crop")
        sampled_angles(p)

    def process(self, inputs):
        source = inputs["image"]
        height, width = source.shape[:2]
        if min(width, height) < 1 or max(width, height) > MAX_SIDE:
            raise ValidationError("Radon projection requires a nonempty image up to 4,096 pixels per axis")
        size = (width, height)
        p = self.parameters
        crop = p["circle_crop"]
        angles = sampled_angles(p)
        side, origin = projection_geometry(width, height, crop)
        if side > MAX_SIDE or side * side * len(angles) > MAX_ROTATED_PIXELS:
            raise ValidationError("Radon projection exceeds the rotation budget. Crop the image or use fewer angles.")
        transform = getattr(getattr(cv2, "ximgproc", None), "RadonTransform", None)
        if transform is None:
            raise ValidationError("Radon projection requires OpenCV contrib ximgproc")
        domain = selection_mask(inputs, source)
        if crop:
            domain = circle_domain(width, height, side, origin, domain)
        work = native_input(source, domain)
        signed_step = math.copysign(float(p["angle_step"]), float(p["end_angle"]) - angles[0])
        raw = transform(
            work, theta=signed_step, start_angle=float(angles[0]),
            end_angle=float(p["end_angle"]), crop=crop, norm=False,
        )
        if raw is None or raw.shape != (side, len(angles)) or not np.isfinite(raw).all():
            raise ValidationError("OpenCV did not return a finite projection on the requested grid")
        return self.result_for(source, raw, origin, angles, domain)

    def result_for(self, source, raw, origin, angles, domain):
        size = (source.shape[1], source.shape[0])
        axes = {
            "model": "radon_axes_v1",
            "angle_degrees": angles.tolist(),
            "distance_count": len(raw),
            "distance_start_pixels": -(len(raw) // 2),
            "distance_step_pixels": 1,
            "origin": list(origin),
            "source_size": list(size),
            "circle_crop": self.parameters["circle_crop"],
        }
        peaks, lines = projection_peaks(raw, angles, origin, size, domain, self.parameters)
        preview = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        drawings = {}
        if self.generate_graphics and lines:
            drawings["projection lines"] = GraphicDrawing(tuple(
                graphic_object("polyline", line["geometry"]["points"], ACCEPTED, thickness=1)
                for line in lines
            ))
        note = f"{len(angles)} angles, {len(peaks)} peaks"
        if peaks:
            note += f" · strongest line {peaks[0]['line_angle_degrees']:.1f}°"
        return Result(
            {
                "image": source,
                "projection": raw,
                "preview": preview,
                "axes": axes,
                "peaks": {"model": "radon_peaks_v1", "items": peaks},
                "collection": collection("lines", lines, source, self.input_payloads["image"]),
            },
            payloads={name: {"radon_axes": axes} for name in ("projection", "preview")},
            drawings=drawings,
            message=note,
        )
