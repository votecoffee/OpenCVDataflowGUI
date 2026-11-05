"""Batched statistics over OpenCV's 45-degree integral-image rectangles."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .histogram import HISTOGRAM_CHANNELS
from .images import image_format
from .parameter_validation import ValidationError
from .processing import image_value
from .selection import SELECTION_INPUTS, selection_mask


def validate_windows(windows):
    if not isinstance(windows, list) or not 1 <= len(windows) <= 1024:
        raise ValidationError("Provide 1 to 1,024 tilted rectangles", "windows")
    for index, window in enumerate(windows, 1):
        if (
            not isinstance(window, list)
            or len(window) != 4
            or any(type(value) is not int for value in window)
            or min(window[:2]) < 0
            or min(window[2:]) < 1
        ):
            raise ValidationError(
                f"Window {index} needs [x, y, width, height] with whole pixels and positive sides",
                "windows",
            )


def window_vertices(windows, shape):
    height, width = shape
    vertices = []
    for index, (x, y, w, h) in enumerate(windows, 1):
        if x - h < 0 or x + w > width or y + w + h > height:
            raise ValidationError(
                f"Tilted window {index} extends outside the source image", "windows"
            )
        vertices.append(((x, y), (x + w, y + w), (x + w - h, y + w + h), (x - h, y + h)))
    return np.array(vertices)


def tilted_sums(image, vertices):
    table = cv2.integral3(image, sdepth=cv2.CV_64F, sqdepth=cv2.CV_64F)[2]
    x, y = np.moveaxis(vertices, -1, 0)
    top, right, bottom, left = table[y, x].T
    return (top - right) + (bottom - left)


class TiltedWindowStatistics(Block):
    type_id, title, category = (
        "cv.tilted_statistics",
        "Tilted window statistics",
        "Image processing",
    )
    block_type, subtype = "Image processing", "Statistics"
    inputs = (
        *SELECTION_INPUTS,
        Port("windows", "data", required=False, required_when_connected=True),
    )
    outputs = (Port("measurements", "data"), Port("count", "number"))
    defaults = {"windows": [[32, 0, 16, 16]], "include_alpha": False}
    parameter_help = {
        "windows": "Each [x, y, width, height] starts at the top integral-grid vertex. "
        "Width extends down-right and height extends down-left at 45 degrees.",
        "include_alpha": "Measure alpha as its own channel on four-channel input.",
    }
    input_help = {"windows": "Optional list of tilted windows that replaces the configured list."}

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        validate_windows(p["windows"])

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source)
        windows = inputs.get("windows", self.parameters["windows"])
        if "windows" in inputs:
            validate_windows(windows)
        vertices = window_vertices(windows, image.shape[:2])
        mask = selection_mask(inputs, source)
        selected = None if mask is None else mask != 0
        if selected is None:
            counts = np.array([2 * w * h for _, _, w, h in windows])
        else:
            counts = np.rint(tilted_sums(selected.astype(np.uint8), vertices)).astype(np.int64)

        records = [
            {
                "window": window,
                "integral_vertices": points,
                "sample_count": int(count),
                "channels": [],
            }
            for window, points, count in zip(windows, vertices.tolist(), counts, strict=True)
        ]
        labels = HISTOGRAM_CHANNELS[image_format(source)]
        if len(labels) == 4 and not self.parameters["include_alpha"]:
            labels = labels[:3]
        planes = image[..., None] if image.ndim == 2 else image
        valid = counts > 0
        if valid.any():
            index = 0 if selected is None else int(np.argmax(selected))
            divisor = np.maximum(counts, 1)
            for channel, label in enumerate(labels):
                plane = planes[..., channel].astype(np.float64)
                if selected is not None:
                    plane[~selected] = 0
                magnitude = float(np.max(np.abs(plane)))
                scale = magnitude if magnitude > 1e150 or 0 < magnitude < 1e-150 else 1.0
                if scale != 1:
                    plane /= scale
                offset = float(plane.flat[index])
                plane -= offset
                if selected is not None:
                    plane[~selected] = 0
                sums = tilted_sums(plane, vertices)
                plane *= plane
                squares = tilted_sums(plane, vertices)
                centered_mean = np.where(valid, sums / divisor, 0)
                variance = np.maximum(squares / divisor - centered_mean**2, 0)
                with np.errstate(over="ignore", invalid="ignore"):
                    total = (sums + counts * offset) * scale
                    mean = (centered_mean + offset) * scale
                    deviation = np.sqrt(variance) * scale
                if not np.isfinite((total[valid], mean[valid], deviation[valid])).all():
                    raise ValidationError(
                        "Tilted window statistics overflowed. Reduce the sample range"
                    )
                for j, record in enumerate(records):
                    record["channels"].append(
                        {
                            "channel": label,
                            "sum": float(total[j]) if valid[j] else 0.0,
                            "mean": float(mean[j]) if valid[j] else None,
                            "standard_deviation": float(deviation[j]) if valid[j] else None,
                        }
                    )
        else:
            for record in records:
                record["channels"] = [
                    {"channel": label, "sum": 0.0, "mean": None, "standard_deviation": None}
                    for label in labels
                ]
        drawings = {}
        if self.generate_graphics:
            # Convert integral-grid vertices to the boundaries of their sampled pixel centers.
            boundary = vertices - (1, 0.5)
            drawings["tilted windows"] = GraphicDrawing(
                tuple(
                    graphic_object(
                        "polyline",
                        points,
                        (70, 220, 140) if keep else (115, 115, 115),
                        thickness=1,
                        closed=True,
                    )
                    for points, keep in zip(boundary, valid, strict=True)
                )
            )
        return Result(
            {
                "measurements": {
                    "image_size": [image.shape[1], image.shape[0]],
                    "windows": records,
                },
                "count": len(records),
            },
            drawings=drawings,
        )
