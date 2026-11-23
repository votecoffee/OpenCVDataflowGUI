"""Native ximgproc segments in the shared image-coordinate line collection."""

import cv2
import numpy as np

from .core import Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .images import image_format
from .objects import collection, line_item
from .parameter_validation import NumericRange, ValidationError
from .processing import ImageTool, gray_image
from .result_graphics import ACCEPTED
from .selection import contains, selection_mask


class FastLineSegments(ImageTool):
    type_id, title, subtype = "inspect.fast_line_segments", "Fast line detector", "Features"
    category = block_type = "Inspection"
    description = "Find directed straight segments in an image or an upstream edge map."
    inputs = (
        Port("image"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("collection", "data"),
        Port("segments", "data"),
        Port("count", "number"),
    )
    defaults = {
        "length_threshold": 10,
        "distance_threshold": 1.414213562,
        "canny_low": 50.0,
        "canny_high": 50.0,
        "canny_aperture": "3",
        "merge_segments": False,
    }
    parameter_choices = {"canny_aperture": ("0", "3", "5", "7")}
    parameter_ranges = {
        "length_threshold": NumericRange(1, 32768, whole=True),
        "distance_threshold": NumericRange(0, 32768),
        "canny_low": NumericRange(0, 1_000_000),
        "canny_high": NumericRange(0, 1_000_000),
    }
    parameter_help = {
        "length_threshold": "Minimum segment length in pixels.",
        "distance_threshold": "Positive maximum distance in pixels from a point to a fitted line.",
        "canny_low": "Positive lower hysteresis threshold on the converted 8-bit image.",
        "canny_high": "Positive upper hysteresis threshold. At least the lower threshold.",
        "canny_aperture": "Sobel aperture for internal Canny. Zero uses a precomputed edge image.",
        "merge_segments": "Use the native incremental merge for nearby aligned segments.",
    }
    input_help = {
        "image": "Image converted to 8-bit grayscale for Canny. Aperture zero expects a mono "
        "8-bit edge map where nonzero pixels mark edges.",
        "mask": "Keep segments whose two endpoints fall in this mask.",
        "region": "Keep segments whose two endpoints fall in this region and any wired mask.",
    }
    output_help = {
        "image": "Original image. Directed arrows are available as separate result graphics.",
        "collection": "Line measurements in source pixels with spatial and pyramid context.",
        "segments": "Native directed [x1, y1, x2, y2] rows in source pixels.",
    }
    preview_safe = parallel_safe = process_inputs_readonly = True
    payload_reference = "image"

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if np.float32(float(p["distance_threshold"])) <= 0:
            raise ValidationError(
                "Distance threshold must be positive at OpenCV's float32 precision",
                "distance_threshold",
            )
        if not 0 < float(p["canny_low"]) <= float(p["canny_high"]):
            raise ValidationError(
                "Canny thresholds must be positive and ordered",
                "canny_low",
                "canny_high",
            )

    def process(self, inputs):
        source = inputs["image"]
        height, width = source.shape[:2]
        if min(width, height) < 6:
            raise ValidationError("Fast line detection requires images at least 6 × 6 pixels")
        if max(width, height) > 16384 or width * height > 64_000_000:
            raise ValidationError("Line collections are limited to 16,384 per axis and 64M pixels")
        selection = selection_mask(inputs, source)
        p = self.parameters
        aperture = int(p["canny_aperture"])
        if aperture:
            work = gray_image(source, (np.uint8,), auto_depth=True)
        else:
            if source.dtype != np.uint8 or source.ndim != 2 or image_format(source) != "gray":
                raise ValidationError("Aperture zero requires a monochrome uint8 edge map")
            # Native FLD consumes edge pixels in place when Canny is disabled.
            work = np.array(source, copy=True, order="C")

        create = getattr(getattr(cv2, "ximgproc", None), "createFastLineDetector", None)
        if create is None:
            raise ValidationError("Fast line detection requires OpenCV contrib ximgproc")
        detector = create(
            int(p["length_threshold"]),
            float(p["distance_threshold"]),
            float(p["canny_low"]),
            float(p["canny_high"]),
            aperture,
            p["merge_segments"],
        )
        found = detector.detect(work)
        items, segments = [], []
        for index, row in enumerate(found.reshape(-1, 4) if found is not None else ()):
            if not contains(selection, row[:2]) or not contains(selection, row[2:]):
                continue
            if len(items) == 100_000:
                raise ValidationError("Too many segments. Increase the minimum segment length.")
            items.append(line_item(row, index))
            segments.append(row.tolist())
        result = Result(
            {
                "image": source,
                "collection": collection("lines", items, source, self.input_payloads["image"]),
                "segments": segments,
                "count": len(items),
            },
            message=f"{len(items)} line segments",
        )
        if self.generate_graphics and segments:
            result.drawings["fast line segments"] = GraphicDrawing(
                tuple(
                    graphic_object("arrow", [row[:2], row[2:]], ACCEPTED, thickness=1)
                    for row in segments
                )
            )
        return result
