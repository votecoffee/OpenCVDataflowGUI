"""Threshold-sweep blob detection with OpenCV's native shape filters."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_circle
from .objects import circle_item, collection
from .parameter_validation import NumericRange, ValidationError
from .processing import gray_image
from .result_graphics import ACCEPTED
from .selection import SELECTION_INPUTS, selection_mask

_SHAPE_FILTERS = {
    "area": ("Area", 25.0, 5000.0, True),
    "circularity": ("Circularity", 0.8, 1.01, False),
    "inertia": ("InertiaRatio", 0.1, 1.01, True),
    "convexity": ("Convexity", 0.95, 1.01, True),
}
_SWEEP = {
    "min_threshold": "minThreshold",
    "max_threshold": "maxThreshold",
    "threshold_step": "thresholdStep",
    "min_distance": "minDistBetweenBlobs",
    "min_repeatability": "minRepeatability",
}


class SimpleBlobDetection(Block):
    type_id, title, category = "inspect.simple_blob_detector", "Simple blob detector", "Inspection"
    block_type, subtype = "Inspection", "Features"
    process_inputs_readonly = True
    inputs = SELECTION_INPUTS
    outputs = (Port("collection", "data"), Port("points", "data"), Port("count", "number"))
    defaults = {
        "min_threshold": 50.0,
        "max_threshold": 220.0,
        "threshold_step": 10.0,
        "min_distance": 10.0,
        "min_repeatability": 2,
        "polarity": "dark",
        **{
            key: value
            for name, (_, low, high, enabled) in _SHAPE_FILTERS.items()
            for key, value in (
                (f"filter_{name}", enabled), (f"min_{name}", low), (f"max_{name}", high)
            )
        },
        "minimum_count": 0,
    }
    parameter_choices = {"polarity": ("dark", "light", "any")}
    parameter_ranges = {
        "min_threshold": NumericRange(0, 255),
        "max_threshold": NumericRange(0, 256),
        "threshold_step": NumericRange(0.1, 256),
        "min_distance": NumericRange(0.001, 16384),
        "min_repeatability": NumericRange(1, 2560, whole=True),
        "min_area": NumericRange(0.001, 64_000_001),
        "max_area": NumericRange(0.001, 64_000_001),
        **{
            f"{bound}_{shape}": NumericRange(0.000001, 2)
            for shape in ("circularity", "inertia", "convexity")
            for bound in ("min", "max")
        },
        "minimum_count": NumericRange(0, 100_000, whole=True),
    }
    parameter_help = {
        "max_threshold": "The threshold sweep stops before this value.",
        "min_distance": "Minimum distance in pixels for treating nearby centers as separate blobs.",
        "min_repeatability": "Minimum observations needed to retain a grouped blob.",
        "minimum_count": "Minimum detected blobs for a pass. Zero disables the count check.",
        **{
            f"max_{shape}": "Upper limit is exclusive. Set above 1 to accept an exact ratio of 1."
            for shape in ("circularity", "inertia", "convexity")
        },
        "max_area": "Upper limit for contour area in square pixels, exclusive.",
    }
    input_help = {"mask": "Accept only blob centers inside the mask. The source is not clipped."}

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for name in ("threshold", *_SHAPE_FILTERS):
            low, high = f"min_{name}", f"max_{name}"
            if float(p[low]) >= float(p[high]):
                raise ValidationError(
                    f"{name.capitalize()} maximum must exceed its minimum", low, high
                )
        if (
            float(p["min_threshold"]) + float(p["threshold_step"]) >= float(p["max_threshold"])
            and p["min_repeatability"] != 1
        ):
            raise ValidationError(
                "A single threshold requires minimum repeatability of one",
                "min_repeatability", "min_threshold", "max_threshold", "threshold_step",
            )

    def process(self, inputs):
        image = gray_image(inputs["image"], (np.uint8,))
        if not image.size or max(image.shape) > 16384 or image.size > 64_000_000:
            raise ValidationError("Use a nonempty image up to 16,384 per axis and 64M pixels")
        p = self.parameters
        options = cv2.SimpleBlobDetector_Params()
        for key, field in _SWEEP.items():
            value = int(p[key]) if key == "min_repeatability" else float(p[key])
            setattr(options, field, value)
        options.filterByColor = p["polarity"] != "any"
        options.blobColor = 255 if p["polarity"] == "light" else 0
        for name, (native, _, _, _) in _SHAPE_FILTERS.items():
            flag = "Inertia" if name == "inertia" else native
            setattr(options, "filterBy" + flag, p[f"filter_{name}"])
            setattr(options, "min" + native, float(p[f"min_{name}"]))
            setattr(options, "max" + native, float(p[f"max_{name}"]))
        mask = selection_mask(inputs, image)
        detector = cv2.SimpleBlobDetector_create(options)
        keypoints = detector.detect(image, mask)
        if len(keypoints) > 100_000:
            raise ValidationError("Too many blob keypoints; tighten the area or shape filters")
        items = [
            circle_item((*point.pt, point.size / 2), None, index)
            for index, point in enumerate(keypoints, 1)
        ]
        passed = len(items) >= p["minimum_count"]
        result = Result(
            {
                "collection": collection("circles", items, image, self.input_payloads["image"]),
                "points": [list(point.pt) for point in keypoints],
                "count": len(items),
            },
            passed=passed,
            fail_reasons=[] if passed else ["simple_blob_count_below_minimum"],
            message=f"{len(items)} blob keypoints in input image coordinates",
        )
        if self.generate_graphics and items:
            result.drawings["blob keypoints"] = GraphicDrawing(tuple(
                graphic_circle(*point.pt, point.size / 2, ACCEPTED) for point in keypoints
            ))
        return result
