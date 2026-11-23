"""Detect scale-aware Harris corners and publish keypoints for downstream descriptors."""

import cv2
import numpy as np

from .core import Port, Result
from .graphics_layers import GraphicDrawing, graphic_circle
from .parameter_validation import NumericRange, ValidationError
from .processing import ImageTool, gray_image
from .result_graphics import ACCEPTED
from .selection import SELECTION_INPUTS, selection_mask

MAX_PIXELS = 1_048_576


class HarrisLaplaceInterestPoints(ImageTool):
    type_id, title, subtype = "inspect.harris_laplace", "Harris–Laplace interest points", "Features"
    category = block_type = "Inspection"
    description = "Find Harris corners at local Difference-of-Gaussians scale maxima."
    inputs = SELECTION_INPUTS
    outputs = (
        Port("image"),
        Port("keypoints", "data"),
        Port("points", "data"),
        Port("count", "number"),
    )
    payload_reference = "image"
    preview_safe = parallel_safe = process_inputs_readonly = True
    defaults = {
        "octaves": 6,
        "layers": "4",
        "corner_threshold": 0.01,
        "dog_threshold": 0.01,
        "max_corners": 512,
        "preview_count": 100,
    }
    parameter_choices = {"layers": ("2", "4")}
    parameter_ranges = {
        "octaves": NumericRange(1, 8, whole=True),
        "corner_threshold": NumericRange(0, 1),
        "dog_threshold": NumericRange(0, 1),
        "max_corners": NumericRange(1, 5000, whole=True),
        "preview_count": NumericRange(0, 512, whole=True),
    }
    parameter_help = {
        "octaves": "Requested scale-space octaves. OpenCV bounds this by the input dimensions.",
        "layers": "Number of intermediate scales per octave. OpenCV supports 2 or 4.",
        "corner_threshold": "Minimum Harris response as a fraction of the strongest response "
        "at each scale.",
        "dog_threshold": "Minimum Difference-of-Gaussians response on normalized intensity.",
        "max_corners": "Keep at most this many selected corners, strongest first.",
        "preview_count": "Number of scale circles to draw in response order. Zero hides drawings.",
    }
    input_help = {
        "image": "Grayscale or color image converted to 8-bit grayscale for detection. "
        "The native doubled pyramid limits the input to 1,048,576 pixels.",
        "mask": "Restrict corner centers to this mask before retaining the strongest keypoints.",
        "region": "Restrict corner centers to this region and any wired mask.",
    }
    output_help = {
        "image": "Original input image or selected pyramid level.",
        "keypoints": "Source-frame center, native diameter, Harris response and detector octave. "
        "Orientation is marked unknown.",
        "points": "Keypoint centers in the selected input image's pixel frame.",
    }

    def process(self, inputs):
        source = inputs["image"]
        height, width = source.shape[:2]
        if min(height, width) < 16 or max(height, width) > 4096 or height * width > MAX_PIXELS:
            raise ValidationError(
                "Harris–Laplace requires 16–4096 pixels per axis and at most 1,048,576 pixels"
            )
        create = getattr(
            getattr(cv2, "xfeatures2d", None), "HarrisLaplaceFeatureDetector_create", None
        )
        if create is None:
            raise ValidationError(
                "Harris–Laplace interest points require OpenCV contrib xfeatures2d"
            )
        image = gray_image(source, (np.uint8,), auto_depth=True)
        mask = selection_mask(inputs, image)
        p = self.parameters
        keys = ()
        if mask is None or cv2.countNonZero(mask):
            detector = create(
                numOctaves=int(p["octaves"]),
                corn_thresh=float(p["corner_threshold"]),
                DOG_thresh=float(p["dog_threshold"]),
                maxCorners=int(p["max_corners"]),
                num_layers=int(p["layers"]),
            )
            keys = detector.detect(image, mask)
        points = [list(key.pt) for key in keys]
        result = Result(
            {
                "image": source,
                "keypoints": {
                    "algorithm": "HarrisLaplace",
                    "image_size": [width, height],
                    "keypoints": [
                        {
                            "point": point,
                            "size": key.size,
                            "angle": -1.0,
                            "response": key.response,
                            "octave": key.octave,
                        }
                        for point, key in zip(points, keys, strict=True)
                    ],
                },
                "points": points,
                "count": len(keys),
            },
            message=f"{len(keys)} scale-aware corners",
        )
        if self.generate_graphics and keys and p["preview_count"]:
            display = keys[: int(p["preview_count"])]
            result.drawings["Harris–Laplace scale circles"] = GraphicDrawing(
                tuple(graphic_circle(*key.pt, key.size / 2, ACCEPTED) for key in display)
            )
        return result
