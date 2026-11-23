"""Contour-arc ellipse detection with source-frame collection geometry."""

import math

import cv2
import numpy as np

from .core import Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .objects import collection
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .processing import ImageTool, gray_image
from .selection import SELECTION_INPUTS, contains, selection_mask


class FindEllipses(ImageTool):
    type_id, title = "inspect.find_ellipses", "Find contour ellipses"
    category, block_type, subtype = "Inspection", "Inspection", "Features"
    description = "Find and rank rotated ellipses in an image using OpenCV's contour detector."
    inputs = SELECTION_INPUTS
    outputs = (
        Port("image"),
        Port("ellipses", "data"),
        Port("collection", "data"),
        Port("best_fit", "data"),
        Port("count", "number"),
    )
    payload_reference = "image"
    parallel_safe = True
    process_inputs_readonly = True
    defaults = {
        "score_threshold": 0.7,
        "reliability_threshold": 0.5,
        "center_distance_threshold": 0.05,
        "minimum_count": 1,
        "maximum_results": 64,
    }
    parameter_ranges = {
        "score_threshold": NumericRange(0, 1),
        "reliability_threshold": NumericRange(0, 1),
        "center_distance_threshold": NumericRange(0, 1),
        "minimum_count": NumericRange(0, 256, whole=True),
        "maximum_results": NumericRange(1, 256, whole=True),
    }
    parameter_help = {
        "score_threshold": "Native contour score threshold. The reported score also uses "
        "reliability, so it can be lower than this threshold.",
        "reliability_threshold": "Native threshold on the extent of supporting contour arcs.",
        "center_distance_threshold": "Allowed discrepancy between contour-arc center estimates "
        "as a fraction of the image diagonal.",
        "minimum_count": "Require at least this many retained detections to pass. Zero allows "
        "an empty result.",
        "maximum_results": "Keep at most this many detections, ranked by reported score.",
    }
    input_help = {
        "image": "Color or monochrome image converted to 8-bit grayscale for detection. "
        "The output image retains its original encoding and depth.",
        "mask": "Keep detections whose centers fall inside this mask.",
        "region": "Keep detections whose centers fall inside this region.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["minimum_count"] > p["maximum_results"]:
            raise ValidationError(
                "Minimum count must be at most maximum results", "minimum_count", "maximum_results"
            )

    def process(self, inputs):
        source = inputs["image"]
        height, width = source.shape[:2]
        if max(height, width) > 4096 or height * width > 2_097_152:
            raise ValidationError(
                "Ellipse detection accepts up to 2,097,152 pixels and 4096 per axis"
            )
        detect = getattr(getattr(cv2, "ximgproc", None), "findEllipses", None)
        if detect is None:
            raise ValidationError("Contour ellipse detection requires OpenCV contrib ximgproc")
        gray = gray_image(source, (np.uint8,), auto_depth=True)
        selection = selection_mask(inputs, gray)
        p = self.parameters
        found = None
        if (selection is None or cv2.countNonZero(selection)) and min(height, width) >= 3:
            found = detect(
                gray,
                scoreThreshold=float(p["score_threshold"]),
                reliabilityThreshold=float(p["reliability_threshold"]),
                centerDistanceThreshold=float(p["center_distance_threshold"]),
            )

        ellipses = []
        if found is not None:
            # OpenCV 5 returns two semiaxes, then score, then angle in radians.
            for row in found.reshape(-1, 6):
                if not np.isfinite(row).all():
                    continue
                x, y, a, b, score, radians = map(float, row)
                if min(a, b) <= 0 or not (0 <= x < width and 0 <= y < height):
                    continue
                if not contains(selection, (x, y)):
                    continue
                if a < b:
                    a, b, radians = b, a, radians + math.pi / 2
                ellipses.append(
                    {
                        "shape": "ellipse",
                        "center": [x, y],
                        "axis_lengths": [2 * a, 2 * b],
                        "angle_degrees": math.degrees(radians) % 180,
                        "score": score,
                    }
                )

        ellipses.sort(key=lambda fit: -fit["score"])
        ellipses = ellipses[: int(p["maximum_results"])]
        items, graphics = [], []
        phases = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        for index, fit in enumerate(ellipses):
            x, y = fit["center"]
            major, minor = fit["axis_lengths"]
            angle = math.radians(fit["angle_degrees"])
            u = np.array([math.cos(angle), math.sin(angle)])
            v = np.array([-u[1], u[0]])
            half_width = math.hypot(major * u[0], minor * v[0]) / 2
            half_height = math.hypot(major * u[1], minor * v[1]) / 2
            center = np.array([x, y])
            points = (
                center
                + np.outer(np.cos(phases) * major / 2, u)
                + np.outer(np.sin(phases) * minor / 2, v)
            )
            items.append(
                {
                    "id": index,
                    "geometry": {**fit, "points": points.tolist()},
                    "metrics": {
                        "x": x - half_width,
                        "y": y - half_height,
                        "center_x": x,
                        "center_y": y,
                        "width": 2 * half_width,
                        "height": 2 * half_height,
                        "aspect_ratio": half_width / half_height,
                        "major_size": major,
                        "minor_size": minor,
                        "rotated_aspect_ratio": major / minor,
                        "angle": fit["angle_degrees"],
                        "area": math.pi * major * minor / 4,
                        "strength": fit["score"],
                    },
                }
            )
            if self.generate_graphics:
                color = (0, 220, 120)
                axis = major / 2 * u
                graphics.extend(
                    (
                        graphic_object("polyline", points, color, closed=True),
                        graphic_object(
                            "polyline", (center - axis, center + axis), color, thickness=1
                        ),
                    )
                )

        count = len(ellipses)
        passed = count >= p["minimum_count"]
        return Result(
            {
                "image": source,
                "ellipses": ellipses,
                "collection": collection("ellipses", items, gray, self.input_payloads["image"]),
                "best_fit": ellipses[0] if count else NO_VALUE,
                "count": count,
            },
            drawings={"ellipses": GraphicDrawing(tuple(graphics))} if graphics else {},
            passed=passed,
            score=ellipses[0]["score"] if count else None,
            message=f"{count} contour ellipses retained",
            fail_reasons=[] if passed else ["too_few_ellipses"],
        )
