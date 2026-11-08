"""Coverage and Euclidean background clearance at registered mask samples."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import require_same_frame
from .result_graphics import ACCEPTED, REJECTED
from .selection import selection_mask

_MISSING = (30, 210, 250)


def _sample_drawing(mask, color):
    indexes = np.flatnonzero(mask)
    if indexes.size > 256:
        indexes = indexes[np.linspace(0, indexes.size - 1, 256, dtype=np.intp)]
    rows, cols = np.divmod(indexes, mask.shape[1])
    return GraphicDrawing(tuple(
        graphic_object("polyline", ((x, y),), color, closed=True, thickness=3)
        for y, x in zip(rows, cols, strict=True)
    ))


class MaskClearance(Block):
    type_id, title = "inspect.mask_clearance", "Mask clearance at reference"
    category, block_type, subtype = "Inspection", "Inspection", "Measurements"
    inputs = (
        Port("image"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
        Port("reference", visual_type="region"),
    )
    outputs = (
        Port("clearance"),
        Port("measurements", "data"),
        Port("coverage", "number"),
        Port("pass_fraction", "number"),
        Port("missing_mask", required=False, visual_type="region"),
        Port("out_of_range_mask", required=False, visual_type="region"),
    )
    payload_reference = "image"
    process_inputs_readonly = True
    parallel_safe = True
    defaults = {
        "minimum_clearance": 1.0,
        "maximum_clearance": 0.0,
        "minimum_coverage": 1.0,
        "minimum_in_range_fraction": 1.0,
        "alignment": "same_frame",
    }
    parameter_ranges = {
        "minimum_clearance": NumericRange(0),
        "maximum_clearance": NumericRange(0),
        "minimum_coverage": NumericRange(0, 1),
        "minimum_in_range_fraction": NumericRange(0, 1),
    }
    parameter_choices = {"alignment": ("same_frame", "pixel_grid")}
    parameter_help = {
        "minimum_clearance": "Minimum Euclidean distance in pixels from a covered sample "
        "center to a background pixel center, including background outside the image.",
        "maximum_clearance": "Maximum accepted distance. Zero disables the upper limit.",
        "minimum_coverage": "Minimum fraction of reference samples covered by foreground. "
        "At least one reference sample must be covered.",
        "minimum_in_range_fraction": "Minimum fraction of covered samples within clearance "
        "limits. Missing samples are checked separately by coverage.",
        "alignment": "Pixel grid compares equal-size images by index across frame IDs. "
        "Use it for an independently registered reference.",
    }
    input_help = {
        "image": "Measured 8-bit monochrome mask. Nonzero pixels mark foreground.",
        "reference": "8-bit monochrome mask of expected sample locations, such as a centerline.",
        "mask": "Limit reference sample locations. Clearance uses the full foreground mask.",
        "region": "Limit reference sample locations. Clearance uses the full foreground mask.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        high = float(p["maximum_clearance"])
        if high and high < float(p["minimum_clearance"]):
            raise ValidationError(
                "Maximum clearance must be at least the minimum or zero", "maximum_clearance"
            )

    def process(self, inputs):
        image, reference = inputs["image"], inputs["reference"]
        if (
            image.ndim != 2 or image.dtype != np.uint8
            or reference.shape != image.shape or reference.dtype != np.uint8
        ):
            raise ValidationError("Use equal-size 8-bit monochrome foreground and reference masks")
        if not 0 < image.size <= 2_100_000:
            raise ValidationError("Mask clearance needs a nonempty image up to 2.1 million pixels")
        p = self.parameters
        if p["alignment"] == "same_frame":
            require_same_frame((image, reference), message="Masks must share a coordinate frame")
        samples = reference != 0
        selection = selection_mask(inputs, image)
        if selection is not None:
            samples &= selection != 0
        sample_count = int(np.count_nonzero(samples))
        if not sample_count:
            raise ValidationError("The selection must contain at least one reference sample")

        covered = samples & (image != 0)
        missing = samples & ~covered
        covered_count = int(np.count_nonzero(covered))
        clearance = np.zeros(image.shape, np.float32)
        if covered_count:
            padded = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
            distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
            clearance[covered] = distance[covered]
        passing = covered & (clearance >= float(p["minimum_clearance"]))
        high = float(p["maximum_clearance"])
        if high:
            passing &= clearance <= high
        outside = covered & ~passing
        passing_count = int(np.count_nonzero(passing))
        coverage = covered_count / sample_count
        fraction = passing_count / sample_count
        in_range = passing_count / covered_count if covered_count else 0.0
        values = clearance[covered]
        measurements = {
            "units": "px",
            "sample_count": sample_count,
            "covered_count": covered_count,
            "missing_count": sample_count - covered_count,
            "out_of_range_count": covered_count - passing_count,
            "in_range_fraction": in_range,
            "minimum": float(values.min()) if covered_count else None,
            "mean": float(values.mean(dtype=np.float64)) if covered_count else None,
            "maximum": float(values.max()) if covered_count else None,
        }
        reasons = []
        if not covered_count or coverage < float(p["minimum_coverage"]):
            reasons.append({
                "code": "mask_coverage_below_minimum",
                "message": f"Foreground covers {covered_count} of {sample_count} reference samples",
            })
        if covered_count and in_range < float(p["minimum_in_range_fraction"]):
            reasons.append({
                "code": "mask_clearance_out_of_range",
                "message": f"{covered_count - passing_count} covered samples are out of range",
            })
        drawings = {}
        if self.generate_graphics:
            for name, pixels, color in (
                ("Within limits", passing, ACCEPTED),
                ("Clearance out of range", outside, REJECTED),
                ("Missing foreground", missing, _MISSING),
            ):
                drawing = _sample_drawing(pixels, color)
                if drawing.objects:
                    drawings[name] = drawing
        return Result(
            {
                "clearance": clearance, "measurements": measurements, "coverage": coverage,
                "pass_fraction": fraction, "missing_mask": missing.astype(np.uint8) * 255,
                "out_of_range_mask": outside.astype(np.uint8) * 255,
            },
            drawings=drawings, passed=not reasons, score=fraction, fail_reasons=reasons,
            message=f"{coverage:.1%} covered and {fraction:.1%} passing at reference samples",
        )
