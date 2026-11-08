"""Compare binary edge maps with symmetric, pixel-distance tolerance."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .images import ImageArray
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, require_same_frame
from .result_graphics import ACCEPTED, REJECTED
from .selection import SELECTION_INPUTS, selection_mask

_MISSING = (30, 210, 250)


def _matched(edges, other, tolerance):
    if tolerance == 0 or not edges.any() or not other.any():
        return edges & other
    distance = cv2.distanceTransform((~other).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return edges & (distance <= tolerance)


class CompareEdgeMaps(Block):
    type_id, title = "inspect.compare_edge_maps", "Compare edge maps"
    category, block_type, subtype = "Inspection", "Inspection", "Measurements"
    inputs = (*SELECTION_INPUTS, Port("reference"))
    outputs = (
        Port("precision", "number"),
        Port("recall", "number"),
        Port("f1", "number"),
        Port("measurements", "data"),
        Port("map", required=False),
    )
    defaults = {
        "tolerance_pixels": 1.0,
        "minimum_precision": 0.0,
        "minimum_recall": 0.0,
        "alignment": "same_frame",
        "draw_map": True,
    }
    parameter_choices = {"alignment": ("same_frame", "pixel_grid")}
    parameter_ranges = {
        "tolerance_pixels": NumericRange(0, 100_000),
        "minimum_precision": NumericRange(0, 1),
        "minimum_recall": NumericRange(0, 1),
    }
    parameter_help = {
        "tolerance_pixels": "Inclusive Euclidean distance to a nearest edge in the other map.",
        "minimum_precision": "Minimum fraction of detected pixels near the reference.",
        "minimum_recall": "Minimum fraction of reference pixels covered by detected edges.",
        "alignment": "Pixel grid compares equal-size maps by index across frame IDs.",
        "draw_map": "Emit a color map of matched, unexpected, and missed edges.",
    }
    input_help = {
        "image": "Detected edges in an 8-bit monochrome image. Nonzero pixels are edges.",
        "reference": "Known edge positions in an equal-size 8-bit monochrome image.",
        "mask": "Restrict both edge sets to the selected pixels.",
        "region": "Restrict both edge sets to the selected pixels.",
    }
    payload_reference = "image"
    process_inputs_readonly = True

    def process(self, inputs):
        image, reference = inputs["image"], inputs["reference"]
        if (
            image.ndim != 2
            or image.dtype != np.uint8
            or reference.dtype != np.uint8
            or reference.shape != image.shape
        ):
            raise ValidationError("Use equal-size 8-bit monochrome edge maps")
        if image.size > 16_777_216:
            raise ValidationError("Edge comparison is limited to 16,777,216 pixels")
        p = self.parameters
        if p["alignment"] == "same_frame":
            require_same_frame(
                (image, reference), message="Edge maps must share a coordinate frame"
            )
        sample, expected = image != 0, reference != 0
        selection = selection_mask(inputs, image)
        if selection is not None:
            if not selection.any():
                raise ValidationError("Edge comparison requires a nonempty selection")
            sample &= selection != 0
            expected &= selection != 0

        tolerance = float(p["tolerance_pixels"])
        matched = _matched(sample, expected, tolerance)
        covered = _matched(expected, sample, tolerance)
        sample_count, reference_count = (int(np.count_nonzero(x)) for x in (sample, expected))
        matched_count, covered_count = (int(np.count_nonzero(x)) for x in (matched, covered))
        precision = matched_count / sample_count if sample_count else float(not reference_count)
        recall = covered_count / reference_count if reference_count else float(not sample_count)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        reasons = []
        if precision < float(p["minimum_precision"]):
            reasons.append("edge_precision_below_minimum")
        if recall < float(p["minimum_recall"]):
            reasons.append("edge_recall_below_minimum")
        view = NO_VALUE
        if p["draw_map"]:
            view = np.zeros((*image.shape, 3), np.uint8)
            view[sample & ~matched] = REJECTED
            view[expected & ~covered] = _MISSING
            view[matched] = ACCEPTED
            view = ImageArray(view, "bgr")
        counts = {
            "sample_count": sample_count,
            "reference_count": reference_count,
            "matched_sample_count": matched_count,
            "covered_reference_count": covered_count,
            "unexpected_count": sample_count - matched_count,
            "missed_count": reference_count - covered_count,
        }
        return Result(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "measurements": counts,
                "map": view,
            },
            passed=not reasons,
            score=f1,
            fail_reasons=reasons,
            message=f"Edge precision {precision:.1%} · recall {recall:.1%}",
        )
