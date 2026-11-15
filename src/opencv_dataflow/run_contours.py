"""Bounded native run-linked boundaries, with explicit flat-list and polygon semantics."""

import cv2
import numpy as np

from .blocks import Contours
from .core import Port
from .images import image_format
from .parameter_validation import NumericRange, ValidationError


def count_runs(image, limit):
    """Count horizontal nonzero runs using at most 64K pixels of scratch per strip."""
    rows = max(1, 65536 // image.shape[1])
    count = 0
    for start in range(0, image.shape[0], rows):
        foreground = image[start : start + rows] != 0
        count += int(np.count_nonzero(foreground[:, 0]))
        count += int(np.count_nonzero(foreground[:, 1:] & ~foreground[:, :-1]))
        if count > limit:
            raise ValidationError(
                "Mask exceeds Max runs before native contour extraction", "max_runs"
            )
    return count


class RunContours(Contours):
    type_id, title = "inspect.run_contours", "Run-linked contours"
    outputs = (Port("overlay", required=False), Port("mask", required=False), *Contours.outputs[2:])
    defaults = {
        **Contours.defaults,
        "retrieval": "list",
        "approximation": "simple",
        "max_runs": 250000,
        "max_contours": 10000,
        "max_points": 250000,
        "draw_overlay": False,
        "draw_mask": False,
    }
    parameter_choices = {
        **Contours.parameter_choices,
        "retrieval": ("list",),
        "approximation": ("simple",),
    }
    parameter_ranges = {
        **Contours.parameter_ranges,
        "max_runs": NumericRange(1, 2_000_000, whole=True),
        "max_contours": NumericRange(1, 100000, whole=True),
        "max_points": NumericRange(1, 2_000_000, whole=True),
    }
    parameter_help = {
        **Contours.parameter_help,
        "retrieval": "Flat list only. This tool does not supply hole parents or nested topology.",
        "approximation": (
            "Native run endpoints; ordering, winding and hole polygons differ "
            "from ordinary Simple contours."
        ),
        "min_area": "Filter by the area of native run polygons, not foreground pixel count.",
        "offset": "Whole-pixel [x,y] shift; raster outputs stay on the input grid.",
        "max_runs": "Reject before the native call when the search mask has too many row runs.",
        "max_contours": "Reject larger native results before measuring collections; no truncation.",
        "max_points": "Total native vertex budget before filtering or graphics; no sampling.",
        "draw_overlay": "Optional color raster; drawable graphics are separate.",
        "draw_mask": "Raster of selected native polygons; may differ from source pixels.",
    }
    input_help = {
        "image": "Nonempty uint8 monochrome mask; every nonzero value is foreground.",
        "mask": "Center mode keeps whole polygons; Clip modifies the search mask.",
        "region": "Optional region, intersected with Mask. Coordinate frames must match.",
    }
    preview_safe = parallel_safe = process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for name in ("draw_overlay", "draw_mask"):
            if not isinstance(p[name], bool):
                raise ValidationError(f"{name} must be a checkbox value", name)

    def process(self, inputs):
        image = inputs["image"]
        if (
            not isinstance(image, np.ndarray)
            or image.ndim != 2
            or image.dtype != np.uint8
            or not image.size
            or image_format(image) != "gray"
        ):
            raise ValidationError("Run-linked contours requires a nonempty uint8 monochrome mask")
        if max(image.shape) > 16384 or image.size > 64_000_000:
            raise ValidationError("Run-linked contours is limited to 64M pixels and 16384 per axis")
        offset = np.asarray(self.parameters["offset"], np.float64)
        low, high = np.iinfo(np.int32).min, np.iinfo(np.int32).max
        if np.any(offset < low) or np.any(offset + np.array(image.shape[::-1]) - 1 > high):
            raise ValidationError(
                "Offset and shifted image grid must fit signed 32-bit coordinates", "offset"
            )
        return super().process(inputs)

    def extract_contours(self, image, retrieval, approximation, offset):
        runs = count_runs(image, int(self.parameters["max_runs"]))
        contours, _ = cv2.findContoursLinkRuns(image)
        if len(contours) > int(self.parameters["max_contours"]):
            raise ValidationError("Native result exceeds Max contours", "max_contours")
        points = sum(c.shape[0] for c in contours)
        if points > int(self.parameters["max_points"]):
            raise ValidationError("Native result exceeds Max points", "max_points")
        # Always publish List semantics, including on a future wheel that supplies
        # the documented two-level hierarchy instead of this wheel's flat links.
        count = len(contours)
        hierarchy = np.full((1, count, 4), -1, np.int32)
        if count > 1:
            hierarchy[0, :-1, 0] = np.arange(1, count)
            hierarchy[0, 1:, 1] = np.arange(count - 1)
        if offset != (0, 0):
            shift = np.array(offset, np.int64).reshape(1, 1, 2)
            contours = tuple((c.astype(np.int64) + shift).astype(np.int32) for c in contours)
        return (
            contours,
            hierarchy,
            {
                "algorithm": "findContoursLinkRuns",
                "boundary_model": "native_run_endpoints",
                "hierarchy_semantics": "flat_list",
                "hole_parents_available": False,
                "horizontal_runs": runs,
                "raw_contour_count": count,
                "raw_point_count": points,
            },
        )
