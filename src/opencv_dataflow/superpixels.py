"""OpenCV SLIC-family segmentation with selectable label and boundary maps."""

import cv2
import numpy as np

from .core import Port, Result
from .parameter_validation import NumericRange, ValidationError
from .processing import ImageTool
from .region_proposals import segmentation_image
from .selection import SELECTION_INPUTS, selection_mask

_ALGORITHMS = ("slic", "slico", "mslic")
_MAX_PIXELS = 2_100_000


class SuperpixelsSLIC(ImageTool):
    type_id, title, subtype = "cv.slic_superpixels", "SLIC superpixels", "Segmentation"
    inputs = SELECTION_INPUTS
    outputs = (Port("labels"), Port("boundaries"), Port("count", "integer"))
    defaults = {
        "algorithm": "slico",
        "region_size": 10,
        "ruler": 10.0,
        "iterations": 10,
        "connectivity_percent": 25,
        "thick_boundaries": False,
        "color_space": "lab",
    }
    parameter_choices = {"algorithm": _ALGORITHMS, "color_space": ("lab", "bgr")}
    parameter_ranges = {
        "region_size": NumericRange(3, 512, whole=True),
        "ruler": NumericRange(0.01, 1000),
        "iterations": NumericRange(1, 50, whole=True),
        "connectivity_percent": NumericRange(0, 100, whole=True),
    }
    parameter_help = {
        "algorithm": "SLIC uses fixed compactness. SLICO adapts it. MSLIC adapts region sizes.",
        "region_size": "Approximate pixel spacing. Must fit the shorter dimension. "
        "MSLIC needs room for at least two starting regions.",
        "ruler": "Native compactness factor. Higher values favor regular shapes in SLIC.",
        "iterations": "Number of native refinement iterations.",
        "connectivity_percent": "Merge small components as a percentage of typical region area. "
        "Zero skips the final connectivity pass.",
        "thick_boundaries": "Use OpenCV's thick contour-mask option.",
        "color_space": "Use Lab or BGR for color input. Grayscale stays grayscale.",
    }
    input_help = {
        "image": "Grayscale, BGR or RGB uint8 image with at most 2.1 million pixels.",
        "mask": "Optional output selection. Zero marks unselected labels.",
        "region": "Optional output selection. Segmentation uses the full image.",
    }
    process_inputs_readonly = True
    parallel_safe = True

    def process(self, inputs):
        source = inputs["image"]
        height, width = source.shape[:2]
        p = self.parameters
        size = int(p["region_size"])
        if size > min(height, width):
            raise ValidationError("Region size must fit the shorter image dimension", "region_size")
        if p["algorithm"] == "mslic" and 2 * max(height, width) < 3 * size:
            raise ValidationError(
                "MSLIC needs at least two starting regions. Reduce the region size.", "region_size"
            )
        if height * width > _MAX_PIXELS:
            raise ValidationError("SLIC superpixels are limited to 2.1 million pixels")
        image = segmentation_image(source)
        selection = selection_mask(inputs, source)
        if image.ndim == 3 and p["color_space"] == "lab":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        slic = cv2.ximgproc.createSuperpixelSLIC(
            image, getattr(cv2.ximgproc, p["algorithm"].upper()), size, float(p["ruler"])
        )
        slic.iterate(int(p["iterations"]))
        slic.enforceLabelConnectivity(int(p["connectivity_percent"]))
        labels = slic.getLabels()
        boundaries = slic.getLabelContourMask(thick_line=p["thick_boundaries"])
        if (
            labels is None or labels.shape != (height, width) or labels.dtype != np.int32
            or labels.min() < 0
            or boundaries is None or boundaries.shape != labels.shape
            or boundaries.dtype != np.uint8
        ):
            raise ValidationError("OpenCV returned an invalid superpixel map")
        labels = labels + 1  # Keep zero for pixels excluded by the selection.
        if selection is not None:
            outside = selection == 0
            labels[outside] = 0
            boundaries[outside] = 0
        unique = np.unique(labels)
        count = int(unique.size - (unique[0] == 0))
        return Result(
            {"labels": labels, "boundaries": boundaries, "count": count},
            message=f"{count} {p['algorithm'].upper()} superpixels in the output selection",
        )
