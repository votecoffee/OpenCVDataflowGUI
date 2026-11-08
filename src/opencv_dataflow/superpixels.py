"""OpenCV superpixel segmentation with selectable label and boundary maps."""

from math import ceil, isqrt

import cv2
import numpy as np

from .core import Port, Result
from .parameter_validation import NumericRange, ValidationError
from .processing import ImageTool
from .region_proposals import segmentation_image
from .selection import SELECTION_INPUTS, selection_mask

_ALGORITHMS = ("slic", "slico", "mslic")
_MAX_PIXELS = 2_100_000


class _Superpixels(ImageTool):
    subtype = "Segmentation"
    inputs = SELECTION_INPUTS
    outputs = (Port("labels"), Port("boundaries"), Port("count", "integer"))
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
        if min(height, width) < 3 or height * width > _MAX_PIXELS:
            raise ValidationError("Superpixels need at least 3 × 3 and at most 2.1 million pixels")
        image = segmentation_image(source)
        selection = selection_mask(inputs, source)
        color_space = self.parameters["color_space"]
        if image.ndim == 3 and color_space != "bgr":
            code = cv2.COLOR_BGR2Lab if color_space == "lab" else cv2.COLOR_BGR2HSV
            image = cv2.cvtColor(image, code)
        segmenter, name = self.segment(image)
        labels = segmenter.getLabels()
        boundaries = segmenter.getLabelContourMask(thick_line=self.parameters["thick_boundaries"])
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
            message=f"{count} {name} superpixels in the output selection",
        )


class SuperpixelsSLIC(_Superpixels):
    type_id, title = "cv.slic_superpixels", "SLIC superpixels"
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
    def segment(self, image):
        height, width = image.shape[:2]
        p = self.parameters
        size = int(p["region_size"])
        if size > min(height, width):
            raise ValidationError("Region size must fit the shorter image dimension", "region_size")
        if p["algorithm"] == "mslic" and 2 * max(height, width) < 3 * size:
            raise ValidationError(
                "MSLIC needs at least two starting regions. Reduce the region size.", "region_size"
            )
        slic = cv2.ximgproc.createSuperpixelSLIC(
            image, getattr(cv2.ximgproc, p["algorithm"].upper()), size, float(p["ruler"])
        )
        slic.iterate(int(p["iterations"]))
        slic.enforceLabelConnectivity(int(p["connectivity_percent"]))
        return slic, p["algorithm"].upper()


class SuperpixelsLSC(_Superpixels):
    type_id, title = "cv.lsc_superpixels", "LSC superpixels"
    defaults = {
        "region_size": 10,
        "ratio": 0.075,
        "iterations": 10,
        "min_component_pixels": 25,
        "thick_boundaries": False,
        "color_space": "lab",
    }
    parameter_choices = {"color_space": ("lab", "bgr")}
    parameter_ranges = {
        "region_size": NumericRange(3, 512, whole=True),
        "ratio": NumericRange(0.001, 10),
        "iterations": NumericRange(1, 50, whole=True),
        "min_component_pixels": NumericRange(0, 10_000, whole=True),
    }
    parameter_help = {
        "region_size": "Approximate spacing in pixels. Must leave room for a seed grid.",
        "ratio": "Spatial compactness weight relative to color variation.",
        "iterations": "Number of refinement iterations.",
        "min_component_pixels": "Merge small components by area in pixels. Zero skips this pass.",
        "thick_boundaries": "Use OpenCV's thick contour-mask option.",
        "color_space": "Use Lab or BGR for color input. Grayscale stays grayscale.",
    }

    def segment(self, image):
        p = self.parameters
        height, width = image.shape[:2]
        size = int(p["region_size"])
        # Match the native seed calculation before it divides by the grid dimensions.
        seeds = int(np.float32(height * width) / (size * size))
        columns = isqrt(seeds * width // height)
        if size > min(height, width) or not 1 <= columns <= min(seeds, width):
            raise ValidationError("Region size must leave room for an LSC seed grid", "region_size")
        if not image.any():
            # Native LSC divides by maximum intensity. Any positive uniform image
            # gives the same normalized color coordinates for an all-black input.
            image = np.ones_like(image)
        segmenter = cv2.ximgproc.createSuperpixelLSC(image, size, float(p["ratio"]))
        segmenter.iterate(int(p["iterations"]))
        if p["min_component_pixels"]:
            segmenter.enforceLabelConnectivity(int(p["min_component_pixels"]))
        return segmenter, "LSC"


class SuperpixelsSEEDS(_Superpixels):
    type_id, title = "cv.seeds_superpixels", "SEEDS superpixels"
    defaults = {
        "target_count": 400,
        "levels": 4,
        "prior": 2,
        "histogram_bins": 5,
        "double_step": False,
        "iterations": 4,
        "thick_boundaries": False,
        "color_space": "lab",
    }
    parameter_choices = {"color_space": ("lab", "hsv", "bgr")}
    parameter_ranges = {
        "target_count": NumericRange(10, 10_000, whole=True),
        "levels": NumericRange(2, 8, whole=True),
        "prior": NumericRange(0, 5, whole=True),
        "histogram_bins": NumericRange(2, 16, whole=True),
        "iterations": NumericRange(1, 50, whole=True),
    }
    parameter_help = {
        "target_count": "Requested number of regions. The native seed grid may yield fewer.",
        "levels": "Maximum number of hierarchy levels. OpenCV can use fewer on a small image.",
        "prior": "Shape smoothing from zero (disabled) to five.",
        "histogram_bins": "Bins per channel. Histogram buffers are limited to 96 MiB.",
        "double_step": "Make an additional refinement pass at every block level.",
        "iterations": "Number of final pixel refinement iterations.",
        "thick_boundaries": "Use OpenCV's thick contour-mask option.",
        "color_space": "Use Lab, HSV, or BGR for color input. Grayscale stays grayscale.",
    }

    input_help = {
        **_Superpixels.input_help,
        "image": "Grayscale, BGR or RGB uint8 image, from 6 × 6 to 2.1 million pixels.",
    }

    def segment(self, image):
        p = self.parameters
        height, width = image.shape[:2]
        # Very small images can raise a native exception during SEEDS refinement.
        if min(height, width) < 6:
            raise ValidationError("SEEDS needs an image of at least 6 × 6 pixels")
        channels = 1 if image.ndim == 2 else 3
        count, levels, bins = (int(p[key]) for key in ("target_count", "levels", "histogram_bins"))
        check_seeds_grid(width, height, count, levels, bins, channels)
        segmenter = cv2.ximgproc.createSuperpixelSEEDS(
            width, height, channels, count, levels, int(p["prior"]), bins, p["double_step"]
        )
        segmenter.iterate(image, int(p["iterations"]))
        return segmenter, "SEEDS"


def check_seeds_grid(width, height, count, levels, bins, channels):
    """Validate native SEEDS hierarchy dimensions and histogram allocation."""
    rows = int(np.sqrt(np.float32(count) * height / width))
    columns = rows * width // height
    if not (1 <= columns <= width and 1 <= rows <= height):
        raise ValidationError(
            "SEEDS target count cannot form a grid for this image", "target_count"
        )
    if 2 * columns > width or 2 * rows > height:
        raise ValidationError(
            "SEEDS needs at least two pixels per starting region along each axis. "
            "Reduce the target count.", "target_count"
        )
    scale = 1 << (levels - 1)
    while columns * scale > width or rows * scale > height:
        scale //= 2
    cell_width = ceil(np.float32(width) / columns / scale)
    cell_height = ceil(np.float32(height) / rows / scale)
    fine_width, fine_height = width // cell_width, height // cell_height
    if min(fine_width, fine_height) < scale:
        raise ValidationError(
            "SEEDS needs at least one region per axis at every level. Reduce levels or "
            "increase the target count.", "levels", "target_count"
        )
    cells = sum(
        (fine_width >> level) * (fine_height >> level) for level in range(scale.bit_length())
    )
    aligned_bins = (bins ** channels + 15) // 16 * 16
    if cells * aligned_bins * 4 > 96 * 1024**2:
        raise ValidationError(
            "SEEDS histogram buffers exceed 96 MiB. Reduce bins, levels, or target count.",
            "histogram_bins", "levels", "target_count"
        )
