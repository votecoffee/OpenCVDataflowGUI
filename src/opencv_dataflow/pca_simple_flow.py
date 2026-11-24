"""OpenCV PCAFlow and SimpleFlow adapters for consecutive inspection images."""

import math

import cv2
import numpy as np

from .images import ImageArray, format_image, image_format
from .parameter_validation import NumericRange, ValidationError
from .processing import gray_image, image_value
from .segmentation import bgr8
from .sequence_tools import _DenseFlowSequence

_INTENSITY_DEPTHS = (np.uint8, np.uint16, np.float32, np.float64)
_MAX_WORKSPACE = 512 * 1024**2
_MAX_SIMPLE_WORK = 2_500_000_000
_FLOW_RANGES = {
    **_DenseFlowSequence.parameter_ranges,
    "max_gap_ms": NumericRange(0),
    "visual_max_motion": NumericRange(1e-6),
}
_FLOW_HELP = {
    **_DenseFlowSequence.parameter_help,
    "visual_max_motion": "Displacement in pixels mapped to full preview brightness.",
}
_FLOW_OUTPUT_HELP = {
    "flow": "Float32 displacement in pixels from the previous frame to the current frame.",
    "consistency_mask": "White where the forward and reverse estimates agree within tolerance.",
    "consistency_error": "Round-trip distance in pixels. NaN for unevaluated positions.",
}


class PCAFlowSequence(_DenseFlowSequence):
    type_id, title = "sequence.pcaflow", "PCAFlow optical flow"
    description = "Fit a smooth dense displacement field to sparse motion using OpenCV's PCAFlow."
    process_inputs_readonly = True
    parameter_ranges = _FLOW_RANGES
    parameter_help = _FLOW_HELP
    input_help = {
        "image": "Consecutive textured frames. Converted to 8-bit grayscale for the native solver.",
        "mask": "Select published displacement on the previous image grid.",
        "region": "Intersect an optional region with the output mask.",
    }
    output_help = _FLOW_OUTPUT_HELP

    def working_image(self, image):
        return gray_image(image, (np.uint8,), auto_depth=True)

    def estimate_state_bytes(self, image):
        return image.shape[0] * image.shape[1] + 4096

    def flow_calculator(self, shape):
        height, width = shape[:2]
        # The factory fits an 18 × 14 basis to a (size divided by 8) × 2 raster.
        if width < 72 or height < 56:
            raise ValidationError("PCAFlow requires at least 72 × 56 pixels (width × height)")
        if max(height, width) > 8192 or height * width * 128 > _MAX_WORKSPACE:
            raise ValidationError("PCAFlow workspace exceeds its size limit. Crop the image first.")
        create = getattr(getattr(cv2, "optflow", None), "createOptFlow_PCAFlow", None)
        if create is None:
            raise ValidationError("PCAFlow requires OpenCV contrib optflow")
        solver = create()

        def calculate(previous, current, _initial):
            for image in (previous, current):
                low, high, _, _ = cv2.minMaxLoc(image)
                if low == high:
                    raise ValidationError("PCAFlow needs trackable texture in both frames")
            return solver.calc(previous, current, None), None

        return calculate


def check_simple_flow_budget(shape, p):
    height, width = shape
    pixels = height * width
    pyramid_pixels = pixels
    for _ in range(int(p["layers"]) - 1):
        if min(height, width) <= 1:
            raise ValidationError("SimpleFlow pyramid has too many levels for this image", "layers")
        height, width = (height + 1) // 2, (width + 1) // 2
        pyramid_pixels += height * width

    def kernel_area(radius):
        return (2 * radius + 1) ** 2

    # A zero bilateral radius asks OpenCV to derive it from the spatial sigma.
    post = int(p["postprocess_radius"]) or math.ceil(1.5 * float(p["postprocess_spatial_sigma"]))
    upscale = int(p["upscale_radius"]) or math.ceil(1.5 * float(p["upscale_spatial_sigma"]))
    search = kernel_area(int(p["search_radius"])) * kernel_area(int(p["averaging_radius"]))
    work = (
        2 * pyramid_pixels * search
        + pixels * kernel_area(post)
        + 2 * (pyramid_pixels - pixels) * kernel_area(upscale)
    )
    pad = 2 * max(post, upscale)
    workspace = 96 * pixels + 20 * (shape[0] + pad) * (shape[1] + pad)
    if work > _MAX_SIMPLE_WORK or workspace > _MAX_WORKSPACE:
        raise ValidationError(
            "SimpleFlow exceeds its processing budget. Crop or reduce the search or filters.",
            "search_radius",
            "averaging_radius",
            "postprocess_radius",
            "upscale_radius",
        )


class SimpleFlowSequence(_DenseFlowSequence):
    type_id, title = "sequence.simpleflow", "SimpleFlow optical flow"
    description = "Estimate dense motion using local color matching and multiscale interpolation."
    process_inputs_readonly = True
    defaults = {
        **_DenseFlowSequence.defaults,
        "layers": 3,
        "averaging_radius": 2,
        "search_radius": 4,
        "spatial_sigma": 4.1,
        "color_sigma": 25.5,
        "postprocess_radius": 18,
        "postprocess_spatial_sigma": 55.0,
        "postprocess_color_sigma": 25.5,
        "occlusion_threshold": 0.35,
        "upscale_radius": 18,
        "upscale_spatial_sigma": 55.0,
        "upscale_color_sigma": 25.5,
        "speed_up_threshold": 10.0,
    }
    parameter_ranges = {
        **_FLOW_RANGES,
        "layers": NumericRange(1, 6, whole=True),
        "averaging_radius": NumericRange(0, 32, whole=True),
        "search_radius": NumericRange(1, 32, whole=True),
        "spatial_sigma": NumericRange(1e-3, 255),
        "color_sigma": NumericRange(1e-3, 255),
        "postprocess_radius": NumericRange(0, 64, whole=True),
        "postprocess_spatial_sigma": NumericRange(1e-3, 255),
        "postprocess_color_sigma": NumericRange(1e-3, 255),
        "occlusion_threshold": NumericRange(0, 100),
        "upscale_radius": NumericRange(0, 64, whole=True),
        "upscale_spatial_sigma": NumericRange(1e-3, 255),
        "upscale_color_sigma": NumericRange(1e-3, 255),
        "speed_up_threshold": NumericRange(0, 100),
    }
    parameter_help = {
        **_FLOW_HELP,
        "layers": "Number of levels in the image pyramid, including full resolution.",
        "averaging_radius": "Radius of the local color-matching neighborhood, in level pixels.",
        "search_radius": "Maximum candidate shift along each axis at each pyramid level.",
        "spatial_sigma": "Spatial weighting sigma for the matching neighborhood.",
        "color_sigma": "Color weighting sigma for the matching neighborhood.",
        "postprocess_radius": "Final cross-bilateral radius. Zero derives it from spatial sigma.",
        "postprocess_spatial_sigma": "Spatial sigma of the final cross-bilateral filter.",
        "postprocess_color_sigma": "Color sigma of the final cross-bilateral filter.",
        "occlusion_threshold": "Native occlusion threshold in squared level pixels.",
        "upscale_radius": "Cross-bilateral radius during upscaling. Zero derives it from sigma.",
        "upscale_spatial_sigma": "Spatial sigma of the upscaling filter.",
        "upscale_color_sigma": "Color sigma of the upscaling filter.",
        "speed_up_threshold": "Local flow variation below which finer levels reuse estimates.",
    }
    input_help = {
        "image": "Consecutive textured images, converted to 8-bit BGR for SimpleFlow.",
        "mask": "Select published displacement on the previous image grid.",
        "region": "Intersect an optional region with the output mask.",
    }
    output_help = _FLOW_OUTPUT_HELP

    def working_image(self, image):
        if image.dtype == np.uint8:
            return bgr8(image)
        if image.dtype == np.float64 and image_format(image) in ("hsv", "lab"):
            image = ImageArray(image.astype(np.float32), image_format(image))
        return format_image(image_value(image, _INTENSITY_DEPTHS, canonical=True), "bgr")

    def estimate_state_bytes(self, image):
        return image.shape[0] * image.shape[1] * 3 + 4096

    def flow_calculator(self, shape):
        p = self.parameters
        height, width = shape[:2]
        if min(height, width) < 2 or max(height, width) >= 32767:
            raise ValidationError("SimpleFlow requires 2–32,766 pixels per axis")
        check_simple_flow_budget((height, width), p)
        calc = getattr(getattr(cv2, "optflow", None), "calcOpticalFlowSF", None)
        if calc is None:
            raise ValidationError("SimpleFlow requires OpenCV contrib optflow")
        options = (
            int(p["layers"]),
            int(p["averaging_radius"]),
            int(p["search_radius"]),
            float(p["spatial_sigma"]),
            float(p["color_sigma"]),
            int(p["postprocess_radius"]),
            float(p["postprocess_spatial_sigma"]),
            float(p["postprocess_color_sigma"]),
            float(p["occlusion_threshold"]),
            int(p["upscale_radius"]),
            float(p["upscale_spatial_sigma"]),
            float(p["upscale_color_sigma"]),
            float(p["speed_up_threshold"]),
        )

        def calculate(previous, current, _initial):
            return calc(previous, current, *options), None

        return calculate
