"""OpenCV ridge filtering for dark cracks and bright linear features."""

import cv2
import numpy as np

from .core import Port, Result
from .parameter_validation import NumericRange, ValidationError
from .parameters import BORDERS, OUTPUT_DEPTHS
from .processing import ImageTool, gray_image
from .selection import RegionalOperation

_DERIVATIVE_DEPTHS = {"float32": (cv2.CV_32F, np.float32), "float64": (cv2.CV_64F, np.float64)}


class RidgeResponse(RegionalOperation, ImageTool):
    type_id, title, subtype = "cv.ridge_response", "Ridge response", "Filtering"
    inputs = (
        Port("image"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    region_background = "zero"
    defaults = {
        "polarity": "dark",
        "kernel_size": "3",
        "dx": 1,
        "dy": 1,
        "derivative_depth": "float32",
        "output_depth": "float32",
        "scale": 1.0,
        "delta": 0.0,
        "border": "reflect_101",
    }
    parameter_choices = {
        "polarity": ("dark", "bright", "both"),
        "kernel_size": ("1", "3", "5", "7"),
        "derivative_depth": tuple(_DERIVATIVE_DEPTHS),
        "output_depth": tuple(depth for depth in OUTPUT_DEPTHS if depth != "same"),
        "border": ("constant", "replicate", "reflect_101", "reflect_edge"),
    }
    parameter_ranges = {
        "dx": NumericRange(1, 6, whole=True),
        "dy": NumericRange(1, 6, whole=True),
        "scale": NumericRange(),
        "delta": NumericRange(),
    }
    parameter_help = {
        "polarity": "Dark uses the native response. Bright inverts luminance before filtering. "
        "Both keeps the stronger response.",
        "kernel_size": "Native Sobel aperture. Size 1 uses a three-tap derivative axis.",
        "dx": "Horizontal derivative order in each Sobel pass.",
        "dy": "Vertical derivative order in each Sobel pass.",
        "derivative_depth": "Precision of the native Sobel and ridge calculations.",
        "output_depth": "Use float to retain signed values. Integer depths clip to their range.",
        "scale": "Native scale applied in each Sobel pass.",
        "delta": "Native bias added in each Sobel pass.",
        "border": "Pixel extrapolation at the input image edge.",
    }
    input_help = {
        "image": "Grayscale or color uint8, uint16 or float32. Limited to 2.1 million pixels.",
        "mask": "Zero the response outside this mask. Sobel uses the full neighborhood.",
        "region": "Zero the response outside this region.",
    }
    process_inputs_readonly = True
    parallel_safe = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        aperture = max(int(p["kernel_size"]), 3)
        for name in ("dx", "dy"):
            if int(p[name]) >= aperture:
                raise ValidationError("Derivative order must be smaller than the Sobel axis", name)

    def process(self, inputs):
        source = inputs["image"]
        if not 0 < source.shape[0] * source.shape[1] <= 2_100_000:
            raise ValidationError("Ridge filtering needs a nonempty image up to 2.1 million pixels")
        gray = gray_image(source)
        p = self.parameters
        native_depth, dtype = _DERIVATIVE_DEPTHS[p["derivative_depth"]]
        ridge = cv2.ximgproc.RidgeDetectionFilter_create(
            native_depth, int(p["dx"]), int(p["dy"]), int(p["kernel_size"]), native_depth,
            float(p["scale"]), float(p["delta"]), BORDERS[p["border"]],
        )
        polarity = p["polarity"]
        if polarity == "dark":
            response = ridge.getRidgeFilteredImage(gray)
        else:
            response = ridge.getRidgeFilteredImage(-gray.astype(dtype))
            if polarity == "both":
                np.maximum(response, ridge.getRidgeFilteredImage(gray), out=response)
        if not np.isfinite(response).all():
            raise ValidationError("Ridge response is nonfinite. Reduce scale or use float64.")
        target = OUTPUT_DEPTHS[p["output_depth"]]
        if target != native_depth:
            response = cv2.add(response, 0, dtype=target)
            if not np.isfinite(response).all():
                raise ValidationError("Ridge response exceeds the selected output depth")
        return Result({"image": response}, message=f"{polarity.title()} ridge response")
