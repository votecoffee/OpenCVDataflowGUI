"""OpenCV local thresholding based on neighborhood mean and variation."""

import cv2
import numpy as np

from .core import Result
from .parameter_validation import NumericRange, ValidationError
from .parameters import THRESHOLD_MODES
from .processing import ImageTool, gray_image
from .selection import RegionalOperation

_METHODS = ("niblack", "sauvola", "wolf", "nick")


class LocalThreshold(RegionalOperation, ImageTool):
    type_id, title = "cv.local_threshold", "Local threshold (Niblack family)"
    subtype = "Segmentation"
    region_background = "zero"
    process_inputs_readonly = True
    parallel_safe = True
    defaults = {
        "method": "sauvola",
        "mode": "binary",
        "block_size": 31,
        "k": 0.2,
        "sauvola_range": 128.0,
        "max_value": 255,
    }
    parameter_choices = {"method": _METHODS, "mode": tuple(THRESHOLD_MODES)}
    parameter_ranges = {
        "block_size": NumericRange(3, 255, whole=True, odd=True),
        "k": NumericRange(-10, 10),
        "sauvola_range": NumericRange(1, 255),
        "max_value": NumericRange(0, 255, whole=True),
    }
    parameter_help = {
        "method": "Local threshold formula. Wolf also uses image-wide intensity and variation.",
        "mode": "Binary and inverse make masks. Trunc and to-zero modes retain grayscale values.",
        "block_size": "Odd width and height of the local neighborhood in pixels.",
        "k": "Native OpenCV formula coefficient. Niblack adds k times local standard deviation.",
        "sauvola_range": "Standard-deviation reference used only for Sauvola.",
        "max_value": "Foreground value in binary modes.",
    }
    input_help = {
        "image": "Grayscale or color, converted to uint8 luminance. Up to 16,777,216 pixels.",
        "mask": "Restrict where output is written. Neighborhoods still use the full source.",
        "region": "Restrict where output is written. Intersected with a wired mask.",
    }

    def process(self, inputs):
        source = inputs["image"]
        if not 0 < source.shape[0] * source.shape[1] <= 16_777_216:
            raise ValidationError("Local threshold needs a nonempty image up to 16,777,216 pixels")
        image = gray_image(source, (np.uint8,), auto_depth=True)
        p = self.parameters
        method, mode = p["method"], THRESHOLD_MODES[p["mode"]]
        maximum = int(p["max_value"])
        if method == "wolf":
            low, high, _, _ = cv2.minMaxLoc(image)
            if low == high:
                # The native formula divides by the maximum local deviation. In a
                # flat frame, use the uniform intensity as the threshold.
                _, output = cv2.threshold(image, low, maximum, mode)
                return Result({"image": output}, message="Wolf · uniform image")
        output = cv2.ximgproc.niBlackThreshold(
            image, maximum, mode, int(p["block_size"]), float(p["k"]),
            binarizationMethod=getattr(cv2.ximgproc, f"BINARIZATION_{method.upper()}"),
            r=float(p["sauvola_range"]),
        )
        return Result({"image": output}, message=f"{method.title()} · {p['block_size']} px window")
