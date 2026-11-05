"""Observable, eager OpenCV defaults for common preprocessing operations."""

import cv2
import numpy as np

from .core import Result
from .images import ImageArray
from .parameters import BORDERS, scalar_value
from .processing import ImageTool, encoded_result, finite, image_value, integer
from .selection import RegionalOperation

ALGORITHM_HINTS = {
    "default": cv2.ALGO_HINT_DEFAULT,
    "accurate": cv2.ALGO_HINT_ACCURATE,
    "approximate": cv2.ALGO_HINT_APPROX,
}


class Grayscale(ImageTool):
    process_inputs_readonly = True
    type_id, title, subtype = "cv.gray", "Grayscale", "Color and depth"
    defaults = {"algorithm_hint": "default"}
    parameter_choices = {"algorithm_hint": tuple(ALGORITHM_HINTS)}

    def process(self, inputs):
        image = image_value(inputs["image"], (np.uint8, np.uint16, np.float32), canonical=True)
        if image.ndim == 3:
            code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
            image = cv2.cvtColor(
                image, code, hint=ALGORITHM_HINTS[self.parameters["algorithm_hint"]]
            )
        return Result({"image": ImageArray(image, "gray")})


class GaussianBlur(RegionalOperation, ImageTool):
    process_inputs_readonly = True
    type_id, title, subtype = "cv.gaussian_blur", "Gaussian blur", "Filtering"
    defaults = {
        "kernel": 5,
        "sigma": 1.2,
        "kernel_y": 0,
        "sigma_y": 0.0,
        "border": "reflect",
        "border_value": [0.0, 0.0, 0.0, 0.0],
        "algorithm_hint": "default",
    }
    parameter_choices = {"border": tuple(BORDERS), "algorithm_hint": tuple(ALGORITHM_HINTS)}
    parameter_help = {
        "kernel": "Odd X size; 0 derives it from positive sigma X.",
        "kernel_y": "0 uses kernel X; if both are zero, derive Y size from sigma.",
        "sigma_y": "0 uses sigma X, following OpenCV.",
        "border_value": "Used only by the constant border mode, in source channel order.",
    }

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source)
        p = self.parameters
        kx = integer(p["kernel"], "Gaussian kernel X", 0)
        ky = integer(p["kernel_y"], "Gaussian kernel Y", 0) or kx
        sx, sy = (finite(p[k], k, 0) for k in ("sigma", "sigma_y"))
        if any(k and k % 2 == 0 for k in (kx, ky)):
            raise ValueError("Gaussian kernel sizes must be positive odd integers or zero")
        if (kx == 0 and sx <= 0) or (ky == 0 and (sy or sx) <= 0):
            raise ValueError("Automatic Gaussian kernel sizing requires positive sigma")
        border = BORDERS[p["border"]]
        value = scalar_value(p["border_value"])
        px = py = 0
        if border == cv2.BORDER_CONSTANT and any(value):
            # GaussianBlur has no borderValue argument. Pad beyond its support,
            # filter, then crop so a nonzero constant has explicit semantics.
            px = kx // 2 if kx else int(np.ceil(8 * sx + 8))
            py = ky // 2 if ky else int(np.ceil(8 * (sy or sx) + 8))
            image = cv2.copyMakeBorder(image, py, py, px, px, border, value=value)
        output = cv2.GaussianBlur(
            image,
            (kx, ky),
            sx,
            sigmaY=sy,
            borderType=border,
            hint=ALGORITHM_HINTS[p["algorithm_hint"]],
        )
        if px or py:
            output = output[py : py + source.shape[0], px : px + source.shape[1]].copy()
        return encoded_result(output, source)


class Canny(RegionalOperation, ImageTool):
    process_inputs_readonly = True
    type_id, title, subtype = "cv.canny", "Canny edges", "Edges"
    region_background = "zero"
    defaults = {"low": 60.0, "high": 150.0, "aperture": "3", "l2_gradient": False}
    parameter_choices = {"aperture": ("3", "5", "7")}

    def process(self, inputs):
        image = image_value(inputs["image"], (np.uint8,), canonical=True)
        if image.ndim != 2:
            raise ValueError("Canny requires uint8 grayscale; connect Grayscale first")
        p = self.parameters
        low, high = (finite(p[k], k, 0) for k in ("low", "high"))
        if low > high:
            raise ValueError("Thresholds must satisfy 0 <= low <= high")
        output = cv2.Canny(
            image, low, high, apertureSize=int(p["aperture"]), L2gradient=p["l2_gradient"]
        )
        return Result({"image": ImageArray(output, "gray")})


BASIC_PROCESSING_BLOCKS = (Grayscale, GaussianBlur, Canny)
