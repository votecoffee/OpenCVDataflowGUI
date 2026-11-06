"""Map scalar image samples to reusable OpenCV false-color views."""

import cv2
import numpy as np

from .core import Port, Result
from .images import DepthArray, ImageArray, image_format, opencv_image
from .parameter_validation import NumericRange, ValidationError
from .parameters import COLORMAPS
from .processing import ImageTool
from .selection import SELECTION_INPUTS, selection_mask

_SCALAR_DEPTHS = (np.uint8, np.uint16, np.int16, np.int32, np.float32, np.float64)


def _palette(image):
    encoding = image_format(image)
    if (
        image.dtype != np.uint8
        or image.shape[:2] not in ((1, 256), (256, 1))
        or (image.ndim == 2 and encoding != "gray")
        or (image.ndim == 3 and (image.shape[2] != 3 or encoding not in ("rgb", "bgr")))
    ):
        raise ValidationError("Palette must be a 256-pixel uint8 strip in gray, BGR or RGB")
    pixels = opencv_image(image)
    shape = (256, 1) if pixels.ndim == 2 else (256, 1, 3)
    return np.ascontiguousarray(pixels.reshape(shape))


def _quantize(samples, low, high):
    bounded = np.clip(samples, low, high)
    if not np.isfinite(high - low):
        # Opposite float64 extremes can overflow their difference.
        bounded *= 0.5
        low, high = low * 0.5, high * 0.5
    return np.rint((bounded - low) / (high - low) * 255).astype(np.uint8)


class ColorizeScalar(ImageTool):
    type_id, title, subtype = "cv.colorize_scalar", "Colorize scalar image", "Color and depth"
    process_inputs_readonly = True
    inputs = (*SELECTION_INPUTS, Port("palette", required=False, required_when_connected=True))
    outputs = (
        Port("image"),
        Port("levels"),
        Port("valid_mask", visual_type="region"),
        Port("range", "data"),
    )
    defaults = {"colormap": "turbo", "range_mode": "auto", "minimum": 0.0, "maximum": 255.0}
    parameter_choices = {"colormap": tuple(COLORMAPS), "range_mode": ("auto", "fixed")}
    parameter_ranges = {"minimum": NumericRange(), "maximum": NumericRange()}
    parameter_help = {
        "colormap": "OpenCV palette. A wired palette strip takes precedence.",
        "range_mode": "Auto uses the minimum and maximum of selected valid pixels.",
        "minimum": "Fixed mode maps this sample value and lower values to palette index 0.",
        "maximum": "Fixed mode maps this sample value and higher values to palette index 255.",
    }
    input_help = {
        "image": "Single-channel scalar image or physical depth plane.",
        "palette": "Optional 1 × 256 or 256 × 1 uint8 strip in gray, BGR or RGB order.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["range_mode"] == "fixed" and float(p["minimum"]) >= float(p["maximum"]):
            raise ValidationError("Fixed minimum must be less than maximum", "minimum", "maximum")

    def process(self, inputs):
        source = inputs["image"]
        image = np.asarray(source)
        if image.ndim != 2 or image.dtype not in _SCALAR_DEPTHS:
            raise ValidationError("Colorize needs a grayscale scalar image or physical depth")
        p = self.parameters
        palette = _palette(inputs["palette"]) if "palette" in inputs else COLORMAPS[p["colormap"]]
        mask = selection_mask(inputs, source)
        valid = np.isfinite(image)
        if isinstance(source, DepthArray):
            valid &= image > 0
        if mask is not None:
            valid &= mask != 0
        samples = image[valid].astype(np.float64, copy=False)
        count = len(samples)
        low, high = None, None
        if p["range_mode"] == "fixed":
            low, high = float(p["minimum"]), float(p["maximum"])
        elif count:
            low, high = float(samples.min()), float(samples.max())
        levels = np.zeros(image.shape, dtype=np.uint8)
        if count and low < high:
            levels[valid] = _quantize(samples, low, high)
        output = cv2.applyColorMap(levels, palette)
        output[~valid] = 0
        return Result(
            {
                "image": ImageArray(output, image_format(output)),
                "levels": levels,
                "valid_mask": valid.astype(np.uint8) * 255,
                "range": {
                    "minimum": low,
                    "maximum": high,
                    "valid_count": count,
                    "units": source.units if isinstance(source, DepthArray) else None,
                },
            },
            passed=bool(count),
            fail_reasons=[] if count else ["no_valid_color_samples"],
            message=f"Mapped {count} selected valid pixels to palette colors",
        )
