"""Observable, eager OpenCV defaults for common preprocessing operations."""

import cv2
import numpy as np

from .core import Port, Result
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .parameters import BORDERS, scalar_value
from .payloads import metadata, require_same_frame
from .processing import ImageTool, encoded_result, gray_image, image_value
from .selection import SELECTION_INPUTS, RegionalOperation

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
    parameter_ranges = {
        "kernel": NumericRange(1, 4095, whole=True, odd=True, zero_allowed=True),
        "kernel_y": NumericRange(1, 4095, whole=True, odd=True, zero_allowed=True),
        "sigma": NumericRange(0),
        "sigma_y": NumericRange(0),
    }
    parameter_help = {
        "kernel": "Odd X size; 0 derives it from positive sigma X.",
        "kernel_y": "0 uses kernel X; if both are zero, derive Y size from sigma.",
        "sigma_y": "0 uses sigma X, following OpenCV.",
        "border_value": "Used only by the constant border mode, in source channel order.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["kernel"] == 0 and float(p["sigma"]) <= 0:
            raise ValidationError(
                "Automatic Gaussian sizing requires positive sigma X", "kernel", "sigma"
            )

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source)
        p = self.parameters
        kx, ky = int(p["kernel"]), int(p["kernel_y"]) or int(p["kernel"])
        sx, sy = float(p["sigma"]), float(p["sigma_y"])
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


class _CannyThresholds(RegionalOperation, ImageTool):
    process_inputs_readonly = True
    region_background = "zero"
    defaults = {"low": 60.0, "high": 150.0, "l2_gradient": False}
    parameter_ranges = {"low": NumericRange(0), "high": NumericRange(0)}

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if float(p["low"]) > float(p["high"]):
            raise ValidationError("Thresholds must satisfy 0 <= low <= high", "low", "high")


class Canny(_CannyThresholds):
    type_id, title, subtype = "cv.canny", "Canny edges", "Edges"
    defaults = {**_CannyThresholds.defaults, "aperture": "3"}
    parameter_choices = {"aperture": ("3", "5", "7")}

    def process(self, inputs):
        image = gray_image(inputs["image"], (np.uint8,), auto_depth=True)
        p = self.parameters
        low, high = float(p["low"]), float(p["high"])
        output = cv2.Canny(
            image, low, high, apertureSize=int(p["aperture"]), L2gradient=p["l2_gradient"]
        )
        return Result({"image": ImageArray(output, "gray")})


class SpatialGradient(RegionalOperation, ImageTool):
    process_inputs_readonly = True
    type_id, title, subtype = "cv.spatial_gradient", "Paired Sobel derivatives", "Edges"
    outputs = (Port("dx"), Port("dy"))
    region_outputs = ("dx", "dy")
    region_background = "zero"
    defaults = {"border": "reflect_101"}
    parameter_choices = {"border": ("reflect_101", "replicate")}
    input_help = {"image": "8-bit image. Color inputs are converted to grayscale."}

    def process(self, inputs):
        image = gray_image(inputs["image"], (np.uint8,))
        dx, dy = cv2.spatialGradient(image, ksize=3, borderType=BORDERS[self.parameters["border"]])
        return Result({"dx": ImageArray(dx, "gray"), "dy": ImageArray(dy, "gray")})


class CannyDerivatives(_CannyThresholds):
    type_id, title, subtype = "cv.canny_derivatives", "Canny from derivatives", "Edges"
    inputs = (Port("dx"), Port("dy"), *SELECTION_INPUTS[1:])
    region_source = "dx"
    payload_reference = "dx"
    input_help = {
        "dx": "Signed int16 X derivatives in a matching one- or three-channel field.",
        "dy": "Signed int16 Y derivatives in the same grid and channel order as Dx.",
    }
    parameter_help = {
        "low": "Lower hysteresis threshold in derivative units.",
        "high": "Upper hysteresis threshold in derivative units.",
        "l2_gradient": "Use Euclidean magnitude instead of the sum of absolute components.",
    }

    def process(self, inputs):
        first, second = inputs["dx"], inputs["dy"]
        dx, dy = (image_value(value, (np.int16,)) for value in (first, second))
        if dx.shape != dy.shape or (dx.ndim == 3 and dx.shape[2] != 3):
            raise ValidationError("Derivatives must have matching one- or three-channel dimensions")
        if image_format(first) != image_format(second):
            raise ValidationError("Derivative fields must have matching channel order")
        require_same_frame(
            (first, second), message="Derivatives must share the same coordinate frame"
        )
        p = self.parameters
        edges = cv2.Canny(dx, dy, float(p["low"]), float(p["high"]), L2gradient=p["l2_gradient"])
        return Result({"image": ImageArray(edges, "gray")})


BAYER_ALGORITHMS = {"bilinear": "", "edge_aware": "_EA", "vng": "_VNG"}


class DemosaicBayer(ImageTool):
    process_inputs_readonly = True
    type_id, title, subtype = "cv.demosaic_bayer", "Demosaic Bayer image", "Color and depth"
    defaults = {
        "pattern": "RGGB",
        "algorithm": "bilinear",
        "color_format": "bgr",
        "column_offset": "even",
        "row_offset": "even",
        "use_sensor_metadata": False,
    }
    parameter_choices = {
        "pattern": ("RGGB", "BGGR", "GRBG", "GBRG"),
        "algorithm": tuple(BAYER_ALGORITHMS),
        "color_format": ("bgr", "rgb", "bgra", "rgba", "gray"),
        "column_offset": ("even", "odd"),
        "row_offset": ("even", "odd"),
    }
    parameter_help = {
        "pattern": "Sensor's 2 by 2 tile in row order, starting at its original top-left pixel.",
        "use_sensor_metadata": (
            "Use sensor phase metadata, following integer crops/flips/rotations."
            " Resampled mosaics are rejected."
        ),
        "algorithm": "VNG requires uint8 and at least eight pixels on both axes.",
        "color_format": "Grayscale and opaque alpha outputs use bilinear interpolation.",
        "column_offset": "Parity of the first input column relative to the sensor origin.",
        "row_offset": "Parity of the first input row relative to the sensor origin.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["algorithm"] != "bilinear" and p["color_format"] not in ("bgr", "rgb"):
            raise ValidationError("Grayscale and alpha formats require bilinear", "color_format")

    def process(self, inputs):
        image = image_value(inputs["image"], (np.uint8, np.uint16))
        p = self.parameters
        if image.ndim != 2:
            raise ValidationError("Demosaicing requires a single-channel Bayer mosaic")
        if min(image.shape) < 3:
            raise ValidationError("Demosaicing requires at least three rows and columns")
        if p["algorithm"] == "vng" and (image.dtype != np.uint8 or min(image.shape) < 8):
            raise ValidationError("VNG requires uint8 and at least eight rows and columns")

        tile = p["pattern"]
        ox = int(p["column_offset"] == "odd")
        oy = int(p["row_offset"] == "odd")
        if p["use_sensor_metadata"]:
            from .raw_sensor import bayer_pattern_from_sensor

            tile = bayer_pattern_from_sensor(self.input_payloads["image"])
            ox = oy = 0
        pattern = "".join(
            tile[2 * ((y + oy) % 2) + (x + ox) % 2] for y in range(2) for x in range(2)
        )
        encoding = p["color_format"]
        suffix = BAYER_ALGORITHMS[p["algorithm"]]
        code = getattr(cv2, f"COLOR_Bayer{pattern}2{encoding.upper()}{suffix}")
        output = ImageArray(cv2.demosaicing(image, code), encoding)
        result = Result({"image": output})
        raw = self.input_payloads["image"].get("raw_sensor")
        if isinstance(raw, dict) and raw.get("model") == "raw_sensor_v1":
            payload = metadata(inputs["image"])
            raw = payload["raw_sensor"]
            raw["mosaic"] = False
            raw["demosaicing"] = {
                "pattern": pattern,
                "algorithm": p["algorithm"],
                "color_format": encoding,
            }
            result.payloads["image"] = payload
            maximum = raw.get("max_code")
            if isinstance(maximum, int):
                maximum = min(maximum, int(np.iinfo(image.dtype).max))
            if (
                self.capture_graphics
                and isinstance(maximum, int)
                and 0 < maximum <= 65535
                and image.max() <= maximum
            ):
                result.graphics["sensor color preview"] = ImageArray(
                    cv2.convertScaleAbs(output, alpha=255 / maximum), encoding
                )
        return result


BASIC_PROCESSING_BLOCKS = (
    Grayscale,
    GaussianBlur,
    Canny,
    SpatialGradient,
    CannyDerivatives,
    DemosaicBayer,
)
