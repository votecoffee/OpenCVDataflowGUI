"""Reference-based and estimated illumination correction in the camera pixel grid."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .parameters import BORDERS
from .processing import image_value
from .selection import SELECTION_INPUTS, selection_mask

INTENSITY_FORMATS = ("gray", "bgr", "rgb", "bgra", "rgba")


class FlatFieldCorrection(Block):
    type_id, title = "cv.flat_field", "Flat-field correction"
    category, block_type, subtype = "Image processing", "Image processing", "Color and depth"
    inputs = (*SELECTION_INPUTS, Port("flat"), Port("dark", required=False))
    outputs = (
        Port("image"),
        Port("invalid_mask", visual_type="region"),
        Port("corrected_pixels", "number"),
    )
    defaults = {"target_mode": "reference_mean", "target_level": 0.5, "min_response": 0.01}
    parameter_choices = {"target_mode": ("reference_mean", "fixed")}
    parameter_ranges = {"target_level": NumericRange(0, 1), "min_response": NumericRange(0, 1)}
    parameter_help = {
        "target_level": "Fixed target as a fraction of integer full scale or float sample units.",
        "min_response": "Minimum flat minus dark, in full-scale fractions or float sample units.",
    }
    input_help = {
        "flat": "Uniform bright reference in the same camera pixel grid, encoding, and depth.",
        "dark": "Optional dark reference captured with the same camera settings.",
    }

    def process(self, inputs):
        source = inputs["image"]
        encoding = image_format(source)
        samples = image_value(source, (np.uint8, np.uint16, np.float32))
        if encoding not in INTENSITY_FORMATS:
            raise ValidationError(
                "Use grayscale, RGB, or BGR intensity data for flat-field correction"
            )
        has_alpha = encoding in ("rgba", "bgra")
        if has_alpha:
            samples = samples[..., :3]
        samples = samples.astype(np.float64)

        references = {}
        for name in ("flat", "dark"):
            if name not in inputs:
                continue
            reference = inputs[name]
            if (
                reference.shape != source.shape
                or reference.dtype != source.dtype
                or image_format(reference) != encoding
            ):
                raise ValidationError(
                    f"{name.capitalize()} must match image dimensions, encoding, and depth"
                )
            pixels = image_value(reference, (np.uint8, np.uint16, np.float32))
            references[name] = (pixels[..., :3] if has_alpha else pixels).astype(np.float64)

        dark = references.get("dark", 0)
        response = references["flat"] - dark
        integer = np.issubdtype(source.dtype, np.integer)
        scale = np.iinfo(source.dtype).max if integer else 1.0
        minimum = float(self.parameters["min_response"]) * scale
        valid_channels = (response > 0) & (response >= minimum)
        valid_reference = valid_channels.all(axis=2) if response.ndim == 3 else valid_channels

        selection = selection_mask(inputs, source)
        selected = selection != 0 if selection is not None else np.ones(source.shape[:2], bool)
        valid = selected & valid_reference
        invalid = selected & ~valid_reference
        count = int(np.count_nonzero(valid))
        output = np.asarray(source).copy()
        if count:
            if self.parameters["target_mode"] == "reference_mean":
                target = response[valid_reference].mean(axis=0)
            else:
                target = float(self.parameters["target_level"]) * scale
            corrected = cv2.divide(samples - dark, np.where(valid_channels, response, 1.0))
            corrected *= target
            limit = scale if integer else np.finfo(source.dtype).max
            np.clip(corrected, 0, limit, out=corrected)
            if integer:
                np.rint(corrected, out=corrected)
            destination = output[..., :3] if has_alpha else output
            destination[valid] = corrected[valid]

        invalid_count = int(np.count_nonzero(invalid))
        passed = count > 0 or invalid_count == 0
        return Result(
            {
                "image": ImageArray(output, encoding),
                "invalid_mask": invalid.astype(np.uint8) * 255,
                "corrected_pixels": count,
            },
            passed=passed,
            fail_reasons=[]
            if passed
            else [
                {
                    "code": "flat_field_no_valid_pixels",
                    "message": "No selected pixels have a usable flat-field response",
                }
            ],
            message=f"Corrected {count} pixels · {invalid_count} invalid reference pixels",
        )


class AdaptiveBackgroundCorrection(Block):
    type_id, title = "cv.adaptive_background", "Adaptive background correction"
    category, block_type, subtype = "Image processing", "Image processing", "Color and depth"
    inputs = SELECTION_INPUTS
    outputs = (Port("image"), Port("background"), Port("invalid_mask", visual_type="region"))
    defaults = {
        "estimator": "closing",
        "radius": 15,
        "shape": "ellipse",
        "sigma": 0.0,
        "border": "reflect_101",
        "operation": "subtract",
        "target_level": 0.5,
        "min_background": 0.001,
        "output_depth": "same",
    }
    parameter_choices = {
        "estimator": ("gaussian", "opening", "closing"),
        "shape": ("ellipse", "rect"),
        "border": ("reflect_101", "replicate"),
        "operation": ("subtract", "divide"),
        "output_depth": ("same", "float32", "float64"),
    }
    parameter_ranges = {
        "radius": NumericRange(1, 63, whole=True),
        "sigma": NumericRange(0, 100),
        "target_level": NumericRange(0),
        "min_background": NumericRange(0),
    }
    parameter_help = {
        "estimator": "Opening ignores small bright details. Closing fills small dark details.",
        "radius": "Half-width of the background window in pixels.",
        "shape": "Window shape for opening or closing.",
        "sigma": "Gaussian sigma in pixels. Zero derives it from the radius.",
        "operation": "Subtract an additive background or divide out multiplicative shading.",
        "target_level": "Baseline after correction, as integer full-scale fraction or float units.",
        "min_background": "Division floor, as integer full-scale fraction or float units.",
        "output_depth": "Floating output retains signed differences in raw sample units.",
    }

    def estimate_background(self, signal):
        p = self.parameters
        size = 2 * int(p["radius"]) + 1
        border = BORDERS[p["border"]]
        if p["estimator"] == "gaussian":
            return cv2.GaussianBlur(signal, (size, size), float(p["sigma"]), borderType=border)
        shape = cv2.MORPH_ELLIPSE if p["shape"] == "ellipse" else cv2.MORPH_RECT
        kernel = cv2.getStructuringElement(shape, (size, size))
        operation = cv2.MORPH_OPEN if p["estimator"] == "opening" else cv2.MORPH_CLOSE
        return cv2.morphologyEx(signal, operation, kernel, borderType=border)

    def process(self, inputs):
        p = self.parameters
        source = inputs["image"]
        image = image_value(source, (np.uint8, np.uint16, np.float32, np.float64))
        encoding = image_format(source)
        if not image.size or encoding not in INTENSITY_FORMATS:
            raise ValidationError(
                "Background correction requires nonempty grayscale or RGB/BGR data"
            )
        alpha = encoding in ("rgba", "bgra")
        signal = (image[..., :3] if alpha else image).astype(np.float64)
        background = self.estimate_background(signal)
        if not np.isfinite(background).all():
            raise ValidationError("Background estimate overflowed. Reduce the sample range.")
        scale = np.iinfo(image.dtype).max if image.dtype.kind == "u" else 1.0
        level = float(p["target_level"])
        if image.dtype.kind == "u" and level > 1:
            raise ValidationError(
                "Integer images require a target fraction from 0 to 1", "target_level"
            )
        target = level * scale
        selection = selection_mask(inputs, source)
        selected = np.ones(image.shape[:2], bool) if selection is None else selection != 0
        valid = selected
        if p["operation"] == "divide":
            minimum = float(p["min_background"]) * scale
            usable = (background > 0) & (background >= minimum)
            valid = selected & (usable.all(axis=2) if usable.ndim == 3 else usable)
            corrected = signal[valid] / background[valid] * target
        else:
            corrected = signal[valid] - background[valid] + target
        if not np.isfinite(corrected).all():
            raise ValidationError("Background correction overflowed. Reduce the sample range.")

        depth = image.dtype if p["output_depth"] == "same" else np.dtype(p["output_depth"])
        if depth.kind == "u":
            corrected = np.rint(corrected).clip(0, np.iinfo(depth).max)
        elif depth != image.dtype and np.max(np.abs(image)) > np.finfo(depth).max:
            raise ValidationError("Input samples exceed the requested output depth")
        if depth.kind == "f" and np.any(np.abs(corrected) > np.finfo(depth).max):
            raise ValidationError("Corrected samples exceed the requested output depth")
        output = image.astype(depth, copy=True)
        destination = output[..., :3] if alpha else output
        destination[valid] = corrected
        invalid = selected & ~valid
        count = int(np.count_nonzero(valid))
        invalid_count = int(np.count_nonzero(invalid))
        passed = count > 0 or invalid_count == 0
        background_format = encoding.removesuffix("a") if alpha else encoding
        return Result(
            {
                "image": ImageArray(output, encoding),
                "background": ImageArray(background, background_format),
                "invalid_mask": invalid.astype(np.uint8) * 255,
            },
            passed=passed,
            fail_reasons=[]
            if passed
            else [
                {
                    "code": "adaptive_background_no_valid_pixels",
                    "message": "No selected pixels have a usable background for division",
                }
            ],
            message=f"Corrected {count} pixels, {invalid_count} invalid",
        )
