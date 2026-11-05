"""Reference-based illumination correction in the camera's pixel grid."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
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
