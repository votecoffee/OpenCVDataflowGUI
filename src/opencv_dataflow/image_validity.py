"""Validate and repair selected pixel samples with source-aligned masks."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .selection import SELECTION_INPUTS, selection_mask

_DEPTHS = (np.uint8, np.uint16, np.int16, np.int32, np.float32, np.float64)
_SAMPLE_TESTS = {
    "nan": np.isnan,
    "positive_infinity": np.isposinf,
    "negative_infinity": np.isneginf,
}
_REPLACEMENT_MODES = {
    "all_nonfinite": tuple(_SAMPLE_TESTS),
    "nan_only": ("nan",),
    "infinity_only": ("positive_infinity", "negative_infinity"),
}


class CheckImageValidity(Block):
    type_id, title = "inspect.image_validity", "Check image samples"
    category, block_type, subtype = "Inspection", "Inspection", "Measurements"
    inputs = SELECTION_INPUTS
    outputs = (
        Port("valid_mask", visual_type="region"),
        Port("invalid_mask", visual_type="region"),
        Port("report", "data"),
        Port("invalid_count", "number"),
        Port("invalid_fraction", "number", required=False),
    )
    process_inputs_readonly = True
    defaults = {
        "check_range": False,
        "minimum": 0.0,
        "maximum": 1.0,
        "max_invalid_fraction": 0.0,
    }
    parameter_ranges = {
        "minimum": NumericRange(),
        "maximum": NumericRange(),
        "max_invalid_fraction": NumericRange(0, 1),
    }
    parameter_help = {
        "check_range": "Also test the stored sample values against the lower and upper bounds.",
        "minimum": "Inclusive lower bound, applied to every stored channel.",
        "maximum": "Exclusive upper bound, applied to every stored channel.",
        "max_invalid_fraction": "Maximum fraction of selected pixels allowed to be invalid.",
    }
    input_help = {"image": "Grayscale, color or physical depth. Tests the stored sample values."}

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["check_range"] and float(p["minimum"]) >= float(p["maximum"]):
            raise ValidationError("Minimum must be less than Maximum", "minimum", "maximum")

    def process(self, inputs):
        source = inputs["image"]
        image = np.asarray(source)
        if image.dtype not in _DEPTHS or (image.ndim == 3 and image.shape[2] not in (3, 4)):
            raise ValidationError("Use grayscale, three-channel or four-channel numeric images")
        selection = selection_mask(inputs, source)
        selected = np.ones(image.shape[:2], bool) if selection is None else selection != 0
        finite = selected
        if image.dtype.kind == "f":
            finite = selected & (cv2.finiteMask(image) != 0)
        valid = finite
        p = self.parameters
        if p["check_range"]:
            low, high = np.float64(p["minimum"]), np.float64(p["maximum"])
            with np.errstate(invalid="ignore"):
                in_range = (image >= low) & (image < high)
            valid = finite & (in_range.all(axis=2) if image.ndim == 3 else in_range)
        invalid = selected & ~valid
        count = int(np.count_nonzero(selected))
        finite_count = int(np.count_nonzero(finite))
        valid_count = int(np.count_nonzero(valid))
        invalid_count = count - valid_count
        fraction = invalid_count / count if count else None
        first = None
        if invalid_count:
            y, x = divmod(int(invalid.argmax()), image.shape[1])
            first = [x, y]

        reasons = []
        limit = float(p["max_invalid_fraction"])
        if fraction is None:
            reasons.append({"code": "no_selected_pixels", "message": "No pixels were selected"})
        elif fraction > limit:
            reasons.append(
                {
                    "code": "invalid_pixel_fraction_exceeds_limit",
                    "message": f"{fraction:.2%} invalid pixels exceeds the {limit:.2%} limit",
                }
            )
        drawings = {}
        if self.generate_graphics and first is not None:
            drawings["first invalid pixel"] = GraphicDrawing(
                (graphic_object("marker", [first], (35, 70, 245)),)
            )
        return Result(
            {
                "valid_mask": valid.astype(np.uint8) * 255,
                "invalid_mask": invalid.astype(np.uint8) * 255,
                "report": {
                    "selected_count": count,
                    "valid_count": valid_count,
                    "nonfinite_count": count - finite_count,
                    "out_of_range_count": finite_count - valid_count,
                    "first_invalid_pixel": first,
                },
                "invalid_count": invalid_count,
                "invalid_fraction": fraction if fraction is not None else NO_VALUE,
            },
            passed=not reasons,
            score=valid_count / count if count else 0.0,
            fail_reasons=reasons,
            drawings=drawings,
            message=f"{valid_count} of {count} selected pixels valid",
        )


class ReplaceNonfiniteSamples(Block):
    type_id, title = "cv.replace_nonfinite", "Replace nonfinite samples"
    category, block_type, subtype = "Image processing", "Image processing", "Color and depth"
    inputs = SELECTION_INPUTS
    outputs = (
        Port("image"),
        Port("replaced_mask", visual_type="region"),
        Port("replaced_pixel_count", "number"),
        Port("report", "data"),
    )
    process_inputs_readonly = True
    defaults = {
        "mode": "all_nonfinite",
        "nan_replacement": 0.0,
        "positive_infinity_replacement": 0.0,
        "negative_infinity_replacement": 0.0,
    }
    parameter_choices = {"mode": tuple(_REPLACEMENT_MODES)}
    parameter_ranges = {f"{kind}_replacement": NumericRange() for kind in _SAMPLE_TESTS}
    parameter_help = {
        "mode": "Choose which nonfinite sample types to replace inside the selection.",
        "nan_replacement": "Replacement for NaN in the input's stored units.",
        "positive_infinity_replacement": "Replacement for positive infinity in stored units.",
        "negative_infinity_replacement": "Replacement for negative infinity in stored units.",
    }
    input_help = {"image": "Float32 or float64 grayscale, color, or physical depth samples."}

    def process(self, inputs):
        source = inputs["image"]
        image = np.asarray(source)
        color = image.ndim == 3
        if image.dtype not in (np.float32, np.float64) or (color and image.shape[2] not in (3, 4)):
            raise ValidationError("Use float32 or float64 grayscale, color, or physical depth")

        p = self.parameters
        limit = float(np.finfo(image.dtype).max)
        selection = selection_mask(inputs, source)
        selected = None if selection is None else selection != 0
        if selected is not None and color:
            selected = selected[..., None]
        output = source.copy()
        replaced = np.zeros(image.shape[:2], dtype=bool)
        counts = dict.fromkeys(_SAMPLE_TESTS, 0)
        for kind in _REPLACEMENT_MODES[p["mode"]]:
            field = f"{kind}_replacement"
            value = float(p[field])
            if abs(value) > limit:
                raise ValidationError(f"Replacement exceeds the {image.dtype} range", field)
            affected = _SAMPLE_TESTS[kind](image)
            if selected is not None:
                affected &= selected
            count = int(np.count_nonzero(affected))
            counts[kind] = count
            if not count:
                continue
            if kind == "nan" and selected is None:
                cv2.patchNaNs(output, value)
            else:
                np.copyto(output, value, where=affected)
            replaced |= affected.any(axis=2) if color else affected

        pixels = int(np.count_nonzero(replaced))
        samples = sum(counts.values())
        return Result(
            {
                "image": output,
                "replaced_mask": replaced.astype(np.uint8) * 255,
                "replaced_pixel_count": pixels,
                "report": {"replaced_samples": samples, "by_kind": counts},
            },
            message=f"Replaced {samples} samples in {pixels} pixels",
        )
