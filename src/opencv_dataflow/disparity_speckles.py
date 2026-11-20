"""Filter connected disparity islands with explicit fixed-point and invalid-value handling."""

import cv2
import numpy as np

from .core import Port, Result
from .geometry3d_common import Tool3D
from .images import DepthArray, ImageArray
from .parameter_validation import NumericRange, ValidationError
from .selection import selection_mask

_MAX_SAMPLES = 16_777_216
_CODE_MIN = -32768
_CODE_SPAN = 65535


def _native_work_image(source, valid, floating, scale, difference):
    samples = source[valid].astype(np.float64)
    if floating:
        with np.errstate(over="ignore", invalid="ignore"):
            samples = np.rint(samples * scale)
        if not np.isfinite(samples).all() or np.max(np.abs(samples)) >= 2**53:
            raise ValidationError("Disparity values exceed the selected float quantization")
    offsets = samples - samples.min()
    span = float(offsets.max())
    if span <= difference:
        # Any adjacent valid pair qualifies. A binary work image avoids a large int16 threshold.
        return valid.astype(np.int16), 0, 0
    if span > _CODE_SPAN:
        raise ValidationError("Disparity span exceeds int16 at the selected quantization")
    if difference > 32767:
        raise ValidationError(
            "Neighbor difference exceeds the native int16 threshold", "max_difference_px"
        )
    if span < _CODE_SPAN:
        sentinel = _CODE_MIN
        codes = offsets.astype(np.int32) + (_CODE_MIN + 1)
    else:
        codes = offsets.astype(np.int32) + _CODE_MIN
        missing = np.flatnonzero(np.bincount(codes - _CODE_MIN, minlength=_CODE_SPAN + 1) == 0)
        if not len(missing):
            raise ValidationError("Native disparity range has no unused code for invalid pixels")
        sentinel = int(missing[0]) + _CODE_MIN
    work = np.full(source.shape, sentinel, dtype=np.int16)
    work[valid] = codes
    return work, sentinel, difference


class FilterDisparitySpeckles(Tool3D):
    type_id, title, subtype = (
        "3d.filter_disparity_speckles",
        "Remove disparity speckles",
        "Stereo",
    )
    inputs = (Port("disparity"), Port("valid_mask", required=False, required_when_connected=True))
    outputs = (Port("disparity"), Port("valid_mask"), Port("removed_mask"), Port("report", "data"))
    defaults = {
        "max_speckle_size": 50,
        "max_difference_px": 1.0,
        "fixed_point_scale": 16.0,
        "float_samples_per_pixel": 16.0,
        "invalid_value_enabled": False,
        "invalid_value": 0.0,
    }
    parameter_ranges = {
        "max_speckle_size": NumericRange(0, _MAX_SAMPLES, whole=True),
        "max_difference_px": NumericRange(0),
        "fixed_point_scale": NumericRange(1e-6, 4096),
        "float_samples_per_pixel": NumericRange(1, 4096),
        "invalid_value": NumericRange(),
    }
    parameter_help = {
        "max_speckle_size": "Remove connected islands at or below this pixel count. Zero disables.",
        "max_difference_px": "Maximum disparity difference between adjacent pixels in one island.",
        "fixed_point_scale": "Integer input samples per disparity pixel. OpenCV stereo uses 16.",
        "float_samples_per_pixel": "Temporary int16 resolution for float disparity inputs.",
        "invalid_value_enabled": "Also mask out a specific source sample value.",
        "invalid_value": "Invalid code in the source's original sample units, before any scaling.",
    }
    input_help = {
        "disparity": "Float pixel disparity, or signed/unsigned 16-bit fixed-point samples.",
        "valid_mask": "Optional nonzero valid pixels on this disparity grid.",
    }
    payload_reference = "disparity"
    parallel_safe = process_inputs_readonly = preview_safe = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if type(p["invalid_value_enabled"]) is not bool:
            raise ValidationError(
                "Enable invalid value with a boolean setting", "invalid_value_enabled"
            )

    def process(self, inputs):
        original = inputs["disparity"]
        source = np.asarray(original)
        if (
            isinstance(original, DepthArray)
            or source.ndim != 2
            or source.dtype not in (np.int16, np.uint16, np.float32, np.float64)
        ):
            raise ValidationError("Use monochrome float, int16 or uint16 disparity")
        if source.size > _MAX_SAMPLES or max(source.shape) > 32768:
            raise ValidationError(
                "Speckle filtering supports up to 16,777,216 pixels and 32,768 per axis"
            )
        p = self.parameters
        floating = source.dtype.kind == "f"
        scale = float(p["float_samples_per_pixel" if floating else "fixed_point_scale"])
        valid = np.isfinite(source)
        if p["invalid_value_enabled"]:
            invalid = float(p["invalid_value"])
            if floating and abs(invalid) > float(np.finfo(source.dtype).max):
                raise ValidationError(
                    "Invalid code must be representable by the float input", "invalid_value"
                )
            if not floating:
                limits = np.iinfo(source.dtype)
                if not invalid.is_integer() or not limits.min <= invalid <= limits.max:
                    raise ValidationError(
                        "Invalid code must be representable by the integer input", "invalid_value"
                    )
            valid &= source != invalid
        mask = selection_mask({"mask": inputs.get("valid_mask")}, original)
        if mask is not None:
            valid &= mask != 0
        input_count = int(np.count_nonzero(valid))
        max_size = int(p["max_speckle_size"])
        max_difference = float(p["max_difference_px"])
        scaled = min(max_difference * scale, _CODE_SPAN)
        difference = int(np.floor(np.nextafter(scaled, np.inf)))
        kept = valid
        if max_size and input_count:
            encoded, sentinel, threshold = _native_work_image(
                source, valid, floating, scale, difference
            )
            cv2.filterSpeckles(encoded, sentinel, max_size, threshold)
            kept = encoded != sentinel
        removed = valid & ~kept
        count = int(np.count_nonzero(kept))
        output_type = source.dtype if floating else np.float32
        disparity = np.full(source.shape, np.nan, dtype=output_type)
        disparity[kept] = source[kept] if floating else source[kept].astype(np.float64) / scale
        report = {
            "model": "disparity_speckles_v1",
            "source_dtype": source.dtype.name,
            "input_valid": input_count,
            "output_valid": count,
            "removed": input_count - count,
            "max_speckle_size": max_size,
            "samples_per_pixel": scale,
            "max_difference_px": max_difference,
            "effective_difference_px": difference / scale,
            "invalid_value": float(p["invalid_value"]) if p["invalid_value_enabled"] else None,
        }
        return Result(
            {
                "disparity": ImageArray(disparity, "gray"),
                "valid_mask": ImageArray(np.uint8(kept) * 255, "gray"),
                "removed_mask": ImageArray(np.uint8(removed) * 255, "gray"),
                "report": report,
            },
            passed=bool(count),
            score=count / input_count if input_count else 0.0,
            fail_reasons=[] if count else ["disparity_speckles_no_valid_pixels"],
            message=f"{input_count - count} disparity pixels removed, {count} valid",
        )
