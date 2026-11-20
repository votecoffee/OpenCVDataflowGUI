"""Physical depth conversion and reversible uint16 image encoding."""

import cv2
import numpy as np

from .core import Port, Result
from .geometry3d_common import Tool3D
from .images import DepthArray
from .parameter_validation import NumericRange, ValidationError

ENCODING = "depth_encoding"
UINT16_LIMIT = 65_535
MAX_PIXELS = 16_777_216
STEP_RANGE = NumericRange(1e-9, 1000)
FLOAT_FORMATS = {"float32": np.float32, "float64": np.float64}


def _depth_values(depth):
    if depth.size > MAX_PIXELS:
        raise ValidationError(f"Depth conversion accepts at most {MAX_PIXELS:,} samples")
    return np.asarray(depth)


def _metric_units(depth, fallback):
    units = getattr(depth, "units", fallback)
    if units not in ("m", "mm"):
        raise ValidationError("Floating depth must use m or mm", "input_units")
    return units


def _encode_metadata(step, invalid):
    return {"model": "linear_uint16_v1", "meters_per_code": step, "invalid_code": invalid}


def _decode_metadata(record):
    if (
        not isinstance(record, dict)
        or record.get("model") != "linear_uint16_v1"
        or type(record.get("meters_per_code")) not in (int, float)
        or type(record.get("invalid_code")) is not int
        or not 0 <= record["invalid_code"] <= UINT16_LIMIT
    ):
        raise ValidationError("Invalid linear uint16 depth encoding metadata")
    try:
        step = STEP_RANGE.check(record["meters_per_code"], "Encoding meters per code")
    except (OverflowError, ValidationError) as exc:
        raise ValidationError("Depth encoding has an invalid meters-per-code value") from exc
    return step, record["invalid_code"]


def _mask(condition):
    return condition.astype(np.uint8) * 255


class NormalizeMetricDepth(Tool3D):
    type_id = "3d.normalize_depth"
    title, subtype = "Decode and scale metric depth", "Depth"
    description = "Convert sensor depth codes or floating depth to a physical depth image."
    payload_reference = "samples"
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = (Port("samples", "depth"),)
    outputs = (
        Port("depth", "depth"),
        Port("valid_mask", visual_type="region"),
        Port("report", "data"),
    )
    defaults = {
        "input_units": "m",
        "meters_per_code": 0.001,
        "invalid_code": UINT16_LIMIT,
        "output_units": "m",
        "output_format": "float32",
        "display_max_m": 5.0,
    }
    parameter_choices = {
        "input_units": ("m", "mm"),
        "output_units": ("m", "mm"),
        "output_format": tuple(FLOAT_FORMATS),
    }
    parameter_ranges = {
        "meters_per_code": STEP_RANGE,
        "invalid_code": NumericRange(0, UINT16_LIMIT, whole=True),
        "display_max_m": NumericRange(0.001, 100000),
    }
    parameter_help = {
        "input_units": "Units of untagged floating inputs. A typed depth supplies its own units.",
        "meters_per_code": "Physical step for a uint16 input without encoding metadata.",
        "invalid_code": "Additional invalid code for untagged uint16. Zero is always invalid.",
        "output_units": "Units of the decoded metric values.",
        "output_format": "Values outside the selected float format's range are invalid.",
        "display_max_m": "Depth preview upper bound in meters. Independent of conversion.",
    }
    input_help = {
        "samples": "Positive camera Z or uint16 sensor codes with optional scale metadata."
    }

    def process(self, inputs):
        samples, p = inputs["samples"], self.parameters
        raw = _depth_values(samples)
        target_scale = 1000.0 if p["output_units"] == "mm" else 1.0
        payload = dict(self.input_payloads["samples"])
        has_encoding = ENCODING in payload
        record = payload.pop(ENCODING, None)
        integer = raw.dtype == np.uint16
        valid = np.isfinite(raw) & (raw > 0)
        if integer:
            step, invalid = (
                _decode_metadata(record)
                if has_encoding
                else (float(p["meters_per_code"]), int(p["invalid_code"]))
            )
            record = _encode_metadata(step, invalid)
            valid &= raw != invalid
            values = cv2.rescaleDepth(raw, cv2.CV_64F, depth_factor=1.0 / (step * target_scale))
        else:
            if has_encoding:
                raise ValidationError("Encoded depth metadata requires a uint16 image")
            input_scale = 0.001 if _metric_units(samples, p["input_units"]) == "mm" else 1.0
            values = cv2.rescaleDepth(raw, cv2.CV_64F)
            with np.errstate(over="ignore", under="ignore"):
                values *= input_scale * target_scale
        dtype = FLOAT_FORMATS[p["output_format"]]
        with np.errstate(over="ignore", under="ignore"):
            values = values.astype(dtype, copy=False)
        usable = valid & np.isfinite(values) & (values > 0)
        values[~usable] = np.nan
        depth = DepthArray(
            values, p["output_units"], 0, float(p["display_max_m"]) * target_scale, dtype=dtype
        )
        input_count, count = int(np.count_nonzero(valid)), int(np.count_nonzero(usable))
        outputs = {
            "depth": depth,
            "valid_mask": _mask(usable),
            "report": {
                "units": depth.units,
                "output_format": p["output_format"],
                "input_encoding": record,
                "valid_samples": count,
                "invalid_input_samples": raw.size - input_count,
                "unrepresentable_samples": input_count - count,
            },
        }
        return Result(
            outputs,
            payloads={name: payload for name in outputs},
            passed=bool(count),
            score=count / raw.size,
            fail_reasons=[] if count else ["no_decoded_depth"],
            message=f"Decoded {count} of {raw.size} samples in {depth.units}",
        )


def _error_summary(error):
    if not error.size:
        return {"samples": 0, "max_absolute_m": None, "rms_m": None}
    peak = float(np.max(np.abs(error)))
    rms = peak * float(np.sqrt(np.mean((error / peak) ** 2))) if peak else 0.0
    return {"samples": int(error.size), "max_absolute_m": peak, "rms_m": rms}


class QuantizeMetricDepth(Tool3D):
    type_id = "3d.quantize_depth"
    title, subtype = "Quantize metric depth to uint16", "Depth"
    description = "Encode physical depth as uint16 with a reserved invalid code and error report."
    payload_reference = "depth"
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = (Port("depth", "depth"),)
    outputs = (
        Port("codes"),
        Port("valid_mask", visual_type="region"),
        Port("out_of_range_mask", visual_type="region"),
        Port("report", "data"),
    )
    defaults = {
        "input_units": "m",
        "meters_per_code": 0.001,
        "invalid_code": UINT16_LIMIT,
        "out_of_range": "invalidate",
    }
    parameter_choices = {
        "input_units": ("m", "mm"),
        "invalid_code": (0, UINT16_LIMIT),
        "out_of_range": ("invalidate", "clip"),
    }
    parameter_ranges = {"meters_per_code": STEP_RANGE}
    parameter_help = {
        "input_units": "Units of untagged floating inputs. A typed depth supplies its own units.",
        "meters_per_code": "Physical step size. Round in-range values to the nearest code.",
        "invalid_code": "Reserve code zero or 65535. Code zero never represents positive depth.",
        "out_of_range": "Invalidate or clamp positive depth outside the representable range.",
    }

    def process(self, inputs):
        depth, p = inputs["depth"], self.parameters
        raw = _depth_values(depth)
        if raw.dtype == np.uint16 or ENCODING in self.input_payloads["depth"]:
            raise ValidationError("Connect floating metric depth. Decode sensor codes first.")
        units = _metric_units(depth, p["input_units"])
        step, invalid = float(p["meters_per_code"]), int(p["invalid_code"])
        maximum = UINT16_LIMIT - (invalid == UINT16_LIMIT)
        valid = np.isfinite(raw) & (raw > 0)
        physical = np.asarray(raw[valid], np.float64)
        if units == "mm":
            physical *= 0.001

        in_range = (physical >= step) & (physical <= maximum * step)
        keep = np.ones(len(physical), bool) if p["out_of_range"] == "clip" else in_range
        # Clamp before dividing by the step so even an extreme input remains safe to round.
        bounded = np.clip(physical[keep], step, maximum * step)
        rounded = np.rint(bounded / step).astype(np.uint16)
        error = rounded.astype(np.float64) * step - physical[keep]
        codes = np.full(raw.shape, invalid, np.uint16)
        emitted, clipped = np.zeros(raw.shape, bool), np.zeros(raw.shape, bool)
        emitted[valid] = keep
        clipped[valid] = ~in_range
        codes[emitted] = rounded
        encoding = _encode_metadata(step, invalid)
        count = int(len(rounded))
        clipped_count = int(np.count_nonzero(~in_range))
        report = {
            "encoding": encoding,
            "minimum_depth_m": step,
            "maximum_depth_m": maximum * step,
            "valid_samples": count,
            "invalid_input_samples": raw.size - int(np.count_nonzero(valid)),
            "out_of_range_samples": clipped_count,
            "clamped_samples": clipped_count if p["out_of_range"] == "clip" else 0,
            "in_range_error": _error_summary(error[in_range[keep]]),
            "retained_error": _error_summary(error),
        }
        return Result(
            {
                "codes": codes,
                "valid_mask": _mask(emitted),
                "out_of_range_mask": _mask(clipped),
                "report": report,
            },
            payloads={"codes": {**self.input_payloads["depth"], ENCODING: encoding}},
            passed=bool(count),
            score=count / raw.size,
            fail_reasons=[] if count else ["no_encoded_depth"],
            message=f"Encoded {count} of {raw.size} samples · {clipped_count} outside range",
        )
