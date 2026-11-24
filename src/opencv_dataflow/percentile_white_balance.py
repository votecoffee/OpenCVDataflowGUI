"""Per-channel percentile stretch with optional OpenCV histogram fitting."""

import cv2
import numpy as np

from .core import Port, Result
from .images import ImageArray, image_format, opencv_image
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, metadata
from .processing import ImageTool, image_value
from .selection import selection_mask

_CHANNELS = ("B", "G", "R")
_ENCODING_CHANNELS = {"bgr": 3, "rgb": 3, "bgra": 4, "rgba": 4}
_DEPTHS = (np.uint8, np.uint16, np.float32)
_RANGE_FIELDS = ("input_min", "input_max", "output_min", "output_max")


def _stretch_channels(image, selection, tail_percent, input_range, output_range):
    input_min, input_max = input_range
    output_min, output_max = output_range
    count = image.shape[0] * image.shape[1] if selection is None else np.count_nonzero(selection)
    tail = int(count * tail_percent / 100)
    high_index = count - 1 - tail
    result = np.empty_like(image)
    limits = {}
    for index, name in enumerate(_CHANNELS):
        channel = image[..., index]
        samples = channel.ravel() if selection is None else channel[selection]
        ordered = np.partition(samples, (tail, high_index))
        low, high = np.clip(ordered[[tail, high_index]].astype(float), input_min, input_max)
        limits[name] = [float(low), float(high)]
        if high > low:
            clipped = np.clip(channel.astype(float), low, high)
            mapped = output_min + (clipped - low) * (output_max - output_min) / (high - low)
        else:
            # The selected samples cannot support a stretch of this channel.
            mapped = np.clip(channel, output_min, output_max)
        if image.dtype.kind == "u":
            mapped = np.rint(mapped)
        result[..., index] = mapped
    return result, limits


def _native_balance(image, tail_percent, input_range, output_range):
    if image.dtype == np.uint16:
        raise ValidationError("Use exact percentiles for uint16 images", "method")
    if image.dtype == np.float32:
        low, high = np.float32(input_range)
        step = (float(high) - float(low)) / 4095
        if step <= np.spacing(max(abs(low), abs(high))):
            raise ValidationError(
                "The input interval is too narrow for the native float histogram",
                "input_min", "input_max",
            )
    factory = getattr(getattr(cv2, "xphoto", None), "createSimpleWB", None)
    if factory is None:
        raise ValidationError("OpenCV histogram balance requires OpenCV contrib xphoto", "method")
    balancer = factory()
    balancer.setP(tail_percent)
    balancer.setInputMin(input_range[0])
    balancer.setInputMax(input_range[1])
    balancer.setOutputMin(output_range[0])
    balancer.setOutputMax(output_range[1])
    clipped = np.clip(image, *np.asarray(input_range, dtype=image.dtype))
    output = balancer.balanceWhite(clipped)
    if output is None or not np.isfinite(output).all():
        raise ValidationError("OpenCV returned nonfinite white balance samples", "method")
    return np.clip(output, *np.asarray(output_range, dtype=image.dtype), out=output)


class PercentileWhiteBalance(ImageTool):
    type_id, title = "cv.percentile_white_balance", "Percentile white balance"
    category, block_type, subtype = "Calibration", "Calibration", "Color balance"
    description = "Fit independent B/G/R stretches after excluding bright and dark tails."
    inputs = (
        Port("image"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (Port("image", required=False), Port("report", "data"))
    defaults = {
        "method": "exact_percentiles",
        "tail_percent": 2.0,
        "range_mode": "image_depth",
        "input_min": 0.0, "input_max": 255.0,
        "output_min": 0.0, "output_max": 255.0,
    }
    parameter_choices = {
        "method": ("exact_percentiles", "opencv_histogram"),
        "range_mode": ("image_depth", "explicit"),
    }
    parameter_ranges = {
        "tail_percent": NumericRange(0, 49.99),
        **{field: NumericRange(-1e9, 1e9) for field in _RANGE_FIELDS},
    }
    parameter_help = {
        "method": "Exact ranks with optional sample selection, or OpenCV SimpleWB histogram "
        "fitting on the whole uint8 or float32 image.",
        "tail_percent": "Percent of samples trimmed independently from each end of each channel.",
        "range_mode": "Image depth uses 0–255, 0–65535 or 0–1. Choose explicit to use the limits below.",
        "input_min": "Minimum sample value used in explicit mode. Lower samples are clipped.",
        "input_max": "Maximum sample value used in explicit mode. Higher samples are clipped.",
        "output_min": "Lower output limit in explicit mode, in source sample units.",
        "output_max": "Upper output limit in explicit mode, in source sample units.",
    }
    input_help = {
        "image": "uint8, uint16 or float32 RGB/BGR. Straight alpha is preserved.",
        "mask": "Select pixels used to fit exact percentiles. The stretch applies to the whole image.",
        "region": "Optional exact-percentile fit region, intersected with the mask if connected.",
    }
    output_help = {
        "image": "Balanced image in the original encoding, depth and coordinate frame.",
        "report": "Fitting method, ranges, sample count, and exact B/G/R endpoints when available.",
    }
    preview_safe = parallel_safe = process_inputs_readonly = True
    payload_reference = "image"

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["range_mode"] == "explicit":
            for kind in ("input", "output"):
                low, high = f"{kind}_min", f"{kind}_max"
                if float(p[low]) >= float(p[high]):
                    raise ValidationError("Minimum must be below maximum", low, high)

    def ranges(self, dtype):
        p = self.parameters
        peak = float(np.iinfo(dtype).max) if dtype.kind == "u" else 1.0
        if p["range_mode"] == "image_depth":
            return (0.0, peak), (0.0, peak)
        values = tuple(float(p[field]) for field in _RANGE_FIELDS)
        if dtype.kind == "u" and any(not x.is_integer() or not 0 <= x <= peak for x in values):
            raise ValidationError(f"Integer images require whole limits in [0, {peak:g}]", *_RANGE_FIELDS)
        if dtype == np.float32 and any(lo >= hi for lo, hi in (
            np.float32(values[:2]), np.float32(values[2:]),
        )):
            raise ValidationError("Explicit range endpoints must differ in float32", *_RANGE_FIELDS)
        return values[:2], values[2:]

    def process(self, inputs):
        source = inputs["image"]
        pixels = image_value(source, _DEPTHS)
        encoding = image_format(source)
        if not pixels.size or pixels.ndim != 3 or pixels.shape[2] != _ENCODING_CHANNELS.get(encoding):
            raise ValidationError("Use a nonempty BGR or RGB image with three color channels")
        if pixels.shape[2] == 4 and metadata(source).get("alpha_representation", "straight") != "straight":
            raise ValidationError("Unassociate premultiplied alpha before white balance")
        if pixels.shape[0] * pixels.shape[1] > 16_000_000:
            raise ValidationError("Percentile white balance is limited to 16 million pixels")
        p = self.parameters
        method, tail = p["method"], float(p["tail_percent"])
        input_range, output_range = self.ranges(pixels.dtype)
        mask = selection_mask(inputs, source)
        if mask is not None and method == "opencv_histogram":
            raise ValidationError("Use exact percentiles to fit from a mask or region", "method")
        selection = None if mask is None else mask != 0
        count = pixels.shape[0] * pixels.shape[1] if selection is None else int(np.count_nonzero(selection))
        report = {
            "method": method, "tail_percent": tail, "sample_count": count,
            "input_range": list(input_range), "output_range": list(output_range),
            "fitted_bgr": None,
        }
        if not count:
            message = "The white balance selection contains no pixels"
            return Result(
                {"image": NO_VALUE, "report": report}, passed=False, message=message,
                fail_reasons=[{"code": "percentile_white_balance_empty_selection", "message": message}],
            )
        image = opencv_image(source)[..., :3]
        if method == "opencv_histogram":
            output = _native_balance(image, tail, input_range, output_range)
        else:
            output, report["fitted_bgr"] = _stretch_channels(
                image, selection, tail, input_range, output_range
            )
        if encoding in ("rgb", "rgba"):
            output = output[..., ::-1]
        if pixels.shape[2] == 4:
            rgba = pixels.copy()
            rgba[..., :3] = output
            output = rgba
        return Result(
            {"image": ImageArray(output, encoding), "report": report},
            message=f"Percentile white balance · {count:,} estimation pixels",
        )
