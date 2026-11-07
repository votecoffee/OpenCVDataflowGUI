"""Reference-image metrics and spatial similarity maps for registered views."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .parameter_validation import ValidationError
from .payloads import NO_VALUE
from .processing import finite, matching_images
from .selection import selection_mask

ABSOLUTE_ERROR_METRICS = ("mae", "rmse", "max_absolute_error")


def reduce_absolute_error(samples, metric):
    maximum = float(samples.max())
    if not maximum or metric == "max_absolute_error":
        return maximum
    # Scale extreme values to avoid overflow or underflow in native accumulations.
    scale = maximum if maximum < 1e-140 or maximum > 1e140 else 1.0
    values = (samples / scale if scale != 1.0 else samples).reshape(-1, 1)
    if metric == "mae":
        mean = cv2.norm(values, cv2.NORM_L1) / values.size
    else:
        mean = cv2.norm(values, cv2.NORM_L2) / np.sqrt(values.size)
    return min(float(mean), maximum / scale) * scale


class CompareImageQuality(Block):
    type_id, title = "inspect.image_quality", "Compare image quality"
    category, block_type, subtype = "Inspection", "Inspection", "Measurements"
    inputs = (
        Port("image_a"),
        Port("image_b"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False),
    )
    outputs = (
        Port("value", "number"),
        Port("measurements", "data"),
        Port("map", required=False),
    )
    defaults = {
        "metric": "ssim",
        "mask_alignment": "same_frame",
        "check_threshold": False,
        "threshold": 0.95,
        "peak_value": 255.0,
    }
    parameter_choices = {
        "metric": ("ssim", "psnr", "mse", *ABSOLUTE_ERROR_METRICS),
        "mask_alignment": ("same_frame", "pixel_grid"),
    }
    parameter_help = {
        "mask_alignment": "Pixel grid applies an equal-size mask by index across frame IDs.",
        "threshold": "Minimum for SSIM/PSNR, maximum for error metrics.",
        "peak_value": "Maximum sample value for PSNR. Default is 255 for uint8 images.",
    }

    def process(self, inputs):
        a, b = matching_images(inputs)
        selected = selection_mask(
            inputs, a, match_mask_frame=self.parameters["mask_alignment"] == "same_frame"
        )
        selected = selected != 0 if selected is not None else None
        if selected is not None and not selected.any():
            raise ValueError("Quality comparison requires at least one selected pixel")
        metric = self.parameters["metric"]
        quality_map = NO_VALUE
        if metric == "ssim":
            if a.dtype != np.uint8:
                raise ValueError("OpenCV SSIM uses an 8-bit scale. Convert both inputs to uint8.")
            _, quality_map = cv2.quality.QualitySSIM_compute(a, b)
            quality_map = quality_map.astype(np.float32)
            samples = quality_map if selected is None else quality_map[selected]
            value = float(np.mean(samples, dtype=np.float64))
        elif metric == "mse":
            _, quality_map = cv2.quality.QualityMSE_compute(a, b)
            samples = quality_map if selected is None else quality_map[selected]
            value = float(np.mean(samples, dtype=np.float64))
        elif metric == "psnr":
            peak = finite(self.parameters["peak_value"], "Peak value", np.finfo(float).eps)
            if selected is not None:
                a, b = np.ascontiguousarray(a[selected]), np.ascontiguousarray(b[selected])
            value = float(cv2.PSNR(a, b, peak))
        else:
            quality_map = cv2.absdiff(a.astype(np.float64), b.astype(np.float64))
            if not np.isfinite(quality_map).all():
                raise ValidationError("Absolute image error exceeds the float64 range")
            samples = quality_map if selected is None else quality_map[selected]
            value = reduce_absolute_error(samples, metric)

        check = self.parameters["check_threshold"]
        threshold = finite(self.parameters["threshold"], "Threshold") if check else None
        passed = not check or (
            value >= threshold if metric in ("ssim", "psnr") else value <= threshold
        )
        units = {"ssim": "unitless", "mse": "sampleÂ²", "psnr": "dB"}.get(metric, "sample")
        graphics = {}
        if self.generate_graphics and metric in ABSOLUTE_ERROR_METRICS:
            # Stretch only the preview. Keep the full-resolution error map in sample units.
            error = quality_map if quality_map.ndim == 2 else np.max(quality_map, axis=2)
            peak_error = float(error.max())
            graphics["absolute error"] = (
                np.rint(error / peak_error * 255).astype(np.uint8)
                if peak_error
                else np.zeros(error.shape, np.uint8)
            )
        return Result(
            {
                "value": value,
                "measurements": {"metric": metric, "value": value, "units": units},
                "map": quality_map,
            },
            graphics=graphics,
            passed=passed,
            message=f"{metric.upper()} {value:.6g} {units}",
            fail_reasons=[] if passed else ["image_quality_out_of_tolerance"],
        )
