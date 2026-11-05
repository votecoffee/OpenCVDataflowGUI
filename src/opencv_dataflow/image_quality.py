"""Reference-image metrics and spatial similarity maps for registered views."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .payloads import NO_VALUE
from .processing import finite, matching_images
from .selection import SELECTION_INPUTS, selection_mask


class CompareImageQuality(Block):
    type_id, title = "inspect.image_quality", "Compare image quality"
    category, block_type, subtype = "Inspection", "Inspection", "Measurements"
    inputs = (Port("image_a"), Port("image_b"), *SELECTION_INPUTS[1:])
    outputs = (
        Port("value", "number"),
        Port("measurements", "data"),
        Port("map", required=False),
    )
    defaults = {
        "metric": "ssim",
        "check_threshold": False,
        "threshold": 0.95,
        "peak_value": 255.0,
    }
    parameter_choices = {"metric": ("ssim", "psnr", "mse")}
    parameter_help = {
        "threshold": "Minimum for SSIM/PSNR, maximum for MSE.",
        "peak_value": "Maximum sample value for PSNR. Default is 255 for uint8 images.",
    }

    def process(self, inputs):
        a, b = matching_images(inputs)
        selected = selection_mask(inputs, a)
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
        else:
            peak = finite(self.parameters["peak_value"], "Peak value", np.finfo(float).eps)
            if selected is not None:
                a, b = np.ascontiguousarray(a[selected]), np.ascontiguousarray(b[selected])
            value = float(cv2.PSNR(a, b, peak))

        check = self.parameters["check_threshold"]
        threshold = finite(self.parameters["threshold"], "Threshold") if check else None
        passed = not check or (value <= threshold if metric == "mse" else value >= threshold)
        units = {"ssim": "unitless", "mse": "sample²", "psnr": "dB"}[metric]
        return Result(
            {
                "value": value,
                "measurements": {"metric": metric, "value": value, "units": units},
                "map": quality_map,
            },
            passed=passed,
            message=f"{metric.upper()} {value:.6g} {units}",
            fail_reasons=[] if passed else ["image_quality_out_of_tolerance"],
        )
