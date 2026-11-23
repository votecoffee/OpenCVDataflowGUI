"""OpenCV weighted median filtering with separate sample and output selections."""

import threading

import cv2
import numpy as np

from .core import Port
from .images import ImageArray, convert_depth, image_format
from .parameter_validation import NumericRange, ValidationError
from .processing import ImageTool, encoded_result, image_value
from .restoration import _check_aligned_guide
from .segmentation import bgr8
from .selection import SELECTION_INPUTS, selection_mask

_WEIGHT_TYPES = {
    "gaussian": "WMF_EXP",
    "inverse_linear": "WMF_IV1",
    "inverse_squared": "WMF_IV2",
    "cosine": "WMF_COS",
    "jaccard": "WMF_JAC",
    "unweighted": "WMF_OFF",
}
_DEPTHS = (np.uint8, np.uint16, np.int16, np.float32)
# The pinned native implementation mutates shared histogram scratch and color hashes.
_NATIVE_LOCK = threading.Lock()


def _guide_uint8(guide):
    reference = image_value(guide, _DEPTHS)
    if reference.dtype == np.int16:
        reference = np.rint((reference.astype(np.int32) + 32768) / 257).astype(np.uint8)
    elif reference.dtype != np.uint8:
        if reference.dtype == np.float32 and (reference.min() < 0 or reference.max() > 1):
            raise ValidationError("Supply a guide with float intensity within [0, 1]")
        reference = convert_depth(reference, "uint8")
    if reference.ndim == 3:
        reference = bgr8(ImageArray(reference, image_format(guide)))
    return np.ascontiguousarray(reference)


def _check_weights(guide, method, sigma, replace_missing_centers):
    if method in ("cosine", "jaccard") and guide.ndim == 3:
        if (guide < 4).all(axis=2).any():
            raise ValidationError(
                "Cosine and Jaccard require a nonzero guide color in every 6-bit bin", "weight"
            )
        # A shared nonzero channel guarantees positive similarity to samples at masked centers.
        if replace_missing_centers and not (guide.min(axis=(0, 1)) >= 4).any():
            raise ValidationError(
                "Masked-center cosine or Jaccard needs a shared nonzero guide channel",
                "weight",
            )
    elif method == "jaccard" and guide.ndim == 2 and not guide.min():
        raise ValidationError("Jaccard requires positive monochrome guide values", "weight")

    if method == "gaussian" and replace_missing_centers:
        span = guide.max(axis=(0, 1)).astype(float) - guide.min(axis=(0, 1))
        # Keep the smallest possible neighborhood weight above float32 underflow.
        if float(np.sum(span**2)) > 160 * sigma**2:
            raise ValidationError(
                "Increase color sigma to give excluded centers usable sample weights",
                "sigma_color",
            )


class WeightedMedianFilter(ImageTool):
    type_id, title, subtype = "cv.weighted_median_filter", "Weighted median filter", "Filters"
    description = "Remove impulse noise using a weighted median guided by an aligned image."
    inputs = (
        *SELECTION_INPUTS,
        Port("guide", required=False, required_when_connected=True),
        Port("sample_mask", required=False, visual_type="region", required_when_connected=True),
    )
    defaults = {
        "radius": 3,
        "sigma_color": 25.5,
        "weight": "gaussian",
        "guide_alignment": "same_frame",
    }
    parameter_choices = {
        "weight": tuple(_WEIGHT_TYPES),
        "guide_alignment": ("same_frame", "pixel_grid"),
    }
    parameter_ranges = {
        "radius": NumericRange(1, 63, whole=True),
        "sigma_color": NumericRange(1e-3, 4096),
    }
    parameter_help = {
        "radius": "Half-width of the square neighborhood in pixels.",
        "sigma_color": "Range scale in 8-bit guide levels for Gaussian and inverse weights.",
        "weight": "Guide weighting mode. Cosine gives equal weights for a monochrome guide.",
        "guide_alignment": "Pixel grid accepts preregistered guides and sample masks.",
    }
    input_help = {
        "image": "Mono or RGB/BGR(A), uint8, int16, uint16 or float32. Preserves alpha.",
        "guide": "Optional aligned guide. Uses the source when disconnected. Converted to 8-bit.",
        "sample_mask": "Only nonzero pixels contribute samples to the neighborhood median.",
        "mask": "Only selected output pixels are replaced.",
        "region": "Select output pixels to replace, intersected with the output mask.",
    }
    preview_safe = parallel_safe = process_inputs_readonly = True
    payload_reference = "image"

    def process(self, inputs):
        p = self.parameters
        source = inputs["image"]
        height, width = source.shape[:2]
        if max(height, width) > 8192 or height * width > 2_097_152:
            raise ValidationError(
                "Weighted median permits up to 2,097,152 pixels and 8,192 per axis"
            )
        image = image_value(source, _DEPTHS)
        guide = inputs.get("guide", source)
        same_frame = p["guide_alignment"] == "same_frame"
        _check_aligned_guide(source, guide, check_frame=same_frame)
        output_mask = selection_mask(inputs, source)
        sample_mask = selection_mask(
            {"mask": inputs.get("sample_mask")}, source, match_mask_frame=same_frame
        )
        radius, sigma = int(p["radius"]), float(p["sigma_color"])
        replace = output_mask != 0 if output_mask is not None else np.ones((height, width), bool)
        if sample_mask is not None:
            # Native medians have no usable value where the entire neighborhood was excluded.
            side = 2 * radius + 1
            replace &= cv2.dilate(sample_mask, np.ones((side, side), np.uint8)) != 0
        output = image.copy()
        if not replace.any():
            return encoded_result(output, source)

        weight = p["weight"]
        guide8 = (
            np.zeros((height, width), np.uint8) if weight == "unweighted" else _guide_uint8(guide)
        )
        missing_centers = sample_mask is not None and bool(np.any(replace & (sample_mask == 0)))
        _check_weights(guide8, weight, sigma, missing_centers)
        has_alpha = image.ndim == 3 and image.shape[2] == 4
        signal = image[..., :3] if has_alpha else image
        if image.dtype == np.float32:
            if float(signal.max()) - float(signal.min()) > float(np.finfo(np.float32).max) / 2:
                raise ValidationError("Source range exceeds native float histogram limits")
        native_signal = np.ascontiguousarray(
            signal if image.dtype == np.uint8 else signal.astype(np.float32, copy=False)
        )
        api = getattr(cv2, "ximgproc", None)
        native_filter = getattr(api, "weightedMedianFilter", None)
        if native_filter is None:
            raise ValidationError("Weighted median filtering requires OpenCV contrib ximgproc")
        with _NATIVE_LOCK:
            filtered = native_filter(
                guide8,
                native_signal,
                radius,
                sigma=sigma,
                weightType=getattr(api, _WEIGHT_TYPES[weight]),
                mask=sample_mask,
            )
        values = filtered[replace]
        if not np.isfinite(values).all():
            raise ValidationError("Native weighted median returned nonfinite samples")
        destination = output[..., :3] if has_alpha else output
        destination[replace] = values.astype(image.dtype, copy=False)
        return encoded_result(output, source)
