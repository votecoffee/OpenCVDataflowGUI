"""Blend aligned images with a weight for each pixel."""

import cv2
import numpy as np

from .core import Port, Result
from .images import ImageArray, image_format
from .parameter_validation import ValidationError
from .payloads import metadata, require_same_frame
from .processing import ImageTool, image_value
from .selection import SELECTION_INPUTS, RegionalOperation
from .tickets import ticket_key

_IMAGE_DEPTHS = (np.uint8, np.uint16, np.int16, np.float32)
_WEIGHT_DEPTHS = (np.uint8, np.uint16, np.float32, np.float64)
# At this scale the native denominator's 1e-5 guard is below float32 resolution.
_WEIGHT_SCALE = 1024.0


def _weight_map(value, shape, name):
    weights = image_value(value, _WEIGHT_DEPTHS)
    if weights.shape != shape:
        raise ValidationError(f"{name} must be monochrome and match the image size", name)
    if weights.dtype.kind == "u":
        return weights.astype(np.float64) / np.iinfo(weights.dtype).max
    if weights.min() < 0 or weights.max() > 1:
        raise ValidationError(f"{name} floating weights must be in [0, 1]", name)
    return weights.astype(np.float64, copy=False)


def _blend(a, b, weight_a, weight_b):
    integer16 = a.dtype in (np.uint16, np.int16)
    if integer16:
        output = cv2.blendLinear(a.astype(np.float32), b.astype(np.float32), weight_a, weight_b)
        limits = np.iinfo(a.dtype)
        return np.clip(np.rint(output), limits.min, limits.max).astype(a.dtype)
    output = cv2.blendLinear(a, b, weight_a, weight_b)
    if a.dtype == np.float32:
        bad = ~np.isfinite(output)
        if bad.any():
            # A large finite input can overflow the native float32 numerator.
            pixels = bad.any(axis=2) if a.ndim == 3 else bad
            alpha = weight_a[pixels].astype(np.float64)
            beta = weight_b[pixels].astype(np.float64)
            total = alpha + beta
            if a.ndim == 3:
                alpha, beta, total = alpha[:, None], beta[:, None], total[:, None]
            limit = float(np.finfo(np.float32).max)
            output[pixels] = np.clip((a[pixels] * alpha + b[pixels] * beta) / total, -limit, limit)
    return output


class BlendWeightMaps(RegionalOperation, ImageTool):
    type_id, title, subtype = "cv.blend_weight_maps", "Blend with weight maps", "Image math"
    region_source = "image_a"
    process_inputs_readonly = True
    inputs = (
        Port("image_a"),
        Port("image_b"),
        Port("weight_a"),
        Port("weight_b", required=False, required_when_connected=True),
        *SELECTION_INPUTS[1:],
    )
    defaults = {"alignment": "same_frame", "zero_weight": "keep_a"}
    parameter_choices = {
        "alignment": ("same_frame", "pixel_grid"),
        "zero_weight": ("keep_a", "zero"),
    }
    parameter_help = {
        "alignment": "Pixel grid pairs equal-size images and weights by index across frame IDs.",
        "zero_weight": "When both weights are zero, keep Image A or write zero.",
    }
    input_help = {
        "image_a": "First image and output frame. Supplies pixels outside the selection.",
        "image_b": "Second image with the same size, pixel depth and channel encoding.",
        "weight_a": "Monochrome weights for Image A. Integer full scale or floating [0, 1].",
        "weight_b": "Optional weights for Image B. The default is one minus Weight A.",
    }

    def process(self, inputs):
        source = inputs["image_a"]
        a, b = (image_value(inputs[key], _IMAGE_DEPTHS) for key in ("image_a", "image_b"))
        encoding = image_format(source)
        if (
            a.shape != b.shape
            or a.dtype != b.dtype
            or encoding != image_format(inputs["image_b"])
            or encoding in ("hsv", "lab")
        ):
            raise ValidationError(
                "Use matching grayscale or RGB/BGR images with the same pixel depth"
            )

        keys = ("image_a", "image_b", "weight_a", "weight_b")
        sources = [inputs[key] for key in keys if key in inputs]
        payloads = [metadata(value) for value in sources]
        payload = payloads[0]
        ticketed = [p for p in payloads if "ticket" in p]
        if len({ticket_key(p["ticket"]) for p in ticketed}) > 1:
            raise ValidationError("Blend images and weights from the same ticket")
        if ticketed and "ticket" not in payload:
            payload.update(
                {
                    key: ticketed[0][key]
                    for key in ("ticket", "processing_time")
                    if key in ticketed[0]
                }
            )

        first = _weight_map(inputs["weight_a"], a.shape[:2], "weight_a")
        if "weight_b" in inputs:
            second = _weight_map(inputs["weight_b"], a.shape[:2], "weight_b")
            total = first + second
            covered = total != 0
            alpha = np.full_like(first, 1.0 if self.parameters["zero_weight"] == "keep_a" else 0.0)
            beta = np.zeros_like(second)
            np.divide(first, total, out=alpha, where=covered)
            np.divide(second, total, out=beta, where=covered)
        else:
            alpha, beta = first, 1 - first
        if self.parameters["alignment"] == "same_frame":
            require_same_frame(
                sources, message="Images and weights must share the same coordinate frame"
            )
        weight_a = (alpha * _WEIGHT_SCALE).astype(np.float32)
        weight_b = (beta * _WEIGHT_SCALE).astype(np.float32)
        return Result(
            {"image": ImageArray(_blend(a, b, weight_a, weight_b), encoding)},
            payloads={"image": payload},
            message="Blended images using pixel weights",
        )
