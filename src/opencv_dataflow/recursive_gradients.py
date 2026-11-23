"""Paired recursive edge derivatives on the source pixel grid."""

import math
from abc import ABC, abstractmethod

import cv2
import numpy as np
from scipy.signal import lfilter, lfilter_zi

from .core import Port, Result
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .processing import ImageTool, image_value
from .selection import RegionalOperation


class _RecursiveGradient(RegionalOperation, ImageTool, ABC):
    subtype = "Edges"
    outputs = (Port("dx"), Port("dy"), Port("magnitude"), Port("preview", required=False))
    region_outputs = ("dx", "dy", "magnitude", "preview")
    region_background = "zero"
    defaults = {"alpha": 1.2, "omega": 0.6}
    parameter_help = {
        "alpha": "Decay parameter in inverse pixels. Smaller values increase spatial support.",
        "omega": "Frequency parameter of the recursive filter in inverse pixels.",
    }
    input_help = {
        "image": "Mono or RGB/BGR(A), uint8, int8, uint16, int16 or float32. Alpha is ignored.",
        "mask": "Select where to publish gradients. Neighborhoods use the full source.",
        "region": "Select published pixels, intersected with the output mask.",
    }
    output_help = {
        "dx": "Signed float32 response. Positive for a rising edge from left to right.",
        "dy": "Signed float32 response. Positive for a rising edge from top to bottom.",
        "magnitude": "Float32 Euclidean gradient magnitude, per source channel.",
        "preview": "Magnitude scaled to the byte range for display.",
    }
    preview_safe = parallel_safe = process_inputs_readonly = True
    payload_reference = "image"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.preview_output = "preview"

    @abstractmethod
    def gradients(self, signal, alpha, omega):
        """Return the horizontal and vertical responses before output selection."""

    def process(self, inputs):
        source = inputs["image"]
        height, width = source.shape[:2]
        if max(height, width) > 8192 or height * width > 2_097_152:
            raise ValidationError(
                "Recursive gradients allow up to 2,097,152 pixels and 8,192 per axis"
            )
        image = image_value(source, (np.uint8, np.int8, np.uint16, np.int16, np.float32))
        encoding = image_format(source)
        if encoding not in ("gray", "rgb", "bgr", "rgba", "bgra"):
            raise ValidationError("Select a grayscale or RGB/BGR intensity image")
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[..., :3]
            encoding = "rgb" if encoding == "rgba" else "bgr"
        p = self.parameters
        if np.array_equal(image.min(axis=(0, 1)), image.max(axis=(0, 1))):
            dx, dy = (np.zeros(image.shape, np.float32) for _ in range(2))
        else:
            dx, dy = self.gradients(image, float(p["alpha"]), float(p["omega"]))
        with np.errstate(over="ignore", invalid="ignore"):
            dx, dy = (axis.astype(np.float32, copy=False) for axis in (dx, dy))
            magnitude = np.hypot(dx, dy)
        if not np.isfinite(magnitude).all():
            raise ValidationError("Recursive gradient overflowed float32. Rescale the input.")
        peak = float(magnitude.max())
        preview = (
            np.rint(magnitude / peak * 255).astype(np.uint8)
            if peak
            else np.zeros_like(magnitude, dtype=np.uint8)
        )
        return Result(
            {
                name: ImageArray(raster, encoding)
                for name, raster in (
                    ("dx", dx),
                    ("dy", dy),
                    ("magnitude", magnitude),
                    ("preview", preview),
                )
            }
        )


class DericheGradient(_RecursiveGradient):
    type_id, title = "cv.deriche_gradient", "Deriche gradients"
    description = "Compute signed derivatives using OpenCV's recursive Deriche filters."
    parameter_ranges = {"alpha": NumericRange(0.05, 16), "omega": NumericRange(0.01, 3.13)}

    def gradients(self, signal, alpha, omega):
        api = getattr(cv2, "ximgproc", None)
        dx, dy = (getattr(api, f"GradientDeriche{axis}", None) for axis in ("X", "Y"))
        if dx is None or dy is None:
            raise ValidationError("Deriche gradients require OpenCV contrib ximgproc")
        height, width = signal.shape[:2]
        # Replicate long enough for the truncated native recurrences to settle.
        halo = math.ceil(16 / alpha)
        if (height + 2 * halo) * (width + 2 * halo) > 4_194_304:
            raise ValidationError("Deriche padded workspace exceeds 4,194,304 pixels")
        padded = cv2.copyMakeBorder(signal, halo, halo, halo, halo, cv2.BORDER_REPLICATE)
        roi = np.s_[halo : halo + height, halo : halo + width]
        return dx(padded, alpha, omega)[roi].copy(), dy(padded, alpha, omega)[roi].copy()


def _causal(signal, axis, numerator, denominator):
    values = np.moveaxis(signal, axis, 0)
    state = lfilter_zi(numerator, denominator).reshape((-1,) + (1,) * (signal.ndim - 1))
    output, _ = lfilter(numerator, denominator, values, axis=0, zi=state * values[0])
    return np.moveaxis(output, 0, axis)


def _paillou_pair(signal, alpha, omega):
    """Separable Paillou derivative and smoothing with constant continuation."""
    slow, fast = math.exp(-alpha + omega), math.exp(-alpha - omega)
    pole_sum, pole_product = slow + fast, slow * fast
    dc = -math.expm1(-alpha + omega) * -math.expm1(-alpha - omega)
    denominator = (1.0, -pole_sum, pole_product)
    decay_sinh, decay_cosh = (slow - fast) / 2, pole_sum / 2
    scale = dc / (2 * alpha * decay_sinh - omega * math.expm1(-2 * alpha))
    center = omega * scale
    next_tap = scale * (alpha * decay_sinh - omega * decay_cosh)
    forward_smoother = (center, next_tap)
    reverse_smoother = (0.0, next_tap + center * pole_sum, -center * pole_product)

    gradients = []
    for axis in (1, 0):
        reversed_source = np.flip(signal, axis)
        causal = _causal(signal, axis, (dc,), denominator)
        reverse = _causal(reversed_source, axis, (dc,), denominator)
        derivative = np.flip(reverse, axis) - causal
        cross_axis = 1 - axis
        smooth = _causal(derivative, cross_axis, forward_smoother, denominator)
        smooth += np.flip(
            _causal(np.flip(derivative, cross_axis), cross_axis, reverse_smoother, denominator),
            cross_axis,
        )
        gradients.append(smooth)
    return tuple(gradients)


class PaillouGradient(_RecursiveGradient):
    type_id, title = "cv.paillou_gradient", "Paillou gradients"
    description = "Compute signed derivatives using the separable Paillou recurrence."
    parameter_ranges = {"alpha": NumericRange(0.05, 16), "omega": NumericRange(0.01, 15.999)}
    parameter_help = {
        **_RecursiveGradient.parameter_help,
        "omega": "Separates the real poles. Values closer to alpha extend the response.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if float(p["alpha"]) - float(p["omega"]) < 0.001 - 1e-12:
            raise ValidationError("Omega must be at least 0.001 below alpha", "alpha", "omega")

    def gradients(self, signal, alpha, omega):
        return _paillou_pair(signal, alpha, omega)
