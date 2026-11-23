"""L0 gradient smoothing with a consistent boundary condition."""

import math

import numpy as np
from scipy.fft import dctn, idctn

from .images import convert_depth, image_format
from .parameter_validation import NumericRange, ValidationError
from .processing import ImageTool, encoded_result, image_value
from .selection import RegionalOperation

_MAX_PENALTY = 100_000
_MAX_TRANSFORM_SAMPLE_STEPS = 160_000_000
_DCT = {"axes": (0, 1), "norm": "ortho", "workers": 1}


def _penalty_steps(weight, growth):
    if not weight:
        return 0
    return math.ceil(math.log(_MAX_PENALTY / (2 * weight)) / math.log(growth))


def _l0_smooth(work, weight, growth, steps):
    """Alternate gradient thresholding with a Neumann least-squares solve."""
    rows, cols = work.shape[:2]
    row_eigenvalues = 4 * np.sin(np.pi * np.arange(rows) / (2 * rows)) ** 2
    col_eigenvalues = 4 * np.sin(np.pi * np.arange(cols) / (2 * cols)) ** 2
    laplacian = (row_eigenvalues[:, None] + col_eigenvalues).astype(work.dtype)
    data = dctn(work, **_DCT)
    estimate = work
    horizontal, vertical, adjoint = (np.zeros_like(work) for _ in range(3))
    penalty = 2 * weight
    for _ in range(steps):
        # The outward differences stay zero. The divergence below is their exact adjoint.
        np.subtract(estimate[:, 1:], estimate[:, :-1], out=horizontal[:, :-1])
        np.subtract(estimate[1:], estimate[:-1], out=vertical[:-1])
        magnitude = np.sum(horizontal**2 + vertical**2, axis=2, keepdims=True)
        keep = magnitude > weight / penalty
        horizontal *= keep
        vertical *= keep
        adjoint.fill(0)
        adjoint[:, :-1] -= horizontal[:, :-1]
        adjoint[:, 1:] += horizontal[:, :-1]
        adjoint[:-1] -= vertical[:-1]
        adjoint[1:] += vertical[:-1]
        spectrum = dctn(adjoint, **_DCT)
        # The adjoint sums to zero. Keep roundoff out of the preserved DC coefficient.
        spectrum[0, 0] = 0
        spectrum *= penalty
        spectrum += data
        spectrum /= (1 + penalty * laplacian)[..., None]
        estimate = idctn(spectrum, **_DCT)
        penalty *= growth
    return estimate


class L0Smoothing(RegionalOperation, ImageTool):
    type_id, title, subtype = "cv.l0_smoothing", "L0 image smoothing", "Filters"
    description = "Suppress small gradients with a global fit that preserves larger boundaries."
    defaults = {"lambda_weight": 0.02, "kappa": 2.0}
    parameter_ranges = {
        "lambda_weight": NumericRange(1e-6, 1, zero_allowed=True),
        "kappa": NumericRange(1.01, 16),
    }
    parameter_help = {
        "lambda_weight": "Penalty for keeping a nonzero gradient. Zero copies the input.",
        "kappa": "Factor that increases the gradient-fit weight between solver steps.",
    }
    input_help = {
        "image": "Mono or RGB/BGR(A), uint8, uint16 or normalized float. Keeps depth and alpha.",
        "mask": "Select output pixels to replace. Smoothing uses the full source.",
        "region": "Select output pixels to replace, intersected with the mask.",
    }
    preview_safe = parallel_safe = process_inputs_readonly = True
    payload_reference = "image"

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if _penalty_steps(float(p["lambda_weight"]), float(p["kappa"])) > 128:
            raise ValidationError(
                "This continuation schedule exceeds 128 steps. Increase kappa.",
                "lambda_weight",
                "kappa",
            )

    def process(self, inputs):
        source = inputs["image"]
        height, width = source.shape[:2]
        if max(height, width) > 8192 or height * width > 2_097_152:
            raise ValidationError("L0 smoothing permits up to 2,097,152 pixels and 8,192 per axis")
        image = image_value(source, (np.uint8, np.uint16, np.float32, np.float64))
        if image_format(source) not in ("gray", "bgr", "rgb", "bgra", "rgba"):
            raise ValidationError("Select a grayscale or RGB/BGR intensity image")
        weight, growth = float(self.parameters["lambda_weight"]), float(self.parameters["kappa"])
        steps = _penalty_steps(weight, growth)
        if not steps:
            return encoded_result(image.copy(), source)
        has_alpha = image.ndim == 3 and image.shape[2] == 4
        signal = image[..., :3] if has_alpha else image
        if image.dtype.kind == "f":
            work = signal
        else:
            work = convert_depth(signal, "float32")
        if work.min() < 0 or work.max() > 1:
            raise ValidationError("Floating-point intensity samples must be within [0, 1]")
        if signal.size * (work.itemsize // 4) * steps > _MAX_TRANSFORM_SAMPLE_STEPS:
            raise ValidationError("L0 smoothing work exceeds the limit. Increase kappa or crop.")
        mono = signal.ndim == 2
        fitted = _l0_smooth(work[..., None] if mono else work, weight, growth, steps)
        if not np.isfinite(fitted).all():
            raise ValidationError("L0 smoothing produced nonfinite samples")
        fitted = np.clip(fitted[..., 0] if mono else fitted, 0, 1)
        output = convert_depth(fitted, image.dtype.name) if image.dtype.kind != "f" else fitted
        if has_alpha:
            output = np.concatenate((output, image[..., 3:]), axis=2)
        return encoded_result(output, source)
