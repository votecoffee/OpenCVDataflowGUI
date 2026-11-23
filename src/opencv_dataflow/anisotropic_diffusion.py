"""Edge-preserving diffusion with fixed intensity units and replicated borders."""

import cv2
import numpy as np

from .images import convert_depth, image_format
from .parameter_validation import NumericRange, ValidationError
from .processing import ImageTool, encoded_result, image_value
from .selection import RegionalOperation

_NEIGHBORS = ((1, 0), (1, 2), (0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (2, 2))
_MAX_EXPLICIT_SAMPLE_STEPS = 128_000_000


def _explicit_diffusion(signal, alpha, sensitivity, iterations):
    """Eight-neighbor Perona–Malik update for full-range bytes and high-depth data."""
    if signal.size * iterations > _MAX_EXPLICIT_SAMPLE_STEPS:
        raise ValidationError("Diffusion work exceeds the limit. Reduce iterations or crop.")
    byte_samples = signal.dtype == np.uint8
    work = signal.astype(np.float32) if byte_samples else convert_depth(signal, "float32")
    if not byte_samples and (work.min() < 0 or work.max() > 1):
        raise ValidationError("Float diffusion samples must be normalized within [0, 1]")
    sigma = np.float32(sensitivity * (255 if byte_samples else 1))
    height, width = work.shape[:2]
    for _ in range(iterations):
        padded = cv2.copyMakeBorder(work, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        update = np.zeros_like(work)
        for y, x in _NEIGHBORS:
            delta = padded[y : y + height, x : x + width] - work
            distance = np.abs(delta)
            if work.ndim == 3:
                distance = distance.mean(axis=2, keepdims=True)
            update += delta * np.exp(-np.square(distance / sigma))
        work += np.float32(alpha) * update
        if byte_samples:
            np.rint(work, out=work)
    if byte_samples:
        return np.clip(work, 0, 255).astype(np.uint8)
    return convert_depth(np.clip(work, 0, 1), signal.dtype.name)


class AnisotropicDiffusion(RegionalOperation, ImageTool):
    type_id, title, subtype = "cv.anisotropic_diffusion", "Anisotropic diffusion", "Filters"
    description = "Reduce within-surface noise while limiting diffusion across intensity edges."
    defaults = {"time_step": 0.1, "edge_sensitivity": 0.12, "iterations": 8}
    parameter_ranges = {
        "time_step": NumericRange(1e-6, 0.125),
        "edge_sensitivity": NumericRange(1e-6, 16),
        "iterations": NumericRange(0, 128, whole=True),
    }
    parameter_help = {
        "time_step": "Step size per pass. At most 1/8 for stability with eight neighbors.",
        "edge_sensitivity": "OpenCV K in normalized intensity units. Smaller values retain edges.",
        "iterations": "Number of diffusion passes. Zero copies the source.",
    }
    input_help = {
        "image": "Mono or RGB/BGR(A), uint8, uint16 or normalized float32. Preserves alpha.",
        "mask": "Select output pixels to replace. Neighbor calculations use the full source.",
        "region": "Select output pixels to replace, intersected with the mask.",
    }
    preview_safe = parallel_safe = process_inputs_readonly = True
    payload_reference = "image"

    def process(self, inputs):
        p = self.parameters
        source = inputs["image"]
        height, width = source.shape[:2]
        if max(height, width) > 8192 or height * width > 2_097_152:
            raise ValidationError("Diffusion permits up to 2,097,152 pixels and 8,192 per axis")
        image = image_value(source, (np.uint8, np.uint16, np.float32))
        if image_format(source) not in ("gray", "rgb", "bgr", "rgba", "bgra"):
            raise ValidationError("Select a grayscale or RGB/BGR intensity image")
        iterations = int(p["iterations"])
        if not iterations:
            return encoded_result(image.copy(), source)
        alpha, sensitivity = float(p["time_step"]), float(p["edge_sensitivity"])
        has_alpha = image.ndim == 3 and image.shape[2] == 4
        signal = image[..., :3] if has_alpha else image
        # The native table omits the maximum three-channel difference. A channel with
        # narrower bounds stays within those bounds under a stable eight-neighbor step.
        if image.dtype != np.uint8 or np.all(
            (signal.min(axis=(0, 1)) == 0) & (signal.max(axis=(0, 1)) == 255)
        ):
            result = _explicit_diffusion(signal, alpha, sensitivity, iterations)
        else:
            native = getattr(getattr(cv2, "ximgproc", None), "anisotropicDiffusion", None)
            if native is None:
                raise ValidationError("Anisotropic diffusion requires OpenCV contrib ximgproc")
            result = (
                cv2.cvtColor(signal, cv2.COLOR_GRAY2BGR)
                if signal.ndim == 2
                else np.ascontiguousarray(signal)
            )
            # A single call per pass reconstructs the native replicated border each time.
            for _ in range(iterations):
                result = native(result, alpha, sensitivity, 1)
            if signal.ndim == 2:
                result = result[..., 0]
        if has_alpha:
            result = np.concatenate((result, image[..., 3:]), axis=2)
        return encoded_result(result, source)
