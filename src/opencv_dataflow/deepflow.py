"""OpenCV DeepFlow defaults on consecutive, ticketed frames."""

import cv2
import numpy as np

from .parameter_validation import NumericRange, ValidationError
from .processing import gray_image
from .sequence_tools import _DenseFlowSequence

_MAX_NATIVE_BYTES = 512 * 1024**2


def _workspace_bytes(shape):
    height, width = shape
    resolution = height * width
    pyramid = resolution
    # Current OpenCV defaults for scale stepping, minimum size and layer count.
    for _ in range(200):
        height, width = int(height * 0.95 + 0.5), int(width * 0.95 + 0.5)
        if min(height, width) <= 25:
            break
        pyramid += height * width
    # Two retained image pyramids plus full-resolution refinement and remap buffers.
    # Rounded dimensions account for the native padded red/black work buffers.
    padded = (shape[0] + 2) * (shape[1] + 5)
    return 4 * (2 * pyramid + 40 * padded)


class DeepFlowSequence(_DenseFlowSequence):
    type_id, title = "sequence.deepflow", "DeepFlow optical flow"
    description = "Estimate dense displacement using the native OpenCV DeepFlow solver."
    parameter_ranges = {
        **_DenseFlowSequence.parameter_ranges,
        "max_gap_ms": NumericRange(0),
        "visual_max_motion": NumericRange(1e-6),
    }
    parameter_help = {
        **_DenseFlowSequence.parameter_help,
        "visual_max_motion": "Displacement in pixels mapped to full preview brightness.",
    }
    input_help = {
        "image": "Unsigned 8-bit, unsigned 16-bit, or normalized floating-point intensity image.",
        "mask": "Select published flow pixels on the previous image grid.",
        "region": "Select published pixels on the previous image grid, intersected with any mask.",
    }
    output_help = {
        "flow": "Float32 previous-to-current displacement vectors in pixels.",
        "consistency_mask": "White where forward and backward estimates agree within tolerance.",
        "consistency_error": "Round-trip distance in pixels. NaN for unevaluated pixels.",
    }

    def working_image(self, image):
        gray = gray_image(image, (np.uint8, np.float32), auto_depth=True)
        if gray.dtype == np.uint8:
            return gray
        if gray.min() < 0 or gray.max() > 1:
            raise ValidationError("Floating-point intensity samples must be within [0, 1]")
        # DeepFlow converts its input to float without changing the intensity scale.
        return gray * np.float32(255)

    def estimate_state_bytes(self, image):
        sample_bytes = 1 if image.dtype == np.uint8 else 4
        return image.shape[0] * image.shape[1] * sample_bytes + 4096

    def flow_calculator(self, shape):
        if min(shape) < 3 or max(shape) >= 32767:
            raise ValidationError("DeepFlow requires image dimensions between 3 and 32,766")
        if _workspace_bytes(shape) > _MAX_NATIVE_BYTES:
            raise ValidationError("DeepFlow workspace exceeds 512 MiB. Crop the image first.")
        create = getattr(getattr(cv2, "optflow", None), "createOptFlow_DeepFlow", None)
        if create is None:
            raise ValidationError("DeepFlow requires OpenCV contrib optflow")
        solver = create()

        def calculate(previous, current, _initial):
            return solver.calc(previous, current, None), None

        return calculate
