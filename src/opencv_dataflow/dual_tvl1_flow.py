"""OpenCV contrib Dual TV-L1 on consecutive frames."""

import cv2
import numpy as np

from .parameter_validation import NumericRange, ValidationError
from .processing import gray_image
from .sequence import SEQUENCE_CHOICES
from .sequence_tools import _DenseFlowSequence

_MAX_NATIVE_BYTES = 512 * 1024**2


def _pyramid_budget(shape, count, ratio, illumination):
    """Count the scales OpenCV can use and bound its float work arrays."""
    rows, cols = shape
    scales = 1
    pyramid_pixels = rows * cols
    for _ in range(1, count):
        rows, cols = round(rows * ratio), round(cols * ratio)
        if min(rows, cols) < 16:
            break
        pyramid_pixels += rows * cols
        scales += 1
    # Full-resolution derivative, warping and optimization arrays plus the
    # image/flow pyramids. The illumination term adds a third flow component.
    full_resolution_pixels = shape[0] * shape[1]
    planes = 5 if illumination else 4
    native_bytes = np.dtype(np.float32).itemsize * (
        32 * full_resolution_pixels + planes * pyramid_pixels
    )
    return scales, native_bytes


class DualTVL1Sequence(_DenseFlowSequence):
    type_id, title = "sequence.dual_tvl1", "Dual TV-L1 optical flow"
    description = "Estimate dense displacement between consecutive images with Dual TV-L1."
    defaults = {
        **_DenseFlowSequence.defaults,
        "tau": 0.25,
        "lambda_weight": 0.15,
        "theta": 0.3,
        "scales": 5,
        "warps": 5,
        "epsilon": 0.01,
        "inner_iterations": 30,
        "outer_iterations": 10,
        "scale_step": 0.8,
        "gamma": 0.0,
        "median_filter": "5",
        "initial_flow": "none",
    }
    parameter_choices = {
        **SEQUENCE_CHOICES,
        "median_filter": ("1", "3", "5"),
        "initial_flow": ("none", "previous"),
    }
    parameter_ranges = {
        **_DenseFlowSequence.parameter_ranges,
        "visual_max_motion": NumericRange(1e-6),
        "max_gap_ms": NumericRange(0),
        "tau": NumericRange(1e-6, 1),
        "lambda_weight": NumericRange(1e-6, 10),
        "theta": NumericRange(1e-6, 10),
        "scales": NumericRange(1, 12, whole=True),
        "warps": NumericRange(1, 20, whole=True),
        "epsilon": NumericRange(1e-6, 1),
        "inner_iterations": NumericRange(1, 100, whole=True),
        "outer_iterations": NumericRange(1, 100, whole=True),
        "scale_step": NumericRange(0.01, 0.999999),
        "gamma": NumericRange(0, 10),
    }
    parameter_help = {
        **_DenseFlowSequence.parameter_help,
        "visual_max_motion": "Displacement in pixels mapped to full preview brightness.",
        "tau": "Time step of the iterative solver.",
        "lambda_weight": "Weight of the image data term relative to flow regularization.",
        "theta": "Coupling between the flow estimate and the auxiliary variables.",
        "scales": "Maximum pyramid levels. Coarse levels smaller than 16 pixels are omitted.",
        "warps": "Maximum target-image warps at each level.",
        "epsilon": "Stopping tolerance for changes in the flow estimate.",
        "inner_iterations": "Maximum solver iterations between median filtering steps.",
        "outer_iterations": "Maximum rounds of inner solver iterations per warp.",
        "scale_step": "Ratio between successive pyramid levels.",
        "gamma": "Weight of the illumination variation term. Zero disables it.",
        "median_filter": "Median window side in pixels. One disables filtering.",
        "initial_flow": "Previous seeds the forward solve with the preceding unmasked field.",
    }
    input_help = {
        "image": "An 8-bit, 16-bit, or normalized floating-point intensity image.",
        "mask": "Select output pixels on the previous image grid after estimating flow.",
        "region": "Select output pixels on the previous image grid, intersected with any mask.",
    }
    output_help = {
        "flow": "Float32 dx and dy in pixels, previous image to current image.",
        "consistency_mask": "White where forward and backward estimates agree within tolerance.",
        "consistency_error": "Round-trip error in pixels. NaN where it cannot be evaluated.",
    }

    def working_image(self, image):
        gray = gray_image(image, (np.uint8, np.float32), auto_depth=True)
        if gray.dtype == np.float32 and (gray.min() < 0 or gray.max() > 1):
            raise ValidationError("Floating-point intensity samples must be within [0, 1]")
        return gray

    def estimate_state_bytes(self, image):
        retained_frame = 1 if image.dtype == np.uint8 else 4
        retained_flow = 8 if self.parameters["initial_flow"] == "previous" else 0
        return image.shape[0] * image.shape[1] * (retained_frame + retained_flow) + 4096

    def flow_calculator(self, shape):
        if min(shape) < 3 or max(shape) >= 32767:
            raise ValidationError("Dual TV-L1 requires image dimensions between 3 and 32,766")
        p = self.parameters
        ratio, gamma = float(p["scale_step"]), float(p["gamma"])
        scales, native_bytes = _pyramid_budget(shape, int(p["scales"]), ratio, gamma != 0)
        if native_bytes > _MAX_NATIVE_BYTES:
            raise ValidationError(
                "Dual TV-L1 workspace exceeds 512 MiB. Crop the image or reduce the pyramid."
            )
        factory = getattr(getattr(cv2, "optflow", None), "DualTVL1OpticalFlow_create", None)
        if factory is None:
            raise ValidationError("Dual TV-L1 requires OpenCV contrib optflow")
        solver = factory(
            float(p["tau"]),
            float(p["lambda_weight"]),
            float(p["theta"]),
            scales,
            int(p["warps"]),
            float(p["epsilon"]),
            int(p["inner_iterations"]),
            int(p["outer_iterations"]),
            ratio,
            gamma,
            int(p["median_filter"]),
            False,
        )

        def calculate(previous, current, initial):
            solver.setUseInitialFlow(initial is not None)
            return solver.calc(previous, current, initial), None

        return calculate
