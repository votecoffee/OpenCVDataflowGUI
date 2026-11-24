"""Wrap the OpenCV contrib low-variation neighborhood smoother."""

import cv2
import numpy as np

from .images import image_format, opencv_image
from .parameter_validation import NumericRange, ValidationError
from .payloads import metadata
from .processing import ImageTool, encoded_result, image_value
from .selection import RegionalOperation

MAX_PIXELS = 1_048_576
MAX_WINDOW_SAMPLES = 32_000_000


class VarianceAdaptiveSmoothing(RegionalOperation, ImageTool):
    type_id, title, subtype = (
        "cv.variance_adaptive_smoothing",
        "Variance-adaptive edge smoothing",
        "Filters",
    )
    description = "Smooth low-variation neighborhoods to suppress noise around sharp edges."
    defaults = {"diameter": 7, "threshold": 30.0}
    parameter_ranges = {
        "diameter": NumericRange(3, 31, whole=True),
        "threshold": NumericRange(0, 442),
    }
    parameter_help = {
        "diameter": "Side of the native square neighborhood in pixels. Both raster axes must be larger.",
        "threshold": "Maximum mean Euclidean color distance from the window mean, in 8-bit units. "
        "Zero copies the input. Raising it admits more varied windows and can blur across an edge.",
    }
    input_help = {
        "image": "8-bit grayscale or RGB/BGR(A). Straight alpha and the input encoding are preserved.",
        "mask": "Select output pixels to replace. Neighborhood estimates use the full source.",
        "region": "Select output pixels to replace, intersected with the mask.",
    }
    output_help = {"image": "Same pixel grid, encoding and uint8 depth as the input."}
    preview_safe = parallel_safe = process_inputs_readonly = True
    payload_reference = "image"

    def process(self, inputs):
        source = inputs["image"]
        diameter = int(self.parameters["diameter"])
        threshold = float(self.parameters["threshold"])
        height, width = source.shape[:2]
        if min(height, width) <= diameter:
            raise ValidationError("Both image axes must be larger than the diameter", "diameter")
        if height * width > MAX_PIXELS:
            raise ValidationError(f"Variance-adaptive smoothing permits up to {MAX_PIXELS:,} pixels")
        window_samples = (height - diameter) * (width - diameter) * diameter**2
        if threshold and window_samples > MAX_WINDOW_SAMPLES:
            raise ValidationError("Smoothing work exceeds the limit. Reduce diameter or crop.", "diameter")
        if source.dtype != np.uint8:
            raise ValidationError("Variance-adaptive smoothing needs 8-bit input. Convert explicitly.", "image")
        image = image_value(source, (np.uint8,))
        encoding = image_format(source)
        if encoding not in ("gray", "rgb", "bgr", "rgba", "bgra"):
            raise ValidationError("Convert color to grayscale or RGB/BGR before smoothing", "image")
        if threshold == 0:
            return encoded_result(image.copy(), source)

        alpha = encoding in ("rgba", "bgra")
        if alpha and metadata(source).get("alpha_representation", "straight") != "straight":
            raise ValidationError("Unassociate premultiplied alpha before smoothing", "image")
        native = getattr(getattr(cv2, "ximgproc", None), "edgePreservingFilter", None)
        if native is None:
            raise ValidationError("Variance-adaptive smoothing requires OpenCV contrib ximgproc")
        work = opencv_image(source)
        if image.ndim == 2:
            work = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
        else:
            work = np.ascontiguousarray(work[..., :3])
        output = native(work, diameter, threshold)
        if image.ndim == 2:
            output = output[..., 0].copy()
        elif encoding in ("rgb", "rgba"):
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        if alpha:
            output = np.concatenate((output, image[..., 3:]), axis=2)
        return encoded_result(output, source)
