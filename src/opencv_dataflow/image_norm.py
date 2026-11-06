"""Measure OpenCV norms over one image or a registered pair."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .images import image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import require_same_frame
from .processing import image_value
from .selection import selection_mask

NORMS = {
    "infinity": cv2.NORM_INF,
    "l1": cv2.NORM_L1,
    "l2": cv2.NORM_L2,
    "l2_squared": cv2.NORM_L2SQR,
    "hamming": cv2.NORM_HAMMING,
    "hamming2": cv2.NORM_HAMMING2,
}


def measure_norm(samples, reference, metric, relative):
    norm_type = NORMS[metric]
    scale = 1.0
    if metric in ("hamming", "hamming2"):
        if samples.dtype != np.uint8:
            raise ValidationError("Hamming norms require uint8 samples")
    else:
        samples = samples.astype(np.float64, copy=False)
        if reference is not None:
            reference = reference.astype(np.float64, copy=False)
            if not relative:
                samples = cv2.absdiff(samples, reference)
                reference = None
                if not np.isfinite(samples).all():
                    raise ValidationError("Image difference exceeds the float64 range")
        peak = float(np.max(np.abs(samples)))
        if reference is not None:
            peak = max(peak, float(np.max(np.abs(reference))))
        # Avoid premature overflow or underflow when OpenCV accumulates squares.
        if 0 < peak < 1e-140 or peak > 1e140:
            scale = peak
            samples = samples / scale
            if reference is not None:
                reference = reference / scale

    if reference is None:
        value = cv2.norm(samples, norm_type)
    else:
        value = cv2.norm(samples, reference, norm_type)
    if relative:
        denominator = cv2.norm(reference, norm_type)
        if denominator == 0:
            raise ValidationError("Relative norm needs a nonzero norm in the selected reference")
        value /= denominator
    else:
        value *= scale
        if metric == "l2_squared":
            value *= scale
    if not np.isfinite(value):
        raise ValidationError("Image norm exceeds the float64 range")
    return value


class MeasureImageNorm(Block):
    type_id, title = "cv.image_norm", "Measure image norm"
    category, block_type, subtype = "Image processing", "Image processing", "Statistics"
    process_inputs_readonly = True
    payload_reference = "image"
    inputs = (
        Port("image"),
        Port("reference", required=False, required_when_connected=True),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (Port("value", "number"), Port("measurements", "data"))
    defaults = {
        "metric": "l2",
        "mode": "absolute",
        "alignment": "same_frame",
        "include_alpha": False,
        "check_threshold": False,
        "maximum": 1.0,
    }
    parameter_choices = {
        "metric": tuple(NORMS),
        "mode": ("absolute", "relative"),
        "alignment": ("same_frame", "pixel_grid"),
    }
    parameter_ranges = {"maximum": NumericRange(0)}
    parameter_help = {
        "metric": "Infinity is the maximum absolute sample. L1 sums absolute samples. "
        "L2 is Euclidean. Hamming counts differing bits. Hamming2 counts differing bit pairs.",
        "mode": "Relative divides the difference norm by the norm of the selected reference.",
        "alignment": "Pixel grid compares equal-size images by index across frame IDs.",
        "include_alpha": "Include the alpha channel for four-channel input.",
        "check_threshold": "Fail when the measured norm exceeds Maximum.",
        "maximum": "Inclusive upper limit when Check threshold is enabled.",
    }
    input_help = {
        "image": "Image whose selected samples are measured.",
        "reference": "Optional reference with the same dimensions, depth and channel encoding.",
    }

    def process(self, inputs):
        source = inputs["image"]
        p = self.parameters
        image = image_value(source)
        reference = inputs.get("reference")
        relative = p["mode"] == "relative"
        if reference is None and relative:
            raise ValidationError("Connect a reference image for a relative norm", "mode")
        if reference is not None:
            other = image_value(reference)
            if (
                other.shape != image.shape
                or other.dtype != image.dtype
                or image_format(reference) != image_format(source)
            ):
                raise ValidationError("Reference must match image size, depth and channel encoding")
            if p["alignment"] == "same_frame":
                try:
                    require_same_frame(
                        (source, reference),
                        message="Image and reference must share a coordinate frame",
                    )
                except ValueError as exc:
                    raise ValidationError(str(exc), "alignment") from exc
        else:
            other = None

        mask = selection_mask(inputs, source)
        selected = None if mask is None else mask != 0
        count = image.shape[0] * image.shape[1] if selected is None else int(selected.sum())
        if not count:
            raise ValidationError("Image norm requires at least one selected pixel")

        def samples(array):
            if array.ndim == 3 and array.shape[2] == 4 and not p["include_alpha"]:
                array = array[..., :3]
            return (array if selected is None else array[selected]).reshape(-1, 1)

        metric = p["metric"]
        value = measure_norm(
            samples(image), None if other is None else samples(other), metric, relative
        )
        mode = "relative" if relative else "difference" if other is not None else "magnitude"
        passed = not p["check_threshold"] or value <= float(p["maximum"])
        return Result(
            {
                "value": value,
                "measurements": {
                    "metric": metric,
                    "mode": mode,
                    "value": value,
                    "selected_count": count,
                },
            },
            passed=passed,
            fail_reasons=[] if passed else ["image_norm_above_limit"],
            message=f"{metric.upper()} {mode} {value:.6g} over {count} selected pixels",
        )
