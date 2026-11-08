"""Model-free image focus measurements on a selected inspection area."""

import math

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .images import ImageArray
from .parameter_validation import NumericRange, ValidationError
from .parameters import BORDERS
from .payloads import NO_VALUE
from .processing import gray_image
from .result_graphics import ACCEPTED, REJECTED
from .selection import SELECTION_INPUTS, selection_mask

_METHODS = {"laplacian_variance": "Laplacian variance", "tenengrad": "Tenengrad"}


class MeasureFocus(Block):
    type_id, title = "inspect.focus", "Measure image focus"
    category, block_type, subtype = "Inspection", "Inspection", "Measurements"
    inputs = SELECTION_INPUTS
    outputs = (
        Port("sharpness", "number"),
        Port("measurements", "data"),
        Port("image"),
        Port("response", required=False),
    )
    defaults = {
        "method": "laplacian_variance",
        "aperture": "3",
        "border": "reflect_101",
        "check_threshold": False,
        "minimum_sharpness": 0.0,
        "emit_response": False,
    }
    parameter_choices = {
        "method": tuple(_METHODS),
        "aperture": ("1", "3", "5", "7"),
        "border": ("constant", "replicate", "reflect_101", "reflect_edge"),
    }
    parameter_ranges = {"minimum_sharpness": NumericRange(0)}
    parameter_help = {
        "method": "Variance of the signed Laplacian, or mean squared Sobel gradient (Tenengrad).",
        "aperture": "Derivative kernel size. Keep this fixed when comparing scores.",
        "border": "Pixel extrapolation at the edge of the source image.",
        "minimum_sharpness": "Inclusive lower limit when threshold checking is enabled.",
        "emit_response": "Output absolute Laplacian or squared gradient; zero outside selection.",
    }
    input_help = {
        "image": "Gray or color uint8, uint16 or float32. Scores use squared luminance units.",
        "mask": "Optional scoring mask. Derivatives still use the full image neighborhood.",
        "region": "Optional scoring region, intersected with the mask when both are supplied.",
    }
    process_inputs_readonly = True
    parallel_safe = True

    def process(self, inputs):
        source = inputs["image"]
        height, width = source.shape[:2]
        if min(height, width) < 2 or height * width > 16_777_216:
            raise ValidationError(
                "Focus measurement needs at least 2 × 2 and at most 16,777,216 pixels"
            )
        gray = gray_image(source).astype(np.float64)
        mask = selection_mask(inputs, source)
        count = gray.size if mask is None else cv2.countNonZero(mask)
        if count < 2:
            raise ValidationError("Focus measurement needs at least two selected pixels")

        p = self.parameters
        method = p["method"]
        options = {"ksize": int(p["aperture"]), "borderType": BORDERS[p["border"]]}
        if method == "laplacian_variance":
            response = cv2.Laplacian(gray, cv2.CV_64F, **options)
            _, stddev = cv2.meanStdDev(response, mask=mask)
            sharpness = float(stddev.item()) ** 2
        else:
            dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, **options)
            dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, **options)
            cv2.multiply(dx, dx, dst=dx)
            cv2.multiply(dy, dy, dst=dy)
            response = cv2.add(dx, dy, dst=dx)
            sharpness = float(cv2.mean(response, mask=mask)[0])
        if not math.isfinite(sharpness):
            raise ValidationError("Focus measurement produced a nonfinite score")

        passed = not p["check_threshold"] or sharpness >= float(p["minimum_sharpness"])
        message = f"{_METHODS[method]} {sharpness:.6g}"
        measurements = {"method": method, "sharpness": sharpness, "selected_pixels": int(count)}
        result = Result(
            {
                "sharpness": sharpness,
                "measurements": measurements,
                "image": source,
                "response": NO_VALUE,
            },
            passed=passed,
            message=message,
            fail_reasons=[] if passed else ["focus_below_minimum"],
            event_data=measurements,
        )
        if p["emit_response"]:
            if method == "laplacian_variance":
                np.abs(response, out=response)
            if mask is not None:
                response[mask == 0] = 0
            result.outputs["response"] = ImageArray(response, "gray")
        if self.generate_graphics:
            color = ACCEPTED if passed else REJECTED
            label = graphic_object("text", ((8, 22),), color, text=message)
            result.drawings["focus measurement"] = GraphicDrawing((label,))
        return result
