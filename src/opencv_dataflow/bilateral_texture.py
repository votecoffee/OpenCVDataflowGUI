"""OpenCV bilateral texture filtering with source-grid write selection."""

import math

import cv2
import numpy as np

from .core import Port
from .images import convert_depth, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import metadata
from .processing import ImageTool, encoded_result, image_value
from .selection import selection_mask

MAX_PIXELS = 2_097_152
MAX_NEIGHBOR_WORK = 360_000_000
FLOAT_MAX = float(np.finfo(np.float32).max)
MIN_SIGMA_AVG = 1 / math.sqrt(FLOAT_MAX)
DEPTHS = (np.uint8, np.uint16, np.float32, np.float64)
CHANNELS = {"gray": 1, "bgr": 3, "rgb": 3, "bgra": 4, "rgba": 4}


class BilateralTextureFilter(ImageTool):
    type_id, title, subtype = "cv.bilateral_texture", "Bilateral texture smoothing", "Filters"
    description = "Reduce surface texture while preserving structural edges with OpenCV's native filter."
    inputs = (
        Port("image"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    defaults = {"radius": 3, "iterations": 1, "sigma_alpha": -1.0, "sigma_avg": -1.0}
    parameter_ranges = {
        "radius": NumericRange(1, 63, whole=True),
        "iterations": NumericRange(1, 50, whole=True),
        "sigma_alpha": NumericRange(maximum=FLOAT_MAX),
        "sigma_avg": NumericRange(),
    }
    parameter_help = {
        "radius": "Native fine-window radius. The final bilateral window uses twice this radius.",
        "iterations": "Number of complete filtering passes.",
        "sigma_alpha": "Edge-to-texture transition sharpness. Negative uses automatic 5 × radius.",
        "sigma_avg": "Range sigma in unit intensity. Larger values admit greater color differences. "
                     "Negative uses the native automatic value.",
    }
    input_help = {
        "image": "Gray or RGB/BGR(A), uint8, uint16 or float in [0,1]. Straight alpha is preserved.",
        "mask": "Only selected pixels are replaced. Surrounding source pixels still contribute.",
        "region": "Intersect this source-grid region with the write mask.",
    }
    output_help = {"image": "Filtered image in the source depth, color encoding, and coordinate frame."}
    payload_reference = "image"
    preview_safe = parallel_safe = process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if 0 <= float(p["sigma_avg"]) < MIN_SIGMA_AVG:
            raise ValidationError(
                f"Sigma avg must be negative for automatic mode or at least {MIN_SIGMA_AVG:.3g}",
                "sigma_avg",
            )

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source, DEPTHS)
        encoding = image_format(source)
        channels = 1 if image.ndim == 2 else image.shape[2]
        if not image.size or CHANNELS.get(encoding) != channels:
            raise ValidationError("Use a nonempty gray or RGB/BGR(A) image", "image")
        if channels == 4 and metadata(source).get("alpha_representation", "straight") != "straight":
            raise ValidationError("Unassociate premultiplied alpha before texture smoothing", "image")

        selected = selection_mask(inputs, source)
        height, width = image.shape[:2]
        x, y, cols, rows = (0, 0, width, height) if selected is None else cv2.boundingRect(selected)
        if not cols or not rows:
            return encoded_result(image.copy(), source, message="Empty selection")
        p = self.parameters
        radius, iterations = int(p["radius"]), int(p["iterations"])
        # Per pass, the gradient reads one extra pixel, the guide depends on
        # two radius-r neighborhoods, and the final bilateral radius is 2r.
        halo = (4 * radius + 1) * iterations
        left, top = max(0, x - halo), max(0, y - halo)
        right, bottom = min(width, x + cols + halo), min(height, y + rows + halo)
        area = (right - left) * (bottom - top)
        work = area * (4 * radius + 1)**2 * iterations * min(channels, 3)
        if area > MAX_PIXELS or work > MAX_NEIGHBOR_WORK:
            raise ValidationError(
                "Texture filtering exceeds the working limit. Reduce radius, iterations, or region.",
                "radius", "iterations",
            )
        patch = image[top:bottom, left:right]
        if channels == 4:
            patch = patch[..., :3]
        filtered = self.filter_patch(patch, radius, iterations)
        if selected is None and channels != 4:
            return encoded_result(filtered, source)

        result = image.copy()
        target = result[y:y + rows, x:x + cols]
        if channels == 4:
            target = target[..., :3]
        dy, dx = y - top, x - left
        core = filtered[dy:dy + rows, dx:dx + cols]
        if selected is None:
            target[:] = core
        else:
            write = selected[y:y + rows, x:x + cols] != 0
            target[write] = core[write]
        return encoded_result(result, source)

    def filter_patch(self, patch, radius, iterations):
        p = self.parameters
        if patch.dtype.kind == "f" and (patch.min() < 0 or patch.max() > 1):
            raise ValidationError("Floating intensity values must be in [0,1]", "image")
        native = getattr(getattr(cv2, "ximgproc", None), "bilateralTextureFilter", None)
        if native is None:
            raise ValidationError("Bilateral texture smoothing requires OpenCV contrib ximgproc")
        dtype = patch.dtype
        work = patch if dtype in (np.uint8, np.float32) else convert_depth(patch, "float32")
        filtered = native(
            np.ascontiguousarray(work), fr=radius, numIter=iterations,
            sigmaAlpha=float(p["sigma_alpha"]), sigmaAvg=float(p["sigma_avg"]),
        )
        if (
            filtered is None or filtered.shape != work.shape or filtered.dtype != work.dtype
            or not np.isfinite(filtered).all()
        ):
            raise ValidationError("OpenCV did not return a finite filtered image")
        if dtype == np.uint16:
            return convert_depth(filtered, "uint16")
        return filtered.astype(dtype, copy=False)
