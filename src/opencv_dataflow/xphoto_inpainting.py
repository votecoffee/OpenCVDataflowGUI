"""Masked xphoto restoration with explicit synthesized-pixel provenance."""

import cv2
import numpy as np

from .core import Port, Result
from .images import ImageArray, convert_depth, image_format, opencv_image
from .parameter_validation import ValidationError
from .processing import ImageTool, image_value
from .selection import selection_mask

_METHODS = {
    "fsr_fast": "INPAINT_FSR_FAST",
    "fsr_best": "INPAINT_FSR_BEST",
    "shift_map": "INPAINT_SHIFTMAP",
}
_MAX_SHIFTMAP_PIXELS = 800 * 600
_MAX_SHIFTMAP_GRAPH_PIXELS = 40_000


def _check_shiftmap(known):
    # Keep the native patch search at source resolution and bound its label graph.
    if min(known.shape) < 32 or known.size > _MAX_SHIFTMAP_PIXELS:
        raise ValidationError("ShiftMap needs a crop at least 32 × 32 and at most 480,000 pixels")
    patch = np.ones((8, 8), np.uint8)
    donors = cv2.erode(known, patch, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    if not cv2.countNonZero(donors):
        raise ValidationError("ShiftMap needs at least one fully known 8 × 8 patch")
    # Upstream adds a four-pixel halo to each missing pixel for graph stitching.
    expanded = cv2.erode(known, np.ones((9, 9), np.uint8))
    if known.size - cv2.countNonZero(expanded) > _MAX_SHIFTMAP_GRAPH_PIXELS:
        raise ValidationError("ShiftMap selection and its four-pixel halo exceed 40,000 pixels")


class ExtendedInpaint(ImageTool):
    type_id, title, subtype = "cv.extended_inpaint", "Extended inpainting", "Filters"
    description = "Fill selected defects using OpenCV frequency or patch reconstruction."
    inputs = (
        Port("image"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (Port("image"), Port("repair_mask"))
    defaults = {"method": "fsr_fast"}
    parameter_choices = {"method": tuple(_METHODS)}
    parameter_help = {
        "method": "FSR reconstructs frequency content. Shift Map searches for similar patches.",
    }
    input_help = {
        "image": "Mono or RGB/BGR(A). Unsigned 8/16-bit or floating point in [0, 1].",
        "mask": "Nonzero pixels are repaired. The rest provide measured context.",
        "region": "Select pixels to repair. With a mask, only their intersection is repaired.",
    }
    output_help = {
        "image": "Reconstructed pixels in the source depth and encoding.",
        "repair_mask": "White pixels identify estimates created by inpainting.",
    }
    preview_safe = parallel_safe = process_inputs_readonly = True
    payload_reference = "image"

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source, (np.uint8, np.uint16, np.float32, np.float64))
        encoding = image_format(source)
        if encoding not in ("gray", "bgr", "rgb", "bgra", "rgba"):
            raise ValidationError(
                "Extended inpainting requires grayscale or RGB/BGR intensity data"
            )
        mask = selection_mask(inputs, source)
        if mask is None:
            raise ValidationError("Connect an inpainting mask or region")
        count = cv2.countNonZero(mask)
        if not count:
            return Result(
                {
                    "image": ImageArray(image.copy(), encoding),
                    "repair_mask": ImageArray(mask, "gray"),
                },
                message="No selected pixels to repair",
            )
        if count == mask.size:
            raise ValidationError("Inpainting needs known pixels outside the selected area")

        method = self.parameters["method"]
        xphoto = getattr(cv2, "xphoto", None)
        flag = getattr(xphoto, _METHODS[method], None)
        if flag is None or not hasattr(xphoto, "inpaint"):
            raise ValidationError("The selected method requires OpenCV contrib xphoto inpainting")
        known = cv2.bitwise_not(mask)
        if method == "shift_map":
            _check_shiftmap(known)

        canonical = opencv_image(source)
        colors = canonical[..., :3] if canonical.ndim == 3 else canonical
        signal = np.array(colors, copy=True, order="C")
        selected = mask != 0
        signal[selected] = 0
        if signal.dtype.kind == "f" and (float(signal.min()) < 0 or float(signal.max()) > 1):
            raise ValidationError("Known floating-point intensity samples must be in [0, 1]")

        fsr = method != "shift_map"
        # Frequency reconstruction always writes a byte destination in OpenCV.
        working = convert_depth(signal, "uint8") if fsr and image.dtype != np.uint8 else signal
        reconstructed = np.zeros_like(working)
        cv2.xphoto.inpaint(working, known, reconstructed, flag)
        values = reconstructed[selected]
        if not np.isfinite(values).all():
            raise ValidationError("OpenCV produced nonfinite repaired pixels")
        if fsr and image.dtype == np.float64:
            values = values.astype(np.float64) / 255
        elif fsr and image.dtype != np.uint8:
            values = convert_depth(values, image.dtype.name)
        if encoding in ("rgb", "rgba"):
            values = values[:, ::-1]
        output = image.copy()
        target = output[..., :3] if output.ndim == 3 else output
        target[selected] = values
        return Result(
            {
                "image": ImageArray(output, encoding),
                "repair_mask": ImageArray(mask, "gray"),
            },
            message=f"Estimated {count:,} selected pixels",
        )
