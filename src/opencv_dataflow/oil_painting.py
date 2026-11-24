"""Local modal histogram smoothing with OpenCV xphoto."""

import cv2
import numpy as np

from .core import Port, Result
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import metadata
from .processing import ImageTool, image_value
from .selection import selection_mask

_HISTOGRAM_PLANES = {
    "grayscale": cv2.COLOR_BGR2GRAY,
    "lab_lightness": cv2.COLOR_BGR2Lab,
    "hsv_hue": cv2.COLOR_BGR2HSV,
}
_CHANNELS = {"gray": 1, "bgr": 3, "rgb": 3, "bgra": 4, "rgba": 4}


def smooth_color_plane(bgr, radius, ratio, code):
    # The pinned xphoto build's multichannel conversion yields unstable bins.
    # Compute the requested plane separately and accumulate its local color means.
    if bgr.shape[0] * bgr.shape[1] > 8_000_000:
        raise ValidationError("Restrict color-plane smoothing to 8 million pixels")
    plane = cv2.cvtColor(bgr, code)[..., 0]
    keys = np.rint(plane.astype(np.float64) * (1.0 / ratio)).astype(np.uint8)
    bins = np.unique(keys)
    if keys.size * len(bins) > 90_000_000:
        raise ValidationError(
            "Use a smaller selection or larger bin divisor for color-plane smoothing",
            "dynamic_ratio",
        )
    side = 2 * radius + 1
    settings = {
        "ddepth": cv2.CV_32S,
        "ksize": (side, side),
        "normalize": False,
        "borderType": cv2.BORDER_CONSTANT,
    }
    best = np.zeros(keys.shape, np.int32)
    output = np.empty_like(bgr)
    for level in bins:
        members = (keys == level).astype(np.uint8)
        count = cv2.boxFilter(members, **settings)
        replace = count > best  # In ties, retain the lower histogram bin.
        if not replace.any():
            continue
        totals = cv2.boxFilter(bgr * members[..., None], **settings)
        output[replace] = np.rint(totals[replace] / count[replace, None])
        best[replace] = count[replace]
    return output


class OilPaintingFilter(ImageTool):
    type_id, title, subtype = "cv.oil_painting", "Oil painting texture filter", "Filters"
    description = "Reduce local texture by averaging pixels in the most common histogram bin."
    inputs = (
        Port("image"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    defaults = {"radius": 4, "dynamic_ratio": 12, "histogram_plane": "grayscale"}
    parameter_choices = {"histogram_plane": tuple(_HISTOGRAM_PLANES)}
    parameter_ranges = {
        "radius": NumericRange(1, 127, whole=True),
        "dynamic_ratio": NumericRange(1, 127, whole=True),
    }
    parameter_help = {
        "radius": "Neighborhood half-width in pixels. Radius 4 gives a 9 × 9 interior window.",
        "dynamic_ratio": "OpenCV histogram bin divisor. Larger values group more intensities.",
        "histogram_plane": "Color grouping uses gray intensity, Lab lightness or linear HSV hue. "
        "Monochrome always uses intensity.",
    }
    input_help = {
        "image": "8-bit gray, RGB or BGR. Straight alpha, when present, is preserved.",
        "mask": "Only nonzero pixels may change. Neighbors outside the mask still contribute.",
        "region": "Limit writes to a region. With a mask, write only to their intersection.",
    }
    preview_safe = parallel_safe = process_inputs_readonly = True
    payload_reference = "image"

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source, (np.uint8,))
        encoding = image_format(source)
        channels = 1 if image.ndim == 2 else image.shape[2]
        if not image.size or channels != _CHANNELS.get(encoding):
            raise ValidationError("Use a nonempty uint8 gray or BGR/RGB image")
        if channels == 4 and metadata(source).get("alpha_representation", "straight") != "straight":
            raise ValidationError("Unassociate premultiplied alpha before texture smoothing")

        selected = selection_mask(inputs, source)
        height, width = image.shape[:2]
        radius = int(self.parameters["radius"])
        x, y, cols, rows = (0, 0, width, height) if selected is None else cv2.boundingRect(selected)
        if not cols or not rows:
            return Result({"image": ImageArray(image.copy(), encoding)}, message="Empty selection")
        # Include neighboring samples at the selection boundary and clip only at image edges.
        left, top = max(0, x - radius), max(0, y - radius)
        right, bottom = min(width, x + cols + radius), min(height, y + rows + radius)
        area = (right - left) * (bottom - top)
        if area > 16_000_000 or area * min(bottom - top, 2 * radius + 1) > 300_000_000:
            raise ValidationError(
                "Reduce the radius or restrict smoothing to a smaller region", "radius"
            )

        patch = image[top:bottom, left:right]
        if patch.ndim == 3:
            patch = patch[..., :3]
        swap_channels = encoding in ("rgb", "rgba")
        if swap_channels:
            patch = patch[..., ::-1]
        patch = np.ascontiguousarray(patch)
        ratio = int(self.parameters["dynamic_ratio"])
        code = _HISTOGRAM_PLANES[self.parameters["histogram_plane"]]
        if patch.ndim == 2 or code == cv2.COLOR_BGR2GRAY:
            native = getattr(getattr(cv2, "xphoto", None), "oilPainting", None)
            if native is None:
                raise ValidationError("Oil painting texture smoothing requires OpenCV contrib xphoto")
            filtered = native(patch, radius, ratio, code=cv2.COLOR_BGR2GRAY)
        else:
            filtered = smooth_color_plane(patch, radius, ratio, code)
        if swap_channels:
            filtered = filtered[..., ::-1]
        if selected is None and channels != 4:
            return Result({"image": ImageArray(filtered, encoding)})

        output = image.copy()
        target = output[y : y + rows, x : x + cols]
        if target.ndim == 3:
            target = target[..., :3]
        core = filtered[y - top : y - top + rows, x - left : x - left + cols]
        if selected is None:
            target[:] = core
        else:
            keep = selected[y : y + rows, x : x + cols] != 0
            target[keep] = core[keep]
        return Result({"image": ImageArray(output, encoding)})
