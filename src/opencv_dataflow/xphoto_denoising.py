"""Bounded overlapping-patch denoising with OpenCV xphoto."""

import cv2
import numpy as np

from .core import Port, Result
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .processing import ImageTool, image_value
from .selection import selection_mask


class DctDenoise(ImageTool):
    type_id, title, subtype = "cv.dct_denoise", "DCT denoising", "Filters"
    description = "Suppress noise with overlapping discrete cosine transform patches."
    inputs = (
        Port("image"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    defaults = {"noise_sigma": 10.0, "patch_size": 16}
    parameter_ranges = {
        "noise_sigma": NumericRange(0, 255),
        "patch_size": NumericRange(2, 32, whole=True),
    }
    parameter_help = {
        "noise_sigma": "Expected noise standard deviation on a 0–255 scale at any depth. "
        "Zero bypasses filtering.",
        "patch_size": "Even side length of each overlapping patch in pixels. "
        "Both image dimensions must be at least this large.",
    }
    input_help = {
        "image": "Mono or RGB/BGR(A), uint8, uint16, or floating point in [0, 1].",
        "mask": "Nonzero pixels may be changed. Neighboring pixels still inform denoising.",
        "region": "Limit writes to this region. With a mask, write only to their intersection.",
    }
    output_help = {
        "image": "Denoised image in the source depth and encoding, with alpha preserved."
    }
    preview_safe = parallel_safe = process_inputs_readonly = True
    payload_reference = "image"

    def validate_parameters(self, parameters=None):
        parameters = self.parameters if parameters is None else parameters
        super().validate_parameters(parameters)
        if int(parameters["patch_size"]) % 2:
            raise ValidationError("Patch size must be even", "patch_size")

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source, (np.uint8, np.uint16, np.float32, np.float64))
        encoding = image_format(source)
        if encoding not in ("gray", "bgr", "rgb", "bgra", "rgba"):
            raise ValidationError("DCT denoising requires grayscale or RGB/BGR intensity data")
        height, width = image.shape[:2]
        if not height or not width:
            raise ValidationError("DCT denoising requires a nonempty image")
        if image.dtype.kind == "f" and (image.min() < 0 or image.max() > 1):
            raise ValidationError("Floating-point image samples must be in [0, 1]")

        mask = selection_mask(inputs, source)
        sigma = float(self.parameters["noise_sigma"])
        output = ImageArray(image.copy(), encoding, getattr(source, "metadata", None))
        if not sigma or (mask is not None and not cv2.countNonZero(mask)):
            return Result({"image": output}, message="No denoising requested for these pixels")

        patch = int(self.parameters["patch_size"])
        if min(height, width) < patch:
            raise ValidationError(
                "Image width and height must be at least the patch size", "patch_size"
            )
        denoise = getattr(getattr(cv2, "xphoto", None), "dctDenoising", None)
        if denoise is None:
            raise ValidationError("DCT denoising requires OpenCV contrib xphoto")
        full_scale = np.iinfo(image.dtype).max if image.dtype.kind == "u" else 1.0
        sigma *= full_scale / 255

        if mask is None:
            x, y, cols, rows = 0, 0, width, height
        else:
            x, y, cols, rows = cv2.boundingRect(mask)
        # OpenCV allocates a float32 patch per window. Limit patch data to 36 MiB
        # per call and include every window that can contribute to the tile.
        halo = patch - 1
        tile_size = min(256, 3072 // patch - halo)
        color = image[..., :3] if image.ndim == 3 else image
        target = np.asarray(output)[..., :3] if output.ndim == 3 else np.asarray(output)
        reverse = encoding in ("rgb", "rgba")
        for top in range(y, y + rows, tile_size):
            bottom = min(y + rows, top + tile_size)
            for left in range(x, x + cols, tile_size):
                right = min(x + cols, left + tile_size)
                selected = None if mask is None else mask[top:bottom, left:right] != 0
                if selected is not None and not selected.any():
                    continue
                sy, sx = max(0, top - halo), max(0, left - halo)
                crop = color[sy : min(height, bottom + halo), sx : min(width, right + halo)]
                if reverse:
                    crop = crop[..., ::-1]
                # Native iteration omits the last legal start. Its final row
                # and column are uncovered, so give it a disposable border.
                padded = cv2.copyMakeBorder(crop, 0, 1, 0, 1, cv2.BORDER_REPLICATE)
                filtered = np.empty_like(padded)
                denoise(padded, filtered, sigma, patch)
                tile = filtered[top - sy : bottom - sy, left - sx : right - sx]
                if reverse:
                    tile = tile[..., ::-1]
                if image.dtype.kind == "f":
                    np.clip(tile, 0, 1, out=tile)
                destination = target[top:bottom, left:right]
                if selected is None:
                    destination[:] = tile
                else:
                    destination[selected] = tile[selected]
        return Result({"image": output})
