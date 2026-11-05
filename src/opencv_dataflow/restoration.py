"""Native denoising and repair operations with image-space selection."""

import cv2
import numpy as np

from .images import image_format
from .processing import ImageTool, encoded_result, finite, image_value, integer
from .selection import SELECTION_INPUTS, RegionalOperation, selection_mask


class NonLocalMeans(RegionalOperation, ImageTool):
    type_id, title, subtype = "cv.non_local_means", "Non-local means denoise", "Filters"
    defaults = {
        "mode": "channels",
        "strength": 3.0,
        "color_strength": 10.0,
        "norm": "l2",
        "template_window": 7,
        "search_window": 21,
    }
    parameter_choices = {"mode": ("channels", "luminance_chroma"), "norm": ("l2", "l1")}
    parameter_help = {
        "strength": "One value or a list per channel. Use one luminance value in color mode.",
        "norm": "Channel mode only. L1 supports uint16.",
        "color_strength": "Chroma denoising strength in luminance/chroma mode.",
    }

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source, (np.uint8, np.uint16))
        patch = integer(self.parameters["template_window"], "Template window", 1, 63, odd=True)
        window = integer(self.parameters["search_window"], "Search window", patch, 127, odd=True)
        channels = image.shape[2] if image.ndim == 3 else 1
        strengths = np.asarray(self.parameters["strength"], np.float32).reshape(-1)
        if (
            len(strengths) not in (1, channels)
            or not np.isfinite(strengths).all()
            or (strengths < 0).any()
        ):
            raise ValueError("Strength must be one nonnegative value or one per channel")
        if self.parameters["mode"] == "channels":
            norm = cv2.NORM_L1 if self.parameters["norm"] == "l1" else cv2.NORM_L2
            if image.dtype == np.uint16 and norm != cv2.NORM_L1:
                raise ValueError("uint16 denoising requires the L1 norm")
            output = cv2.fastNlMeansDenoising(image, strengths.tolist(), None, patch, window, norm)
        else:
            encoding = image_format(source)
            if len(strengths) != 1 or image.dtype != np.uint8 or encoding not in ("bgr", "rgb"):
                raise ValueError("Luminance/chroma mode requires uint8 BGR/RGB and one strength")
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if encoding == "rgb" else image
            color = finite(self.parameters["color_strength"], "Color strength", 0)
            output = cv2.fastNlMeansDenoisingColored(
                bgr, None, float(strengths[0]), color, patch, window
            )
            if encoding == "rgb":
                output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return encoded_result(output, source)


class Inpaint(ImageTool):
    type_id, title, subtype = "cv.inpaint", "Inpaint selected pixels", "Filters"
    inputs = SELECTION_INPUTS
    defaults = {"method": "telea", "radius": 3.0}
    parameter_choices = {"method": ("telea", "navier_stokes")}

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source, (np.uint8, np.uint16, np.float32))
        if image_format(source) not in ("gray", "bgr", "rgb", "bgra", "rgba"):
            raise ValueError("Inpainting accepts monochrome, BGR, RGB, BGRA, or RGBA")
        mask = selection_mask(inputs, image)
        if mask is None:
            raise ValueError("Connect an inpainting mask or region")
        if np.all(mask):
            raise ValueError("Inpainting needs unselected neighboring pixels")
        radius = finite(self.parameters["radius"], "Radius", 1)
        if radius > 100:
            raise ValueError("Inpainting radius must be at most 100 pixels")
        method = cv2.INPAINT_TELEA if self.parameters["method"] == "telea" else cv2.INPAINT_NS
        if image.ndim == 2:
            return encoded_result(cv2.inpaint(image, mask, radius, method), source)
        colors = image[..., :3]
        if image.dtype == np.uint8:
            output = cv2.inpaint(colors, mask, radius, method)
        else:
            output = cv2.merge(
                [cv2.inpaint(channel, mask, radius, method) for channel in cv2.split(colors)]
            )
        if image.shape[2] == 4:
            output = np.concatenate((output, image[..., 3:]), axis=2)
        return encoded_result(output, source)


RESTORATION_BLOCKS = (NonLocalMeans, Inpaint)
