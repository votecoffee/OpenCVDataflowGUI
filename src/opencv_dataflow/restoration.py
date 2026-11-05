"""Native denoising and repair operations with image-space selection."""

import cv2
import numpy as np

from .core import Port
from .images import convert_depth, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import require_same_frame
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


class GuidedFilter(ImageTool):
    type_id, title, subtype = "cv.guided_filter", "Guided filter", "Filters"
    inputs = (*SELECTION_INPUTS, Port("guide", required=False))
    defaults = {"radius": 8, "epsilon": 0.001, "scale": 1.0, "output_depth": "unchanged"}
    parameter_choices = {"output_depth": ("unchanged", "uint8", "uint16", "float32")}
    parameter_ranges = {
        "radius": NumericRange(1, 4095, whole=True),
        "epsilon": NumericRange(1e-8, 1e8),
        "scale": NumericRange(0.01, 1),
    }
    parameter_help = {
        "radius": "Neighborhood radius in pixels.",
        "epsilon": "Regularization in squared guide units. Integer guide samples are normalized.",
        "scale": "Internal subsampling fraction. One filters at full resolution.",
    }
    input_help = {"guide": "Optional aligned guide. Otherwise the block guides with the source."}

    def process(self, inputs):
        source = inputs["image"]
        guide = inputs.get("guide", source)
        image_value(source, (np.uint8, np.uint16, np.float32))
        image_value(guide, (np.uint8, np.uint16, np.float32))
        if guide.shape[:2] != source.shape[:2]:
            raise ValidationError("Guide must have the same dimensions as the source")
        try:
            require_same_frame((source, guide))
        except ValueError as exc:
            raise ValidationError("Guide must share the source pixel frame") from exc
        if image_format(source) in ("hsv", "lab") or image_format(guide) in ("hsv", "lab"):
            raise ValidationError("Use grayscale or RGB/BGR intensity data for guided filtering")

        p = self.parameters
        scale = float(p["scale"])
        if scale * min(source.shape[:2]) < 1:
            raise ValidationError("Scale is too small for the image dimensions")
        work = convert_depth(source, "float32")
        reference = work if guide is source else convert_depth(guide, "float32")
        has_alpha = work.ndim == 3 and work.shape[2] == 4
        signal = work[..., :3] if has_alpha else work
        if reference.ndim == 3:
            reference = reference[..., :3]
        result = cv2.ximgproc.guidedFilter(
            reference, signal, int(p["radius"]), float(p["epsilon"]), scale=scale
        )
        if not np.isfinite(result).all():
            raise ValidationError("Guided filter produced nonfinite samples. Check the input range")
        if has_alpha:
            result = np.concatenate((result, work[..., 3:]), axis=2)
        mask = selection_mask(inputs, source)
        if mask is not None:
            result[mask == 0] = work[mask == 0]
        depth = source.dtype.name if p["output_depth"] == "unchanged" else p["output_depth"]
        return encoded_result(convert_depth(result, depth), source)


RESTORATION_BLOCKS = (NonLocalMeans, Inpaint, GuidedFilter)
