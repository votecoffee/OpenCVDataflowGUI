"""OpenCV photo filters with native controls and optional selected writes."""

import cv2
import numpy as np

from .core import Port, Result
from .images import ImageArray, image_format, opencv_image
from .parameter_validation import NumericRange, ValidationError
from .processing import ImageTool, image_value
from .selection import RegionalOperation

_EDGE_METHODS = {
    "recursive": cv2.RECURS_FILTER,
    "normalized_convolution": cv2.NORMCONV_FILTER,
}


class _PhotoFilter(RegionalOperation, ImageTool):
    subtype = "Edge-aware rendering"
    defaults = {"sigma_s": 60.0, "sigma_r": 0.4}
    parameter_ranges = {"sigma_s": NumericRange(0.0001, 200), "sigma_r": NumericRange(0.0001, 1)}
    parameter_help = {
        "sigma_s": "Spatial scale in pixels. Larger values smooth across wider neighborhoods.",
        "sigma_r": "Relative color tolerance. Larger values smooth across stronger edges.",
    }
    input_help = {"image": "8-bit BGR or RGB image at least two pixels wide and high."}
    process_inputs_readonly = True

    def work_image(self, source):
        image_value(source, (np.uint8,))
        if image_format(source) not in ("bgr", "rgb") or min(source.shape[:2]) < 2:
            raise ValidationError("Use a uint8 BGR or RGB image at least 2 × 2 pixels")
        return opencv_image(source)

    def sigmas(self):
        return {name: float(self.parameters[name]) for name in ("sigma_s", "sigma_r")}

    @staticmethod
    def color_output(output, source):
        encoding = image_format(source)
        if encoding == "rgb":
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return ImageArray(output, encoding)


class EdgePreservingPhotoFilter(_PhotoFilter):
    type_id, title = "cv.edge_preserving_photo", "Edge-preserving photo filter"
    defaults = {**_PhotoFilter.defaults, "method": "recursive"}
    parameter_choices = {"method": tuple(_EDGE_METHODS)}
    parameter_help = {**_PhotoFilter.parameter_help, "method": "OpenCV domain-filter mode."}

    def process(self, inputs):
        source = inputs["image"]
        output = cv2.edgePreservingFilter(
            self.work_image(source), flags=_EDGE_METHODS[self.parameters["method"]], **self.sigmas()
        )
        return Result({"image": self.color_output(output, source)})


class DetailEnhance(_PhotoFilter):
    type_id, title = "cv.detail_enhance", "Enhance photo detail"
    defaults = {"sigma_s": 10.0, "sigma_r": 0.15}

    def process(self, inputs):
        source = inputs["image"]
        output = cv2.detailEnhance(self.work_image(source), **self.sigmas())
        return Result({"image": self.color_output(output, source)})


class PencilSketch(_PhotoFilter):
    type_id, title = "cv.pencil_sketch", "Pencil sketch"
    defaults = {"sigma_s": 60.0, "sigma_r": 0.07, "shade_factor": 0.02}
    outputs = (Port("image"), Port("gray"))
    region_outputs = ("image", "gray")
    parameter_ranges = {**_PhotoFilter.parameter_ranges, "shade_factor": NumericRange(0, 0.1)}
    parameter_help = {**_PhotoFilter.parameter_help, "shade_factor": "Sketch brightness scaling."}

    def process(self, inputs):
        source = inputs["image"]
        gray, color = cv2.pencilSketch(
            self.work_image(source), shade_factor=float(self.parameters["shade_factor"]),
            **self.sigmas(),
        )
        return Result({"image": self.color_output(color, source), "gray": gray})


class StylizePhoto(_PhotoFilter):
    type_id, title = "cv.stylize_photo", "Stylize photo"
    defaults = {"sigma_s": 60.0, "sigma_r": 0.45}

    def process(self, inputs):
        source = inputs["image"]
        output = cv2.stylization(self.work_image(source), **self.sigmas())
        return Result({"image": self.color_output(output, source)})


PHOTO_FILTER_BLOCKS = (EdgePreservingPhotoFilter, DetailEnhance, PencilSketch, StylizePhoto)
