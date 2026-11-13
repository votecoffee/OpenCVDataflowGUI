"""OpenCV photo filters and masked gradient-domain appearance edits."""

import copy
from abc import ABC, abstractmethod

import cv2
import numpy as np

from .core import Port, Result
from .graphics_layers import GraphicDrawing, graphic_box, graphic_object
from .images import ImageArray, image_format, opencv_image
from .parameter_validation import NumericRange, ValidationError
from .payloads import metadata
from .processing import ImageTool, image_value
from .selection import SELECTION_INPUTS, RegionalOperation, selection_mask

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
            self.work_image(source),
            shade_factor=float(self.parameters["shade_factor"]),
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


class _MaskedPhotoCorrection(ImageTool, ABC):
    """Use the selection in the native solve and restrict writes to that selection."""

    subtype = "Illumination and texture"
    inputs = SELECTION_INPUTS
    outputs = (
        Port("image"),
        Port("selected_mask", visual_type="region"),
        Port("gradient_mask", visual_type="region"),
        Port("change"),
        Port("report", "data"),
    )
    process_inputs_readonly = parallel_safe = True
    input_help = {
        "image": "uint8 BGR/RGB, optionally with straight alpha; at least 8×8, up to 2.1M pixels.",
        "mask": "Optional uint8 binary selection in the image frame; intersected with region.",
        "region": "Optional region on the same grid. No mask/region means the full image.",
    }

    @abstractmethod
    def correct(self, image, mask):
        """Return a native BGR correction without changing the caller's inputs."""

    def process(self, inputs):
        source = inputs["image"]
        samples = image_value(source, (np.uint8,))
        encoding = image_format(source)
        if (
            encoding not in ("bgr", "rgb", "bgra", "rgba")
            or samples.ndim != 3
            or samples.shape[2] != (4 if encoding.endswith("a") else 3)
        ):
            raise ValidationError("Use uint8 BGR/RGB color, optionally with straight alpha")
        height, width = samples.shape[:2]
        if min(height, width) < 8 or height * width > 2_100_000:
            raise ValidationError("Photo correction requires at least 8×8 and at most 2.1M pixels")
        incoming = metadata(source)
        if samples.shape[2] == 4 and incoming.get("alpha_representation", "straight") != "straight":
            raise ValidationError("Unassociate premultiplied alpha before photo correction")
        mask = selection_mask(inputs, source)
        if mask is None:
            mask = np.full((height, width), 255, np.uint8)
        # The native implementation erodes its private mask by three 3x3 passes.
        gradient_mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=3)
        selected = mask != 0
        selected_pixels = int(np.count_nonzero(selected))
        gradient_pixels = int(np.count_nonzero(gradient_mask))
        skip = (
            "empty_selection"
            if not selected_pixels
            else "no_gradient_interior"
            if not gradient_pixels
            else None
        )
        output = samples.copy()
        if skip is None:
            color = samples[..., :3]
            native = np.ascontiguousarray(color[..., ::-1] if encoding.startswith("rgb") else color)
            edited = self.correct(native, mask.copy())
            if encoding.startswith("rgb"):
                edited = edited[..., ::-1]
            output[..., :3][selected] = edited[selected]
        difference = np.abs(output[..., :3].astype(np.int16) - samples[..., :3])
        change = difference.max(axis=2).astype(np.uint8)
        mean_change = float(difference[selected].mean()) if selected_pixels else 0.0
        report = {
            "native_binding": self.native_binding,
            "parameters": copy.deepcopy(self.parameters),
            "selected_pixels": selected_pixels,
            "gradient_pixels": gradient_pixels,
            "changed_pixels": int(np.count_nonzero(change)),
            "mean_absolute_change": mean_change,
            "skip_reason": skip,
            "channel_order": encoding,
            "native_mask_erosion_pixels": 3,
            "outside_selection": "original_samples",
            "photometric_calibration": False,
        }
        outputs = {
            "image": ImageArray(output, encoding),
            "selected_mask": mask,
            "gradient_mask": gradient_mask,
            "change": change,
            "report": report,
        }
        payloads = {key: copy.deepcopy(incoming) for key in outputs}
        for key in ("selected_mask", "gradient_mask", "change", "report"):
            payloads[key].pop("alpha_representation", None)
            payloads[key].pop("raw_sensor", None)
        drawings = {}
        if self.generate_graphics and selected_pixels:
            x, y, w, h = cv2.boundingRect(mask)
            drawings["photo edit bounds"] = GraphicDrawing(
                (
                    graphic_box(x, y, w, h, (60, 210, 120)),
                    graphic_object(
                        "text",
                        [(x, y)],
                        (60, 210, 120),
                        text=f"Edit bounds: mean change {mean_change:.2f} DN",
                    ),
                )
            )
        return Result(
            outputs,
            payloads=payloads,
            drawings=drawings,
            message=(f"Unchanged: {skip}" if skip else f"Edited {selected_pixels} selected pixels"),
        )


class IlluminationChange(_MaskedPhotoCorrection):
    type_id, title = "cv.illumination_change", "Change selected illumination"
    native_binding = "cv2.illuminationChange"
    defaults = {"alpha": 0.2, "beta": 0.4}
    parameter_ranges = {"alpha": NumericRange(0, 2), "beta": NumericRange(0, 2)}
    parameter_help = {
        "alpha": "Native gradient scale (0..2), not a direct brightness gain.",
        "beta": "Native gradient exponent (0..2). Alters apparent illumination in the selection.",
    }

    def correct(self, image, mask):
        return cv2.illuminationChange(
            image, mask, alpha=float(self.parameters["alpha"]), beta=float(self.parameters["beta"])
        )


class TextureFlattening(_MaskedPhotoCorrection):
    type_id, title = "cv.texture_flattening", "Flatten selected texture"
    native_binding = "cv2.textureFlattening"
    defaults = {"low_threshold": 30.0, "high_threshold": 45.0, "kernel_size": 3}
    parameter_ranges = {
        "low_threshold": NumericRange(0, 65535),
        "high_threshold": NumericRange(0, 65535),
    }
    parameter_choices = {"kernel_size": ("3", "5", "7")}
    parameter_help = {
        "low_threshold": "Native Canny low gradient threshold; no greater than high_threshold.",
        "high_threshold": "Canny high threshold; retain strong edges and suppress weaker texture.",
        "kernel_size": "Native Sobel/Canny aperture. Threshold sensitivity changes with aperture.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["low_threshold"] > p["high_threshold"]:
            raise ValidationError(
                "Low threshold must not exceed high threshold", "low_threshold", "high_threshold"
            )

    def correct(self, image, mask):
        p = self.parameters
        return cv2.textureFlattening(
            image,
            mask,
            low_threshold=float(p["low_threshold"]),
            high_threshold=float(p["high_threshold"]),
            kernel_size=int(p["kernel_size"]),
        )


PHOTO_FILTER_BLOCKS = (
    EdgePreservingPhotoFilter,
    DetailEnhance,
    PencilSketch,
    StylizePhoto,
    IlluminationChange,
    TextureFlattening,
)
