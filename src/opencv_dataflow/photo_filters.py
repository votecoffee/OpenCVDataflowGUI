"""OpenCV photo filters and masked gradient-domain appearance edits."""

import copy
from abc import ABC, abstractmethod

import cv2
import numpy as np

from .core import Port, Result
from .graphics_layers import GraphicDrawing, graphic_box, graphic_object
from .images import ImageArray, image_format, opencv_image
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, metadata
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


def _photo_color_input(source):
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
    return samples, encoding, incoming


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
        samples, encoding, incoming = _photo_color_input(source)
        height, width = samples.shape[:2]
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


class ColorChange(_MaskedPhotoCorrection):
    type_id, title = "cv.color_change", "Change selected color"
    subtype = "Color and depth"
    native_binding = "cv2.colorChange"
    defaults = {"red_mul": 1.0, "green_mul": 1.0, "blue_mul": 1.0}
    parameter_ranges = {name: NumericRange(0.5, 2.5) for name in defaults}
    parameter_help = {
        "red_mul": "Native red gradient multiplier (0.5..2.5), rather than direct pixel scaling.",
        "green_mul": "Native green gradient multiplier (0.5..2.5).",
        "blue_mul": "Native blue gradient multiplier (0.5..2.5).",
    }

    def correct(self, image, mask):
        return cv2.colorChange(image, mask, **{k: float(v) for k, v in self.parameters.items()})


class Decolor(ImageTool):
    type_id, title, subtype = "cv.decolor", "Contrast-preserving grayscale", "Color and depth"
    inputs = SELECTION_INPUTS
    outputs = (
        Port("image"),
        Port("color_boost"),
        Port("ordinary_gray"),
        Port("selected_mask", visual_type="region"),
        Port("change"),
        Port("report", "data"),
    )
    process_inputs_readonly = parallel_safe = True
    input_help = {
        **_MaskedPhotoCorrection.input_help,
        "mask": "Write selection only: the whole color image fits the native decolor mapping.",
        "region": "Intersect with mask. Outside: ordinary gray and original color boost samples.",
    }

    def process(self, inputs):
        source = inputs["image"]
        samples, encoding, incoming = _photo_color_input(source)
        color = samples[..., :3]
        native = np.ascontiguousarray(color[..., ::-1] if encoding.startswith("rgb") else color)
        ordinary = cv2.cvtColor(native, cv2.COLOR_BGR2GRAY)
        mask = selection_mask(inputs, source)
        if mask is None:
            mask = np.full(samples.shape[:2], 255, np.uint8)
        selected = mask != 0
        count = int(np.count_nonzero(selected))
        uniform = not np.ptp(native.reshape(-1, 3), axis=0).any()
        achromatic = np.array_equal(native[..., 0], native[..., 1]) and np.array_equal(
            native[..., 1], native[..., 2]
        )
        skip = (
            "empty_selection"
            if not count
            else "uniform_color"
            if uniform
            else "achromatic_input"
            if achromatic
            else None
        )
        gray, boost = ordinary.copy(), samples.copy()
        native_range = None
        passed = True
        if skip is None:
            solved, enhanced = cv2.decolor(native)
            native_range = int(np.ptp(solved))
            passed = native_range > 0
            if passed:
                gray[selected] = solved[selected]
                if encoding.startswith("rgb"):
                    enhanced = enhanced[..., ::-1]
                boost[..., :3][selected] = enhanced[selected]
        change = cv2.absdiff(gray, ordinary) if passed else NO_VALUE
        report = {
            "native_binding": "cv2.decolor",
            "mapping_context": "full_image",
            "selected_pixels": count,
            "native_gray_range": native_range,
            "mean_absolute_change": float(change[selected].mean()) if passed and count else None,
            "outside_gray": "ordinary_grayscale",
            "outside_color_boost": "original_samples",
            "skip_reason": skip,
            "status": "ok" if passed else "degenerate_native_mapping",
            "passed": passed,
        }
        outputs = {
            "image": ImageArray(gray, "gray") if passed else NO_VALUE,
            "color_boost": ImageArray(boost, encoding) if passed else NO_VALUE,
            "ordinary_gray": ImageArray(ordinary, "gray"),
            "selected_mask": mask,
            "change": change,
            "report": report,
        }
        payloads = {key: copy.deepcopy(incoming) for key in outputs}
        for key in outputs.keys() - {"color_boost"}:
            payloads[key].pop("alpha_representation", None)
            payloads[key].pop("raw_sensor", None)
        drawings = {}
        if self.generate_graphics and count:
            x, y, w, h = cv2.boundingRect(mask)
            color = (60, 210, 120) if passed else (60, 160, 240)
            drawings["decolor write bounds"] = GraphicDrawing((graphic_box(x, y, w, h, color),))
        return Result(
            outputs,
            payloads=payloads,
            drawings=drawings,
            passed=passed,
            message=(
                f"Unchanged: {skip}"
                if skip
                else "Decolor mapping fitted on full image"
                if passed
                else "Native decolor collapsed nonuniform color; output withheld"
            ),
        )


PHOTO_FILTER_BLOCKS = (
    EdgePreservingPhotoFilter,
    DetailEnhance,
    PencilSketch,
    StylizePhoto,
    IlluminationChange,
    TextureFlattening,
    ColorChange,
    Decolor,
)
