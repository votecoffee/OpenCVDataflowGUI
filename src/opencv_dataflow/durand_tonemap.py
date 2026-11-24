"""Local HDR display mapping with a bilateral log-luminance base."""

import cv2
import numpy as np

from .core import Port, Result
from .images import ImageArray, convert_depth, image_format
from .parameter_validation import NumericRange, ValidationError
from .processing import ImageTool, image_value


def portable_tonemap(rgb, maximum, contrast, saturation, sigma_color, sigma_space):
    relative = rgb / maximum
    luminance = cv2.cvtColor(relative, cv2.COLOR_RGB2GRAY)
    positive = luminance > 0
    floor = np.finfo(np.float32).tiny
    log_luminance = np.log2(np.maximum(luminance, floor))
    base = cv2.bilateralFilter(log_luminance, -1, sigma_color, sigma_space)
    # Exclude black from the base range since log(0) has no finite value.
    lit_base = base[positive]
    span = float(lit_base.max() - lit_base.min())
    scale = contrast / span if span > contrast else 1.0
    display_log = (base - lit_base.max()) * scale + (log_luminance - base)
    display_log -= display_log[positive].max()
    display = np.exp2(np.minimum(display_log, 0))
    display[~positive] = 0
    ratios = relative / np.maximum(luminance, floor)[..., None]
    return np.power(ratios, saturation) * display[..., None]


def native_tonemap(contrast, saturation, sigma_color, sigma_space):
    factory = getattr(getattr(cv2, "xphoto", None), "createTonemapDurand", None)
    if factory is not None:
        try:
            return factory(1.0, contrast, saturation, sigma_color, sigma_space)
        except cv2.error as exc:
            if exc.code != -213:
                raise
    raise ValidationError(
        "Native Durand tone mapping requires an OpenCV contrib build with "
        "OPENCV_ENABLE_NONFREE. Choose the portable backend on the stock wheel.",
        "backend",
    )


class ToneMapDurand(ImageTool):
    type_id, title, subtype = "cv.tonemap_durand", "Tone map HDR (Durand)", "Exposure stacks"
    outputs = (Port("image"), Port("info", "data", required=False))
    defaults = {
        "backend": "portable",
        "contrast": 4.0,
        "saturation": 1.0,
        "sigma_color": 2.0,
        "sigma_space": 2.0,
        "gamma": 1.0,
        "output_depth": "uint8",
    }
    parameter_choices = {"backend": ("portable", "opencv"), "output_depth": ("uint8", "float32")}
    parameter_ranges = {
        "contrast": NumericRange(0.1, 20),
        "saturation": NumericRange(0.01, 5),
        "sigma_color": NumericRange(0.01, 16),
        "sigma_space": NumericRange(0.1, 16),
        "gamma": NumericRange(0.1, 8),
    }
    parameter_help = {
        "backend": "Portable log-bilateral mapping, or Durand from an enabled OpenCV contrib build.",
        "contrast": "Base contrast limit in stops for portable. Target natural-log span for OpenCV.",
        "saturation": "Exponent applied to channel-to-luminance ratios. One preserves the ratios.",
        "sigma_color": "Bilateral range sigma. Stops for portable, natural-log units for OpenCV.",
        "sigma_space": "Bilateral spatial sigma in pixels.",
        "gamma": "Display gamma. Values above one brighten midtones.",
        "output_depth": "8-bit display image or float32 display samples in [0, 1].",
    }
    input_help = {"image": "Nonnegative float32 gray, BGR or RGB radiance for the whole frame."}
    process_inputs_readonly = True

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source, (np.float32,))
        encoding = image_format(source)
        if encoding not in ("gray", "bgr", "rgb") or not image.size or float(image.min()) < 0:
            raise ValidationError("Use nonempty, nonnegative float32 gray, BGR or RGB radiance")
        p = self.parameters
        space = float(p["sigma_space"])
        diameter = 2 * max(1, round(1.5 * space)) + 1
        pixels = image.shape[0] * image.shape[1]
        if pixels > 8_000_000 or pixels * diameter**2 > 1_000_000_000:
            raise ValidationError("Reduce the spatial sigma or crop the HDR image", "sigma_space")
        if encoding == "gray":
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif encoding == "bgr":
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = image

        settings = (float(p["contrast"]), float(p["saturation"]), float(p["sigma_color"]), space)
        backend = p["backend"]
        mapper = native_tonemap(*settings) if backend == "opencv" else None
        maximum = np.float32(rgb.max())
        if np.all(rgb == rgb[0, 0]):
            # A spatially uniform frame has no base contrast to compress.
            output = rgb / maximum if maximum > 0 else np.zeros_like(rgb)
        elif mapper is None:
            output = portable_tonemap(rgb, maximum, *settings)
        else:
            output = mapper.process(np.ascontiguousarray(rgb))
            if output is not None:
                # Native min-max normalization can produce zero luminance before division.
                minimum = float(rgb.min())
                black = minimum if float(maximum) - minimum > np.finfo(np.float64).eps else 0
                output[np.all(rgb == black, axis=2)] = 0

        if output is None or not np.isfinite(output).all():
            raise ValidationError("Durand tone mapping returned nonfinite display samples", "backend")
        np.clip(output, 0, 1, out=output)
        gamma = float(p["gamma"])
        if gamma != 1:
            np.power(output, 1.0 / gamma, out=output)
        if encoding == "gray":
            output = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
        elif encoding == "bgr":
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        depth = p["output_depth"]
        if depth == "uint8":
            output = convert_depth(output, depth)
        return Result({
            "image": ImageArray(output, encoding),
            "info": {"backend": backend, "output_depth": depth},
        }, message=f"Tone mapped with Durand ({backend})")
