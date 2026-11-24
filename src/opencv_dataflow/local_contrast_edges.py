"""Edge localization after subtracting a broad local illumination estimate."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .parameter_validation import NumericRange, ValidationError
from .parameters import BORDERS
from .processing import gray_image
from .selection import selection_mask

MAX_PIXELS = 4_194_304
DERIVATIVE_SCALE = 32
MAX_DERIVATIVE = np.iinfo(np.int16).max / DERIVATIVE_SCALE


def intensity_8bit_units(image):
    with np.errstate(over="ignore"):
        gray = gray_image(image, auto_depth=True)
        scale = {np.dtype("uint8"): 1.0, np.dtype("uint16"): 1.0 / 257}.get(gray.dtype, 255.0)
        intensity = gray.astype(np.float32) * np.float32(scale)
    if not np.isfinite(intensity).all():
        raise ValidationError("Image intensity exceeds the working range. Rescale the image.")
    return intensity


def local_contrast(intensity, short, long, mode, border):
    fine = cv2.GaussianBlur(intensity, (short, short), 0, borderType=border)
    if mode == "box":
        background = cv2.blur(intensity, (long, long), borderType=border)
    else:
        background = cv2.GaussianBlur(intensity, (long, long), 0, borderType=border)
    return cv2.subtract(fine, background)


class LocalContrastEdges(Block):
    type_id, title = "cv.local_contrast_edges", "Local-contrast edges"
    category = block_type = "Image processing"
    subtype = "Edges"
    description = "Subtract broad illumination before tracing weak edges at a finer spatial scale."
    inputs = (
        Port("image"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image", visual_type="mask"),
        Port("contrast"),
        Port("strength"),
        Port("report", "data"),
    )
    payload_reference = "image"
    parallel_safe = preview_safe = process_inputs_readonly = True
    defaults = {
        "short_range": 3,
        "long_range": 25,
        "background": "box",
        "low": 0.6,
        "high": 1.2,
        "border": "reflect_101",
    }
    parameter_choices = {
        "background": ("box", "gaussian"),
        "border": ("reflect_101", "reflect_edge", "replicate"),
    }
    parameter_ranges = {
        "short_range": NumericRange(1, 63, whole=True, odd=True),
        "long_range": NumericRange(3, 255, whole=True, odd=True),
        "low": NumericRange(0, MAX_DERIVATIVE),
        "high": NumericRange(0, MAX_DERIVATIVE),
    }
    parameter_help = {
        "short_range": "Fine Gaussian window side in pixels. A size of 1 skips fine smoothing.",
        "long_range": "Background window side in pixels. Choose a scale wider than the edge transition.",
        "background": "Use a box mean or a broader Gaussian to estimate local illumination.",
        "low": "Lower hysteresis threshold in 8-bit-equivalent intensity units per pixel.",
        "high": "Upper seed threshold in the same units as Low.",
        "border": "Pixel extension at the image boundary for smoothing and gradient estimation.",
    }
    input_help = {
        "image": "Monochrome or color intensity. Float 0–1 and uint16 full scale map to the byte range.",
        "mask": "Publish detections only within this source-grid mask. Neighborhoods use the full image.",
        "region": "Intersect this region with the mask to select published pixels.",
    }
    output_help = {
        "image": "White Canny edge pixels on a black background.",
        "contrast": "Signed float32 fine-minus-background intensity in 8-bit-equivalent units.",
        "strength": "Float32 gradient magnitude in 8-bit-equivalent intensity units per pixel.",
        "report": "Selected area, edge pixel count and peak gradient strength.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["short_range"] >= p["long_range"]:
            raise ValidationError(
                "Short range must be smaller than long range", "short_range", "long_range"
            )
        if float(p["low"]) > float(p["high"]):
            raise ValidationError("Thresholds must satisfy 0 <= low <= high", "low", "high")

    def process(self, inputs):
        source = inputs["image"]
        height, width = source.shape[:2]
        if min(height, width) < 1 or height * width > MAX_PIXELS:
            raise ValidationError(f"Local-contrast edges require a nonempty image up to {MAX_PIXELS:,} pixels")
        selected = selection_mask(inputs, source)
        p = self.parameters
        border = BORDERS[p["border"]]
        contrast = local_contrast(
            intensity_8bit_units(source), int(p["short_range"]), int(p["long_range"]),
            p["background"], border,
        )
        # Normalize the 3×3 Sobel step to intensity units per pixel.
        dx = cv2.Sobel(contrast, cv2.CV_32F, 1, 0, ksize=3, scale=1/8, borderType=border)
        dy = cv2.Sobel(contrast, cv2.CV_32F, 0, 1, ksize=3, scale=1/8, borderType=border)
        strength = cv2.magnitude(dx, dy)
        if not np.isfinite(strength).all() or strength.max() > MAX_DERIVATIVE:
            raise ValidationError("Edge gradients exceed the supported range. Rescale the image.")
        # OpenCV's derivative Canny path requires signed 16-bit input.
        qx, qy = (np.rint(axis * DERIVATIVE_SCALE).astype(np.int16) for axis in (dx, dy))
        edges = cv2.Canny(
            qx, qy, float(p["low"]) * DERIVATIVE_SCALE, float(p["high"]) * DERIVATIVE_SCALE,
            L2gradient=True,
        )
        if selected is not None:
            outside = selected == 0
            edges[outside] = contrast[outside] = strength[outside] = 0
        count = cv2.countNonZero(edges)
        payload = {
            key: value for key, value in self.input_payloads["image"].items()
            if key not in ("alpha_representation", "raw_sensor")
        }
        return Result(
            {
                "image": edges,
                "contrast": contrast,
                "strength": strength,
                "report": {
                    "edge_pixels": count,
                    "selected_pixels": int(edges.size if selected is None else cv2.countNonZero(selected)),
                    "peak_strength": float(strength.max()),
                },
            },
            payloads={key: payload for key in ("image", "contrast", "strength", "report")},
            message=f"{count:,} edge pixels",
        )
