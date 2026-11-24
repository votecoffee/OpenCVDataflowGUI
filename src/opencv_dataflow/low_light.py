"""Bio-inspired multi-exposure fusion with a portable sparse illumination solve.

The illumination model follows OpenCV's BIMEF implementation by Zhenqiang.Ying
(2017, MIT). The SciPy solve and bounded discrete exposure search are independent
implementations.
"""

import re

import cv2
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import cg

from .core import Port, Result
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import metadata
from .processing import ImageTool, image_value
from .selection import SELECTION_INPUTS, selection_mask

_CHANNELS = {"gray": 1, "bgr": 3, "rgb": 3, "bgra": 4, "rgba": 4}
_LIMIT = 2_100_000


def illumination_system(light):
    """Build I + L for the texture-weighted, periodic four-neighbor grid."""
    shape = light.shape
    pixels = np.arange(light.size, dtype=np.int32).reshape(shape)
    indices = pixels.ravel()
    weights, neighbors = [], []
    for axis, kernel_shape in ((1, (1, 5)), (0, (5, 1))):
        gradient = np.roll(light, -1, axis=axis) - light
        summed = cv2.filter2D(gradient, -1, np.ones(kernel_shape), borderType=cv2.BORDER_CONSTANT)
        weights.append((0.5 / (np.abs(gradient * summed) + 0.001)).ravel())
        neighbors.append(np.roll(pixels, -1, axis=axis).ravel())
    edges = coo_matrix(
        (np.concatenate(weights), (np.tile(indices, 2), np.concatenate(neighbors))),
        shape=(light.size, light.size),
    ).tocsr()
    adjacency = edges + edges.T
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    return diags(1 + degree, format="csr") - adjacency


def smooth_illumination(image):
    height, width = image.shape[:2]
    light = image.max(axis=2) if image.ndim == 3 else image
    coarse = cv2.resize(light, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    if not np.ptp(coarse):
        return np.full(light.shape, coarse.flat[0], light.dtype), 0.0
    system = illumination_system(coarse)
    rhs = coarse.ravel().astype(np.float64)
    result, status = cg(
        system, rhs, x0=rhs.copy(), M=diags(1 / system.diagonal(), format="csr"),
        rtol=1e-5, maxiter=400,
    )
    if status or not np.isfinite(result).all():
        raise ValidationError("The BIMEF illumination solver did not converge")
    residual = float(np.linalg.norm(system @ result - rhs) / np.linalg.norm(rhs))
    resized = cv2.resize(result.reshape(coarse.shape), (width, height))
    return np.clip(resized, 0, 1), residual


def _expose(image, ratio, a, b):
    power = ratio ** a
    return np.power(image, power) * np.exp((1 - power) * b)


def _automatic_exposure(image, illumination, a, b):
    shadow = cv2.resize((illumination < 0.5).astype(np.uint8), (50, 50)).astype(bool)
    if not shadow.any():
        return 1.0
    reduced = cv2.resize(image, (50, 50))
    if image.ndim == 3:
        reduced = np.cbrt(np.prod(reduced, axis=2))
    values = reduced[shadow]
    best_ratio, best_entropy = 1.0, -1.0
    for k in np.linspace(1, 7, 49):
        samples = np.rint(_expose(values, k, a, b) * 255).clip(0, 255).astype(np.uint8)
        counts = np.bincount(samples)
        probabilities = counts[counts > 0] / samples.size
        entropy = -np.sum(probabilities * np.log2(probabilities))
        if entropy > best_entropy + 1e-12:
            best_ratio, best_entropy = float(k), entropy
    return best_ratio


def _portable(image, automatic, ratio, mu, a, b):
    illumination, residual = smooth_illumination(image)
    if automatic:
        ratio = _automatic_exposure(image, illumination, a, b)
    if ratio == 1 or mu == 0:
        return image, ratio, residual
    exposed = _expose(image, ratio, a, b)
    if automatic:
        exposed -= 0.01
    else:
        exposed = np.minimum(exposed, 1)
    weights = illumination ** mu
    if image.ndim == 3:
        weights = weights[..., None]
    return image * weights + exposed * (1 - weights), ratio, residual


def _native_function(automatic):
    api = getattr(cv2, "intensity_transform", None)
    binding = getattr(api, "BIMEF" if automatic else "BIMEF2", None)
    if binding is None or re.search(r"(?im)^\s*Eigen:\s*NO\b", cv2.getBuildInformation()):
        return None
    return binding


class EnhanceLowLight(ImageTool):
    type_id, title, subtype = "cv.bimef", "Enhance low light (BIMEF)", "Contrast"
    inputs = SELECTION_INPUTS
    outputs = (Port("image"), Port("report", "data"))
    defaults = {
        "backend": "auto", "exposure_mode": "automatic", "exposure_ratio": 4.0,
        "enhancement_ratio": 0.5, "response_a": -0.3293, "response_b": 1.1258,
    }
    parameter_choices = {
        "backend": ("auto", "opencv", "scipy"),
        "exposure_mode": ("automatic", "explicit"),
    }
    parameter_ranges = {
        "exposure_ratio": NumericRange(1, 16), "enhancement_ratio": NumericRange(0, 4),
        "response_a": NumericRange(-1, 0), "response_b": NumericRange(0, 4),
    }
    parameter_help = {
        "backend": "Auto uses native OpenCV for uint8 with Eigen, or SciPy otherwise.",
        "exposure_mode": "Estimate exposure from shadows or use the specified exposure ratio.",
        "exposure_ratio": "Simulated second exposure ratio. Used in explicit mode.",
        "enhancement_ratio": "Illumination exponent mu. Zero keeps the input.",
        "response_a": "Camera response exponent coefficient a.",
        "response_b": "Camera response gain coefficient b.",
    }
    input_help = {
        "image": "Gray, RGB or BGR. Use uint8, uint16 or float32 in [0, 1], with optional alpha.",
        "mask": "Only selected pixels are updated. Exposure estimation uses the full input.",
        "region": "Optional image-space write selection, intersected with mask.",
    }
    output_help = {
        "image": "Enhanced image with the original encoding, depth, alpha and grid.",
        "report": "Backend, exposure when known, mean intensity and upper saturation in selection.",
    }
    process_inputs_readonly = True
    payload_reference = "image"

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source, (np.uint8, np.uint16, np.float32))
        height, width = image.shape[:2]
        if min(height, width) < 4 or height * width > _LIMIT:
            raise ValidationError("Use images at least 4 × 4 pixels and up to 2.1 million pixels")
        encoding = image_format(source)
        channels = image.shape[2] if image.ndim == 3 else 1
        if channels != _CHANNELS.get(encoding):
            raise ValidationError("Use grayscale, BGR or RGB with optional straight alpha")
        if channels == 4 and metadata(source).get("alpha_representation", "straight") != "straight":
            raise ValidationError("Unassociate premultiplied alpha before applying BIMEF")
        if image.dtype == np.float32 and (image.min() < 0 or image.max() > 1):
            raise ValidationError("Float BIMEF samples must be in [0, 1]")
        colors = image[..., :3] if channels == 4 else image
        selection = selection_mask(inputs, source)
        picked = np.ones((height, width), bool) if selection is None else selection != 0
        p = self.parameters
        automatic = p["exposure_mode"] == "automatic"
        ratio = None if automatic else float(p["exposure_ratio"])
        mu, a, b = (float(p[key]) for key in ("enhancement_ratio", "response_a", "response_b"))
        backend = p["backend"]
        if backend == "opencv" and colors.dtype != np.uint8:
            raise ValidationError("Native BIMEF requires uint8 input", "backend")
        method = (
            _native_function(automatic) if backend != "scipy" and colors.dtype == np.uint8 else None
        )
        if backend == "opencv" and method is None:
            raise ValidationError("Native BIMEF requires an Eigen-enabled OpenCV contrib build")
        scale = 1.0 if colors.dtype == np.float32 else float(np.iinfo(colors.dtype).max)
        residual, enhanced, ran = None, colors, "none"
        if picked.any():
            if method is not None:
                native = cv2.cvtColor(colors, cv2.COLOR_GRAY2BGR) if colors.ndim == 2 else colors
                try:
                    processed = (
                        method(native, mu=mu, a=a, b=b) if automatic
                        else method(native, ratio, mu, a, b)
                    )
                except cv2.error as exc:
                    if backend != "auto" or exc.code != cv2.Error.StsNotImplemented:
                        raise
                else:
                    if (
                        processed is None or processed.shape != native.shape
                        or processed.dtype != np.uint8
                    ):
                        raise ValidationError("OpenCV returned an invalid BIMEF image")
                    enhanced = processed[..., 0] if colors.ndim == 2 else processed
                    ran = "opencv"
            if ran == "none":
                normalized = colors.astype(np.float64) / scale
                fused, ratio, residual = _portable(normalized, automatic, ratio, mu, a, b)
                result = np.clip(fused, 0, 1) * scale
                enhanced = (result if scale == 1 else np.rint(result)).astype(colors.dtype)
                ran = "scipy"
        output = image.copy()
        destination = output[..., :3] if channels == 4 else output
        destination[picked] = enhanced[picked]
        before, after = colors[picked], destination[picked]
        metrics = {"selected_pixels": int(picked.sum()), "mean_before": None, "mean_after": None,
                   "upper_saturation_before": None, "upper_saturation_after": None}
        if before.size:
            before_peaks = before.reshape(before.shape[0], -1).max(axis=1)
            after_peaks = after.reshape(after.shape[0], -1).max(axis=1)
            metrics.update(
                mean_before=float(before.mean(dtype=np.float64) / scale),
                mean_after=float(after.mean(dtype=np.float64) / scale),
                upper_saturation_before=float(np.mean(before_peaks == scale)),
                upper_saturation_after=float(np.mean(after_peaks == scale)),
            )
        return Result(
            {
                "image": ImageArray(output, encoding),
                "report": {
                    "backend": ran, "exposure_mode": p["exposure_mode"], "exposure_ratio": ratio,
                    "illumination_residual": residual, **metrics,
                },
            },
            message=f"BIMEF {ran} · {metrics['selected_pixels']:,} selected pixels",
        )
