"""Grayscale BM3D with OpenCV and separable DCT collaborative filtering backends."""

from threading import Lock

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import dctn, idctn

from .core import Port, Result
from .images import ImageArray
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, ValuePacket, check_spatial, require_same_frame
from .processing import ImageTool, image_value
from .selection import SELECTION_INPUTS, selection_mask

_DEPTHS = (np.uint8, np.uint16, np.float32, np.float64)
_MAX_PIXELS = 2_097_152
_NATIVE_LOCK = Lock()


def reference_starts(length, patch, stride):
    last = length - patch
    return (*range(0, last, stride), last)


def patch_group(guide_patches, y, x, search, group_size, match_threshold, norm):
    height, width = guide_patches.shape[:2]
    radius = search // 2
    top, left = max(0, y - radius), max(0, x - radius)
    bottom, right = min(height, y + search - radius), min(width, x + search - radius)
    cols = right - left
    candidates = guide_patches[top:bottom, left:right]
    delta = candidates - guide_patches[y, x]
    if norm == "l2":
        difference = np.square(delta, out=delta).mean(axis=(-2, -1))
    else:
        difference = np.abs(delta, out=delta).mean(axis=(-2, -1))
        np.square(difference, out=difference)
    # The reference is always included, including in constant patches with tied distances.
    difference[y - top, x - left] = -1
    distances = difference.ravel()
    best = np.argsort(distances, kind="stable")[:group_size]
    best = best[distances[best] <= match_threshold]
    rows, columns = divmod(best, cols)
    return rows + top, columns + left


def collaborative_filter(noisy, basic, sigma):
    coeffs = dctn(noisy, norm="ortho", workers=1)
    if basic is None:
        keep = np.abs(coeffs) >= 2.7 * sigma
        keep[0, 0, 0] = True
        coeffs *= keep
        weight = 1 / np.count_nonzero(keep)
    else:
        guide_energy = np.square(dctn(basic, norm="ortho", workers=1))
        gain = guide_energy / (guide_energy + sigma**2)
        gain[0, 0, 0] = 1
        coeffs *= gain
        weight = 1 / float(np.sum(gain**2))
    return idctn(coeffs, norm="ortho", workers=1), weight


def portable_stage(image, basic, parameters, cancelled):
    p = parameters
    patch, stride = int(p["patch_size"]), int(p["sliding_step"])
    observed = sliding_window_view(image, (patch, patch))
    guides = observed if basic is None else sliding_window_view(basic, (patch, patch))
    threshold = p["hard_match"] if basic is None else p["wiener_match"]
    window = np.kaiser(patch, float(p["beta"])).astype(np.float32)
    window = np.outer(window, window)
    estimate, total = np.zeros_like(image), np.zeros_like(image)
    xs = reference_starts(image.shape[1], patch, stride)
    for y in reference_starts(image.shape[0], patch, stride):
        if cancelled():
            raise InterruptedError("BM3D cancelled")
        for x in xs:
            ys, columns = patch_group(
                guides, y, x, int(p["search_window"]), int(p["group_size"]), threshold, p["norm"]
            )
            source = observed[ys, columns]
            guide_stack = None if basic is None else guides[ys, columns]
            filtered, confidence = collaborative_filter(
                source, guide_stack, float(p["noise_sigma"])
            )
            weight = window * confidence
            for row, col, samples in zip(ys, columns, filtered, strict=True):
                estimate[row : row + patch, col : col + patch] += samples * weight
                total[row : row + patch, col : col + patch] += weight
    return np.divide(estimate, total, out=image.copy(), where=total > 0)


def native_stages(image, basic, p, cancelled):
    module = getattr(cv2, "xphoto", None)
    denoise = getattr(module, "bm3dDenoising", None)
    if denoise is None:
        raise ValidationError("OpenCV BM3D requires an xphoto build with OPENCV_ENABLE_NONFREE=ON")
    stage = {
        "hard_threshold": module.BM3D_STEP1,
        "wiener": module.BM3D_STEP2,
        "both": module.BM3D_STEPALL,
    }[p["stage"]]
    depth_scale = 257 if image.dtype == np.uint16 else 1
    hard, wiener = (int(p[name]) * depth_scale**2 for name in ("hard_match", "wiener_match"))
    if max(hard, wiener) > np.iinfo(np.int32).max:
        raise ValidationError(
            "Matching thresholds exceed the native uint16 range", "hard_match", "wiener_match"
        )
    first = np.empty_like(image) if basic is None else np.array(basic, order="C", copy=True)
    final = np.empty_like(image)
    # Native Haar function pointers depend on the patch size and are shared across calls.
    while not _NATIVE_LOCK.acquire(timeout=0.05):
        if cancelled():
            raise InterruptedError("BM3D cancelled")
    try:
        if cancelled():
            raise InterruptedError("BM3D cancelled")
        first, final = denoise(
            image,
            dstStep1=first,
            dstStep2=final,
            h=float(p["noise_sigma"]) * depth_scale,
            templateWindowSize=int(p["patch_size"]),
            searchWindowSize=int(p["search_window"]),
            blockMatchingStep1=hard,
            blockMatchingStep2=wiener,
            groupSize=int(p["group_size"]),
            slidingStep=int(p["sliding_step"]),
            beta=float(p["beta"]),
            normType=cv2.NORM_L2 if p["norm"] == "l2" else cv2.NORM_L1,
            step=stage,
            transformType=module.HAAR,
        )
    except cv2.error as exc:
        if "OPENCV_ENABLE_NONFREE" in str(exc):
            raise ValidationError(
                "This OpenCV build disables native BM3D. Choose the portable backend.",
                "backend",
            ) from exc
        raise
    finally:
        _NATIVE_LOCK.release()
    return first, first if p["stage"] == "hard_threshold" else final


def monochrome(source):
    image = image_value(source, _DEPTHS)
    if image.ndim != 2 or not image.size:
        raise ValidationError("BM3D requires a nonempty grayscale image")
    if image.dtype.kind == "f" and (image.min() < 0 or image.max() > 1):
        raise ValidationError("Floating-point BM3D samples must use [0, 1]")
    return image


class Bm3dDenoise(ImageTool):
    type_id, title, subtype = "cv.bm3d_denoise", "BM3D denoising", "Filters"
    description = "Reduce Gaussian noise by collaborative filtering of similar image patches."
    inputs = (*SELECTION_INPUTS, Port("basic", required=False, required_when_connected=True))
    outputs = (Port("image"), Port("basic", required=False), Port("report", "data"))
    defaults = {
        "backend": "portable",
        "stage": "both",
        "noise_sigma": 10.0,
        "patch_size": 8,
        "search_window": 19,
        "hard_match": 2500,
        "wiener_match": 400,
        "group_size": 16,
        "sliding_step": 3,
        "beta": 2.0,
        "norm": "l2",
        "basic_alignment": "same_frame",
    }
    parameter_choices = {
        "backend": ("portable", "opencv"),
        "stage": ("both", "hard_threshold", "wiener"),
        "norm": ("l2", "l1"),
        "basic_alignment": ("same_frame", "pixel_grid"),
    }
    parameter_ranges = {
        "noise_sigma": NumericRange(0, 255),
        "patch_size": NumericRange(2, 64, whole=True),
        "search_window": NumericRange(3, 127, whole=True),
        "hard_match": NumericRange(0, 65025, whole=True),
        "wiener_match": NumericRange(0, 65025, whole=True),
        "group_size": NumericRange(1, 64, whole=True),
        "sliding_step": NumericRange(1, 63, whole=True),
        "beta": NumericRange(0, 16),
    }
    parameter_help = {
        "backend": "Portable orthonormal DCT or native Haar on an enabled OpenCV build.",
        "stage": "Run both stages, the hard-threshold estimate, or Wiener with a basic image.",
        "noise_sigma": "Noise sigma in 8-bit-equivalent levels. Zero bypasses filtering.",
        "patch_size": "Power-of-two square patch width in pixels.",
        "search_window": "Side of the candidate window in patch positions.",
        "hard_match": "First-stage match limit in squared 8-bit-equivalent pixel levels.",
        "wiener_match": "Second-stage match limit in squared 8-bit-equivalent pixel levels.",
        "group_size": "Maximum similar patches in one group.",
        "sliding_step": "Reference stride in pixels, smaller than the patch.",
        "beta": "Kaiser aggregation window parameter. Zero uses uniform patch weights.",
        "norm": "L2 is mean squared difference. L1 compares squared mean absolute difference.",
        "basic_alignment": "Check known coordinate frames or explicitly pair equal pixel grids.",
    }
    input_help = {
        "image": "Grayscale source. Portable supports uint8, uint16, or float in [0, 1].",
        "basic": "Same-depth grayscale estimate required for the Wiener-only stage.",
        "mask": "Limit writes while patch matching uses the full image.",
        "region": "Intersect the source-grid region with the optional write mask.",
    }
    output_help = {
        "image": "Denoised source in its original depth.",
        "basic": "Hard-threshold estimate, or its input counterpart in Wiener-only mode.",
        "report": "Backend, stage and noise level used for processing.",
    }
    process_inputs_readonly = preview_safe = True
    payload_reference = "image"

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        patch = int(p["patch_size"])
        if patch & (patch - 1):
            raise ValidationError("Patch size must be a power of two", "patch_size")
        if p["search_window"] <= patch:
            raise ValidationError("Search window must exceed patch size", "search_window")
        if p["sliding_step"] >= patch:
            raise ValidationError("Sliding step must be smaller than the patch", "sliding_step")

    def process(self, inputs):
        p = self.parameters
        source = inputs["image"]
        image = monochrome(source)
        if image.size > _MAX_PIXELS:
            raise ValidationError("BM3D supports up to 2,097,152 grayscale pixels")
        mask = selection_mask(inputs, source)
        selected = mask != 0 if mask is not None else None
        basic = inputs.get("basic")
        use_basic = p["stage"] == "wiener"
        if use_basic != (basic is not None):
            raise ValidationError("Connect Basic only for the Wiener-only stage", "basic", "stage")
        if basic is not None:
            basic = monochrome(basic)
            if basic.shape != image.shape or basic.dtype != image.dtype:
                raise ValidationError(
                    "Basic estimate must match the source size and depth", "basic"
                )
            basic_context = self.input_payloads["basic"]
            check_spatial(basic_context, [image.shape[1], image.shape[0]])
            if p["basic_alignment"] == "same_frame":
                require_same_frame(
                    (
                        ValuePacket(None, self.input_payloads["image"]),
                        ValuePacket(None, basic_context),
                    ),
                    message="Basic estimate uses a different coordinate frame",
                )
        native = p["backend"] == "opencv"
        bypass = not p["noise_sigma"] or (selected is not None and not selected.any())
        if bypass:
            filtered, first = image, image if basic is None else basic
        else:
            self.check_work(image.shape, p)
            if native:
                if image.dtype.kind == "f" or (image.dtype == np.uint16 and p["norm"] != "l1"):
                    raise ValidationError(
                        "Native BM3D uses uint8, or uint16 with L1. Portable supports float.",
                        "backend",
                        "norm",
                    )
                first, filtered = native_stages(image, basic, p, self._cancelled)
            else:
                first, filtered = self.portable(image, basic, p)
        return Result(
            {
                "image": self.published(filtered, image, selected),
                "basic": self.published(first, image, selected)
                if "basic" in self.exposed_outputs
                else NO_VALUE,
                "report": {
                    "backend": p["backend"],
                    "stage": p["stage"],
                    "noise_sigma": p["noise_sigma"],
                    "bypassed": bool(bypass),
                },
            }
        )

    def check_work(self, shape, p):
        height, width = shape
        patch, window, stride = (
            int(p[key]) for key in ("patch_size", "search_window", "sliding_step")
        )
        if min(height, width) < patch:
            raise ValidationError("Image dimensions must fit the patch", "patch_size")
        rows = (height - patch + stride - 1) // stride + 1
        columns = (width - patch + stride - 1) // stride + 1
        passes = 2 if p["stage"] == "both" else 1
        if passes * rows * columns * window**2 * patch**2 > 350_000_000:
            raise ValidationError(
                "BM3D exceeds the patch comparison budget. Crop or reduce the search window.",
                "search_window",
                "sliding_step",
            )

    def portable(self, image, basic, p):
        full_scale = np.iinfo(image.dtype).max if image.dtype.kind == "u" else 1.0
        scale = np.float32(255 / full_scale)
        work = image.astype(np.float32) * scale
        if basic is None:
            first = portable_stage(work, None, p, self._cancelled)
        else:
            first = basic.astype(np.float32) * scale
        filtered = (
            first
            if p["stage"] == "hard_threshold"
            else portable_stage(work, first, p, self._cancelled)
        )
        return first / scale, filtered / scale

    @staticmethod
    def published(values, source, selected):
        if values.shape != source.shape or not np.isfinite(values).all():
            raise ValidationError("BM3D returned invalid samples")
        if values.dtype == source.dtype and source.dtype.kind == "u":
            output = values.copy()
        elif source.dtype.kind == "u":
            output = np.rint(np.clip(values, 0, np.iinfo(source.dtype).max)).astype(source.dtype)
        else:
            output = np.clip(values, 0, 1).astype(source.dtype)
        if selected is not None:
            output[~selected] = source[~selected]
        return ImageArray(output, "gray")
