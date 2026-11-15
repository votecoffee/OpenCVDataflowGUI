"""Bounded mask-aware multiband composition on an already registered pixel grid."""

import math

import cv2
import numpy as np

from .analysis_tools import cast_intensity
from .parameter_validation import NumericRange, ValidationError

MULTIBAND_DEFAULTS = {
    "multiband_backend": "float64",
    "multiband_bands": 4,
    "multiband_native_weights": "float32",
}
MULTIBAND_CHOICES = {
    "multiband_backend": ("float64", "native_detail"),
    "multiband_native_weights": ("float32", "int16"),
}
MULTIBAND_RANGES = {"multiband_bands": NumericRange(0, 6, whole=True)}
MULTIBAND_HELP = {
    "multiband_backend": "Multiband only. Float64 preserves high-depth/HDR samples; native detail "
    "uses the CPU int16 blender and requires uint8 gray/RGB/BGR.",
    "multiband_bands": "Multiband Laplacian reductions, 0..6. Zero averages overlap; more bands "
    "spread low-frequency transitions farther and need more halo/workspace.",
    "multiband_native_weights": "Native detail only: float32 or int16 Gaussian weight maps. "
    "Native integer arithmetic is approximate, even for uint8.",
}
MAX_WORKSPACE = 256 * 1024**2


def band_count(size, requested):
    return min(int(requested), int(math.ceil(math.log2(max(size)))))


def tile_window(origin, tile_size, mosaic_size, bands):
    """Align the decimation phase to the global mosaic and retain filter support."""
    step = 1 << band_count(mosaic_size, bands)
    halo = 8 * step
    left, top = (max(0, (v - halo) // step * step) for v in origin)
    right, bottom = (
        min(limit, math.ceil((start + length + halo) / step) * step)
        for start, length, limit in zip(origin, tile_size, mosaic_size, strict=True)
    )
    return left, top, right, bottom, halo


def _cpu(function, *args):
    # UMat-backed detail blenders otherwise enter OpenCL even with try_gpu=0.
    # OpenCV keeps this switch in its calling-thread TLS; always restore it.
    enabled = cv2.ocl.useOpenCL()
    try:
        cv2.ocl.setUseOpenCL(False)
        return function(*args)
    finally:
        cv2.ocl.setUseOpenCL(enabled)


class MultibandMosaic:
    def __init__(self, shape, dtype, parameters, *, base_bytes=0):
        self.dtype, self.shape = np.dtype(dtype), shape
        self.backend = parameters["multiband_backend"]
        h, w = shape[:2]
        channels = 1 if len(shape) == 2 else shape[2]
        self.bands = band_count((w, h), parameters["multiband_bands"])
        native = self.backend == "native_detail"
        if native and (self.dtype != np.uint8 or channels not in (1, 3)):
            raise ValidationError(
                "Native multiband needs uint8 gray/RGB/BGR; choose Float64 for this input",
                "multiband_backend",
            )
        step = 1 << self.bands
        width, height = (math.ceil(v / step) * step for v in (w, h)) if native else (w, h)
        sizes = []
        for _ in range(self.bands + 1):
            sizes.append((height, width))
            width, height = (width + 1) // 2, (height + 1) // 2
        # Destination accumulators, source pyramids, masks, normalization, copies,
        # source-only restoration and final reconstruction/cast coexist at peak.
        work_channels = max(3, channels) if native else channels
        estimate = base_bytes + sum(a * b for a, b in sizes) * 8 * (4 * work_channels + 4)
        estimate += h * w * (work_channels * (self.dtype.itemsize + 16) + 16)
        if estimate > MAX_WORKSPACE:
            raise ValidationError(
                "Multiband exceeds the 256 MiB workspace guard; reduce tile size or bands",
                "multiband_bands",
            )
        self.info = {
            "backend": self.backend,
            "requested_bands": parameters["multiband_bands"],
            "bands": self.bands,
            "working_depth": "int16" if native else "float64",
            "native_weights": parameters["multiband_native_weights"] if native else None,
            "workspace_upper_bound_bytes": estimate,
            "source_only_pixels_restored": True,
            "native_feed_clipped_samples": 0,
        }
        self.exclusive = np.zeros(shape, self.dtype)
        self.counts = np.zeros((h, w), np.uint8)
        self.scale = 1.0
        self.native = None
        if native:
            weights = (
                cv2.CV_32F if parameters["multiband_native_weights"] == "float32" else cv2.CV_16S
            )
            self.native = _cpu(cv2.detail.MultiBandBlender, 0, self.bands, weights)
            _cpu(self.native.prepare, (0, 0, w, h))
        else:
            self.total = [np.zeros((*s, *shape[2:]), np.float64) for s in sizes]
            self.weights = [np.zeros(s, np.float64) for s in sizes]

    def add(self, image, valid):
        if not valid.any():
            return
        np.copyto(
            self.exclusive,
            cast_intensity(image, self.dtype),
            where=valid if image.ndim == 2 else valid[..., None],
        )
        self.counts += valid
        np.minimum(self.counts, 2, out=self.counts)
        if self.native is not None:
            self.info["native_feed_clipped_samples"] += int(
                np.count_nonzero((image[valid] < 0) | (image[valid] > 255))
            )
            source = cast_intensity(image, np.dtype("uint8"))
            if source.ndim == 2:
                source = np.repeat(source[..., None], 3, axis=2)
            source = source.astype(np.int16)
            source[~valid] = 0
            _cpu(self.native.feed, source, valid.astype(np.uint8) * 255, (0, 0))
            return
        amplitude = max(1.0, float(np.abs(image[valid]).max()))
        if amplitude > self.scale:
            for band in self.total:
                band *= self.scale / amplitude
            self.scale = amplitude
        current = image.astype(np.float64) / self.scale
        current[~valid] = 0
        weight = valid.astype(np.float64)
        for level in range(self.bands + 1):
            if level < self.bands:
                coarse_weight = cv2.pyrDown(weight)
                coarse = cv2.pyrDown(current * (weight if current.ndim == 2 else weight[..., None]))
                divisor = coarse_weight if current.ndim == 2 else coarse_weight[..., None]
                np.divide(coarse, divisor, out=coarse, where=divisor > 0)
                expanded = cv2.pyrUp(coarse, dstsize=current.shape[1::-1])
                band = current - expanded
            else:
                band = current
            self.total[level] += band * (weight if band.ndim == 2 else weight[..., None])
            self.weights[level] += weight
            if level < self.bands:
                current, weight = coarse, coarse_weight

    def finish(self, dtype):
        if np.dtype(dtype) != self.dtype:
            raise ValidationError("Multiband output must preserve the source depth")
        if self.native is not None:
            output, support = _cpu(self.native.blend, None, None)
            if np.any(support[self.counts != 0] == 0):
                raise ValidationError("Native multiband lost valid source coverage")
            output = output[..., 0] if len(self.shape) == 2 else output
            output = cast_intensity(output, self.dtype)
            self.native = None
        else:
            for total, weight in zip(self.total, self.weights, strict=True):
                divisor = weight if total.ndim == 2 else weight[..., None]
                np.divide(total, divisor, out=total, where=divisor > 0)
            output = self.total[-1]
            for band in reversed(self.total[:-1]):
                output = cv2.pyrUp(output, dstsize=band.shape[1::-1]) + band
            with np.errstate(over="ignore", invalid="ignore"):
                output *= self.scale
            if not np.isfinite(output).all():
                raise ValidationError("Multiband reconstruction exceeded the floating range")
            output = cast_intensity(output, self.dtype)
        output[self.counts == 0] = 0
        only = self.counts == 1
        output[only] = self.exclusive[only]
        return output
