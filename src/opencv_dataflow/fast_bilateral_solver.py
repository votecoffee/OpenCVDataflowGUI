"""Confidence-weighted bilateral-space optimization with OpenCV and SciPy."""

import re

import cv2
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg

from .core import Port
from .images import ImageArray, convert_depth, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import require_same_frame
from .processing import ImageTool, encoded_result, image_value
from .restoration import _check_aligned_guide
from .segmentation import bgr8
from .selection import SELECTION_INPUTS, selection_mask

_MAX_PIXELS = 2_097_152
_MAX_VERTICES = 262_144


def _bilateral_grid(guide, spatial, luma, chroma):
    """Return pixel-to-bin indices and the occupied axis-neighbor graph."""
    height, width = guide.shape[:2]
    color = cv2.cvtColor(guide, cv2.COLOR_BGR2YCrCb) if guide.ndim == 3 else guide[..., None]
    color_axes = tuple(
        (color[..., index], sigma)
        for index, sigma in enumerate((luma, chroma, chroma)[: color.shape[2]])
    )
    axes = (
        (np.arange(width)[None, :], spatial),
        (np.arange(height)[:, None], spatial),
        *color_axes,
    )
    keys = np.zeros((height, width), np.int64)
    strides, stride = [], 1
    native_grid_safe = True
    for values, step in axes:
        bins = (values / step).astype(np.int64)
        highest = int(bins.max())
        keys += bins * stride
        strides.append(stride)
        # Leave an unused coordinate between occupied axes to prevent carry into the next axis.
        stride *= highest + 2
        native_grid_safe &= highest < 254
    occupied, pixel_bins = np.unique(keys.ravel(), return_inverse=True)
    size = occupied.size
    if size > _MAX_VERTICES:
        raise ValidationError("Bilateral grid exceeds 262,144 occupied bins. Increase a sigma.")
    rows, cols = [], []
    for delta in strides:
        neighbor = occupied + delta
        index = np.searchsorted(occupied, neighbor)
        np.minimum(index, size - 1, out=index)
        adjacent = occupied[index] == neighbor
        rows.append(np.flatnonzero(adjacent))
        cols.append(index[adjacent])
    first, second = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix(
        (np.ones(2 * first.size), (np.r_[first, second], np.r_[second, first])),
        shape=(size, size),
    ).tocsr()
    return pixel_bins, graph, native_grid_safe


def _supported_bins(adjacency, weights, strength):
    known = weights > 0
    if not strength or known.all():
        return known
    _, groups = connected_components(adjacency, directed=False)
    return np.bincount(groups, weights=known)[groups] > 0


def _scipy_solve(signal, weight, bins, adjacency, totals, active, strength, iterations, tolerance):
    # The 10-neighbor diagonal convention also matches the OpenCV normalization.
    count = np.bincount(bins)
    norm = np.ones_like(totals)
    if strength:
        for _ in range(10):
            norm = np.sqrt(norm * count / (10 * norm + adjacency @ norm))
        # Diagonal self-weights cancel in the graph Laplacian.
        normalized = adjacency.multiply(norm[:, None]).multiply(norm[None, :]).tocsr()
        laplacian = diags(np.asarray(normalized.sum(axis=1)).ravel()) - normalized
        system = (strength * laplacian + diags(totals)).tocsr()[active][:, active]
        preconditioner = diags(1 / system.diagonal()).tocsr()
    channels = signal.reshape(bins.size, -1)
    result = channels.astype(np.float64, copy=True)
    residuals, converged = [], True
    pixels = active[bins]
    for channel in range(channels.shape[1]):
        rhs = np.bincount(bins, weights=channels[:, channel] * weight)
        guess = np.divide(rhs, totals, where=totals > 0, out=np.zeros_like(totals))[active]
        if strength:
            estimate, status = cg(
                system, rhs[active], x0=guess, M=preconditioner, maxiter=iterations, rtol=tolerance
            )
            if status < 0:
                raise ValueError("Bilateral solver failed to solve the sparse system")
            reference = np.linalg.norm(rhs[active])
            residual = np.linalg.norm(system @ estimate - rhs[active])
            residuals.append(float(residual / reference) if reference else float(residual))
            converged &= status == 0
        else:
            estimate = guess
            residuals.append(0.0)
        values = np.zeros_like(totals)
        values[active] = estimate
        result[pixels, channel] = values[bins[pixels]]
    return result.reshape(signal.shape), converged, residuals


def _opencv_factory():
    factory = getattr(getattr(cv2, "ximgproc", None), "createFastBilateralSolverFilter", None)
    if factory is None or re.search(r"(?im)^\s*Eigen:\s*NO\b", cv2.getBuildInformation()):
        return None
    return factory


class FastBilateralSolver(ImageTool):
    type_id, title, subtype = "cv.fast_bilateral_solver", "Fast bilateral solver", "Filters"
    description = "Reconstruct a confidence-weighted signal while preserving edges in a guide."
    inputs = (
        *SELECTION_INPUTS,
        Port("guide"),
        Port("confidence", required=False, required_when_connected=True),
    )
    outputs = (*ImageTool.outputs, Port("supported", required=False), Port("report", "data"))
    defaults = {
        "backend": "auto",
        "alignment": "same_frame",
        "sigma_spatial": 8.0,
        "sigma_luma": 8.0,
        "sigma_chroma": 8.0,
        "lambda_weight": 128.0,
        "iterations": 100,
        "max_tolerance": 1e-5,
    }
    parameter_choices = {
        "backend": ("auto", "scipy", "opencv"),
        "alignment": ("same_frame", "pixel_grid"),
    }
    parameter_ranges = {
        "sigma_spatial": NumericRange(1, 8192),
        "sigma_luma": NumericRange(1, 255),
        "sigma_chroma": NumericRange(1, 255),
        "lambda_weight": NumericRange(0, 1e6),
        "iterations": NumericRange(1, 1000, whole=True),
        "max_tolerance": NumericRange(1e-8, 1),
    }
    parameter_help = {
        "backend": "Auto uses OpenCV with Eigen when the guide and confidence permit it.",
        "alignment": "Pixel grid accepts independently loaded, preregistered images.",
        "sigma_spatial": "Size of a spatial bin in pixels.",
        "sigma_luma": "Size of a luma bin in 8-bit guide levels.",
        "sigma_chroma": "Size of a chroma bin in 8-bit guide levels.",
        "lambda_weight": "Smoothness weight. Zero computes confidence-weighted bin means.",
        "iterations": "Maximum conjugate-gradient iterations per channel.",
        "max_tolerance": "Relative residual tolerance for conjugate gradients.",
    }
    input_help = {
        "image": "Signal to reconstruct. Keeps grayscale or color encoding, depth and alpha.",
        "guide": "Aligned intensity or color guide. Converted to 8-bit luma and chroma bins.",
        "confidence": "Mono uint8 [0,255] or float32 [0,1]. Omit for equal weights.",
        "mask": "Select output pixels to replace. The solve uses confidence across the image.",
        "region": "Select output pixels to replace, intersected with the mask.",
    }
    output_help = {
        "supported": "White where selected pixels have an observation in their connected bins.",
        "report": "Backend, support coverage and convergence information when available.",
    }
    preview_safe = parallel_safe = process_inputs_readonly = True
    payload_reference = "image"

    def process(self, inputs):
        p = self.parameters
        source, guide = inputs["image"], inputs["guide"]
        if np.prod(source.shape[:2]) > _MAX_PIXELS or max(source.shape[:2]) > 8192:
            raise ValidationError(
                "Bilateral solver allows up to 2,097,152 pixels and 8,192 per axis"
            )
        image = image_value(source, (np.uint8, np.int16, np.uint16, np.float32))
        image_value(guide, (np.uint8, np.uint16, np.float32))
        same_frame = p["alignment"] == "same_frame"
        _check_aligned_guide(source, guide, check_frame=same_frame)
        confidence = inputs.get("confidence")
        if confidence is None:
            weights = np.ones(image.shape[:2], np.float64)
        else:
            raw = image_value(confidence, (np.uint8, np.float32))
            if raw.shape != image.shape[:2] or raw.min() < 0:
                raise ValidationError("Confidence must be an aligned nonnegative monochrome image")
            if same_frame:
                require_same_frame(
                    (source, guide, confidence), message="Confidence must share the source frame"
                )
            if raw.dtype == np.float32 and raw.max() > 1:
                raise ValidationError("Float confidence must be within [0, 1]")
            weights = raw.astype(np.float64)
            if raw.dtype == np.uint8:
                weights /= 255

        mask = selection_mask(inputs, source)
        selected = mask != 0 if mask is not None else np.ones(image.shape[:2], dtype=bool)
        sigmas = tuple(float(p[key]) for key in ("sigma_spatial", "sigma_luma", "sigma_chroma"))
        strength, tolerance = float(p["lambda_weight"]), float(p["max_tolerance"])
        iterations = int(p["iterations"])
        guide8 = convert_depth(guide, "uint8")
        if guide.ndim == 3:
            guide8 = bgr8(ImageArray(guide8, image_format(guide)))
        guide8 = np.ascontiguousarray(guide8)
        bins, adjacency, native_safe = _bilateral_grid(guide8, *sigmas)
        totals = np.bincount(bins, weights=weights.ravel(), minlength=adjacency.shape[0])
        active = _supported_bins(adjacency, totals, strength)
        replace = selected & active[bins].reshape(selected.shape)
        backend, reason, converged, residuals = "none", None, None, None
        output = image.copy()
        factory = _opencv_factory() if p["backend"] != "scipy" else None
        if p["backend"] != "scipy":
            if factory is None:
                reason = "OpenCV fast bilateral solver requires an Eigen-enabled build"
            elif not native_safe:
                reason = "Guide bins exceed the native OpenCV coordinate range"
            elif not active.all():
                reason = "Some guide components have no confidence observations"
            if reason is not None and p["backend"] == "opencv":
                raise ValidationError(reason, "backend")
        if replace.any():
            signal = image[..., :3] if image.ndim == 3 and image.shape[2] == 4 else image
            prediction = None
            if factory is not None and reason is None:
                try:
                    solver = factory(guide8, *sigmas, strength, iterations, tolerance)
                    normalized = signal.astype(np.float64)
                    amplitude = max(float(np.abs(normalized).max()), 1.0)
                    prediction = (
                        solver.filter(
                            (normalized / amplitude).astype(np.float32), weights.astype(np.float32)
                        ).astype(np.float64)
                        * amplitude
                    )
                    backend = "opencv"
                except cv2.error as exc:
                    if exc.code != cv2.Error.StsNotImplemented:
                        raise
                    reason = "OpenCV fast bilateral solver requires an Eigen-enabled build"
                    if p["backend"] == "opencv":
                        raise ValidationError(reason, "backend") from exc
            if prediction is None:
                prediction, converged, residuals = _scipy_solve(
                    signal,
                    weights.ravel(),
                    bins,
                    adjacency,
                    totals,
                    active,
                    strength,
                    iterations,
                    tolerance,
                )
                backend = "scipy"
            limits = np.iinfo(image.dtype) if image.dtype.kind in "iu" else np.finfo(image.dtype)
            values = prediction[replace]
            if not np.isfinite(values).all():
                raise ValueError("Bilateral solver produced nonfinite samples")
            if image.dtype.kind in "iu":
                values = np.rint(values)
            target = output[..., :3] if signal.shape != output.shape else output
            target[replace] = np.clip(values, limits.min, limits.max).astype(image.dtype)
        result = encoded_result(output, source)
        result.outputs.update(
            supported=ImageArray(replace.astype(np.uint8) * 255, "gray"),
            report={
                "backend": backend,
                "backend_reason": reason,
                "grid_bins": int(totals.size),
                "observed_pixels": int(np.count_nonzero(weights)),
                "selected_pixels": int(np.count_nonzero(selected)),
                "supported_selected_pixels": int(np.count_nonzero(replace)),
                "converged": converged,
                "relative_residuals": residuals,
            },
        )
        if converged is False:
            result.passed = False
            result.message = "Bilateral solver reached the iteration limit"
            result.fail_reasons = ["bilateral_solver_not_converged"]
        return result
