"""Closed-form alpha matting with exact trimap constraints and a bounded sparse solve."""

import copy

import cv2
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import cg

from .alpha_composition import pack_image, split_layer, unassociate
from .core import Block, Port, Result
from .flow_warp import same_grid
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, check_spatial
from .selection import SELECTION_INPUTS, selection_mask

_MAX_PIXELS = 2_097_152
_MAX_WINDOW_ENTRIES = 8_000_000


def matting_system(guide, foreground, unknown, p, cancelled):
    """Build the unknown-to-unknown part of the Levin matting Laplacian."""
    height, width = unknown.shape
    radius = int(p["window_radius"])
    side = 2 * radius + 1
    if min(height, width) < side:
        raise ValidationError("Image dimensions must fit the matting window", "window_radius")
    local = np.arange(-radius, radius + 1, dtype=np.int32)
    offsets = (local[:, None] * width + local).ravel()
    terms = side * side
    active = cv2.dilate(unknown.astype(np.uint8), np.ones((side, side), np.uint8))
    y, x = np.nonzero(active[radius:-radius, radius:-radius])
    windows = ((y + radius) * width + x + radius).astype(np.int32)
    work = len(windows) * terms * terms
    if work > _MAX_WINDOW_ENTRIES:
        raise ValidationError(
            "Trimap exceeds the matting work limit. Crop, narrow the unknown band, "
            "or reduce the window radius.",
            "window_radius",
        )

    unknown_count = int(unknown.sum())
    index = np.full(height * width, -1, np.int32)
    index[unknown.ravel()] = np.arange(unknown_count, dtype=np.int32)
    fixed = foreground.ravel()
    channels = guide.shape[2]
    colors = guide.reshape(-1, channels)
    rows, columns = (np.empty(work, np.int32) for _ in range(2))
    coefficients, rhs = np.empty(work, np.float64), np.zeros(unknown_count, np.float64)
    identity = np.eye(terms)
    regularization = float(p["color_regularization"]) * np.eye(channels)
    cursor = 0
    for start in range(0, len(windows), 1024):
        if cancelled():
            raise InterruptedError("Matting cancelled")
        ids = windows[start : start + 1024, None] + offsets
        samples = colors[ids]
        centered = samples - samples.mean(axis=1, keepdims=True)
        transposed = centered.swapaxes(1, 2)
        # The scatter matrix is n times the population covariance. Regularization
        # here corresponds to epsilon/n in the covariance formulation.
        scatter = transposed @ centered + regularization
        laplacian = identity - 1 / terms - centered @ np.linalg.solve(scatter, transposed)
        ids_unknown = index[ids]
        keep = ids_unknown >= 0
        contribution = -(laplacian @ fixed[ids][..., None])[..., 0]
        np.add.at(rhs, ids_unknown[keep], contribution[keep])
        both = keep[:, :, None] & keep[:, None, :]
        stop = cursor + int(both.sum())
        rows[cursor:stop] = np.broadcast_to(ids_unknown[:, :, None], both.shape)[both]
        columns[cursor:stop] = np.broadcast_to(ids_unknown[:, None, :], both.shape)[both]
        coefficients[cursor:stop] = laplacian[both]
        cursor = stop
    matrix = coo_matrix(
        (coefficients[:cursor], (rows[:cursor], columns[:cursor])),
        shape=(unknown_count, unknown_count),
    ).tocsr()
    return matrix, rhs, len(windows)


def solve_matte(guide, trimap, p, cancelled):
    foreground = trimap == 255
    background = trimap == 0
    unknown = ~(foreground | background)
    alpha = foreground.astype(np.float64)
    if not unknown.any():
        return alpha, {"windows": 0, "iterations": 0, "relative_residual": 0.0}
    if not foreground.any() or not background.any():
        raise ValidationError("Trimap needs known foreground and background", "trimap")
    matrix, rhs, windows = matting_system(guide, foreground, unknown, p, cancelled)
    diagonal = matrix.diagonal()
    if not np.isfinite(diagonal).all() or np.any(diagonal <= 0):
        raise ValidationError("Trimap does not constrain the unknown alpha system", "trimap")
    preconditioner = diags(1 / diagonal).tocsr()
    iterations = 0

    def step(_estimate):
        nonlocal iterations
        iterations += 1
        if cancelled():
            raise InterruptedError("Matting cancelled")

    solution, status = cg(
        matrix,
        rhs,
        M=preconditioner,
        rtol=float(p["solver_tolerance"]),
        maxiter=int(p["max_iterations"]),
        callback=step,
    )
    if status or not np.isfinite(solution).all():
        raise ValidationError(
            "Trimap matting did not converge. Increase iterations or narrow the unknown band.",
            "max_iterations",
        )
    scale = np.linalg.norm(rhs)
    residual = np.linalg.norm(matrix @ solution - rhs)
    alpha[unknown] = np.clip(solution, 0, 1)
    return alpha, {
        "windows": windows,
        "iterations": iterations,
        "relative_residual": float(residual / scale if scale else residual),
    }


class TrimapMatting(Block):
    type_id, title = "cv.trimap_matting", "Trimap alpha matting"
    category = block_type = "Image processing"
    subtype = "Segmentation"
    description = "Estimate soft foreground opacity from a trimap with closed-form image matting."
    inputs = (*SELECTION_INPUTS, Port("trimap"))
    outputs = (
        Port("alpha"),
        Port("cutout", required=False),
        Port("report", "data"),
    )
    defaults = {
        "trimap_alignment": "pixel_grid",
        "window_radius": 1,
        "color_regularization": 1e-7,
        "solver_tolerance": 1e-7,
        "max_iterations": 1000,
    }
    parameter_choices = {"trimap_alignment": ("pixel_grid", "same_frame")}
    parameter_ranges = {
        "window_radius": NumericRange(1, 3, whole=True),
        "color_regularization": NumericRange(1e-10, 1e-1),
        "solver_tolerance": NumericRange(1e-10, 1e-3),
        "max_iterations": NumericRange(1, 5000, whole=True),
    }
    parameter_help = {
        "trimap_alignment": "Use matching rasters or also check known spatial frame identities.",
        "window_radius": "Radius of each local color model. Default gives 3 × 3 windows.",
        "color_regularization": "Diagonal regularization of the normalized color scatter matrix.",
        "solver_tolerance": "Relative residual tolerance for the conjugate-gradient solve.",
        "max_iterations": "Maximum iterations before the block reports lack of convergence.",
    }
    input_help = {
        "image": "Gray or RGB/BGR composite, including alpha-tagged images.",
        "trimap": "Matching uint8 gray raster. 0 is background, 255 is foreground, others unknown.",
        "mask": "Limit published alpha; the solve retains all trimap constraints.",
        "region": "Intersect a source-grid region with the optional output mask.",
    }
    output_help = {
        "alpha": "Normalized float32 foreground opacity on the original pixel grid.",
        "cutout": "Source colors with the estimated alpha attached as a straight-alpha channel.",
        "report": "Trimap counts and convergence diagnostics for the sparse solve.",
    }
    payload_reference = "image"
    process_inputs_readonly = preview_safe = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_inputs = ["image", "trimap"]
        self.preview_output = "alpha"

    def process(self, inputs):
        image, trimap = inputs["image"], inputs["trimap"]
        height, width = image.shape[:2]
        if not 1 <= height * width <= _MAX_PIXELS:
            raise ValidationError("Trimap matting supports up to 2,097,152 image pixels")
        if trimap.dtype != np.uint8 or trimap.shape != (height, width):
            raise ValidationError(
                "Trimap must be uint8 grayscale matching the image size", "trimap"
            )
        p = self.parameters
        incoming = self.input_payloads["image"]
        trimap_context = self.input_payloads["trimap"]
        for context in (incoming, trimap_context):
            check_spatial(context, [width, height])
        if p["trimap_alignment"] == "same_frame":
            same_grid(trimap_context, incoming, "trimap")
        selected = selection_mask(inputs, image)
        rgb, old_alpha, representation = split_layer(image, "auto", "Matting image")
        if representation == "premultiplied":
            rgb = unassociate(rgb, old_alpha)
        if rgb.min() < 0 or rgb.max() > 1:
            raise ValidationError("Matting image colors must be in [0, 1] after normalization")
        encoding = image_format(image)
        guide = rgb[..., :1] if encoding == "gray" else rgb
        alpha, solve = solve_matte(guide, trimap, p, self._cancelled)
        if selected is not None:
            alpha[selected == 0] = 0
        alpha_image = ImageArray(alpha.astype(np.float32), "gray")
        context = copy.deepcopy(incoming)
        for name in ("alpha_representation", "raw_sensor"):
            context.pop(name, None)
        payloads = {"alpha": context, "report": context}
        cutout = NO_VALUE
        if "cutout" in self.exposed_outputs:
            cutout, _ = pack_image(
                rgb, alpha, "rgba" if encoding in ("rgb", "rgba") else "bgra", image.dtype
            )
            payloads["cutout"] = {**context, "alpha_representation": "straight"}
        report = {
            "backend": "scipy_closed_form",
            "known_background_pixels": int(np.count_nonzero(trimap == 0)),
            "known_foreground_pixels": int(np.count_nonzero(trimap == 255)),
            "unknown_pixels": int(np.count_nonzero((trimap != 0) & (trimap != 255))),
            **solve,
        }
        return Result(
            {"alpha": alpha_image, "cutout": cutout, "report": report},
            payloads=payloads,
            message=f"{report['unknown_pixels']} unknown pixels · {solve['iterations']} iterations",
        )
