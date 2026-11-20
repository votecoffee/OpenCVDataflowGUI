"""Shared validation, statistics and previews for aligned raster comparisons."""

import cv2
import numpy as np

from .images import ImageArray
from .parameter_validation import ValidationError
from .payloads import require_same_frame
from .selection import selection_mask

MAX_EVALUATION_PIXELS = 16_777_216


def comparison_selection(inputs, alignment, label):
    """Check companion rasters and resolve the estimate's optional selection."""
    if alignment == "same_frame":
        try:
            require_same_frame(
                [value for name, value in inputs.items() if name != "region"],
                message=f"{label} rasters must share the same pixel frame",
            )
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
    return selection_mask(inputs, inputs["estimate"], match_mask_frame=False)


def apply_valid_mask(valid, mask, name):
    if mask is not None:
        if mask.shape != valid.shape or mask.dtype != np.uint8:
            raise ValidationError(f"{name.title()} valid mask needs matching uint8 pixels")
        valid &= mask != 0
    return valid


def error_statistics(residual, absolute, units, label):
    keys = (f"bias_{units}", f"mae_{units}", f"mse_{units}2", f"rmse_{units}", f"max_error_{units}")
    if not residual.size:
        return dict.fromkeys(keys)
    with np.errstate(over="ignore", invalid="ignore"):
        mse = float(np.dot(residual, residual) / residual.size)
    if not np.isfinite(mse):
        raise ValidationError(f"{label} error exceeds the numeric range")
    return dict(
        zip(
            keys,
            (float(residual.mean()), float(absolute.mean()), mse, mse**0.5, float(absolute.max())),
            strict=True,
        )
    )


def error_heatmap(scored, absolute, missing, ceiling):
    gray = np.zeros(scored.shape, dtype=np.uint8)
    gray[scored] = np.rint(np.minimum(absolute, ceiling) / ceiling * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    heatmap[~scored] = (0, 0, 0)
    heatmap[missing] = (255, 0, 255)
    return ImageArray(heatmap, "bgr")
