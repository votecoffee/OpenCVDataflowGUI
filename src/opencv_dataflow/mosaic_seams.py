"""Choose a content-aware seam without changing the samples it selects."""

import cv2
import numpy as np

from .parameter_validation import NumericRange, ValidationError

_COSTS = {"color": "COST_COLOR", "color_gradient": "COST_COLOR_GRAD"}
_CONTEXT = 16
_NATIVE_PADDING = 10
_MAX_GRAPH_PIXELS = 524_288

SEAM_DEFAULTS = {
    "seam_cost": "color_gradient",
    "seam_terminal_cost": 10000.0,
    "seam_bad_region_penalty": 1000.0,
}
SEAM_CHOICES = {"seam_cost": tuple(_COSTS)}
SEAM_RANGES = {
    "seam_terminal_cost": NumericRange(1, 1_000_000),
    "seam_bad_region_penalty": NumericRange(0, 1_000_000),
}
SEAM_HELP = {
    "seam_cost": "Graph cut uses color disagreement alone or weights it by local gradients.",
    "seam_terminal_cost": "Graph cut weight for retaining pixels supported by a view.",
    "seam_bad_region_penalty": "Graph cut penalty for crossing invalid support boundaries.",
}


def _cost_images(images, masks):
    signals = [image if image.ndim == 2 else image[..., :3] for image in images]
    values = [signal[mask] for signal, mask in zip(signals, masks, strict=True)]
    dtype = images[0].dtype
    if dtype.kind in "ui":
        limits = np.iinfo(dtype)
        low, high = float(limits.min), float(limits.max)
    else:
        low = min(float(value.min()) for value in values)
        high = max(float(value.max()) for value in values)
        if not all(np.isfinite(value).all() for value in values):
            raise ValidationError("Graph-cut seam requires finite warped pixels")

    # Normalize both views together and avoid overflow for large floating-point samples.
    scale = max(1.0, abs(low), abs(high))
    low, high = low / scale, high / scale
    result = []
    for signal, value, mask in zip(signals, values, masks, strict=True):
        cost = np.zeros(signal.shape, dtype=np.float32)
        if high > low:
            cost[mask] = ((value.astype(np.float64) / scale - low) / (high - low)) * 255
        result.append(cv2.cvtColor(cost, cv2.COLOR_GRAY2BGR) if cost.ndim == 2 else cost)
    return result


def graph_cut_selection(existing, incoming, existing_valid, incoming_valid, settings):
    """Return an incoming-view selector and counts for the true overlap."""
    overlap = existing_valid & incoming_valid
    use_incoming = incoming_valid & ~existing_valid
    total = int(np.count_nonzero(overlap))
    selected = 0
    cost = settings["seam_cost"]
    if total:
        x, y, width, height = cv2.boundingRect(overlap.view(np.uint8))
        rows, cols = overlap.shape
        x0, y0 = max(0, x - _CONTEXT), max(0, y - _CONTEXT)
        x1, y1 = min(cols, x + width + _CONTEXT), min(rows, y + height + _CONTEXT)
        work_pixels = (x1 - x0 + 2 * _NATIVE_PADDING) * (y1 - y0 + 2 * _NATIVE_PADDING)
        if work_pixels > _MAX_GRAPH_PIXELS:
            raise ValidationError(
                f"Graph-cut work area exceeds {_MAX_GRAPH_PIXELS:,} pixels", "blend"
            )
        area = np.s_[y0:y1, x0:x1]
        valid = [existing_valid[area], incoming_valid[area]]
        images = _cost_images((existing[area], incoming[area]), valid)
        finder = cv2.detail.GraphCutSeamFinder(
            _COSTS[cost],
            float(settings["seam_terminal_cost"]),
            float(settings["seam_bad_region_penalty"]),
        )
        old_mask, new_mask = finder.find(
            images, [(0, 0), (0, 0)], [v.astype(np.uint8) for v in valid]
        )
        # Native ties or gaps keep the existing view. Original support is authoritative.
        wins = (new_mask.get() != 0) & (old_mask.get() == 0) & overlap[area]
        use_incoming[area] |= wins
        selected = int(np.count_nonzero(wins))
    return use_incoming, {
        "cost": cost,
        "overlap_pixels": total,
        "overlap_from_existing": total - selected,
        "overlap_from_incoming": selected,
    }
