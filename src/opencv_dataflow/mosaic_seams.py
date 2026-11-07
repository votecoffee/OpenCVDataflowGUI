"""Choose a content-aware seam without changing the samples it selects."""

import cv2
import numpy as np

from .parameter_validation import NumericRange, ValidationError

_COSTS = {"color": "COST_COLOR", "color_gradient": "COST_COLOR_GRAD"}
_RESOLUTIONS = {"full": 1, "half": 2, "quarter": 4}
_CONTEXT = 16
_NATIVE_PADDING = 10
_MAX_GRAPH_PIXELS = 524_288

SEAM_DEFAULTS = {
    "seam_cost": "color_gradient",
    "seam_resolution": "auto",
    "seam_terminal_cost": 10000.0,
    "seam_bad_region_penalty": 1000.0,
}
SEAM_CHOICES = {"seam_cost": tuple(_COSTS), "seam_resolution": ("auto", *_RESOLUTIONS)}
SEAM_RANGES = {
    "seam_terminal_cost": NumericRange(1, 1_000_000),
    "seam_bad_region_penalty": NumericRange(0, 1_000_000),
}
SEAM_HELP = {
    "seam_cost": "Graph cut uses color disagreement alone or weights it by local gradients.",
    "seam_resolution": "Estimate at full, half, or quarter resolution. Auto fits the work limit.",
    "seam_terminal_cost": "Graph cut weight for retaining pixels supported by a view.",
    "seam_bad_region_penalty": "Graph cut penalty for crossing invalid support boundaries.",
}


def _work_grid(width, height, resolution):
    factor = _RESOLUTIONS.get(resolution, 1)
    while True:
        cols, rows = (width + factor - 1) // factor, (height + factor - 1) // factor
        pixels = (cols + 2 * _NATIVE_PADDING) * (rows + 2 * _NATIVE_PADDING)
        if pixels <= _MAX_GRAPH_PIXELS:
            return (cols, rows), factor, pixels
        if resolution != "auto":
            raise ValidationError(
                f"Graph-cut work area exceeds {_MAX_GRAPH_PIXELS:,} pixels", "seam_resolution"
            )
        factor *= 2


def _cost_image(signal, mask, size, normalization):
    low, high, scale = normalization
    reduced = size != mask.shape[::-1]
    weights = (
        cv2.resize(mask.astype(np.float32), size, interpolation=cv2.INTER_AREA) if reduced else None
    )
    valid = weights > 0 if reduced else mask
    cost = np.zeros((*size[::-1], 3), dtype=np.float32)
    if high > low:
        # Reuse one full-resolution plane, including for RGB and HDR views.
        plane = np.zeros(mask.shape, dtype=np.float64)
        channels = (signal,) if signal.ndim == 2 else np.moveaxis(signal, -1, 0)
        for index, channel in enumerate(channels):
            np.divide(channel, scale, out=plane, where=mask, dtype=np.float64)
            np.subtract(plane, low, out=plane, where=mask)
            np.divide(plane, high - low, out=plane, where=mask)
            values = plane
            if reduced:
                values = cv2.resize(plane, size, interpolation=cv2.INTER_AREA)
                np.divide(values, weights, out=values, where=valid)
            np.multiply(values, 255, out=values)
            cost[..., index] = values
    if signal.ndim == 2:
        cost[..., 1:] = cost[..., :1]
    return cost, valid.astype(np.uint8)


def _cost_images(images, masks, size):
    signals = [image if image.ndim == 2 else image[..., :3] for image in images]
    dtype = images[0].dtype
    if dtype.kind in "ui":
        limits = np.iinfo(dtype)
        low, high = float(limits.min), float(limits.max)
    else:
        limits = []
        for signal, mask in zip(signals, masks, strict=True):
            valid = mask if signal.ndim == 2 else mask[..., None]
            limits.append(
                (
                    signal.min(where=valid, initial=np.inf),
                    signal.max(where=valid, initial=-np.inf),
                )
            )
        if not np.isfinite(limits).all():
            raise ValidationError("Graph-cut seam requires finite warped pixels")
        low = min(float(a) for a, _ in limits)
        high = max(float(b) for _, b in limits)

    # Normalize both views together and avoid overflow for large floating-point samples.
    scale = max(1.0, abs(low), abs(high))
    normalization = (low / scale, high / scale, scale)
    costs, work_masks = [], []
    for signal, mask in zip(signals, masks, strict=True):
        cost, support = _cost_image(signal, mask, size, normalization)
        costs.append(cost)
        work_masks.append(support)
    return costs, work_masks


def graph_cut_selection(existing, incoming, existing_valid, incoming_valid, settings):
    """Return an incoming-view selector and counts for the true overlap."""
    overlap = existing_valid & incoming_valid
    use_incoming = incoming_valid & ~existing_valid
    total = int(np.count_nonzero(overlap))
    selected = 0
    cost = settings["seam_cost"]
    work = {}
    if total:
        x, y, width, height = cv2.boundingRect(overlap.view(np.uint8))
        rows, cols = overlap.shape
        x0, y0 = max(0, x - _CONTEXT), max(0, y - _CONTEXT)
        x1, y1 = min(cols, x + width + _CONTEXT), min(rows, y + height + _CONTEXT)
        area_size = (x1 - x0, y1 - y0)
        size, factor, pixels = _work_grid(*area_size, settings["seam_resolution"])
        work = {"downsample_factor": factor, "work_size": list(size), "work_pixels": pixels}
        area = np.s_[y0:y1, x0:x1]
        valid = [existing_valid[area], incoming_valid[area]]
        images, masks = _cost_images((existing[area], incoming[area]), valid, size)
        finder = cv2.detail.GraphCutSeamFinder(
            _COSTS[cost],
            float(settings["seam_terminal_cost"]),
            float(settings["seam_bad_region_penalty"]),
        )
        old_mask, new_mask = finder.find(images, [(0, 0), (0, 0)], masks)
        # Native ties or gaps keep the existing view. Original support is authoritative.
        wins = (new_mask.get() != 0) & (old_mask.get() == 0)
        if size != area_size:
            wins = (
                cv2.resize(wins.astype(np.uint8), area_size, interpolation=cv2.INTER_NEAREST_EXACT)
                != 0
            )
        wins &= overlap[area]
        use_incoming[area] |= wins
        selected = int(np.count_nonzero(wins))
    return use_incoming, {
        "cost": cost,
        "overlap_pixels": total,
        "overlap_from_existing": total - selected,
        "overlap_from_incoming": selected,
        **work,
    }
