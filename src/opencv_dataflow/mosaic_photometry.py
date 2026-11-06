"""Fit overlap exposure and feather valid pixels without changing the mosaic grid."""

import cv2
import numpy as np

from .analysis_tools import cast_intensity
from .parameter_validation import ValidationError

MAX_EXPOSURE_SAMPLES = 32_768
IDENTITY_EXPOSURE = (1.0, 0.0)


def _intensity_at(image, indices):
    if image.ndim == 2:
        return image.ravel()[indices].astype(np.float64)
    channels = image.reshape(-1, image.shape[2])[indices, :3]
    return channels.mean(axis=1, dtype=np.float64)


def fit_overlap_exposure(previous, current, overlap, previous_correction, settings):
    """Estimate a shared correction to the anchor from unsaturated overlap pixels."""
    indices = np.flatnonzero(overlap)
    if len(indices) > MAX_EXPOSURE_SAMPLES:
        sample = np.linspace(0, len(indices) - 1, MAX_EXPOSURE_SAMPLES, dtype=np.intp)
        indices = indices[sample]
    a, b = _intensity_at(previous, indices), _intensity_at(current, indices)
    full_scale = float(np.iinfo(current.dtype).max) if current.dtype.kind == "u" else 1.0
    floor = full_scale / 1024
    eligible = (a > floor) & (a < full_scale) & (b > floor) & (b < full_scale)
    count = int(eligible.sum())
    report = {"samples": count}
    if count < int(settings["min_exposure_samples"]):
        report["status"] = "insufficient_exposure_samples"
        return None, report

    previous_gain, previous_offset = previous_correction
    target = a[eligible] * previous_gain + previous_offset
    source = b[eligible]
    if settings["exposure"] == "gain":
        gain, offset = float(np.median(target / source)), 0.0
    else:
        gain, offset = 1.0, float(np.median(target - source))
    report.update(
        gain=gain,
        offset=offset,
        median_error_before=float(np.median(np.abs(target - source))),
        median_error_after=float(np.median(np.abs(target - (gain * source + offset)))),
    )
    limit = float(settings["max_exposure_gain"])
    if not 1 / limit <= gain <= limit or abs(offset) > full_scale * float(
        settings["max_exposure_offset_fraction"]
    ):
        report["status"] = "exposure_out_of_range"
        return None, report
    report["status"] = "corrected"
    return (gain, offset), report


def corrected_view(image, valid, correction):
    if correction == IDENTITY_EXPOSURE:
        return image
    gain, offset = correction
    adjusted = image.astype(np.float64)
    adjusted[~valid] = 0
    # Alpha remains a separate channel and never participates in the exposure fit.
    signal = adjusted[..., :3] if adjusted.ndim == 3 else adjusted
    with np.errstate(over="ignore", invalid="ignore"):
        signal *= gain
        signal += offset
    if not np.isfinite(adjusted).all():
        raise ValidationError("Exposure correction produced nonfinite samples")
    return adjusted


class FeatherMosaic:
    def __init__(self, shape, width):
        self.width = width
        self.total = np.zeros(shape, dtype=np.float64)
        self.weights = np.zeros(shape[:2], dtype=np.float64)

    def add(self, image, valid):
        # Pad the canvas so a view touching its outer edge still has a boundary.
        padded = cv2.copyMakeBorder(valid.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT)
        weights = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_3)[1:-1, 1:-1]
        weights /= self.width
        np.minimum(weights, 1.0, out=weights)
        self.weights += weights
        if image.ndim == 3:
            weights = weights[..., None]
        with np.errstate(over="ignore", invalid="ignore"):
            weighted = np.multiply(image, weights, dtype=np.float64)
            weighted[~valid] = 0
            self.total += weighted

    def finish(self, dtype):
        weights = self.weights if self.total.ndim == 2 else self.weights[..., None]
        np.divide(self.total, weights, out=self.total, where=weights > 0)
        if not np.isfinite(self.total).all():
            raise ValidationError("Feather blending produced nonfinite samples")
        return cast_intensity(self.total, dtype)
