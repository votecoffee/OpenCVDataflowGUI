"""Estimate a right-to-left exposure correction on rectified stereo correspondences."""

import cv2
import numpy as np

from .core import Port, Result
from .disparity_search import rectification_rois
from .geometry3d_common import Tool3D
from .images import DepthArray, ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, require_same_frame
from .selection import selection_mask
from .stereo3d import STEREO_GRID, check_stereo_grids
from .tickets import ticket_key

_MAX_PIXELS = 8_388_608
_BAND_PIXELS = 262_144
_FORMAT_CHANNELS = {"gray": 1, "rgb": 3, "bgr": 3, "rgba": 4, "bgra": 4}


def _color_view(image):
    enc = image_format(image)
    channels = _FORMAT_CHANNELS.get(enc)
    if (
        isinstance(image, DepthArray)
        or image.dtype not in (np.uint8, np.uint16, np.float32)
        or channels is None
        or (channels == 1 and image.ndim != 2)
        or (channels > 1 and (image.ndim != 3 or image.shape[2] != channels))
    ):
        raise ValidationError("Stereo correction requires gray, RGB/BGR or RGBA/BGRA images")
    base = np.asarray(image)
    if channels == 1:
        view = base[..., None]
    else:
        view = base[..., :3] if enc.startswith("rgb") else base[..., 2::-1]
    if image.dtype == np.float32 and (
        not np.isfinite(view).all() or view.min() < 0 or view.max() > 1
    ):
        raise ValidationError("Float stereo samples must be finite and in [0, 1]")
    scale = float(np.iinfo(image.dtype).max) if image.dtype.kind == "u" else 1.0
    return view, scale


def _support(image, roi, mask, region=None):
    allowed = selection_mask({"mask": mask, "region": region}, image)
    x, y, width, height = roi
    result = np.zeros(image.shape[:2], bool)
    zone = np.s_[y : y + height, x : x + width]
    result[zone] = True if allowed is None else allowed[zone] != 0
    return result


def _matching_points(disparity, left_allowed, right_allowed):
    width = disparity.shape[1]
    guides, left = disparity.ravel(), left_allowed.ravel()
    bands = []
    for start in range(0, disparity.size, _BAND_PIXELS):
        band = slice(start, start + _BAND_PIXELS)
        points = np.flatnonzero(left[band] & np.isfinite(guides[band])) + start
        y, x = np.divmod(points, width)
        xr = x - guides[points].astype(np.float64)
        in_bounds = (xr >= 0) & (xr <= width - 1)
        y, xr, points = y[in_bounds], xr[in_bounds], points[in_bounds]
        low, high = np.floor(xr).astype(np.intp), np.ceil(xr).astype(np.intp)
        supported = right_allowed[y, low] & right_allowed[y, high]
        bands.append(points[supported].astype(np.int32))
    return np.concatenate(bands)


def _sample_pairs(left, right, disparity, left_allowed, right_allowed, limit, margin):
    a, a_scale = left
    b, b_scale = right
    points = _matching_points(disparity, left_allowed, right_allowed)
    geometric_count = len(points)
    if geometric_count > limit:
        points = points[np.linspace(0, geometric_count - 1, limit, dtype=np.intp)]
    y, x = np.divmod(points, disparity.shape[1])
    xr = x - disparity[y, x].astype(np.float64)
    low, high = np.floor(xr).astype(np.intp), np.ceil(xr).astype(np.intp)
    left_signal = a[y, x] / a_scale
    first, second = b[y, low] / b_scale, b[y, high] / b_scale
    usable = np.ones(len(points), bool)
    for signal in (left_signal, first, second):
        usable &= ((signal > margin) & (signal < 1 - margin)).all(axis=1)
    fraction = (xr[usable] - low[usable])[:, None]
    right_signal = first[usable] * (1 - fraction) + second[usable] * fraction
    return y[usable], x[usable], left_signal[usable], right_signal, geometric_count, len(points)


def _weighted_model(source, target, weights, model, minimum_spread):
    w = weights[:, None] / weights.sum()
    center = np.sum(w * source, axis=0)
    reference = np.sum(w * target, axis=0)
    spread = np.sqrt(np.sum(w * (source - center) ** 2, axis=0))
    if np.any(spread < minimum_spread):
        return None
    if model == "gain_bias":
        x, y = source - center, target - reference
        gain = np.sum(w * x * y, axis=0) / spread**2
        return gain, reference - gain * center
    gain = np.sum(w * source * target, axis=0) / np.sum(w * source**2, axis=0)
    return gain, np.zeros_like(gain)


def _model_residual(source, target, fit):
    gain, bias = fit
    return np.abs(source * gain + bias - target)


def _fit(source, target, settings):
    tolerance = float(settings["residual_tolerance"])
    floor = float(settings["minimum_signal_std"])
    model = settings["model"]
    minimum = int(settings["minimum_samples"])
    weights = np.ones(len(source))
    for _ in range(8):
        fit = _weighted_model(source, target, weights, model, floor)
        if fit is None:
            return None, None, "stereo_color_insufficient_signal"
        residual = _model_residual(source, target, fit).max(axis=1)
        weights = tolerance / np.maximum(tolerance, residual)
    for _ in range(2):
        inliers = _model_residual(source, target, fit).max(axis=1) <= tolerance
        if np.count_nonzero(inliers) < minimum:
            return fit, inliers, "stereo_color_insufficient_inliers"
        fit = _weighted_model(source[inliers], target[inliers], weights[inliers], model, floor)
        if fit is None:
            return None, None, "stereo_color_insufficient_signal"
    inliers = _model_residual(source, target, fit).max(axis=1) <= tolerance
    gain, _ = fit
    if np.any((gain < 0.05) | (gain > 20)):
        return fit, inliers, "stereo_color_implausible_model"
    if np.count_nonzero(inliers) < minimum or inliers.mean() < float(
        settings["minimum_inlier_ratio"]
    ):
        return fit, inliers, "stereo_color_insufficient_inliers"
    return fit, inliers, None


def _apply(source, view, scale, allowed, fit):
    output = source.copy()
    target, _ = _color_view(output)
    gain, bias = fit
    rows = max(1, _BAND_PIXELS // view.shape[1])
    for start in range(0, view.shape[0], rows):
        band = slice(start, start + rows)
        corrected = np.clip(view[band] * (gain / scale) + bias, 0, 1) * scale
        if output.dtype.kind == "u":
            np.rint(corrected, out=corrected)
        np.copyto(target[band], corrected.astype(output.dtype), where=allowed[band, :, None])
    return output


class MatchStereoColor(Tool3D):
    type_id, title, subtype = "3d.stereo_color_match", "Match stereo exposure and color", "Stereo"
    inputs = (
        Port("left"),
        Port("right"),
        Port("disparity"),
        Port("rectification", "data", required=False, required_when_connected=True),
        Port("left_mask", required=False, required_when_connected=True, visual_type="region"),
        Port(
            "left_region",
            "data",
            required=False,
            required_when_connected=True,
            visual_type="region",
        ),
        Port("right_mask", required=False, required_when_connected=True, visual_type="region"),
    )
    outputs = (
        Port("right", required=False),
        Port("fit_mask", visual_type="region"),
        Port("report", "data"),
    )
    defaults = {
        "model": "gain_bias",
        "minimum_samples": 128,
        "maximum_samples": 32_768,
        "saturation_margin": 0.004,
        "minimum_signal_std": 0.004,
        "residual_tolerance": 0.04,
        "minimum_inlier_ratio": 0.65,
    }
    parameter_choices = {"model": ("gain_bias", "gain_only")}
    parameter_ranges = {
        "minimum_samples": NumericRange(8, 131_072, whole=True),
        "maximum_samples": NumericRange(8, 131_072, whole=True),
        "saturation_margin": NumericRange(0, 0.25),
        "minimum_signal_std": NumericRange(1e-6, 0.25),
        "residual_tolerance": NumericRange(1e-6, 1),
        "minimum_inlier_ratio": NumericRange(0, 1),
    }
    parameter_help = {
        "model": "Fit one gain per channel, with optional additive bias.",
        "minimum_samples": "Minimum count of photometric inliers needed to emit a corrected image.",
        "maximum_samples": "Uniformly subsample eligible correspondences before fitting.",
        "saturation_margin": "Exclude normalized samples this close to either endpoint.",
        "minimum_signal_std": "Minimum standard deviation in each right channel.",
        "residual_tolerance": "Largest allowed channel residual in normalized units for an inlier.",
        "minimum_inlier_ratio": "Minimum fraction of usable sampled pairs that must agree.",
    }
    input_help = {
        "left": "Rectified reference image. The tool learns the right-to-left correction.",
        "right": "Rectified image to correct without moving its pixels.",
        "disparity": "Float pixels on the left grid. Right X = left X − disparity. NaN is invalid.",
        "rectification": "Calibrated horizontal rectification with per-view ROIs.",
        "left_mask": "Intersect with Left Region and finite disparities.",
        "left_region": "Optional left-grid fitting region. Limits fitting only.",
        "right_mask": "Limit right sampling and correction to supported pixels.",
    }
    payload_reference = "left"
    flow_primary = ("right",)
    parallel_safe = preview_safe = process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["maximum_samples"] < p["minimum_samples"]:
            raise ValidationError(
                "Maximum samples must cover minimum samples", "maximum_samples", "minimum_samples"
            )

    def process(self, inputs):
        p = self.parameters
        left, right, guide = (inputs[name] for name in ("left", "right", "disparity"))
        shape = left.shape[:2]
        if not 0 < shape[0] * shape[1] <= _MAX_PIXELS or right.shape[:2] != shape:
            raise ValidationError(
                "Stereo correction requires matching grids of up to 8,388,608 pixels"
            )
        if (
            isinstance(guide, DepthArray)
            or guide.shape != shape
            or guide.dtype not in (np.float32, np.float64)
        ):
            raise ValidationError("Supply left-grid float disparity in pixels")
        check_stereo_grids(inputs, self.input_payloads, shape[::-1])
        left_payload, guide_payload = (self.input_payloads[name] for name in ("left", "disparity"))
        require_same_frame((left, guide), message="Disparity belongs to a different left grid")
        if left_payload.get(STEREO_GRID) != guide_payload.get(STEREO_GRID):
            raise ValidationError("Disparity must come from the same sealed left rectification")
        tickets = {ticket_key(v["ticket"]) for v in self.input_payloads.values() if "ticket" in v}
        if len(tickets) > 1:
            raise ValidationError("Stereo correction inputs must refer to the same ticket")

        reference, source = _color_view(left), _color_view(right)
        if reference[0].shape[2] != source[0].shape[2]:
            raise ValidationError("Stereo correction needs two gray or two color images")
        a_roi, b_roi = rectification_rois(inputs.get("rectification"), shape[::-1])
        a = _support(left, a_roi, inputs.get("left_mask"), inputs.get("left_region"))
        b = _support(right, b_roi, inputs.get("right_mask"))
        y, x, reference_signal, source_signal, eligible, sampled = _sample_pairs(
            reference,
            source,
            np.asarray(guide),
            a,
            b,
            int(p["maximum_samples"]),
            float(p["saturation_margin"]),
        )
        return self._result(
            right, source, b, y, x, reference_signal, source_signal, eligible, sampled
        )

    def _result(self, right, source, apply_mask, y, x, reference, measured, eligible, sampled):
        p = self.parameters
        fit, inliers, reason = None, None, "stereo_color_insufficient_samples"
        if len(measured) >= p["minimum_samples"]:
            fit, inliers, reason = _fit(measured, reference, p)
        fit_mask = np.zeros(right.shape[:2], np.uint8)
        if inliers is not None:
            fit_mask[y[inliers], x[inliers]] = 255
        matched = int(np.count_nonzero(fit_mask))
        ratio = matched / len(measured) if len(measured) else 0.0
        absolute = np.abs(measured - reference)
        after = _model_residual(measured, reference, fit) if fit is not None else None
        report = {
            "model": "stereo_photometric_match_v1",
            "fit": p["model"],
            "channel_order": "gray" if source[0].shape[2] == 1 else "rgb",
            "eligible_pairs": eligible,
            "sampled_pairs": sampled,
            "saturated_pairs": sampled - len(measured),
            "usable_pairs": len(measured),
            "inlier_pairs": matched,
            "inlier_ratio": ratio,
            "right_to_left_gain": fit[0].tolist() if fit is not None else None,
            "right_to_left_bias": fit[1].tolist() if fit is not None else None,
            "mae_before": float(absolute.mean()) if len(absolute) else None,
            "mae_after": float(after.mean()) if after is not None else None,
            "failure": reason,
        }
        corrected = NO_VALUE if reason else _apply(right, *source, apply_mask, fit)
        graphics = {}
        if self.capture_graphics and after is not None:
            residual = np.zeros(fit_mask.shape, np.uint8)
            limit = float(p["residual_tolerance"])
            residual[y, x] = np.rint(np.minimum(after.max(axis=1) / limit, 1) * 255)
            graphic = cv2.applyColorMap(residual, cv2.COLORMAP_TURBO)
            samples = np.zeros(fit_mask.shape, bool)
            samples[y, x] = True
            graphic[~samples] = 0
            graphics["Fit residual"] = ImageArray(graphic, "bgr")
        status = f"{matched}/{len(measured)} usable stereo pairs agree"
        if reason:
            detail = reason.removeprefix("stereo_color_").replace("_", " ")
            status = f"Correction withheld due to {detail}. {status}"
        return Result(
            {"right": corrected, "fit_mask": fit_mask, "report": report},
            payloads={"right": self.input_payloads["right"]},
            graphics=graphics,
            passed=reason is None,
            score=ratio,
            fail_reasons=[] if reason is None else [reason],
            message=status,
        )
