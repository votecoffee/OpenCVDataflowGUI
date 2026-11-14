"""Robust photometric agreement of aligned inspection images, without registration."""

import copy

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .processing import image_value
from .result_graphics import ACCEPTED, REJECTED
from .selection import selection_mask

MAX_PIXELS = 4_194_304
MAX_SAMPLES = 8_388_608
MAX_INTENSITY = 1e12


def _fit_plane(source, reference, selected, p):
    """Weighted local moments; centered/scaled work avoids raw-intensity squares."""
    origin_x = float(source[selected].mean()) if selected.any() else 0
    origin_y = float(reference[selected].mean()) if selected.any() else 0
    scale = max(
        1.0, float(np.max(np.abs(source - origin_x))), float(np.max(np.abs(reference - origin_y)))
    )
    x, y = (source - origin_x) / scale, (reference - origin_y) / scale
    local = p["scope"] == "local"
    kernel = int(p["window"])

    def total(value):
        if local:
            return cv2.boxFilter(
                value, cv2.CV_64F, (kernel, kernel), normalize=False, borderType=cv2.BORDER_CONSTANT
            )
        return float(np.sum(value))

    weights = selected.astype(np.float64)
    area = total(np.ones_like(x)) if local else float(selected.sum())
    minimum = np.maximum(float(p["minimum_fit_pixels"]), p["minimum_fit_fraction"] * area)
    iterations = int(p["iterations"]) if p["robust_loss"] == "tukey" else 0
    regularizer = (float(p["gain_regularization"]) / scale) ** 2
    floor = max((float(p["minimum_contrast"]) / scale) ** 2, 128 * np.finfo(float).eps)
    for step in range(iterations + 1):
        count = total(weights)
        divisor = np.maximum(count, 1e-12)
        mx, my = total(weights * x) / divisor, total(weights * y) / divisor
        variance = np.maximum(total(weights * x * x) / divisor - mx * mx, 0)
        covariance = total(weights * x * y) / divisor - mx * my
        contrast = variance > floor
        if p["model"] == "gain_offset":
            gain = np.where(
                contrast,
                (covariance + regularizer) / np.maximum(variance + regularizer, 1e-30),
                1.0,
            )
        else:
            gain = np.ones_like(variance)
        offset = my - gain * mx
        gain_roundoff = 64 * np.finfo(float).eps * np.maximum(1, np.abs(gain))
        supported = (
            (count >= minimum)
            & (gain >= p["minimum_gain"] - gain_roundoff)
            & (gain <= p["maximum_gain"] + gain_roundoff)
        )
        if step < iterations:
            residual = (y - gain * x - offset) / (float(p["outlier_limit"]) / scale)
            weights = np.where(
                selected & supported & (np.abs(residual) < 1),
                np.maximum(1 - residual * residual, 0) ** 2,
                0,
            )
    gain = np.broadcast_to(gain, source.shape)
    offset = np.broadcast_to(origin_y - gain * origin_x + offset * scale, source.shape)
    supported = np.broadcast_to(supported, source.shape)
    fallback = (
        np.broadcast_to(True if p["model"] == "offset_only" else ~contrast, source.shape)
        & supported
    )
    return gain, offset, supported, weights > 0, fallback


class DifferentialExposure(Block):
    type_id, title = "inspect.differential_exposure", "Compare exposure-adjusted images"
    category = block_type = "Inspection"
    subtype = "Reference inspection"
    inputs = (
        Port("image"),
        Port("reference"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
        Port("fit_mask", required=False, required_when_connected=True),
        Port("fit_region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("adjusted"),
        Port("residual"),
        Port("raw_residual"),
        Port("valid_mask", visual_type="mask"),
        Port("defect_mask", visual_type="mask"),
        Port("gain"),
        Port("offset"),
        Port("fit_inliers", visual_type="mask"),
        Port("info", "data"),
    )
    defaults = {
        "scope": "local",
        "window": 63,
        "model": "gain_offset",
        "robust_loss": "tukey",
        "iterations": 4,
        "outlier_limit": 8.0,
        "minimum_contrast": 1.0,
        "gain_regularization": 0.0,
        "minimum_gain": 0.25,
        "maximum_gain": 4.0,
        "minimum_fit_pixels": 16,
        "minimum_fit_fraction": 0.1,
        "minimum_valid_fraction": 0.95,
        "defect_threshold": 15.0,
        "maximum_defect_fraction": 0.0,
        "saturation": "integer_endpoints",
        "reference_alignment": "pixel_grid",
    }
    parameter_choices = {
        "scope": ("local", "global"),
        "model": ("gain_offset", "offset_only"),
        "robust_loss": ("tukey", "none"),
        "saturation": ("integer_endpoints", "none"),
        "reference_alignment": ("pixel_grid", "same_frame"),
    }
    parameter_ranges = {
        "window": NumericRange(3, 255, whole=True, odd=True),
        "iterations": NumericRange(1, 10, whole=True),
        "outlier_limit": NumericRange(1e-12, MAX_INTENSITY),
        "minimum_contrast": NumericRange(0, MAX_INTENSITY),
        "gain_regularization": NumericRange(0, MAX_INTENSITY),
        "minimum_gain": NumericRange(1e-6, 1e6),
        "maximum_gain": NumericRange(1e-6, 1e6),
        "minimum_fit_pixels": NumericRange(2, MAX_PIXELS, whole=True),
        "minimum_fit_fraction": NumericRange(0, 1),
        "minimum_valid_fraction": NumericRange(0, 1),
        "defect_threshold": NumericRange(0, MAX_INTENSITY),
        "maximum_defect_fraction": NumericRange(0, 1),
    }
    parameter_help = {
        "scope": "Local weighted windows, or one model per channel for the whole fit selection.",
        "window": "Odd local window. Only actual in-image pixels supply support; no reflection.",
        "model": "Reference = gain * image + offset, or fix gain at one. Alpha is not fitted.",
        "robust_loss": "Tukey refits reject residual outliers; None uses ordinary least squares.",
        "iterations": "Robust reweight/refit steps after the initial fit; unused for None.",
        "outlier_limit": (
            "Tukey cutoff in raw sample units. Residuals at/above it receive zero weight."
        ),
        "minimum_contrast": (
            "Raw standard-deviation floor. Low-contrast windows use offset-only correction."
        ),
        "gain_regularization": (
            "Raw contrast scale of a squared ridge penalty toward gain one; zero disables it."
        ),
        "minimum_gain": "Inclusive lower gain bound; out-of-range models provide no coverage.",
        "maximum_gain": "Inclusive upper gain bound; must be at least Minimum gain.",
        "minimum_fit_pixels": "Minimum effective weighted samples in each model/window.",
        "minimum_fit_fraction": (
            "Effective support / actual local window area, or / initial global fit selection."
        ),
        "minimum_valid_fraction": "Inclusive selected fraction with usable models and samples.",
        "defect_threshold": (
            "Maximum accepted raw-unit absolute residual; any channel exceeding it is a defect."
        ),
        "maximum_defect_fraction": "Allowed valid-pixel defect fraction; zero allows none.",
        "saturation": (
            "Exclude integer minimum/maximum endpoints from fitting and inspection, or disable. "
            "Floating data has no assumed saturation level."
        ),
        "reference_alignment": (
            "Pixel grid assumes equal-index registration; Same frame needs matching known frames."
        ),
    }
    input_help = {
        "image": "Candidate intensity image; owns output tickets, geometry and unchanged alpha.",
        "reference": "Aligned reference with matching dimensions, encoding and depth.",
        "mask": "Inspection selection; intersects Region. Residuals are zero outside it.",
        "region": "Optional inspection region on the candidate grid.",
        "fit_mask": "Stable fitting selection, independent of inspection; intersects Fit region.",
        "fit_region": (
            "Optional fitting region. With no fitting inputs, use the inspection selection."
        ),
    }
    payload_reference = "image"
    parallel_safe = preview_safe = process_inputs_readonly = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = [
            "image",
            "adjusted",
            "residual",
            "raw_residual",
            "valid_mask",
            "defect_mask",
            "info",
        ]

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["minimum_gain"] > p["maximum_gain"]:
            raise ValidationError(
                "Minimum gain exceeds Maximum gain", "minimum_gain", "maximum_gain"
            )
        if p["scope"] == "local" and p["minimum_fit_pixels"] > p["window"] ** 2:
            raise ValidationError(
                "Minimum fit pixels exceeds local window area", "minimum_fit_pixels", "window"
            )

    def process(self, inputs):
        source, reference, p = inputs["image"], inputs["reference"], self.parameters
        raw, target = (image_value(value) for value in (source, reference))
        encoding = image_format(source)
        if (
            raw.shape != target.shape
            or raw.dtype != target.dtype
            or encoding != image_format(reference)
            or encoding not in ("gray", "bgr", "rgb", "bgra", "rgba")
        ):
            raise ValidationError(
                "Use matching aligned intensity images, depths and channel encodings"
            )
        if not raw.size or np.prod(raw.shape[:2]) > MAX_PIXELS or raw.size > MAX_SAMPLES:
            raise ValidationError(
                "Comparison exceeds 4,194,304 pixels or 8,388,608 channel samples"
            )
        if (
            max(raw.shape[:2]) > 16384
            or max(np.max(np.abs(raw.astype(float))), np.max(np.abs(target.astype(float))))
            > MAX_INTENSITY
        ):
            raise ValidationError(
                "Image dimensions or absolute intensity exceed supported bounds (1e12)"
            )
        for name in ("image", "reference"):
            if self.input_payloads[name].get("alpha_representation") == "premultiplied":
                raise ValidationError("Unassociate premultiplied alpha before photometric fitting")
        source_payload, ref_payload = self.input_payloads["image"], self.input_payloads["reference"]
        if p["reference_alignment"] == "same_frame":
            a, b = source_payload["spatial"], ref_payload["spatial"]
            if (
                a["root_id"] is None
                or any(a[k] != b[k] for k in ("root_id", "root_size", "image_size"))
                or not np.allclose(a["to_root"], b["to_root"], atol=1e-9, rtol=0)
            ):
                raise ValidationError("Same-frame comparison requires matching known pixel frames")
        inspect_mask = selection_mask(inputs, source)
        selected = np.ones(raw.shape[:2], bool) if inspect_mask is None else inspect_mask != 0
        if not selected.any():
            raise ValidationError("Select at least one inspection pixel")
        fit_inputs = {
            name.removeprefix("fit_"): inputs[name]
            for name in ("fit_mask", "fit_region")
            if name in inputs
        }
        fit_mask = selection_mask(fit_inputs, source) if fit_inputs else inspect_mask
        fitted = np.ones(raw.shape[:2], bool) if fit_mask is None else fit_mask != 0
        channels = 1 if raw.ndim == 2 else min(raw.shape[2], 3)
        signal = raw[..., None] if raw.ndim == 2 else raw[..., :channels]
        other = target[..., None] if target.ndim == 2 else target[..., :channels]
        usable = np.ones(raw.shape[:2], bool)
        if p["saturation"] == "integer_endpoints" and raw.dtype.kind in "ui":
            limits = np.iinfo(raw.dtype)
            usable = (
                (signal > limits.min)
                & (signal < limits.max)
                & (other > limits.min)
                & (other < limits.max)
            ).all(axis=2)
        fitted &= usable
        valid, inliers = selected & usable, fitted.copy()
        gains, offsets, adjusted = (np.empty(signal.shape, np.float64) for _ in range(3))
        fallback = np.zeros(raw.shape[:2], bool)
        for index in range(channels):
            x, y = signal[..., index].astype(float), other[..., index].astype(float)
            gain, offset, supported, weights, low_contrast = _fit_plane(x, y, fitted, p)
            valid &= supported
            inliers &= weights
            fallback |= low_contrast
            gains[..., index] = np.where(supported, gain, 1)
            offsets[..., index] = np.where(supported, offset, 0)
            adjusted[..., index] = gains[..., index] * x + offsets[..., index]
        if not np.isfinite(adjusted).all():
            raise ValidationError("Exposure fitting produced nonfinite samples")
        difference = np.abs(adjusted - other)
        original_difference = np.abs(signal.astype(float) - other)
        residual = np.where(valid, difference.max(axis=2), 0)
        raw_residual = np.where(selected, original_difference.max(axis=2), 0)
        decision_roundoff = (
            64 * np.finfo(float).eps * np.maximum(np.abs(adjusted), np.abs(other)).max(axis=2)
        )
        defects = valid & (residual > p["defect_threshold"] + decision_roundoff)
        count, defects_count = int(valid.sum()), int(defects.sum())
        fraction = defects_count / count if count else None
        coverage = count / int(selected.sum())
        reasons = []
        if not count or coverage < p["minimum_valid_fraction"]:
            reasons.append(
                {
                    "code": "exposure_insufficient_support",
                    "message": (
                        f"Usable fitting coverage {coverage:.1%}; "
                        f"require {p['minimum_valid_fraction']:.1%}"
                    ),
                }
            )
        if fraction is not None and fraction > p["maximum_defect_fraction"]:
            reasons.append(
                {
                    "code": "exposure_defects",
                    "message": (
                        f"{defects_count} defect pixels ({fraction:.2%}); "
                        f"allow {p['maximum_defect_fraction']:.2%}"
                    ),
                }
            )
        converted = adjusted
        if raw.dtype.kind in "ui":
            limits = np.iinfo(raw.dtype)
            converted = np.rint(converted).clip(limits.min, limits.max)
        if raw.dtype == np.float32 and np.max(np.abs(converted)) > np.finfo(np.float32).max:
            raise ValidationError("Adjusted samples exceed float32")
        output = raw.copy()
        if output.ndim == 2:
            output[valid] = converted[..., 0][valid].astype(raw.dtype)
        else:
            output[..., :channels][valid] = converted[valid].astype(raw.dtype)

        def stats(values):
            return (
                {
                    "mae": float(values.mean()),
                    "rmse": float(np.sqrt(np.mean(values**2))),
                    "maximum": float(values.max()),
                }
                if count
                else None
            )

        info = {
            "model": "differential_exposure_v1",
            "scope": p["scope"],
            "settings": copy.deepcopy(p),
            "fit_convention": "reference = gain * candidate + offset",
            "sample_units": "raw_intensity",
            "residual_before_rounding_or_clipping": True,
            "selected_pixels": int(selected.sum()),
            "fit_pixels": int(fitted.sum()),
            "fit_inlier_pixels": int(inliers.sum()),
            "valid_pixels": count,
            "valid_fraction": coverage,
            "defect_pixels": defects_count,
            "defect_fraction": fraction,
            "offset_only_pixels": int((fallback & valid).sum()),
            "raw": stats(original_difference[valid]),
            "adjusted": stats(difference[valid]),
            "reference_ticket": copy.deepcopy(ref_payload.get("ticket")),
            "reference_spatial": copy.deepcopy(ref_payload["spatial"]),
            "channel_statistics": [
                {
                    "index": i,
                    "name": "Gray" if channels == 1 else encoding[i].upper(),
                    "gain_range": [
                        float(gains[..., i][valid].min()),
                        float(gains[..., i][valid].max()),
                    ]
                    if count
                    else None,
                    "offset_range": [
                        float(offsets[..., i][valid].min()),
                        float(offsets[..., i][valid].max()),
                    ]
                    if count
                    else None,
                    "raw": stats(original_difference[..., i][valid]),
                    "adjusted": stats(difference[..., i][valid]),
                }
                for i in range(channels)
            ],
        }
        maps = (gains[..., 0], offsets[..., 0]) if channels == 1 else (gains, offsets)
        result = Result(
            {
                "image": source,
                "adjusted": ImageArray(output, encoding),
                "gain": ImageArray(maps[0], encoding.removesuffix("a")),
                "offset": ImageArray(maps[1], encoding.removesuffix("a")),
                "residual": residual,
                "raw_residual": raw_residual,
                "valid_mask": valid.astype(np.uint8) * 255,
                "defect_mask": defects.astype(np.uint8) * 255,
                "fit_inliers": inliers.astype(np.uint8) * 255,
                "info": info,
            },
            passed=not reasons,
            fail_reasons=reasons,
            score=coverage * (1 - fraction) if fraction is not None else 0,
            message=f"{count} valid pixels; {defects_count} adjusted-residual defects",
        )
        if self.generate_graphics:
            contours, _ = cv2.findContours(
                defects.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            objects = [
                graphic_object("polyline", c.reshape(-1, 2), REJECTED, closed=True)
                for c in contours[:256]
            ]
            objects.append(
                graphic_object(
                    "text",
                    [(8, 20)],
                    REJECTED if reasons else ACCEPTED,
                    text=f"Exposure comparison: {defects_count} defects; coverage {coverage:.1%}",
                )
            )
            result.drawings["exposure comparison"] = GraphicDrawing(tuple(objects))
        return result
