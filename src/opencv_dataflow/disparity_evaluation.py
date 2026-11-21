"""Measure aligned stereo disparity against reference values in pixel units."""

import numpy as np

from .core import Port, Result
from .disparity_inputs import disparity_image, valid_samples
from .geometry3d_common import Tool3D
from .images import DepthArray, ImageArray
from .parameter_validation import NumericRange, ValidationError
from .raster_evaluation import (
    apply_valid_mask,
    comparison_selection,
    error_heatmap,
    error_statistics,
)


def _metrics(residual, absolute, bad):
    count = residual.size
    bad_count = int(np.count_nonzero(bad))
    return {
        "evaluated_pixels": count,
        "bad_pixels": bad_count,
        "bad_fraction": bad_count / count if count else None,
        **error_statistics(residual, absolute, "px", "Disparity"),
    }


def _confidence_slices(image, scored, residual, absolute, bad, cutoff):
    if image is None:
        return None
    confidences = np.asarray(image)
    if (
        isinstance(image, DepthArray)
        or confidences.shape != scored.shape
        or confidences.dtype not in (np.float32, np.float64)
    ):
        raise ValidationError("Confidence needs a matching float image with values between 0 and 1")
    samples = confidences[scored].astype(np.float64, copy=False)
    known = np.isfinite(samples)
    if np.any(known & ((samples < 0) | (samples > 1))):
        raise ValidationError("Confidence on evaluated pixels must be between 0 and 1")
    low = known & (samples < cutoff)
    strata = {"below_cutoff": low, "at_or_above_cutoff": known & ~low, "unknown": ~known}
    return {
        name: _metrics(residual[selected], absolute[selected], bad[selected])
        for name, selected in strata.items()
    }


class EvaluateDisparity(Tool3D):
    type_id, title, subtype = (
        "3d.evaluate_disparity",
        "Evaluate stereo disparity",
        "Stereo",
    )
    inputs = (
        Port("estimate"),
        Port("reference"),
        Port("estimate_valid", required=False, required_when_connected=True),
        Port("reference_valid", required=False, required_when_connected=True),
        Port("confidence", required=False, required_when_connected=True),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, visual_type="region", required_when_connected=True),
    )
    outputs = (
        Port("absolute_error"),
        Port("evaluated_mask"),
        Port("bad_mask"),
        Port("missing_mask"),
        Port("report", "data"),
    )
    defaults = {
        "alignment": "same_frame",
        "estimate_samples_per_pixel": 16.0,
        "reference_samples_per_pixel": 16.0,
        "estimate_invalid_mode": "nonfinite_only",
        "estimate_invalid_code": 0.0,
        "reference_invalid_mode": "nonfinite_only",
        "reference_invalid_code": 0.0,
        "bad_threshold_px": 1.0,
        "minimum_coverage": 0.0,
        "maximum_bad_fraction": 1.0,
        "confidence_cutoff": 0.5,
        "preview_max_px": 4.0,
    }
    parameter_choices = {
        "alignment": ("same_frame", "pixel_grid"),
        "estimate_invalid_mode": ("nonfinite_only", "also_use_code"),
        "reference_invalid_mode": ("nonfinite_only", "also_use_code"),
    }
    parameter_ranges = {
        "estimate_samples_per_pixel": NumericRange(1e-6, 65536),
        "reference_samples_per_pixel": NumericRange(1e-6, 65536),
        "estimate_invalid_code": NumericRange(),
        "reference_invalid_code": NumericRange(),
        "bad_threshold_px": NumericRange(0),
        "minimum_coverage": NumericRange(0, 1),
        "maximum_bad_fraction": NumericRange(0, 1),
        "confidence_cutoff": NumericRange(0, 1),
        "preview_max_px": NumericRange(1e-6),
    }
    parameter_help = {
        "alignment": "Check known frames, or assert independent rasters use the same pixel grid.",
        "estimate_samples_per_pixel": "Integer estimate samples per disparity pixel.",
        "reference_samples_per_pixel": "Integer reference samples per disparity pixel.",
        "estimate_invalid_mode": "Optionally exclude a finite estimate code in raw sample units.",
        "estimate_invalid_code": "Additional missing-estimate code, before scaling.",
        "reference_invalid_mode": "Optionally exclude a finite reference code in raw sample units.",
        "reference_invalid_code": "Additional unknown-reference code, before scaling.",
        "bad_threshold_px": "An evaluated pixel is bad when its absolute error exceeds this value.",
        "minimum_coverage": "Required evaluated fraction of selected reference-valid pixels.",
        "maximum_bad_fraction": "Allowed bad fraction among evaluated pixels in the selection.",
        "confidence_cutoff": "Split evaluated values below this confidence and at or above it.",
        "preview_max_px": "Error magnitude that saturates the color preview, in disparity pixels.",
    }
    input_help = {
        "estimate": "Monochrome estimated disparity. Float samples are already in pixels.",
        "reference": "Monochrome ground-truth disparity on the estimate pixel grid.",
        "estimate_valid": "Optional uint8 mask of available estimate samples.",
        "reference_valid": "Optional uint8 mask of valid reference samples.",
        "confidence": "Optional normalized float confidence on the estimate grid.",
        "mask": "Optional uint8 selection intersected with Region for the evaluation domain.",
        "region": "Optional image-space region on the estimate grid.",
    }
    payload_reference = "estimate"
    parallel_safe = preview_safe = process_inputs_readonly = True

    def process(self, inputs):
        p = self.parameters
        estimate, reference = (disparity_image(inputs[name]) for name in ("estimate", "reference"))
        if estimate.shape != reference.shape:
            raise ValidationError("Estimate and reference must have the same pixel dimensions")
        selection = comparison_selection(inputs, p["alignment"], "Disparity")
        validity = {}
        for name, values in (("estimate", estimate), ("reference", reference)):
            invalid = (
                float(p[f"{name}_invalid_code"])
                if p[f"{name}_invalid_mode"] == "also_use_code"
                else None
            )
            valid = valid_samples(values, invalid, f"{name}_invalid_code")
            validity[name] = apply_valid_mask(valid, inputs.get(f"{name}_valid"), name)
        truth = validity["reference"]
        if selection is not None:
            truth &= selection != 0
        scored = truth & validity["estimate"]
        missing = truth & ~scored

        def pixels(values, name):
            scale = 1 if values.dtype.kind == "f" else float(p[f"{name}_samples_per_pixel"])
            return values[scored].astype(np.float64) / scale

        with np.errstate(over="ignore", invalid="ignore"):
            residual = pixels(estimate, "estimate") - pixels(reference, "reference")
        absolute = np.abs(residual)
        bad_threshold = float(p["bad_threshold_px"])
        bad = absolute > bad_threshold
        metrics = _metrics(residual, absolute, bad)
        selected_count = estimate.size if selection is None else int(np.count_nonzero(selection))
        truth_count = int(np.count_nonzero(truth))
        count = metrics["evaluated_pixels"]
        missing_count = truth_count - count
        coverage = count / truth_count if truth_count else None
        cutoff = float(p["confidence_cutoff"])
        report = {
            "model": "stereo_disparity_evaluation_v1",
            "selected_pixels": selected_count,
            "reference_valid_pixels": truth_count,
            "missing_pixels": missing_count,
            "coverage": coverage,
            "bad_or_missing_fraction": (
                (metrics["bad_pixels"] + missing_count) / truth_count if truth_count else None
            ),
            "bad_threshold_px": bad_threshold,
            "confidence_cutoff": cutoff if "confidence" in inputs else None,
            "confidence_slices": _confidence_slices(
                inputs.get("confidence"), scored, residual, absolute, bad, cutoff
            ),
            **metrics,
        }
        reasons = []
        if not count:
            reasons.append("disparity_no_overlap" if truth_count else "disparity_no_reference")
        else:
            if coverage < float(p["minimum_coverage"]):
                reasons.append("disparity_coverage_below_minimum")
            if metrics["bad_fraction"] > float(p["maximum_bad_fraction"]):
                reasons.append("disparity_bad_fraction_above_maximum")
        error = np.full(estimate.shape, np.nan, dtype=np.float64)
        error[scored] = absolute
        bad_image = np.zeros(estimate.shape, dtype=np.uint8)
        bad_image[scored] = bad.astype(np.uint8) * 255
        graphics = {}
        if self.generate_graphics:
            graphics["disparity_error"] = error_heatmap(
                scored, absolute, missing, float(p["preview_max_px"])
            )
        return Result(
            {
                "absolute_error": ImageArray(error, "gray"),
                "evaluated_mask": ImageArray(scored.astype(np.uint8) * 255, "gray"),
                "bad_mask": ImageArray(bad_image, "gray"),
                "missing_mask": ImageArray(missing.astype(np.uint8) * 255, "gray"),
                "report": report,
            },
            graphics=graphics,
            passed=not reasons,
            score=(count - metrics["bad_pixels"]) / truth_count if truth_count else 0.0,
            message=(
                f"{count}/{truth_count} reference pixels evaluated, {metrics['bad_pixels']} bad"
            ),
            fail_reasons=reasons,
        )
