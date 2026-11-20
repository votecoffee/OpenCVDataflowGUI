"""Compare metric camera depth on a common raster grid."""

import numpy as np

from .core import Port, Result
from .geometry3d_common import Tool3D
from .images import ImageArray
from .parameter_validation import NumericRange, ValidationError
from .raster_evaluation import (
    MAX_EVALUATION_PIXELS,
    apply_valid_mask,
    comparison_selection,
    error_heatmap,
    error_statistics,
)

_DEPTH_DTYPES = (np.uint16, np.float32, np.float64)


def _depth_samples(image, name, parameters):
    values = np.asarray(image)
    if (
        values.ndim != 2
        or values.dtype not in _DEPTH_DTYPES
        or not 0 < values.size <= MAX_EVALUATION_PIXELS
    ):
        raise ValidationError(
            f"{name.title()} depth needs a monochrome uint16 or float grid "
            "of up to 16,777,216 pixels"
        )
    units = getattr(image, "units", parameters[f"{name}_units"])
    if units not in ("m", "mm"):
        raise ValidationError(f"{name.title()} depth must declare meters or millimeters")
    scale = 1000.0 if units == "m" else 1.0
    if values.dtype == np.uint16:
        scale *= float(parameters[f"{name}_units_per_sample"])
    return values, scale, units


class EvaluateDepth(Tool3D):
    type_id, title, subtype = "3d.evaluate_depth", "Evaluate camera depth", "Depth and RGB-D"
    inputs = (
        Port("estimate", "depth"),
        Port("reference", "depth"),
        Port("estimate_valid", required=False, required_when_connected=True),
        Port("reference_valid", required=False, required_when_connected=True),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, visual_type="region", required_when_connected=True),
    )
    outputs = (
        Port("residual"),
        Port("overlap_mask"),
        Port("bad_mask"),
        Port("missing_mask"),
        Port("estimate_only_mask"),
        Port("report", "data"),
    )
    defaults = {
        "alignment": "same_frame",
        "estimate_units": "mm",
        "reference_units": "mm",
        "estimate_units_per_sample": 1.0,
        "reference_units_per_sample": 1.0,
        "error_tolerance_mm": 2.0,
        "minimum_coverage": 0.0,
        "maximum_bad_fraction": 1.0,
        "preview_max_mm": 10.0,
    }
    parameter_choices = {
        "alignment": ("same_frame", "pixel_grid"),
        "estimate_units": ("m", "mm"),
        "reference_units": ("m", "mm"),
    }
    parameter_ranges = {
        "estimate_units_per_sample": NumericRange(1e-9, 1000),
        "reference_units_per_sample": NumericRange(1e-9, 1000),
        "error_tolerance_mm": NumericRange(0),
        "minimum_coverage": NumericRange(0, 1),
        "maximum_bad_fraction": NumericRange(0, 1),
        "preview_max_mm": NumericRange(1e-6),
    }
    parameter_help = {
        "alignment": "Check known frames, or assert independent rasters use the same pixel grid.",
        "estimate_units": "Physical units for an estimate without a depth-unit tag.",
        "reference_units": "Physical units for a reference without a depth-unit tag.",
        "estimate_units_per_sample": "Physical units per raw uint16 estimate sample.",
        "reference_units_per_sample": "Physical units per raw uint16 reference sample.",
        "error_tolerance_mm": "Absolute difference above which an overlapping pixel is bad.",
        "minimum_coverage": "Required overlap fraction of selected reference-valid pixels.",
        "maximum_bad_fraction": "Allowed bad fraction among overlapping pixels.",
        "preview_max_mm": "Error magnitude that saturates the color preview, in millimeters.",
    }
    input_help = {
        "estimate": "Estimated camera depth on the comparison grid.",
        "reference": "Metric reference depth aligned to the same camera grid.",
        "estimate_valid": "Optional uint8 mask of trustworthy estimate samples.",
        "reference_valid": "Optional uint8 mask of trustworthy reference samples.",
        "mask": "Optional uint8 selection intersected with Region.",
        "region": "Optional image-space region on the estimate grid.",
    }
    payload_reference = "estimate"
    preview_safe = parallel_safe = process_inputs_readonly = True

    def process(self, inputs):
        p = self.parameters
        estimate, estimate_scale, estimate_units = _depth_samples(inputs["estimate"], "estimate", p)
        reference, reference_scale, reference_units = _depth_samples(
            inputs["reference"], "reference", p
        )
        if estimate.shape != reference.shape:
            raise ValidationError(
                "Depth estimate and reference must have the same pixel dimensions"
            )
        selection = comparison_selection(inputs, p["alignment"], "Depth")
        estimate_valid = apply_valid_mask(
            np.isfinite(estimate) & (estimate > 0), inputs.get("estimate_valid"), "estimate"
        )
        truth_valid = apply_valid_mask(
            np.isfinite(reference) & (reference > 0), inputs.get("reference_valid"), "reference"
        )
        if selection is not None:
            estimate_valid &= selection != 0
            truth_valid &= selection != 0

        common = estimate_valid & truth_valid
        missing = truth_valid & ~estimate_valid
        estimate_only = estimate_valid & ~truth_valid
        tolerance = float(p["error_tolerance_mm"])
        with np.errstate(over="ignore", invalid="ignore"):
            residual = (
                estimate[common].astype(np.float64) * estimate_scale
                - reference[common].astype(np.float64) * reference_scale
            )
        absolute = np.abs(residual)
        bad = absolute > tolerance
        metrics = error_statistics(residual, absolute, "mm", "Depth")

        count = residual.size
        truth_count = int(np.count_nonzero(truth_valid))
        estimate_count = int(np.count_nonzero(estimate_valid))
        bad_count = int(np.count_nonzero(bad))
        union_count = truth_count + estimate_count - count
        coverage = count / truth_count if truth_count else None
        bad_fraction = bad_count / count if count else None
        report = {
            "model": "camera_depth_evaluation_v1",
            "units": "mm",
            "estimate_units": estimate_units,
            "reference_units": reference_units,
            "estimate_mm_per_sample": estimate_scale,
            "reference_mm_per_sample": reference_scale,
            "selected_pixels": (
                estimate.size if selection is None else int(np.count_nonzero(selection))
            ),
            "reference_valid_pixels": truth_count,
            "estimate_valid_pixels": estimate_count,
            "evaluated_pixels": count,
            "missing_pixels": truth_count - count,
            "estimate_only_pixels": estimate_count - count,
            "coverage": coverage,
            "validity_iou": count / union_count if union_count else None,
            "error_tolerance_mm": tolerance,
            "bad_pixels": bad_count,
            "bad_fraction": bad_fraction,
            **metrics,
        }
        reasons = []
        if not count:
            reasons.append("depth_no_overlap" if truth_count else "depth_no_reference")
        else:
            if coverage < float(p["minimum_coverage"]):
                reasons.append("depth_coverage_below_minimum")
            if bad_fraction > float(p["maximum_bad_fraction"]):
                reasons.append("depth_bad_fraction_above_maximum")

        error_image = np.full(estimate.shape, np.nan, dtype=np.float64)
        error_image[common] = residual
        bad_image = np.zeros(estimate.shape, dtype=np.uint8)
        bad_image[common] = bad.astype(np.uint8) * 255
        graphics = {}
        if self.generate_graphics:
            graphic = error_heatmap(common, absolute, missing, float(p["preview_max_mm"]))
            graphic[estimate_only] = (128, 128, 128)
            graphics["depth_error"] = graphic
        return Result(
            {
                "residual": ImageArray(error_image, "gray"),
                "overlap_mask": ImageArray(common.astype(np.uint8) * 255, "gray"),
                "bad_mask": ImageArray(bad_image, "gray"),
                "missing_mask": ImageArray(missing.astype(np.uint8) * 255, "gray"),
                "estimate_only_mask": ImageArray(estimate_only.astype(np.uint8) * 255, "gray"),
                "report": report,
            },
            graphics=graphics,
            passed=not reasons,
            score=(count - bad_count) / truth_count if truth_count else 0.0,
            message=(
                f"{count}/{truth_count} reference pixels evaluated, {bad_count} outside tolerance"
            ),
            fail_reasons=reasons,
        )
