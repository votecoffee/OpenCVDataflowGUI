"""Validated row-dependent translation for reference-based wobble correction."""

import copy

import numpy as np

from .core import Block, Port, Result
from .flow_files import MAX_PIXELS, MODEL, endpoint, field_hash
from .focus_stacking import luminance, stack_values
from .graphics_layers import GraphicDrawing, graphic_object
from .images import ImageArray
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, FlowGroup, ValuePacket
from .selection import selection_mask
from .sequence_tools import DenseFlow
from .stabilization import _correspondences, _features


def _design(rows, height, degree):
    return np.polynomial.polynomial.polyvander(2 * rows / (height - 1) - 1, degree)


def fit_rows(reference_points, moving_points, height, p):
    """Deterministic Huber fit with separate validation tracks and strict support."""
    count = len(reference_points)
    if count < p["minimum_tracks"]:
        return None, {"reason": "insufficient_tracks", "tracked": count}
    order = np.lexsort((reference_points[:, 0], reference_points[:, 1]))
    reference_points, moving_points = reference_points[order], moving_points[order]
    holdout = np.arange(count) % 4 == 0
    rows = reference_points[:, 1].astype(np.float64)
    delta = moving_points.astype(np.float64) - reference_points
    design = _design(rows, height, int(p["degree"]))
    training = ~holdout
    weights = np.ones(int(training.sum()))
    a, b = design[training], delta[training]
    coefficients = None
    for _ in range(p["fit_iterations"]):
        root = np.sqrt(weights)
        coefficients, _, rank, _ = np.linalg.lstsq(a * root[:, None], b * root[:, None], rcond=None)
        if rank != design.shape[1] or not np.isfinite(coefficients).all():
            return None, {"reason": "degenerate_rows", "tracked": count}
        residual = np.linalg.norm(a @ coefficients - b, axis=1)
        weights = np.minimum(1, p["inlier_pixels"] / np.maximum(residual, np.finfo(float).tiny))
    error = np.linalg.norm(design @ coefficients - delta, axis=1)
    inliers = (error <= p["inlier_pixels"]) & training
    train_count = int(training.sum())
    if inliers.sum() >= design.shape[1]:
        coefficients, _, rank, _ = np.linalg.lstsq(design[inliers], delta[inliers], rcond=None)
        if rank != design.shape[1]:
            return None, {"reason": "degenerate_rows", "tracked": count}
        error = np.linalg.norm(design @ coefficients - delta, axis=1)
        inliers = (error <= p["inlier_pixels"]) & training
    checked = holdout & (error <= p["validation_pixels"])
    validation_count = int(holdout.sum())
    inlier_rows = rows[inliers]
    span, gap = 0.0, None
    if len(inlier_rows):
        span = float(np.ptp(inlier_rows) / (height - 1))
        gap = float(np.diff(inlier_rows).max(initial=0) / (height - 1))
    report = {
        "reason": None,
        "tracked": count,
        "training_tracks": train_count,
        "training_inliers": int(inliers.sum()),
        "training_inlier_ratio": float(inliers.sum() / train_count),
        "validation_tracks": validation_count,
        "validation_inliers": int(checked.sum()),
        "validation_inlier_ratio": float(checked.sum() / validation_count),
        "training_rms_pixels": float(np.sqrt(np.mean(error[inliers] ** 2)))
        if inliers.any()
        else None,
        "validation_rms_pixels": float(np.sqrt(np.mean(error[holdout] ** 2))),
        "row_span_fraction": span,
        "maximum_row_gap_fraction": gap,
        "reference_points": reference_points.tolist(),
        "moving_points": moving_points.tolist(),
        "training_inlier_mask": inliers.tolist(),
        "validation_mask": holdout.tolist(),
    }
    if (
        inliers.sum() < max(p["minimum_training_inliers"], design.shape[1])
        or report["training_inlier_ratio"] < p["minimum_inlier_ratio"]
    ):
        report["reason"] = "insufficient_consensus"
    elif (
        checked.sum() < p["minimum_validation_tracks"]
        or report["validation_inlier_ratio"] < p["minimum_validation_ratio"]
    ):
        report["reason"] = "validation_rejected"
    elif span < p["minimum_row_span"] or gap is None or gap > p["maximum_row_gap"]:
        report["reason"] = "insufficient_row_coverage"
    if report["reason"]:
        return None, report
    values = _design(np.arange(height), height, int(p["degree"])) @ coefficients
    first, last = float(inlier_rows.min()), float(inlier_rows.max())
    row_support = (np.arange(height) >= first - p["extrapolation_rows"]) & (
        np.arange(height) <= last + p["extrapolation_rows"]
    )
    supported_values = values[row_support]
    step = np.diff(values, axis=0)
    supported_step = row_support[:-1] & row_support[1:]
    minimum_jacobian = float((1 + step[supported_step, 1]).min()) if supported_step.any() else 1.0
    maximum_step = float(np.linalg.norm(step[supported_step], axis=1).max(initial=0))
    report.update(
        coefficients_xy=coefficients.tolist(),
        row_coordinate="u=2*y/(height-1)-1; ascending powers",
        observed_row_range=[first, last],
        supported_rows=int(row_support.sum()),
        maximum_motion_pixels=float(np.abs(supported_values).max(initial=0)),
        minimum_vertical_jacobian=minimum_jacobian,
        maximum_row_step_pixels=maximum_step,
    )
    if report["maximum_motion_pixels"] > p["maximum_motion"]:
        report["reason"] = "excess_motion"
    elif minimum_jacobian < p["minimum_vertical_jacobian"] or maximum_step > p["maximum_row_step"]:
        report["reason"] = "unsafe_row_geometry"
    return None if report["reason"] else (values, row_support), report


class EstimateScanlineMotion(Block):
    type_id, title = "inspect.scanline_motion", "Estimate rolling-shutter motion"
    category = block_type = "Inspection"
    subtype = "Registration"
    parallel_safe = process_inputs_readonly = True
    inputs = (
        Port("image"),
        Port("reference"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
        Port("reference_mask", required=False, required_when_connected=True),
        Port("reference_region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("flow", "any", required=False),
        Port("support", visual_type="region"),
        Port("report", "data"),
    )
    defaults = {
        "degree": "3",
        "max_features": 800,
        "feature_quality": 0.01,
        "feature_spacing": 5.0,
        "tracking_window": 21,
        "max_pyramid_level": 3,
        "forward_backward_pixels": 0.75,
        "max_patch_error": 25.0,
        "minimum_tracks": 32,
        "minimum_training_inliers": 20,
        "minimum_inlier_ratio": 0.7,
        "inlier_pixels": 0.75,
        "fit_iterations": 10,
        "minimum_validation_tracks": 6,
        "minimum_validation_ratio": 0.8,
        "validation_pixels": 1.0,
        "minimum_row_span": 0.75,
        "maximum_row_gap": 0.25,
        "extrapolation_rows": 8,
        "maximum_motion": 32.0,
        "maximum_row_step": 2.0,
        "minimum_vertical_jacobian": 0.25,
        "maximum_work": 64_000_000,
    }
    parameter_choices = {"degree": ("1", "2", "3", "4", "5")}
    parameter_ranges = {
        "max_features": NumericRange(16, 5000, whole=True),
        "feature_quality": NumericRange(1e-6, 1),
        "feature_spacing": NumericRange(1, 100),
        "tracking_window": NumericRange(3, 61, whole=True, odd=True),
        "max_pyramid_level": NumericRange(0, 5, whole=True),
        "forward_backward_pixels": NumericRange(0, 20),
        "max_patch_error": NumericRange(0, 255),
        "minimum_tracks": NumericRange(8, 5000, whole=True),
        "minimum_training_inliers": NumericRange(3, 5000, whole=True),
        "minimum_inlier_ratio": NumericRange(0, 1),
        "inlier_pixels": NumericRange(0.001, 20),
        "fit_iterations": NumericRange(1, 30, whole=True),
        "minimum_validation_tracks": NumericRange(2, 1250, whole=True),
        "minimum_validation_ratio": NumericRange(0, 1),
        "validation_pixels": NumericRange(0.001, 20),
        "minimum_row_span": NumericRange(0.1, 1),
        "maximum_row_gap": NumericRange(0.01, 1),
        "extrapolation_rows": NumericRange(0, 64, whole=True),
        "maximum_motion": NumericRange(0.001, 4096),
        "maximum_row_step": NumericRange(0.001, 100),
        "minimum_vertical_jacobian": NumericRange(0.001, 1),
        "maximum_work": NumericRange(1, 512_000_000, whole=True),
    }
    parameter_help = {
        "degree": "Polynomial degree of horizontal and vertical displacement versus reference row.",
        "max_features": "Maximum Shi-Tomasi corners detected in the reference fitting selection.",
        "feature_quality": "Native minimum corner quality relative to the best corner.",
        "feature_spacing": "Minimum reference corner separation in current pixels.",
        "tracking_window": (
            "Odd LK patch width/height. Selections restrict centers, not full patches."
        ),
        "max_pyramid_level": "Highest LK pyramid level; zero uses the original resolution only.",
        "forward_backward_pixels": "Maximum tracking round-trip error in reference pixels.",
        "max_patch_error": "Maximum native LK mean patch difference in shared 8-bit working units.",
        "minimum_tracks": "Required forward/backward-checked paired corners before fitting.",
        "minimum_training_inliers": (
            "Required fitting inliers, excluding every-fourth validation tracks."
        ),
        "minimum_inlier_ratio": "Required fraction of training tracks within Inlier pixels.",
        "inlier_pixels": "Huber scale and final Euclidean training residual tolerance in pixels.",
        "fit_iterations": "Bounded deterministic Huber iterations, followed by one inlier refit.",
        "minimum_validation_tracks": "Required withheld tracks within the validation tolerance.",
        "minimum_validation_ratio": "Required accepted fraction of all withheld validation tracks.",
        "validation_pixels": "Maximum withheld reference-to-image prediction error in pixels.",
        "minimum_row_span": "Minimum height fraction spanned by fitting inliers.",
        "maximum_row_gap": "Largest allowed gap between fitting-inlier rows, as a height fraction.",
        "extrapolation_rows": (
            "Supported rows beyond the first/last fitting inlier; zero forbids extrapolation."
        ),
        "maximum_motion": "Reject a fit exceeding this per-axis displacement on supported rows.",
        "maximum_row_step": (
            "Reject excessive change of displacement between neighboring supported rows."
        ),
        "minimum_vertical_jacobian": "Minimum 1 + vertical flow step; protects against row folds.",
        "maximum_work": (
            "Preflight max_features * tracking_window squared * (pyramid level + 1) * 30."
        ),
    }
    input_help = {
        "image": "Current wobbled capture; same shape, depth and intensity encoding as reference.",
        "reference": "Sharp stable scene geometry defining the output/flow index grid.",
        "mask": (
            "Current-grid track-center support, intersected with Region. Also wire to the "
            "warper for donor coverage."
        ),
        "region": "Current-grid track-center selection. Does not crop LK patch context.",
        "reference_mask": (
            "Reference fitting selection; exclude moving parts and retain static texture "
            "across rows."
        ),
        "reference_region": "Reference-grid region intersected with Reference mask.",
    }
    output_help = {
        "flow": "Reference-grid target-to-source displacement for Warp image with dense flow.",
        "support": (
            "Rows supported by the validated fit; intersected with donor support by the warper."
        ),
        "report": (
            "Fit/holdout evidence, coefficients, row limits, rejection reason and "
            "image-pair identity."
        ),
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_inputs = ["image", "reference"]
        self.exposed_outputs = ["flow", "support", "report"]
        self.documentation = (
            "<p>Estimate smooth horizontal/vertical displacement versus reference row. Connect "
            "Flow and Support to Warp image with dense flow, with the current image as its Image "
            "and the reference as its Target. The field is a target-to-source pull map; the "
            "warper preserves current pixels/type/ticket and emits explicit coverage and maps.</p>"
            "<p>Reference static texture must span the image height. Exclude expected moving "
            "objects with fitting selections. Forward/backward LK tracks feed a deterministic "
            "robust polynomial fit; every fourth row-sorted track is withheld for validation. "
            "Poor consensus, inadequate row support or unsafe geometry withholds Flow.</p>"
            "<p>This reference-based baseline is not a physical rolling-shutter calibration or "
            "the native C++ wobble suppressor. It models row translation, not x-dependent "
            "perspective/rotation, parallax or arbitrary high-frequency vibration. Validation "
            "cannot prove rigidity or absence of defects. Retain raw captures.</p>"
        )

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if (
            p["minimum_tracks"] > p["max_features"]
            or p["minimum_training_inliers"] + p["minimum_validation_tracks"] > p["max_features"]
        ):
            raise ValidationError(
                "Track requirements exceed Max features",
                "max_features",
                "minimum_tracks",
                "minimum_training_inliers",
                "minimum_validation_tracks",
            )

    def process(self, inputs):
        p = dict(self.parameters)
        for name, limits in self.parameter_ranges.items():
            p[name] = limits.check(p[name], name, name)
        image, reference = inputs["image"], inputs["reference"]
        if not isinstance(image, np.ndarray) or not isinstance(reference, np.ndarray):
            raise ValidationError("Connect current and reference images")
        if image.ndim not in (2, 3) or reference.ndim not in (2, 3):
            raise ValidationError("Use two-dimensional intensity images")
        h, w = image.shape[:2]
        channels = 1 if image.ndim == 2 else image.shape[2]
        if (
            min(h, w) < 24
            or max(h, w) >= 32767
            or h * w > MAX_PIXELS
            or image.nbytes + reference.nbytes + h * w * (128 + channels * 16) > 256 * 1024**2
        ):
            raise ValidationError(
                "Scanline motion needs at least 24x24, at most 2.1M pixels and a 256 MiB workspace"
            )
        if (
            p["max_features"] * p["tracking_window"] ** 2 * (p["max_pyramid_level"] + 1) * 30
            > p["maximum_work"]
        ):
            raise ValidationError("LK work exceeds Maximum work", "maximum_work")
        source_context, target_context = (
            self.input_payloads["image"],
            self.input_payloads["reference"],
        )
        _, encoding = stack_values(
            FlowGroup(
                (ValuePacket(image, source_context), ValuePacket(reference, target_context)),
                ("current", "reference"),
            )
        )
        scale = (
            float(np.iinfo(image.dtype).max)
            if image.dtype.kind == "u"
            else max(float(np.abs(image).max()), float(np.abs(reference).max())) or 1.0
        )
        a, b = (luminance(value, encoding, scale) for value in (reference, image))
        low, high = min(float(a.min()), float(b.min())), max(float(a.max()), float(b.max()))
        first, second = (
            np.rint(np.clip((value - low) / (high - low or 1), 0, 1) * 255).astype(np.uint8)
            for value in (a, b)
        )
        source_mask = selection_mask(inputs, image)
        selected = selection_mask(
            {
                "image": ImageArray(reference, encoding, target_context),
                **{
                    key: inputs["reference_" + key]
                    for key in ("mask", "region")
                    if "reference_" + key in inputs
                },
            },
            reference,
        )
        if encoding in ("rgba", "bgra"):
            source_mask = (
                np.uint8((image[..., 3] > 0) & (True if source_mask is None else source_mask != 0))
                * 255
            )
            selected = (
                np.uint8((reference[..., 3] > 0) & (True if selected is None else selected != 0))
                * 255
            )
        points = _features(first, selected, p, "max_features")
        detected = 0 if points is None else len(points)
        original, moved = np.empty((0, 2)), np.empty((0, 2))
        if detected:
            indices, moved = _correspondences(first, second, points, source_mask, p)
            original = points[indices, 0]
        fitted, report = fit_rows(original, moved, h, p)
        report.update(
            model="scanline_motion_v1",
            detected=detected,
            degree=int(p["degree"]),
            image_size=[w, h],
            direction="target_to_source",
            working_luminance_range=[low, high],
            working_scale=scale,
        )
        context = copy.deepcopy(source_context)
        for key in (
            "spatial",
            "pyramid",
            "alpha_representation",
            "raw_sensor",
            "outcome",
            "processing_time",
            "flow_association",
        ):
            context.pop(key, None)
        context["spatial"] = copy.deepcopy(target_context["spatial"])
        if "pyramid" in target_context:
            context["pyramid"] = copy.deepcopy(target_context["pyramid"])
        flow, support = NO_VALUE, np.zeros((h, w), np.uint8)
        if fitted is not None:
            rows, usable = fitted
            vectors = np.broadcast_to(rows[:, None, :], (h, w, 2)).astype(np.float32).copy()
            vectors[~usable] = 0
            support[usable] = 255
            association = {
                "model": MODEL,
                "shape": list(vectors.shape),
                "sha256": field_hash(vectors),
                "direction": "target_to_source",
                "source": endpoint(image, source_context, "current"),
                "target": endpoint(reference, target_context, "reference"),
            }
            context["flow_association"] = association
            flow = DenseFlow(vectors, context)
            report["pair"] = copy.deepcopy(association)
        drawings = {}
        if self.generate_graphics:
            good = report["reason"] is None
            color = (60, 210, 100) if good else (50, 70, 240)
            objects = [
                graphic_object(
                    "text",
                    [[5, 18]],
                    color,
                    text="Scanline motion: " + (report["reason"] or "validated"),
                )
            ]
            if "reference_points" in report:
                for start, end, accepted, validation in list(
                    zip(
                        report["reference_points"],
                        report["moving_points"],
                        report["training_inlier_mask"],
                        report["validation_mask"],
                        strict=True,
                    )
                )[:: max(1, detected // 128)]:
                    objects.append(
                        graphic_object(
                            "arrow",
                            [start, end],
                            (220, 190, 60)
                            if validation
                            else (60, 210, 100)
                            if accepted
                            else (50, 70, 240),
                        )
                    )
            drawings["scanline fitting tracks"] = GraphicDrawing(tuple(objects), "reference")
        return Result(
            {"flow": flow, "support": ImageArray(support, "gray"), "report": report},
            payloads={key: copy.deepcopy(context) for key in ("flow", "support", "report")},
            passed=fitted is not None,
            score=report.get("validation_inlier_ratio", 0),
            drawings=drawings,
            fail_reasons=[] if fitted is not None else ["scanline_motion_" + report["reason"]],
            message="Scanline motion " + (report["reason"] or "validated"),
        )
