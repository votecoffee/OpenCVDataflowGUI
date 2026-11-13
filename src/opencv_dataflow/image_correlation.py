"""Correlation of already aligned images with explicit constant-region policy."""

import math

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .images import ImageArray
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .processing import image_value
from .result_graphics import ACCEPTED, REJECTED
from .selection import selection_mask


def _planes(source, mode, alpha):
    try:
        image = image_value(source, (np.uint8, np.uint16, np.float32, np.float64), canonical=True)
    except cv2.error as exc:
        raise ValidationError("Unsupported image encoding/depth for color conversion") from exc
    if not np.isfinite(image).all():
        raise ValidationError("Color conversion produced nonfinite samples")
    if image.ndim == 2:
        return (image,), ("Gray",)
    if mode == "gray":
        # Preserve double precision and HDR range; cvtColor does not accept CV_64F.
        gray = np.zeros(image.shape[:2], np.float64)
        for index, weight in enumerate((0.114, 0.587, 0.299)):
            gray += image[..., index].astype(np.float64) * weight
        if not np.isfinite(gray).all():
            raise ValidationError("Grayscale conversion exceeded the float64 range")
        return (gray,), ("Gray",)
    labels = ("B", "G", "R", "A") if alpha and image.shape[2] == 4 else ("B", "G", "R")
    return tuple(image[..., index] for index in range(len(labels))), labels


def _center(samples):
    """Power-of-two scaling avoids overflow without quantizing small variations."""
    maximum = float(np.max(np.abs(samples)))
    if maximum == 0:
        return np.zeros_like(samples), 0.0
    scale = math.ldexp(1.0, math.frexp(maximum)[1] - 1)
    shifted = samples / scale
    shifted -= shifted[0]
    shifted -= shifted.mean()
    norm = float(cv2.norm(shifted, cv2.NORM_L2))
    deviation = min(norm / math.sqrt(samples.size), maximum / scale) * scale
    return (shifted / norm if norm else np.zeros_like(shifted)), deviation


def _equal_constants(a, b, tolerance):
    maximum = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    if maximum == 0:
        return True
    scale = math.ldexp(1.0, math.frexp(maximum)[1] - 1)
    return float(np.max(np.abs(a / scale - b / scale))) <= tolerance / scale


class CompareAlignedImages(Block):
    type_id, title = "inspect.image_correlation", "Compare aligned images (correlation)"
    category, block_type, subtype = "Inspection", "Inspection", "Measurements"
    inputs = (
        Port("image"),
        Port("reference"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("similarity", "number", required=False),
        Port("measurements", "data"),
        Port("residual", required=False),
    )
    defaults = {
        "method": "ecc",
        "channels": "per_channel",
        "aggregation": "minimum",
        "include_alpha": False,
        "reference_alignment": "pixel_grid",
        "constant_policy": "fail",
        "minimum_stddev": 1e-12,
        "constant_tolerance": 0.0,
        "check_threshold": True,
        "minimum_similarity": 0.95,
        "emit_residual": False,
        "residual_preview_max": 3.0,
    }
    parameter_choices = {
        "method": ("ecc", "normalized_correlation"),
        "channels": ("per_channel", "gray"),
        "aggregation": ("minimum", "mean"),
        "reference_alignment": ("pixel_grid", "same_frame"),
        "constant_policy": ("fail", "match_equal"),
    }
    parameter_ranges = {
        "minimum_stddev": NumericRange(0, 1e308),
        "constant_tolerance": NumericRange(0, 1e308),
        "minimum_similarity": NumericRange(-1, 1),
        "residual_preview_max": NumericRange(0.001, 1e6),
    }
    parameter_help = {
        "method": (
            "Native computeECC or double-precision normalized correlation on selected pixels."
        ),
        "channels": (
            "Compare canonical B/G/R separately, or weighted grayscale. HSV/Lab convert to BGR."
        ),
        "aggregation": (
            "Use the lowest similarity or unweighted channel mean; undefined channels fail."
        ),
        "include_alpha": (
            "Include alpha in per-channel comparison when both inputs have it; otherwise ignore it."
        ),
        "reference_alignment": (
            "Pixel grid compares equal-size pixels by index. Same frame requires "
            "matching spatial payloads on both images."
        ),
        "constant_policy": (
            "Fail undefined correlation, or explicitly accept two constant regions "
            "whose values match within tolerance."
        ),
        "minimum_stddev": (
            "A channel at or below this standard deviation is treated as constant, in "
            "original sample units."
        ),
        "constant_tolerance": (
            "Maximum per-pixel absolute difference for match_equal, in original sample "
            "units; no depth rescaling."
        ),
        "check_threshold": (
            "Apply the inclusive minimum similarity; undefined comparisons still fail "
            "when disabled."
        ),
        "minimum_similarity": (
            "Inclusive signed correlation limit on [-1,1]. Positive gain/offset changes"
            " are not defects to this metric."
        ),
        "emit_residual": (
            "Output maximum absolute per-channel z-score difference as float32; zero "
            "outside the selection."
        ),
        "residual_preview_max": (
            "Z-score difference at full heat-map intensity; affects only the diagnostic preview."
        ),
    }
    input_help = {
        "image": (
            "Already aligned candidate. Gray/BGR/RGB/BGRA/RGBA: uint8, uint16, "
            "float32/64. HSV/Lab: uint8/float32."
        ),
        "reference": (
            "Equal-size aligned reference. Per-channel mode needs matching compared "
            "channels. Mixed depths compare raw sample values."
        ),
        "mask": (
            "Optional uint8 selection in the candidate's coordinate frame; intersects the region."
        ),
        "region": (
            "Optional region in the candidate's pixel grid. Both images use the same "
            "selected pixels."
        ),
    }
    process_inputs_readonly = True
    parallel_safe = True

    def process(self, inputs):
        source, reference, p = inputs["image"], inputs["reference"], self.parameters
        if source.shape[:2] != reference.shape[:2]:
            raise ValidationError("Aligned comparison requires matching image dimensions")
        if np.prod(source.shape[:2]) > 8_388_608:
            raise ValidationError("Aligned comparison is limited to 8,388,608 pixels")
        if p["reference_alignment"] == "same_frame":
            a = self.input_payloads["image"].get("spatial")
            b = self.input_payloads["reference"].get("spatial")
            if (
                not a
                or not b
                or a.get("root_id") is None
                or any(a.get(key) != b.get(key) for key in ("root_id", "root_size", "image_size"))
                or not np.allclose(a["to_root"], b["to_root"], atol=1e-9, rtol=0)
            ):
                raise ValidationError(
                    "Same-frame comparison needs matching spatial payloads on both images"
                )
        a, labels = _planes(source, p["channels"], p["include_alpha"])
        b, other_labels = _planes(reference, p["channels"], p["include_alpha"])
        if labels != other_labels:
            raise ValidationError(
                "Per-channel comparison needs matching channels; choose grayscale to mix gray/color"
            )
        mask = selection_mask(inputs, source)
        selected = np.ones(source.shape[:2], bool) if mask is None else mask != 0
        count = int(selected.sum())
        if count < 2:
            raise ValidationError("Correlation needs at least two selected pixels")
        want_residual = p["emit_residual"] or self.capture_graphics
        residual = np.zeros(source.shape[:2], np.float32) if want_residual else None
        channels = []
        reasons = []
        for label, candidate, target in zip(labels, a, b, strict=True):
            left = candidate[selected].astype(np.float64)
            right = target[selected].astype(np.float64)
            x, std_a = _center(left)
            y, std_b = _center(right)
            constant_a, constant_b = std_a <= p["minimum_stddev"], std_b <= p["minimum_stddev"]
            coefficient, similarity, state = None, None, "measured"
            if constant_a or constant_b:
                if (
                    p["constant_policy"] == "match_equal"
                    and constant_a
                    and constant_b
                    and _equal_constants(left, right, p["constant_tolerance"])
                ):
                    similarity, state = 1.0, "constant_equal"
                    x, y = np.zeros_like(x), np.zeros_like(y)
                else:
                    state = (
                        "constant_mismatch"
                        if constant_a and constant_b and p["constant_policy"] == "match_equal"
                        else "undefined"
                    )
                    reasons.append(
                        "correlation_constant_mismatch"
                        if state == "constant_mismatch" and p["constant_policy"] == "match_equal"
                        else "correlation_undefined"
                    )
            else:
                # Center/scale in float64 before native ECC, avoiding integer mean rounding
                # and extreme-range overflow. Each signal retains the same correlation.
                value = (
                    cv2.computeECC(y.reshape(-1, 1), x.reshape(-1, 1))
                    if p["method"] == "ecc"
                    else float(x @ y)
                )
                if not math.isfinite(value):
                    raise ValidationError("Correlation produced a nonfinite coefficient")
                if abs(value) >= 1 - 64 * np.finfo(np.float64).eps:
                    value = math.copysign(1.0, value)
                coefficient = similarity = float(np.clip(value, -1, 1))
            if residual is not None and similarity is not None:
                difference = np.abs(x - y) * math.sqrt(count)
                residual[selected] = np.maximum(residual[selected], difference.astype(np.float32))
            channels.append(
                {
                    "channel": label,
                    "coefficient": coefficient,
                    "similarity": similarity,
                    "state": state,
                    "image_stddev": std_a,
                    "reference_stddev": std_b,
                }
            )
        values = [entry["similarity"] for entry in channels]
        value = (
            None
            if reasons
            else float(min(values) if p["aggregation"] == "minimum" else np.mean(values))
        )
        if value is not None and p["check_threshold"] and value < p["minimum_similarity"]:
            reasons.append("image_correlation_below_minimum")
        reasons = list(dict.fromkeys(reasons))
        report = {
            "model": "aligned_image_correlation_v1",
            "valid": value is not None,
            "method": p["method"],
            "similarity": value,
            "channels": channels,
            "selected_pixels": count,
            "parameters": dict(p),
            "failure_reasons": reasons,
            "residual_units": "maximum absolute channel z-score difference",
        }
        message = (
            f"Similarity {value:.6g}"
            if value is not None
            else "Undefined correlation: constant channel"
        )
        passed = not reasons
        result = Result(
            {
                "image": source,
                "similarity": NO_VALUE if value is None else value,
                "measurements": report,
                "residual": NO_VALUE,
            },
            passed=passed,
            message=message,
            score=None if value is None else (value + 1) / 2,
            fail_reasons=reasons,
            event_data={"similarity": value, "method": p["method"], "selected_pixels": count},
        )
        if value is not None and residual is not None:
            if p["emit_residual"]:
                result.outputs["residual"] = ImageArray(residual, "gray")
            if self.capture_graphics:
                view = np.rint(np.clip(residual / p["residual_preview_max"], 0, 1) * 255).astype(
                    np.uint8
                )
                view = cv2.applyColorMap(view, cv2.COLORMAP_TURBO)
                view[~selected] = 0
                result.graphics["normalized residual"] = view
        if self.generate_graphics:
            label = graphic_object(
                "label", ((8, 22),), ACCEPTED if passed else REJECTED, text=message
            )
            result.drawings["aligned image comparison"] = GraphicDrawing((label,))
        return result
