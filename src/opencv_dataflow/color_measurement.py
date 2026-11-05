"""Measure selected image colors in Lab and compare them with CIE tolerances."""

import math

import cv2
import numpy as np

from .core import Block, Port, Result
from .images import convert_depth, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .processing import finite, image_value
from .selection import SELECTION_INPUTS, selection_mask


def lab_value(value):
    lab = np.asarray(value, dtype=np.float64)
    if (
        lab.shape != (3,)
        or not np.isfinite(lab).all()
        or not 0 <= lab[0] <= 100
        or np.max(np.abs(lab[1:])) > 1000
    ):
        raise ValueError("Lab requires [L, a, b] with L in [0, 100] and finite chroma")
    return lab


def image_lab(source):
    depths = (np.uint8, np.uint16, np.float32)
    if image_format(source) == "lab":
        image = image_value(source, (np.uint8, np.float32))
        if image.dtype == np.uint8:
            lab = image.astype(np.float32)
            lab[..., 0] *= 100 / 255
            lab[..., 1:] -= 128
        else:
            lab = image
        if np.any((lab[..., 0] < 0) | (lab[..., 0] > 100)) or np.any(np.abs(lab[..., 1:]) > 1000):
            raise ValueError("Lab image contains out-of-range lightness or chroma")
        return lab
    image = convert_depth(image_value(source, depths, canonical=True), "float32")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if np.any((image < 0) | (image > 1)):
        raise ValueError("Floating-point color channels must be normalized to [0, 1]")
    return cv2.cvtColor(image, cv2.COLOR_BGR2Lab)


def delta_e_2000(reference, sample, weights):
    """CIEDE2000 as described by Sharma, Wu and Dalal (2005)."""
    l1, a1, b1 = map(float, reference)
    l2, a2, b2 = map(float, sample)
    c_mean = (math.hypot(a1, b1) + math.hypot(a2, b2)) / 2
    adjustment = 0.5 * (1 - math.sqrt(c_mean**7 / (c_mean**7 + 25**7)))
    a1 *= 1 + adjustment
    a2 *= 1 + adjustment
    c1, c2 = math.hypot(a1, b1), math.hypot(a2, b2)
    h1 = math.degrees(math.atan2(b1, a1)) % 360 if c1 else 0.0
    h2 = math.degrees(math.atan2(b2, a2)) % 360 if c2 else 0.0
    hue_difference = h2 - h1
    hue_sum = h1 + h2
    if not c1 * c2:
        hue_difference, h_mean = 0.0, hue_sum
    elif abs(hue_difference) <= 180 + 1e-12:
        h_mean = hue_sum / 2
    else:
        hue_difference -= math.copysign(360, hue_difference)
        h_mean = (hue_sum + (360 if hue_sum < 360 else -360)) / 2
    hue_delta = 2 * math.sqrt(c1 * c2) * math.sin(math.radians(hue_difference / 2))
    l_mean, c_mean = (l1 + l2) / 2, (c1 + c2) / 2
    hue_weight = (
        1
        - 0.17 * math.cos(math.radians(h_mean - 30))
        + 0.24 * math.cos(math.radians(2 * h_mean))
        + 0.32 * math.cos(math.radians(3 * h_mean + 6))
        - 0.20 * math.cos(math.radians(4 * h_mean - 63))
    )
    lightness_term = (l_mean - 50) ** 2
    sl = 1 + 0.015 * lightness_term / math.sqrt(20 + lightness_term)
    sc = 1 + 0.045 * c_mean
    sh = 1 + 0.015 * c_mean * hue_weight
    rotation = (
        -2
        * math.sqrt(c_mean**7 / (c_mean**7 + 25**7))
        * math.sin(math.radians(60 * math.exp(-(((h_mean - 275) / 25) ** 2))))
    )
    lightness = (l2 - l1) / sl / weights[0]
    chroma = (c2 - c1) / sc / weights[1]
    hue = hue_delta / sh / weights[2]
    return math.sqrt(max(0, lightness**2 + chroma**2 + hue**2 + rotation * chroma * hue))


class MeasureLabColor(Block):
    type_id, title = "inspect.measure_lab", "Measure Lab color"
    category, block_type, subtype = "Inspection", "Inspection", "Color measurements"
    inputs = SELECTION_INPUTS
    outputs = (Port("lab", "data"), Port("statistics", "data"), Port("swatch"))

    def process(self, inputs):
        source = inputs["image"]
        mask = selection_mask(inputs, source)
        count = source.shape[0] * source.shape[1] if mask is None else cv2.countNonZero(mask)
        if not count:
            raise ValueError("Lab measurement needs at least one selected pixel")
        mean, std = cv2.meanStdDev(image_lab(source), mask=mask)
        lab = lab_value(mean.ravel()).tolist()
        bgr = cv2.cvtColor(np.float32([[lab]]), cv2.COLOR_Lab2BGR)[0, 0]
        swatch = np.broadcast_to(
            np.rint(np.clip(bgr, 0, 1) * 255).astype(np.uint8), (24, 64, 3)
        ).copy()
        return Result(
            {
                "lab": lab,
                "statistics": {
                    "mean": lab,
                    "standard_deviation": std.ravel().tolist(),
                    "count": count,
                },
                "swatch": swatch,
            },
            payloads={"swatch": {}},
            message=f"Mean Lab ({lab[0]:.2f}, {lab[1]:.2f}, {lab[2]:.2f}) over {count} pixels",
        )


def lab_weights(parameters):
    if parameters["metric"] == "cie76":
        return ()
    return tuple(
        finite(parameters[key], key.replace("_", " ").capitalize(), 1e-6)
        for key in ("lightness_weight", "chroma_weight", "hue_weight")
    )


def lab_distance(reference, sample, metric, weights):
    if metric == "cie76":
        return float(np.linalg.norm(sample - reference))
    return delta_e_2000(reference, sample, weights)


class CompareLabColor(Block):
    type_id, title = "inspect.compare_lab", "Compare Lab colors"
    category, block_type, subtype = "Inspection", "Inspection", "Color measurements"
    inputs = (Port("sample", "data"), Port("reference", "data", required=False))
    outputs = (Port("delta_e", "number"), Port("measurements", "data"))
    defaults = {
        "reference_lab": [50.0, 0.0, 0.0],
        "metric": "ciede2000",
        "maximum_delta_e": 2.0,
        "lightness_weight": 1.0,
        "chroma_weight": 1.0,
        "hue_weight": 1.0,
    }
    parameter_choices = {"metric": ("ciede2000", "cie76")}
    parameter_help = {
        "reference_lab": "Reference [L, a, b]. A wired reference takes precedence.",
        "maximum_delta_e": "Largest color difference that passes.",
        "lightness_weight": "CIEDE2000 kL divisor.",
        "chroma_weight": "CIEDE2000 kC divisor.",
        "hue_weight": "CIEDE2000 kH divisor.",
    }

    def process(self, inputs):
        p = self.parameters
        sample = lab_value(inputs["sample"])
        reference = lab_value(inputs.get("reference", p["reference_lab"]))
        difference = sample - reference
        distance = lab_distance(reference, sample, p["metric"], lab_weights(p))
        passed = distance <= finite(p["maximum_delta_e"], "Maximum delta E", 0)
        measurements = {
            "metric": p["metric"],
            "reference": reference.tolist(),
            "sample": sample.tolist(),
            "delta_lab": difference.tolist(),
            "delta_e": distance,
        }
        return Result(
            {"delta_e": distance, "measurements": measurements},
            passed=passed,
            message=f"{p['metric'].upper()} color difference {distance:.4f}",
            fail_reasons=[] if passed else ["color_difference_above_maximum"],
        )


def lab_palette(value):
    if not isinstance(value, dict) or not 1 <= len(value) <= 256:
        raise ValidationError(
            "References must map 1–256 unique labels to Lab triplets", "references"
        )
    palette = []
    for name, lab in value.items():
        if not isinstance(name, str) or not name.strip() or len(name) > 128:
            raise ValidationError(
                "Color labels must be nonblank and at most 128 characters", "references"
            )
        try:
            palette.append((name, lab_value(lab)))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError(f"Invalid Lab value for {name!r}: {exc}", "references") from exc
    return palette


class ClassifyLabColor(Block):
    type_id, title = "inspect.classify_lab", "Classify Lab color"
    category, block_type, subtype = "Inspection", "Inspection", "Color measurements"
    inputs = (Port("sample", "data"), Port("references", "data", required=False))
    outputs = (
        Port("label", "data", required=False),
        Port("delta_e", "number"),
        Port("classification", "data"),
        Port("swatches"),
    )
    defaults = {
        **{key: value for key, value in CompareLabColor.defaults.items() if key != "reference_lab"},
        "references": {"dark gray": [30, 0, 0], "light gray": [70, 0, 0]},
        "minimum_margin": 0.5,
    }
    parameter_choices = CompareLabColor.parameter_choices
    parameter_ranges = {
        "maximum_delta_e": NumericRange(0),
        "minimum_margin": NumericRange(0),
        "lightness_weight": NumericRange(1e-6),
        "chroma_weight": NumericRange(1e-6),
        "hue_weight": NumericRange(1e-6),
    }
    parameter_help = {
        **{
            key: value
            for key, value in CompareLabColor.parameter_help.items()
            if key != "reference_lab"
        },
        "references": 'JSON object such as {"red": [53, 80, 67]}. Wired values take precedence.',
        "minimum_margin": "Minimum ΔE gap to the runner-up. Zero allows ties in label order.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        values = self.parameters if parameters is None else parameters
        lab_palette(values["references"])

    def process(self, inputs):
        p = self.parameters
        try:
            sample = lab_value(inputs["sample"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError(str(exc)) from exc
        palette = lab_palette(inputs.get("references", p["references"]))
        metric, weights = p["metric"], lab_weights(p)
        ranked = sorted(
            (
                {
                    "label": label,
                    "lab": lab.tolist(),
                    "delta_e": lab_distance(lab, sample, metric, weights),
                }
                for label, lab in palette
            ),
            key=lambda item: (item["delta_e"], item["label"]),
        )
        best = ranked[0]
        distance = best["delta_e"]
        margin = ranked[1]["delta_e"] - distance if len(ranked) > 1 else None
        if distance > float(p["maximum_delta_e"]):
            reason = "color_class_distance_above_maximum"
            status = "outside tolerance"
        elif margin is not None and margin < float(p["minimum_margin"]):
            reason = "color_class_ambiguous"
            status = "ambiguous"
        else:
            reason = None
            status = "accepted"
        bgr = cv2.cvtColor(np.float32([[sample.tolist(), best["lab"]]]), cv2.COLOR_Lab2BGR)
        swatches = np.repeat(
            np.repeat(np.rint(np.clip(bgr, 0, 1) * 255).astype(np.uint8), 24, axis=0), 64, axis=1
        )
        return Result(
            {
                "label": best["label"] if reason is None else NO_VALUE,
                "delta_e": distance,
                "classification": {
                    "status": status,
                    "metric": metric,
                    "sample": sample.tolist(),
                    "candidate_label": best["label"],
                    "margin": margin,
                    "ranking": ranked,
                },
                "swatches": swatches,
            },
            payloads={"swatches": {}},
            passed=reason is None,
            message=f"{best['label']} · ΔE {distance:.4f} · {status}",
            fail_reasons=[reason] if reason else [],
        )


COLOR_MEASUREMENT_BLOCKS = (MeasureLabColor, CompareLabColor, ClassifyLabColor)
