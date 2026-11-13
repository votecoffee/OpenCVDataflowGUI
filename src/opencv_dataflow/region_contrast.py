"""Selected signal/background statistics with explicit contrast and noise formulas."""

import copy
import math

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .histogram import CHANNEL_CHOICES, HISTOGRAM_CHANNELS
from .images import image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .processing import image_value
from .selection import SELECTION_INPUTS, selection_mask

_MAX_PIXELS = 16_777_216
_METRICS = ("cnr", "absolute_difference", "weber", "michelson")


def contrast_planes(source, mode, alpha):
    image = image_value(source)
    encoding = image_format(source)
    if image.ndim == 2:
        return (image,), ("Gray",)
    if mode == "native":
        labels = HISTOGRAM_CHANNELS[encoding]
        if len(labels) == 4 and not alpha:
            labels = labels[:3]
        return tuple(image[..., index] for index in range(len(labels))), labels
    if encoding in ("rgb", "rgba"):
        bgr = image[..., [2, 1, 0]]
    elif encoding in ("bgr", "bgra"):
        bgr = image[..., :3]
    else:
        bgr = image_value(source, canonical=True)[..., :3]
    bgr = np.asarray(bgr, np.float64)
    blue, green, red = (bgr[..., index] for index in range(3))
    # Anchoring at green preserves small changes on a common large DC level.
    with np.errstate(over="ignore", invalid="ignore"):
        gray = green + 0.114 * (blue - green) + 0.299 * (red - green)
    bad = ~np.isfinite(gray)
    if bad.any():
        values = bgr[bad]
        magnitude = np.max(np.abs(values), axis=1)
        gray[bad] = np.sum(values / magnitude[:, None] * [0.114, 0.587, 0.299], axis=1) * magnitude
    if not np.isfinite(gray).all():
        raise ValidationError("Grayscale contrast samples exceed the float64 range")
    return (gray,), ("Gray",)


def raw_units(value, scale):
    result = float(value) * scale
    return result if math.isfinite(result) and (result != 0 or value == 0) else None


def ratio(numerator, denominator):
    if denominator <= 0:
        return None
    result = float(numerator) / float(denominator)
    return result if math.isfinite(result) and (result != 0 or numerator == 0) else None


def centered_statistics(samples, anchor, scale, sample_deviation):
    with np.errstate(over="ignore", invalid="ignore"):
        offsets = (samples - anchor) / scale
    invalid = ~np.isfinite(offsets)
    if invalid.any():
        offsets[invalid] = samples[invalid] / scale - anchor / scale
    offsets = offsets.reshape(-1, 1)
    mean_offset = cv2.mean(offsets)[0]
    residual = offsets - mean_offset
    magnitude = float(np.max(np.abs(residual)))
    sigma = 0.0
    if magnitude:
        _, native = cv2.meanStdDev(residual / magnitude)
        sigma = float(native[0, 0]) * magnitude
    if sample_deviation:
        sigma *= math.sqrt(len(samples) / (len(samples) - 1))
    return mean_offset, sigma


def channel_contrast(signal, background, parameters):
    maximum = max(float(np.max(np.abs(signal))), float(np.max(np.abs(background))))
    scale = maximum if 0 < maximum < 1e-150 or maximum > 1e150 else 1.0
    anchor = float(background[0])
    sample = parameters["deviation"] == "sample"
    s, ss = centered_statistics(signal, anchor, scale, sample)
    b, bs = centered_statistics(background, anchor, scale, sample)
    sm, bm, difference = anchor / scale + s, anchor / scale + b, s - b
    measured_noise = math.hypot(ss, bs) if parameters["noise_model"] == "quadrature" else bs
    floor = float(parameters["noise_floor"])
    floor_scaled = floor / scale
    noise = max(measured_noise, floor_scaled)
    delta = raw_units(difference, scale)
    cnr = ratio(abs(difference), noise)
    if not math.isfinite(floor_scaled):
        cnr = ratio(abs(delta), floor) if delta is not None else None
    nonnegative = bool(np.min(signal) >= 0 and np.min(background) >= 0)
    weber = ratio(difference, bm) if nonnegative else None
    michelson = ratio(difference, sm + bm) if nonnegative else None
    fields = {
        "signal_mean": raw_units(sm, scale),
        "background_mean": raw_units(bm, scale),
        "signal_stddev": raw_units(ss, scale),
        "background_stddev": raw_units(bs, scale),
        "difference": delta,
        "absolute_difference": abs(delta) if delta is not None else None,
        "measured_noise": raw_units(measured_noise, scale),
        "noise": floor if not math.isfinite(floor_scaled) else raw_units(noise, scale),
        "noise_floor_applied": floor_scaled > measured_noise,
        "cnr": cnr,
        "weber": weber,
        "michelson": michelson,
    }
    reasons = {}
    for key, value in fields.items():
        if value is None:
            reasons[key] = "unrepresentable_value"
    if cnr is None and noise == 0:
        reasons["cnr"] = "zero_noise"
    if not nonnegative:
        reasons.update(
            weber="requires_nonnegative_samples", michelson="requires_nonnegative_samples"
        )
    elif bm <= 0:
        reasons["weber"] = "nonpositive_background"
    if nonnegative and sm + bm <= 0:
        reasons["michelson"] = "nonpositive_sum"
    fields["undefined"] = reasons
    return fields, math.copysign(1.0, difference) if difference else 0.0


def selection_outlines(mask, color):
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    limited = len(contours) > 64
    if limited:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:64]
    objects = []
    for contour in contours:
        points = contour
        epsilon = 0.25
        while len(points) > 2048:
            limited = True
            points = cv2.approxPolyDP(contour, epsilon, True)
            epsilon *= 2
        objects.append(graphic_object("polyline", points.reshape(-1, 2), color, closed=True))
    return objects, limited


class MeasureRegionContrast(Block):
    type_id, title = "inspect.region_contrast", "Measure signal/background contrast"
    category, block_type, subtype = "Inspection", "Inspection", "Visibility"
    process_inputs_readonly = True
    payload_reference = "image"
    inputs = (
        *SELECTION_INPUTS,
        Port("signal_mask", required=False, required_when_connected=True, visual_type="mask"),
        Port("signal_region", "data", required=False, required_when_connected=True),
        Port("background_mask", required=False, required_when_connected=True, visual_type="mask"),
        Port("background_region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("value", "number", required=False),
        Port("valid", "data"),
        Port("measurements", "data"),
        Port("signal_mask", visual_type="mask"),
        Port("background_mask", visual_type="mask"),
        Port("signal_count", "number"),
        Port("background_count", "number"),
        *(
            Port(name, "number", required=False)
            for name in (
                "difference",
                "cnr",
                "weber",
                "michelson",
                "noise",
                "signal_mean",
                "background_mean",
                "signal_stddev",
                "background_stddev",
            )
        ),
    )
    defaults = {
        "channel_mode": "gray",
        "channel": "auto",
        "include_alpha": False,
        "background_mode": "explicit",
        "overlap": "reject",
        "minimum_pixels": 4,
        "deviation": "population",
        "noise_model": "quadrature",
        "noise_floor": 0.0,
        "metric": "cnr",
        "check_threshold": False,
        "minimum_value": 3.0,
        "polarity": "either",
    }
    parameter_choices = {
        "channel_mode": ("gray", "native"),
        "channel": ("auto", *CHANNEL_CHOICES),
        "background_mode": ("explicit", "outside_signal"),
        "overlap": ("reject", "exclude_signal", "allow"),
        "deviation": ("population", "sample"),
        "noise_model": ("quadrature", "background"),
        "metric": _METRICS,
        "polarity": ("either", "bright", "dark"),
    }
    parameter_ranges = {
        "minimum_pixels": NumericRange(1, _MAX_PIXELS, whole=True),
        "noise_floor": NumericRange(0),
        "minimum_value": NumericRange(0),
    }
    parameter_help = {
        "channel_mode": (
            "Gray uses floating BT.601 intensity without depth conversion. "
            "Native reports every stored channel."
        ),
        "channel": "Channel used for Value and decisions. Auto chooses the first reported channel.",
        "include_alpha": (
            "Native mode may measure alpha as a numeric channel. Alpha never weights selections."
        ),
        "background_mode": (
            "Explicit requires a background selector. "
            "Outside signal uses the remaining work selection."
        ),
        "overlap": "Reject, exclude signal pixels from background, or allow shared samples.",
        "minimum_pixels": "Minimum final pixel counts. Sample deviation also needs at least two.",
        "noise_model": (
            "Quadrature uses hypot(signal_stddev, background_stddev). "
            "Background uses background_stddev."
        ),
        "noise_floor": (
            "Minimum denominator in raw sample units. "
            "Zero noise without a positive floor leaves CNR undefined."
        ),
        "metric": (
            "Value is the magnitude of the selected metric. "
            "Signed differences and contrasts remain in the report."
        ),
        "check_threshold": "Require minimum Value and requested polarity on the selected channel.",
        "polarity": (
            "Bright/dark checks the sign of signal minus background. "
            "Equal means satisfy only Either."
        ),
    }
    input_help = {
        "mask": "Optional valid/work mask applied to both selections.",
        "region": "Optional work region applied to both selections.",
        "signal_mask": "Signal mask; intersects Signal region when both are connected.",
        "signal_region": "Region on the image grid. Use an editable Shape region block.",
        "background_mask": "Background mask; intersects Background region in Explicit mode.",
        "background_region": "Region on the image grid, used only with Explicit background mode.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = ["image", "value", "valid", "measurements"]

    def process(self, inputs):
        source, p = inputs["image"], self.parameters
        minimum_value = float(p["minimum_value"])
        image = image_value(source)
        if image.shape[0] * image.shape[1] > _MAX_PIXELS:
            raise ValidationError("Region contrast is limited to 16,777,216 pixels")
        work = selection_mask(inputs, source)
        if work is None:
            work = np.full(image.shape[:2], 255, np.uint8)
        masks, reason = {}, None
        for name in ("signal", "background"):
            if name == "background" and p["background_mode"] == "outside_signal":
                if any(f"background_{kind}" in inputs for kind in ("mask", "region")):
                    raise ValidationError("Disconnect background selectors in Outside signal mode")
                masks[name] = cv2.bitwise_and(work, cv2.bitwise_not(masks["signal"]))
                continue
            selectors = {
                kind: inputs[f"{name}_{kind}"]
                for kind in ("mask", "region")
                if f"{name}_{kind}" in inputs
            }
            if not selectors:
                reason = reason or f"missing_{name}_selection"
                masks[name] = np.zeros(image.shape[:2], np.uint8)
            else:
                selected = selection_mask({"image": source, **selectors}, source)
                masks[name] = cv2.bitwise_and(work, selected)
        overlap = int(np.count_nonzero(cv2.bitwise_and(masks["signal"], masks["background"])))
        if overlap:
            if p["overlap"] == "reject":
                reason = reason or "overlapping_selections"
            elif p["overlap"] == "exclude_signal":
                masks["background"] = cv2.bitwise_and(
                    masks["background"], cv2.bitwise_not(masks["signal"])
                )
        counts = {name: int(np.count_nonzero(mask)) for name, mask in masks.items()}
        required = max(int(p["minimum_pixels"]), 2 if p["deviation"] == "sample" else 1)
        for name, count in counts.items():
            if count < required:
                reason = reason or (f"empty_{name}" if not count else f"too_few_{name}_pixels")
        planes, labels = contrast_planes(source, p["channel_mode"], p["include_alpha"])
        channel = labels[0] if p["channel"] == "auto" else p["channel"]
        if channel not in labels:
            raise ValidationError("Selected contrast channel is unavailable", "channel")
        rows, signs = [], {}
        if reason is None:
            for label, plane in zip(labels, planes, strict=True):
                signal = np.asarray(plane[masks["signal"] != 0], np.float64)
                background = np.asarray(plane[masks["background"] != 0], np.float64)
                row, sign = channel_contrast(signal, background, p)
                rows.append({"channel": label, **row})
                signs[label] = sign
        selected = next((row for row in rows if row["channel"] == channel), {})
        chosen = selected.get(p["metric"])
        value = abs(chosen) if chosen is not None else None
        selection_valid = reason is None
        valid = selection_valid and value is not None
        if selection_valid and not valid:
            reason = "undefined_metric"
        value_passed = valid and value >= minimum_value
        sign = signs.get(channel, 0)
        polarity_passed = p["polarity"] == "either" or (
            sign > 0 if p["polarity"] == "bright" else sign < 0
        )
        passed = valid and (not p["check_threshold"] or (value_passed and polarity_passed))
        failures = [f"contrast_{reason}"] if reason else []
        if valid and p["check_threshold"]:
            if not value_passed:
                failures.append("contrast_below_minimum")
            if not polarity_passed:
                failures.append("contrast_wrong_polarity")
        status = reason or ("pass" if passed else "limits_failed")
        measurements = {
            "model": "region_contrast_v1",
            "image_size": [image.shape[1], image.shape[0]],
            "pixel_depth": str(image.dtype),
            "encoding": image_format(source),
            "status": status,
            "valid": valid,
            "selection_valid": selection_valid,
            "channel_mode": p["channel_mode"],
            "channel": channel,
            "metric": p["metric"],
            "value": value,
            "channels": rows,
            "noise_model": p["noise_model"],
            "noise_floor": float(p["noise_floor"]),
            "deviation": p["deviation"],
            "signal_count": counts["signal"],
            "background_count": counts["background"],
            "background_mode": p["background_mode"],
            "minimum_pixels": int(p["minimum_pixels"]),
            "required_pixels": required,
            "overlap_count": overlap,
            "overlap_policy": p["overlap"],
            "criteria": {
                "enabled": p["check_threshold"],
                "minimum_value": minimum_value,
                "polarity": p["polarity"],
                "value_passed": value_passed,
                "polarity_passed": polarity_passed,
            },
        }
        drawings = {}
        if self.generate_graphics:
            limited = False
            for name, color in (("signal", (240, 210, 40)), ("background", (50, 150, 245))):
                objects, clipped = selection_outlines(masks[name], color)
                limited |= clipped
                if objects:
                    drawings[name] = GraphicDrawing(tuple(objects))
            text = f"{channel} {p['metric']}: {value:.5g}" if valid else f"Contrast: {status}"
            label = graphic_object(
                "text",
                [[0, min(16, image.shape[0] - 1)]],
                (50, 220, 80) if passed else (60, 70, 240),
                text=text,
            )
            drawings["contrast"] = GraphicDrawing((label,))
            measurements["graphic_outlines_limited"] = limited
        outputs = {
            "image": source,
            "value": value if valid else NO_VALUE,
            "valid": valid,
            "measurements": measurements,
            "signal_mask": masks["signal"],
            "background_mask": masks["background"],
            "signal_count": counts["signal"],
            "background_count": counts["background"],
            **{
                name: selected.get(name) if selected.get(name) is not None else NO_VALUE
                for name in (
                    "difference",
                    "cnr",
                    "weber",
                    "michelson",
                    "noise",
                    "signal_mean",
                    "background_mean",
                    "signal_stddev",
                    "background_stddev",
                )
            },
        }
        payloads = {}
        for name in ("signal_mask", "background_mask"):
            payloads[name] = copy.deepcopy(self.input_payloads["image"])
            payloads[name].pop("alpha_representation", None)
        score = None
        if valid and p["check_threshold"]:
            score = (
                (min(1.0, value / minimum_value) if minimum_value else 1.0)
                if polarity_passed
                else 0.0
            )
        return Result(
            outputs,
            payloads=payloads,
            drawings=drawings,
            passed=passed,
            score=score,
            fail_reasons=failures,
            event_data={
                "contrast": {"status": status, "metric": p["metric"], "channel": channel, **counts}
            },
            message=f"{channel} {p['metric']}: {value:.5g}"
            if valid
            else f"Contrast invalid: {status}",
        )
