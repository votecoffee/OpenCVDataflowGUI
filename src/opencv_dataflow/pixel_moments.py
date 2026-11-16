"""Per-pixel centered temporal moments, paired covariance and explicit support maps."""

import copy

import cv2
import numpy as np

from .core import Port
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import ValuePacket, require_same_frame
from .processing import image_value
from .sequence import (
    SEQUENCE_CHOICES,
    SEQUENCE_DEFAULTS,
    SEQUENCE_HELP,
    SequenceBlock,
    SequenceStep,
)
from .tickets import ticket_key

CHANNELS = ("gray", "blue", "green", "red", "alpha")
ARRAYS = (
    "count",
    "weight",
    "weight2",
    "origin_x",
    "origin_y",
    "mean_x",
    "mean_y",
    "m2_x",
    "m2_y",
    "cross",
)
MAPS = (
    "mean_x",
    "mean_y",
    "variance_x",
    "variance_y",
    "std_x",
    "std_y",
    "covariance",
    "correlation",
    "count",
    "effective_count",
    "valid",
    "correlation_valid",
)


def plane(image, channel):
    raw = image_value(image)
    if image_format(image) not in ("gray", "bgr", "rgb", "bgra", "rgba"):
        raise ValidationError("Convert HSV/Lab images to gray or RGB before temporal statistics")
    if raw.ndim == 2:
        if channel != "gray":
            raise ValidationError("A color/alpha channel was selected for a monochrome image")
        return np.ascontiguousarray(raw, dtype=np.float64)
    order = (2, 1, 0) if image_format(image) in ("rgb", "rgba") else (0, 1, 2)
    if channel == "gray":
        values = raw.astype(np.float64)
        return (
            0.114 * values[..., order[0]]
            + 0.587 * values[..., order[1]]
            + 0.299 * values[..., order[2]]
        )
    if channel == "alpha":
        if raw.shape[2] != 4:
            raise ValidationError("Alpha requires a four-channel image")
        index = 3
    else:
        index = order[("blue", "green", "red").index(channel)]
    return np.ascontiguousarray(raw[..., index], dtype=np.float64)


class PixelMoments(SequenceBlock):
    process_inputs_readonly = True
    type_id, title, subtype = (
        "sequence.pixel_moments",
        "Temporal pixel statistics",
        "Temporal statistics",
    )
    inputs = (
        *SequenceBlock.inputs,
        Port("other", required=False, required_when_connected=True),
        Port("learn", "boolean", required=False),
    )
    data_outputs = tuple(Port(name) for name in MAPS)
    defaults = {
        **SEQUENCE_DEFAULTS,
        "max_bytes_mb": 256.0,
        "method": "cumulative",
        "normalization": "sample",
        "alpha": 0.05,
        "channel_x": "gray",
        "channel_y": "gray",
        "alignment": "same_frame",
        "ticket_policy": "match_if_present",
        "minimum_samples": 2,
        "maximum_samples": 1_000_000,
        "minimum_coverage": 1.0,
        "maximum_pixels": 4_194_304,
        "maximum_workspace_bytes": 1024**3,
        "preview_metric": "std_x",
        "preview_min": 0.0,
        "preview_max": 10.0,
    }
    parameter_choices = {
        **SEQUENCE_CHOICES,
        "method": ("cumulative", "exponential"),
        "normalization": ("population", "sample"),
        "channel_x": CHANNELS,
        "channel_y": CHANNELS,
        "alignment": ("same_frame", "pixel_indices"),
        "ticket_policy": ("match_if_present", "require_match", "ignore"),
        "preview_metric": ("std_x", "variance_x", "covariance", "correlation", "mean_x"),
    }
    parameter_ranges = {
        "alpha": NumericRange(1e-6, 1),
        "minimum_samples": NumericRange(2, 1e9, whole=True),
        "maximum_samples": NumericRange(2, 1e9, whole=True),
        "minimum_coverage": NumericRange(0.001, 1),
        "maximum_pixels": NumericRange(1, 16_777_216, whole=True),
        "maximum_workspace_bytes": NumericRange(1, 2 * 1024**3, whole=True),
        "preview_min": NumericRange(-1e100, 1e100),
        "preview_max": NumericRange(-1e100, 1e100),
        "warmup": NumericRange(1, 10000, whole=True),
        "max_streams": NumericRange(1, 64, whole=True),
        "max_bytes_mb": NumericRange(0.001, 4096),
        "max_gap_ms": NumericRange(0, 3_600_000),
    }
    parameter_help = {
        **SEQUENCE_HELP,
        "method": "Cumulative or exponentially weighted eligible observations per pixel.",
        "normalization": "Population divides by sum of weights; sample uses W - sum(w²)/W.",
        "alpha": "Exponential observation weight; no decay while paused or outside selection.",
        "channel_x": "Measured channel of Image, in its original numerical sample units.",
        "channel_y": "Channel of Other when connected, otherwise another channel of Image.",
        "alignment": "Check declared grids, or explicitly assert registration by pixel indices.",
        "ticket_policy": "Other-source ticket matching; not proof of exposure synchronization.",
        "minimum_samples": "Per-pixel accepted observations needed for a valid estimate.",
        "maximum_samples": "Freeze each pixel at this count; counts stay exactly representable.",
        "minimum_coverage": "Required valid fraction of currently selected pixels for readiness.",
        "maximum_workspace_bytes": (
            "Conservative working-map estimate, not a hard RSS bound; GUI/runtime copies are extra."
        ),
        "preview_metric": "Diagnostic heatmap only; graphics off/Lean skips it.",
        "preview_min": "Heatmap lower bound. Correlation always uses -1..1.",
        "preview_max": "Heatmap upper bound; does not alter numerical maps.",
    }
    input_help = {
        "image": "Finite uint8/uint16/int16/float32/float64 gray/RGB(A); absolute samples ≤1e100.",
        "other": "Optional paired image with equal dimensions; connected inputs must emit.",
        "learn": "False freezes the model but allows outputs from existing eligible observations.",
        "mask": "Selected pixels learn and appear in maps; excluded pixels are not observations.",
        "region": "Rectangle/circle/polygon intersection with Mask in Image coordinates.",
    }
    output_help = {
        "valid": "255 where selected pixels have enough support; zero-filled values are invalid.",
        "correlation_valid": "255 only where correlation is defined (both variances positive).",
        "count": "Unweighted accepted count per selected pixel, even in exponential mode.",
        "effective_count": "W²/sum(w²): weight concentration, not proof of independent samples.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        visible = {"mean_x", "std_x", "covariance", "count", "valid", "ready", "status"}
        self.exposed_outputs = [name for name in self.exposed_outputs if name in visible]

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for name, limits in self.parameter_ranges.items():
            if type(p[name]) not in (int, float) or limits.whole and type(p[name]) is not int:
                raise ValidationError(f"{name} requires a JSON number of the declared type", name)
        if max(p["minimum_samples"], p["warmup"]) > p["maximum_samples"]:
            raise ValidationError(
                "Minimum samples and Warmup must not exceed Maximum samples",
                "minimum_samples",
                "maximum_samples",
                "warmup",
            )
        if p["method"] == "exponential" and p["normalization"] == "sample" and p["alpha"] == 1:
            raise ValidationError(
                "Sample variance requires Alpha below one", "alpha", "normalization"
            )
        if p["preview_max"] <= p["preview_min"]:
            raise ValidationError(
                "Preview maximum must exceed minimum", "preview_min", "preview_max"
            )

    def estimate_state_bytes(self, image):
        return image.shape[0] * image.shape[1] * len(ARRAYS) * 8 + 8192

    def extra_context(self, inputs):
        image, p = inputs["image"], self.parameters
        if self.estimate_state_bytes(image) > p["max_bytes_mb"] * 1024**2:
            raise ValidationError("Temporal model exceeds Sequence memory MB", "max_bytes_mb")
        pixels = image.shape[0] * image.shape[1]
        other = inputs.get("other", image)
        if (
            pixels > p["maximum_pixels"]
            or pixels * 48 * 8 + image.nbytes + other.nbytes > p["maximum_workspace_bytes"]
        ):
            raise ValidationError(
                "Temporal statistics exceed pixel/workspace limits",
                "maximum_pixels",
                "maximum_workspace_bytes",
            )
        if other.shape[:2] != image.shape[:2]:
            raise ValidationError("Paired images must have equal pixel dimensions")
        for value in (image, other):
            image_value(value)
            if np.max(np.abs(value.astype(np.float64))) > 1e100:
                raise ValidationError("Temporal image samples exceed the absolute 1e100 bound")
        # Validate channel/layout before allocating a temporal model.
        for value, field in ((image, "channel_x"), (other, "channel_y")):
            try:
                plane(value, p[field])
            except ValidationError as exc:
                raise ValidationError(str(exc), field) from exc
        if "other" not in inputs:
            return {"paired_source": False}
        a, b = self.input_payloads["image"], self.input_payloads["other"]
        if p["alignment"] == "same_frame":
            try:
                require_same_frame((ValuePacket(image, a), ValuePacket(other, b)))
            except ValueError as exc:
                raise ValidationError(str(exc), "alignment") from exc
        left, right = a.get("ticket"), b.get("ticket")
        if p["ticket_policy"] == "require_match" and (left is None or right is None):
            raise ValidationError("Both paired sources require tickets")
        if (
            p["ticket_policy"] != "ignore"
            and left is not None
            and right is not None
            and ticket_key(left) != ticket_key(right)
        ):
            raise ValidationError("Paired source tickets do not match")
        return {
            "paired_source": True,
            "shape": list(other.shape),
            "dtype": str(other.dtype),
            "encoding": image_format(other),
            "context": {key: b[key] for key in ("spatial", "pyramid", "recipe_ref") if key in b},
            "source": b.get("ticket", {}).get("source"),
        }

    def step(self, inputs, state, mask):
        p = self.parameters
        image, other = inputs["image"], inputs.get("other", inputs["image"])
        shape = image.shape[:2]
        selected = np.ones(shape, bool) if mask is None else mask != 0
        learn = inputs.get("learn", True)
        if type(learn) is not bool:
            raise ValidationError("Learn must be boolean")
        if "count" not in state:
            state.update({name: np.zeros(shape, np.float64) for name in ARRAYS})
        count, weight, weight2 = (state[key] for key in ("count", "weight", "weight2"))
        update = selected & (count < p["maximum_samples"]) & learn
        changed = bool(update.any())
        if changed:
            x, y = plane(image, p["channel_x"]), plane(other, p["channel_y"])
            fresh = update & (count == 0)
            state["origin_x"][fresh], state["origin_y"][fresh] = x[fresh], y[fresh]
            dx = np.where(update, x - state["origin_x"] - state["mean_x"], 0)
            dy = np.where(update, y - state["origin_y"] - state["mean_y"], 0)
            alpha = p["alpha"] if p["method"] == "exponential" else 1.0
            beta = 1 - alpha if p["method"] == "exponential" else 1.0
            denominator = beta * weight + alpha
            gain, factor = alpha / denominator, alpha * beta * weight / denominator
            native_mask = update.astype(np.uint8) * 255
            if p["method"] == "exponential":
                zeros = np.zeros(shape, np.float64)
                for key in ("m2_x", "m2_y", "cross", "weight"):
                    cv2.accumulateWeighted(zeros, state[key], alpha, mask=native_mask)
                cv2.accumulateWeighted(zeros, weight2, 1 - beta**2, mask=native_mask)
            cv2.accumulate(dx * gain, state["mean_x"], mask=native_mask)
            cv2.accumulate(dy * gain, state["mean_y"], mask=native_mask)
            qx, qy = dx * np.sqrt(factor), dy * np.sqrt(factor)
            cv2.accumulateSquare(qx, state["m2_x"], mask=native_mask)
            cv2.accumulateSquare(qy, state["m2_y"], mask=native_mask)
            cv2.accumulateProduct(qx, qy, state["cross"], mask=native_mask)
            ones = np.ones(shape, np.float64)
            cv2.accumulate(ones, count, mask=native_mask)
            cv2.accumulate(ones * alpha, weight, mask=native_mask)
            cv2.accumulateSquare(ones * alpha, weight2, mask=native_mask)
            state["last_observation_tickets"] = {
                name: copy.deepcopy(self.input_payloads.get(name, {}).get("ticket"))
                for name in ("image", "other")
                if name in inputs
            }
        denominator = weight.copy()
        if p["normalization"] == "sample":
            denominator -= np.divide(weight2, weight, out=np.zeros(shape), where=weight > 0)
        valid = selected & (count >= p["minimum_samples"]) & (denominator > 0)
        divisor = np.where(valid, denominator, 1.0)
        vx, vy = (np.maximum(state[key] / divisor, 0) for key in ("m2_x", "m2_y"))
        covariance = state["cross"] / divisor
        correlation_divisor = np.sqrt(state["m2_x"]) * np.sqrt(state["m2_y"])
        correlation_valid = valid & (correlation_divisor > 0)
        correlation = np.divide(
            state["cross"], correlation_divisor, out=np.zeros(shape), where=correlation_valid
        )
        values = {
            "mean_x": state["origin_x"] + state["mean_x"],
            "mean_y": state["origin_y"] + state["mean_y"],
            "variance_x": vx,
            "variance_y": vy,
            "std_x": np.sqrt(vx),
            "std_y": np.sqrt(vy),
            "covariance": covariance,
            "correlation": np.clip(correlation, -1, 1),
            "count": np.where(selected, count, 0),
            "effective_count": np.where(
                selected, np.divide(weight**2, weight2, out=np.zeros(shape), where=weight2 > 0), 0
            ),
            "valid": valid.astype(np.uint8) * 255,
            "correlation_valid": correlation_valid.astype(np.uint8) * 255,
        }
        for name in MAPS[:8]:
            values[name] = np.where(valid, values[name], 0)
        total, supported = int(selected.sum()), int(valid.sum())
        fraction = supported / total if total else 0.0
        ready = total > 0 and fraction >= p["minimum_coverage"]
        details = {
            "phase": "updating" if changed else "frozen" if total else "empty_selection",
            "selected_pixels": total,
            "valid_pixels": supported,
            "coverage": fraction,
            "updated_pixels": int(update.sum()),
            "paired_source": "other" in inputs,
            "alignment_evidence": "same_image"
            if "other" not in inputs
            else "explicit_pixel_indices"
            if p["alignment"] == "pixel_indices"
            else "matching_declared_frames"
            if all(self.input_payloads[name].get("spatial") for name in ("image", "other"))
            else "dimensions_only",
            "method": p["method"],
            "normalization": p["normalization"],
            "std_x_rms": float(np.sqrt(vx[valid].mean())) if supported else None,
            "effective_count_min": float(values["effective_count"][valid].min())
            if supported
            else 0.0,
            "minimum_count": float(count[selected].min()) if total else 0,
            "maximum_count": float(count[selected].max()) if total else 0,
            "last_observation_tickets": state.get("last_observation_tickets"),
            "variance_units": "sample units squared",
            "mean_units": "sample units",
            "workspace_estimate_bytes": shape[0] * shape[1] * 48 * 8 + image.nbytes + other.nbytes,
        }
        graphics = {}
        if ready and state["learning"] + int(changed) >= p["warmup"] and self.generate_graphics:
            metric = values[p["preview_metric"]]
            low, high = (
                (-1.0, 1.0)
                if p["preview_metric"] == "correlation"
                else (p["preview_min"], p["preview_max"])
            )
            shade = np.rint(np.clip((metric - low) / (high - low), 0, 1) * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(shade, cv2.COLORMAP_TURBO)
            heatmap[~(correlation_valid if p["preview_metric"] == "correlation" else valid)] = 0
            graphics[f"{p['preview_metric']} [{low:g}, {high:g}]"] = heatmap
        return SequenceStep(
            {name: ImageArray(values[name], "gray") for name in MAPS},
            ready=ready,
            learned=changed,
            details=details,
            graphics=graphics,
        )
