"""Bounded capture-stack estimation of reusable bright and dark references."""

import copy

import numpy as np

from .core import Block, Port, Result
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, FlowGroup, check_spatial, frame, metadata, require_same_frame
from .processing import image_value
from .selection import selection_mask


def estimate(frames, method, saturation, noise):
    """Reduce bounded row tiles; normalize per pixel/channel to protect HDR arithmetic."""
    h, w = frames[0].shape[:2]
    shape = frames[0].shape
    result = np.zeros(shape, np.float64)
    deviation = np.zeros(shape, np.float64) if noise else None
    counts = np.zeros((h, w), np.uint8)
    channels = 1 if len(shape) == 2 else shape[2]
    rows = max(1, min(32, 131072 // (w * len(frames) * channels)))
    for top in range(0, h, rows):
        slab = np.stack([image[top : top + rows] for image in frames]).astype(np.float64)
        accepted = np.ones(slab.shape[:3], bool)
        if saturation is not None:
            accepted = slab < saturation
            if slab.ndim == 4:
                accepted = accepted.all(axis=3)
        count = accepted.sum(axis=0)
        keep = accepted[..., None] if slab.ndim == 4 else accepted
        slab = np.where(keep, slab, 0)
        amplitude = np.max(np.abs(slab), axis=0)
        divisor = np.where(amplitude > 0, amplitude, 1)
        scaled = slab / divisor
        n = count[..., None] if slab.ndim == 4 else count
        mean = scaled.sum(axis=0) / np.maximum(n, 1)
        reduced = mean
        if method == "median":
            ordered = np.sort(np.where(keep, scaled, np.inf), axis=0)
            low = np.broadcast_to(np.maximum(n - 1, 0) // 2, amplitude.shape)
            high = np.broadcast_to(n // 2, amplitude.shape)
            a = np.take_along_axis(ordered, low[None], axis=0)[0]
            b = np.take_along_axis(ordered, high[None], axis=0)[0]
            reduced = np.where(n > 0, np.where(a == b, a, a * 0.5 + b * 0.5), 0)
        result[top : top + rows] = reduced.clip(-1, 1) * amplitude
        counts[top : top + rows] = count
        if noise:
            variance = np.where(keep, (scaled - mean) ** 2, 0).sum(axis=0) / np.maximum(n, 1)
            deviation[top : top + rows] = np.sqrt(variance).clip(0, 1) * amplitude
    return result, counts, deviation


class BuildFlatFieldReference(Block):
    type_id, title = "calibration.flat_field_reference", "Build flat-field references"
    category = block_type = "Calibration"
    subtype = "Illumination"
    parallel_safe = process_inputs_readonly = True
    payload_reference = "bright"
    inputs = (
        Port("bright", "any"),
        Port("dark", "any", required=False, required_when_connected=True),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("flat"),
        Port("dark"),
        Port("response"),
        Port("valid_mask"),
        Port("invalid_mask"),
        Port("bright_count"),
        Port("dark_count"),
        Port("info", "data"),
        Port("source"),
        Port("bright_noise", required=False),
        Port("dark_noise", required=False),
        Port("preview", required=False),
    )
    defaults = {
        "method": "mean",
        "minimum_frames": 3,
        "anchor_index": 0,
        "alignment": "same_frame",
        "output_depth": "float64",
        "reject_saturation": True,
        "saturation_level": 0.99,
        "minimum_response": 0.01,
        "minimum_valid_fraction": 0.95,
        "check_noise": False,
        "maximum_noise": 0.02,
        "retain_noise": False,
    }
    parameter_choices = {
        "method": ("mean", "median"),
        "alignment": ("same_frame", "pixel_grid"),
        "output_depth": ("same", "float32", "float64"),
    }
    parameter_ranges = {
        "minimum_frames": NumericRange(1, 64, whole=True),
        "anchor_index": NumericRange(0, 63, whole=True),
        "saturation_level": NumericRange(0),
        "minimum_response": NumericRange(0),
        "minimum_valid_fraction": NumericRange(0, 1),
        "maximum_noise": NumericRange(0),
    }
    parameter_help = {
        "method": "Per-pixel mean or median of accepted samples; even medians average two values.",
        "minimum_frames": "Minimum usable captures per selected pixel in each connected batch.",
        "anchor_index": "Bright capture defining the output grid and Source image.",
        "alignment": "Require matching grids, or assume equal-size pixels are already aligned.",
        "output_depth": "Floating references keep raw sample units; Same rounds integer estimates.",
        "reject_saturation": "Reject a capture pixel if any intensity channel reaches the limit.",
        "saturation_level": "Exclusive ceiling: full-scale fraction or float sample units.",
        "minimum_response": "Inclusive response floor: full-scale fraction or float sample units.",
        "minimum_valid_fraction": "Minimum selected fraction with usable published references.",
        "check_noise": "Reject pixels with excessive population temporal standard deviation.",
        "maximum_noise": "Maximum channel noise: full-scale fraction or float sample units.",
        "retain_noise": "Emit noise maps; Check noise computes them even when not retained.",
    }
    input_help = {
        "bright": "FlowGroup of 1–64 repeated uniform bright captures with fixed camera settings.",
        "dark": "Optional repeated dark captures. When absent, the dark offset is explicitly zero.",
        "mask": "Calibration pixel selection on the anchor grid; references are zero elsewhere.",
        "region": "Optional anchor-grid region intersected with the calibration mask.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = ["flat", "dark", "valid_mask", "info"]
        self.documentation = (
            "<p>Estimate bright and dark references from independent repeated-capture batches. "
            "Keep camera, optics, exposure, gain and lighting settings fixed; registration and "
            "camera-setting verification are operator responsibilities. Use linear intensity "
            "samples, linearizing gamma-encoded data first. Alpha does not participate "
            "in calibration and is retained only on Source. Saturation rejection is per capture "
            "pixel across all intensity channels. Mean/median and optional population temporal "
            "noise use accepted captures. Noise is about the mean, also in Median mode.</p>"
            "<p>Validity requires enough captures in both connected batches, a positive published "
            "flat-minus-dark response above the configured floor in every channel, and optional "
            "noise limits. Invalid and unselected reference pixels are zero, so correction cannot "
            "use them. Floating maps retain original sample units, with no full-scale conversion; "
            "choose Raw sample units in Flat-field correction for a different reference depth. "
            "Each batch is limited to 64 frames; combined signal samples to 64 million, image "
            "pixels to 4,194,304 and each axis to 16384. Reduction uses bounded row tiles.</p>"
        )

    def process(self, inputs):
        p = self.parameters
        groups = {}
        for name in ("bright", "dark"):
            if name not in inputs:
                continue
            group = inputs[name]
            if not isinstance(group, FlowGroup) or len(group.items) < p["minimum_frames"]:
                raise ValidationError(
                    f"{name.capitalize()} needs a capture batch meeting Minimum frames"
                )
            groups[name] = group
        bright = groups["bright"]
        index = int(p["anchor_index"])
        if index >= len(bright.items):
            raise ValidationError("Anchor index is outside the bright batch", "anchor_index")
        anchor = bright.items[index].value
        if not isinstance(anchor, np.ndarray):
            raise ValidationError("Every capture must be an image")
        raw = image_value(anchor, (np.uint8, np.uint16, np.float32, np.float64))
        encoding = image_format(anchor)
        if not raw.size or encoding not in ("gray", "rgb", "bgr", "rgba", "bgra"):
            raise ValidationError("Use nonempty grayscale or RGB(A)/BGR(A) intensity captures")
        if raw.shape[0] * raw.shape[1] > 4_194_304 or max(raw.shape[:2]) > 16384:
            raise ValidationError("Reference image exceeds 4,194,304 pixels or 16384 per axis")
        alpha = encoding in ("rgba", "bgra")
        expected = 1 if encoding == "gray" else 4 if alpha else 3
        if (1 if raw.ndim == 2 else raw.shape[2]) != expected:
            raise ValidationError("Capture encoding does not match its channel layout")
        signals, entries = {}, []
        for name, group in groups.items():
            signals[name] = []
            for entry in group.items:
                image = entry.value
                if not isinstance(image, np.ndarray):
                    raise ValidationError("Every capture must be an image")
                values = image_value(image, (np.uint8, np.uint16, np.float32, np.float64))
                if alpha and metadata(entry).get("alpha_representation", "straight") != "straight":
                    raise ValidationError("Unassociate premultiplied captures before calibration")
                if (
                    values.shape != raw.shape
                    or values.dtype != raw.dtype
                    or image_format(image) != encoding
                ):
                    raise ValidationError(
                        "Captures must share pixel dimensions, depth and channel order"
                    )
                try:
                    check_spatial(metadata(entry), [raw.shape[1], raw.shape[0]])
                except ValueError as exc:
                    raise ValidationError(str(exc)) from exc
                signals[name].append(values[..., :3] if alpha else values)
                entries.append(entry)
        if sum(image.size for images in signals.values() for image in images) > 64_000_000:
            raise ValidationError("Capture batches exceed 64 million intensity samples")
        if p["alignment"] == "same_frame":
            try:
                require_same_frame(entries)
            except ValueError as exc:
                raise ValidationError(str(exc)) from exc
        context = metadata(bright.items[index])
        selected = selection_mask({**inputs, "image": ImageArray(raw, encoding, context)}, anchor)
        selected = np.ones(raw.shape[:2], bool) if selected is None else selected != 0
        if not selected.any():
            raise ValidationError("Select at least one calibration pixel")
        scale = float(np.iinfo(raw.dtype).max) if raw.dtype.kind == "u" else 1.0
        if raw.dtype.kind == "u" and any(
            p[key] > 1 for key in ("saturation_level", "minimum_response", "maximum_noise")
        ):
            raise ValidationError("Unsigned captures require radiometric limits between 0 and 1")
        if p["reject_saturation"] and p["saturation_level"] <= 0:
            raise ValidationError("Saturation level must be positive", "saturation_level")
        ceiling = p["saturation_level"] * scale if p["reject_saturation"] else None
        need_noise = p["check_noise"] or p["retain_noise"]
        flat, bright_count, bright_noise = estimate(
            signals["bright"], p["method"], ceiling, need_noise
        )
        dark, dark_count, dark_noise = (
            estimate(signals["dark"], p["method"], ceiling, need_noise)
            if "dark" in signals
            else (
                np.zeros_like(flat),
                np.zeros(raw.shape[:2], np.uint8),
                np.zeros_like(flat) if need_noise else None,
            )
        )
        depth = raw.dtype if p["output_depth"] == "same" else np.dtype(p["output_depth"])
        limit = np.iinfo(depth).max if depth.kind == "u" else np.finfo(depth).max
        if np.max(np.abs(flat)) > limit or np.max(np.abs(dark)) > limit:
            raise ValidationError(
                "Reference estimates exceed the requested output depth", "output_depth"
            )
        if depth.kind == "u":
            flat, dark = np.rint(flat), np.rint(dark)
        flat, dark = flat.astype(depth), dark.astype(depth)
        with np.errstate(over="ignore", invalid="ignore"):
            response = flat.astype(np.float64) - dark.astype(np.float64)
        if not np.isfinite(response).all():
            raise ValidationError("Reference response exceeds float64; reduce the capture range")
        channels_valid = (response > 0) & (response >= p["minimum_response"] * scale)
        valid = channels_valid.all(axis=2) if response.ndim == 3 else channels_valid
        valid &= selected & (bright_count >= p["minimum_frames"])
        if "dark" in signals:
            valid &= dark_count >= p["minimum_frames"]
        if p["check_noise"]:
            usable = np.maximum(bright_noise, dark_noise) <= p["maximum_noise"] * scale
            valid &= usable.all(axis=2) if usable.ndim == 3 else usable
        fraction = float(valid.sum() / selected.sum())
        passed = bool(valid.any() and fraction >= p["minimum_valid_fraction"])
        for values in (flat, dark, response):
            values[~valid] = 0
        bright_count[~selected] = dark_count[~selected] = 0
        if need_noise:
            bright_noise[~selected] = dark_noise[~selected] = 0
        payload = copy.deepcopy(context)
        for key in ("ticket", "outcome", "processing_time", "alpha_representation"):
            payload.pop(key, None)
        payload.setdefault("spatial", frame([raw.shape[1], raw.shape[0]]))
        reference_info = {
            "model": "flat_field_reference_v1",
            "source_depth": raw.dtype.name,
            "output_depth": np.dtype(depth).name,
            "units": "raw_sample_units",
            "source_full_scale": scale,
            "method": p["method"],
            "alpha_omitted": alpha,
        }
        payload["flat_field_reference"] = reference_info
        info = {
            **reference_info,
            "selected_pixels": int(selected.sum()),
            "valid_pixels": int(valid.sum()),
            "valid_fraction": fraction,
            "minimum_frames": p["minimum_frames"],
            "dark_assumption": "captured" if "dark" in signals else "zero_offset",
            "noise_statistic": "population standard deviation about accepted-sample mean",
            "maximum_bright_noise": float(bright_noise[selected].max()) if need_noise else None,
            "maximum_dark_noise": float(dark_noise[selected].max()) if need_noise else None,
            "captures": {
                name: [
                    {"index": i, "label": group.labels[i], "ticket": metadata(entry).get("ticket")}
                    for i, entry in enumerate(group.items)
                ]
                for name, group in groups.items()
            },
        }
        output_format = encoding.removesuffix("a") if alpha else encoding
        outputs = {
            "flat": ImageArray(flat, output_format),
            "dark": ImageArray(dark, output_format),
            "response": ImageArray(response, output_format),
            "valid_mask": np.uint8(valid) * 255,
            "invalid_mask": np.uint8(selected & ~valid) * 255,
            "bright_count": bright_count,
            "dark_count": dark_count,
            "info": info,
            "source": anchor,
            "bright_noise": ImageArray(bright_noise, output_format)
            if p["retain_noise"]
            else NO_VALUE,
            "dark_noise": ImageArray(dark_noise, output_format) if p["retain_noise"] else NO_VALUE,
            "preview": NO_VALUE,
        }
        graphics = {}
        if self.generate_graphics:
            preview = np.rint(np.clip(response / scale, 0, 1) * 255).astype(np.uint8)
            outputs["preview"] = ImageArray(preview, output_format)
            graphics["usable flat-field response"] = outputs["preview"]
        contexts = {
            name: {
                **payload,
                "flat_field_reference": {
                    **reference_info,
                    "role": name,
                    "units": "capture_count"
                    if name.endswith("count")
                    else ("binary_validity" if name.endswith("mask") else "raw_sample_units"),
                },
            }
            for name in outputs
        }
        contexts["source"] = context
        contexts["preview"]["flat_field_reference"]["units"] = "display_uint8"
        return Result(
            outputs,
            payloads=contexts,
            graphics=graphics,
            passed=passed,
            score=fraction,
            fail_reasons=[] if passed else ["flat_field_reference_insufficient_valid_pixels"],
            message=f"Usable reference {int(valid.sum())}/{int(selected.sum())} pixels",
        )
