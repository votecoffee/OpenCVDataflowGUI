"""Conservative, response-aware HDR accumulation with explicit measurement support."""

import copy

import cv2
import numpy as np

from .core import Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .hdr import _DEBEVEC_RESPONSE, _ROBERTSON_RESPONSE, _TimedExposureStack, response_table
from .images import ImageArray
from .parameter_validation import NumericRange, ValidationError
from .payloads import (
    NO_VALUE,
    FlowGroup,
    ValuePacket,
    image_with_payload,
    metadata,
    require_same_frame,
)
from .selection import selection_mask


def mask_group(value, exposures, shape):
    if value is not None and (
        not isinstance(value, FlowGroup)
        or value.labels != exposures.labels
        or len(value.items) != len(exposures.items)
    ):
        raise ValidationError("Masks must be a batch in the same exposure label/order")
    masks = []
    for index, source in enumerate(exposures.items):
        mask = None
        support = metadata(source).get("exposure_support")
        if support is not None:
            if not isinstance(support, dict) or support.get("model") != "opencv_mtb_support_v1":
                raise ValidationError("Invalid exposure alignment support")
            fields = [support.get(name) for name in ("source_size", "shift", "output_xywh")]
            if any(
                not isinstance(v, list) or len(v) != n or any(type(i) is not int for i in v)
                for v, n in zip(fields, (2, 2, 4), strict=True)
            ):
                raise ValidationError("Invalid exposure support grid fields")
            (w, h), (dx, dy), (left, top, ww, hh) = fields
            if (
                min(w, h) < 2
                or max(w, h) > 16384
                or [ww, hh] != list(shape[::-1])
                or support.get("spatial") != metadata(source).get("spatial")
                or min(left, top) < 0
                or left + ww > w
                or top + hh > h
                or max(abs(dx), abs(dy)) > 16384
            ):
                raise ValidationError("Exposure support must retain its alignment grid")
            x0, y0 = max(0, dx - left), max(0, dy - top)
            x1, y1 = min(ww, w + dx - left), min(hh, h + dy - top)
            mask = np.zeros(shape, bool)
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = True
        if value is not None:
            packet = value.items[index]
            supplied = packet.value
            if (
                not isinstance(supplied, np.ndarray)
                or supplied.dtype != np.uint8
                or supplied.shape != shape
            ):
                raise ValidationError("Each exposure mask must be uint8 and match the image grid")
            require_same_frame(
                (source, packet), message="Exposure and mask coordinate frames differ"
            )
            mask = supplied != 0 if mask is None else mask & (supplied != 0)
        masks.append(mask)
    return masks


def rejected_boundaries(mask, maximum):
    if maximum < 2:
        return []
    h, w = mask.shape
    scale = min(1.0, (262144 / mask.size) ** 0.5)
    size = max(1, round(w * scale)), max(1, round(h * scale))
    preview = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST) if scale < 1 else mask
    contours, _ = cv2.findContours(preview, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    objects, left = [], maximum
    for contour in contours:
        points = contour.reshape(-1, 2)
        if len(points) < 2 or left < 2:
            continue
        points = points[np.linspace(0, len(points) - 1, min(left, len(points)), dtype=int)]
        points = (points + 0.5) * [w / size[0], h / size[1]] - 0.5
        objects.append(graphic_object("polyline", points, (60, 90, 240), closed=True))
        left -= len(points)
    return objects


class MergeMotionHDR(_TimedExposureStack):
    type_id, title = "cv.merge_hdr_motion", "Merge HDR with motion rejection"
    inputs = (
        *_TimedExposureStack.inputs,
        Port("response", "data", required=False, required_when_connected=True),
        Port("masks", "any", required=False, required_when_connected=True),
        Port("mask", required=False, visual_type="region"),
        Port("region", "data", required=False, visual_type="region"),
    )
    outputs = (
        Port("image"),
        Port("display"),
        Port("valid_mask", visual_type="region"),
        Port("motion_mask", visual_type="region"),
        Port("fallback_mask", visual_type="region"),
        Port("contributors"),
        Port("support_masks", "any", required=False),
        Port("report", "data"),
    )
    defaults = {
        **_TimedExposureStack.defaults,
        "absolute_tolerance": 0.02,
        "relative_tolerance": 0.05,
        "clip_low": 4,
        "clip_high": 251,
        "clipping_evidence": "bounds",
        "minimum_contributors": 2,
        "minimum_coverage": 0.8,
        "weighting": "triangle",
        "fallback": "reference",
        "grid_policy": "matching_frames",
        "emit_support_masks": False,
        "maximum_pixels": 4_194_304,
        "maximum_working_bytes": 768 * 1024**2,
        "maximum_graphics_points": 2048,
    }
    parameter_choices = {
        "weighting": ("triangle", "uniform"),
        "fallback": ("reference", "zero"),
        "grid_policy": ("matching_frames", "assert_pixel_alignment"),
        "clipping_evidence": ("bounds", "ignore"),
    }
    parameter_ranges = {
        **_TimedExposureStack.parameter_ranges,
        "absolute_tolerance": NumericRange(0),
        "relative_tolerance": NumericRange(0, 1),
        "clip_low": NumericRange(0, 253, whole=True),
        "clip_high": NumericRange(2, 255, whole=True),
        "minimum_contributors": NumericRange(2, 16, whole=True),
        "minimum_coverage": NumericRange(0, 1),
        "maximum_pixels": NumericRange(4, 16_777_216, whole=True),
        "maximum_working_bytes": NumericRange(1, 2 * 1024**3, whole=True),
        "maximum_graphics_points": NumericRange(0, 100000, whole=True),
    }
    parameter_help = {
        **_TimedExposureStack.parameter_help,
        "absolute_tolerance": "Allowed radiance span in response units per second.",
        "relative_tolerance": "Additional span allowance times the largest channel radiance.",
        "clip_low": "Stored uint8 codes at or below this level are not usable.",
        "clip_high": "Stored uint8 codes at or above this level are not usable.",
        "clipping_evidence": "Use clipped-code bounds to detect contradictions, or ignore them.",
        "minimum_contributors": "Required unclipped agreeing exposures per measured pixel.",
        "minimum_coverage": "Minimum measured fraction of the selected reference domain.",
        "weighting": "Uniform mean or triangular distance from the clipping boundaries.",
        "fallback": "Display only: fill unsupported pixels from the reference, or use zero.",
        "grid_policy": "Require known matching grids, or explicitly assert aligned pixel indices.",
        "emit_support_masks": "Optional accepted masks, retaining member labels and tickets.",
        "maximum_pixels": "Reference-grid pixel limit before radiance allocation.",
        "maximum_working_bytes": "Estimated arrays; excludes runtime copies and native workspaces.",
        "maximum_graphics_points": "Bound optional decimated rejected-region drawing points.",
    }
    input_help = {
        **_TimedExposureStack.input_help,
        "response": "Optional existing Debevec/Robertson response; otherwise assume z/128 linear.",
        "masks": "Optional uint8 mask batch with the same labels/order and per-member image grids.",
        "mask": "Optional reference-domain uint8 selection, intersected with Region.",
        "region": "Optional reference-domain region; does not change exposure registration.",
    }
    output_help = {
        "image": "Measured float32 radiance only; zero outside Valid mask.",
        "display": "Review radiance with optional reference fills; do not use as measured HDR.",
        "valid_mask": "Pixels with enough unclipped, photometrically consistent exposures.",
        "motion_mask": "Observed radiance disagreement proxy; not a physical motion measurement.",
        "fallback_mask": "Display pixels supplied from the reference instead of measured HDR.",
        "contributors": "Accepted exposure count per measured pixel; zero otherwise.",
        "support_masks": "Optional per-member masks for accepted radiance contributions.",
        "report": "Coverage, rejected support, clipping, source provenance and policy.",
    }
    preview_safe = parallel_safe = process_inputs_readonly = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = ["image", "display", "valid_mask", "report"]
        self.description = self.description or "Reject inconsistent HDR pixels before measurement."
        self.documentation = self.documentation or (
            "<p>Response-aware weighted radiance with conservative exposure consistency. "
            "Image is zero outside measured support. Display may fill those pixels from "
            "the reference, marked separately. Align exposures first. This is a photometric "
            "motion proxy, not native Debevec/Robertson merging or radiometric certification. "
            "See docs/hdr-motion.md.</p>"
        )

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for name, limit in self.parameter_ranges.items():
            if type(p[name]) not in (int, float) or (limit.whole and type(p[name]) is not int):
                raise ValidationError("Use typed numeric HDR settings", name)
        if type(p["emit_support_masks"]) is not bool:
            raise ValidationError("Emit support masks must be boolean", "emit_support_masks")
        if p["clip_high"] - p["clip_low"] < 2:
            raise ValidationError(
                "Clipping limits must leave usable codes", "clip_low", "clip_high"
            )

    def process(self, inputs):
        p, group = self.parameters, inputs["exposures"]
        if not isinstance(group, FlowGroup) or not 2 <= len(group.items) <= 16:
            raise ValidationError("Supply 2–16 ordered exposure images")
        shape = getattr(group.items[0].value, "shape", ())
        if len(shape) not in (2, 3) or min(shape[:2]) < 2 or max(shape[:2]) > 16384:
            raise ValidationError("Use nonempty 2D exposure images with axes at most 16384")
        pixels, count = int(np.prod(shape[:2])), len(group.items)
        channels = shape[2] if len(shape) == 3 else 1
        estimate = pixels * (channels * 80 + count * 2 + 64)
        if pixels > p["maximum_pixels"] or estimate > p["maximum_working_bytes"]:
            raise ValidationError(
                "HDR grid exceeds pixel/working budget", "maximum_pixels", "maximum_working_bytes"
            )
        frames, times, encoding, context = self.timed_stack(inputs)
        reference = int(p["reference_index"])
        if p["minimum_contributors"] > count:
            raise ValidationError(
                "Minimum contributors exceeds exposure count", "minimum_contributors"
            )
        if p["grid_policy"] == "matching_frames":
            require_same_frame(
                group.items, message="Align exposure grids or assert pixel alignment"
            )
        response = inputs.get("response")
        model = (
            _ROBERTSON_RESPONSE
            if response is None
            else response.get("model")
            if isinstance(response, dict)
            else None
        )
        if model not in (_DEBEVEC_RESPONSE, _ROBERTSON_RESPONSE):
            raise ValidationError("Connect an existing Debevec or Robertson response")
        table = response_table(response, encoding, model)
        own_masks = mask_group(inputs.get("masks"), group, shape[:2])
        source = image_with_payload(group.items[reference].value, metadata(group.items[reference]))
        domain = selection_mask(inputs, source)
        domain = np.ones(shape[:2], bool) if domain is None else domain != 0
        lower = np.full((*shape[:2], channels), -np.inf)
        upper = np.full_like(lower, np.inf)
        weighted = np.zeros_like(lower)
        mass = np.zeros(shape[:2], np.float64)
        observed = np.zeros(shape[:2], np.uint8)
        evidence_count = np.zeros(shape[:2], np.uint8)
        low_ceiling = (
            table[: p["clip_low"] + 1].reshape(-1, channels).max(axis=0).astype(np.float64)
        )
        high_floor = table[p["clip_high"] :].reshape(-1, channels).min(axis=0).astype(np.float64)
        eligible_masks, sources = [], []
        reference_radiance = None
        for index, (entry, frame, seconds, mask) in enumerate(
            zip(group.items, frames, times, own_masks, strict=True)
        ):
            minimum = frame.min(axis=2) if frame.ndim == 3 else frame
            maximum = frame.max(axis=2) if frame.ndim == 3 else frame
            unclipped = (minimum > p["clip_low"]) & (maximum < p["clip_high"])
            selected = domain if mask is None else domain & mask
            lookup = cv2.LUT(frame, table)
            radiance = lookup.astype(np.float64) / float(seconds)
            if radiance.ndim == 2:
                radiance = radiance[..., None]
            if not np.isfinite(radiance).all() or np.any(radiance < 0):
                raise ValidationError("Camera response/time produced invalid radiance")
            positive = np.all(radiance > 0, axis=2)
            eligible = unclipped & selected & positive
            eligible_masks.append(eligible)
            if index == reference:
                reference_radiance = radiance.copy()
            if p["clipping_evidence"] == "bounds":
                raw = frame[..., None] if frame.ndim == 2 else frame
                bound_low = np.where(
                    raw <= p["clip_low"],
                    0,
                    np.where(raw >= p["clip_high"], high_floor / float(seconds), radiance),
                )
                bound_high = np.where(
                    raw >= p["clip_high"],
                    np.inf,
                    np.where(raw <= p["clip_low"], low_ceiling / float(seconds), radiance),
                )
                constrained = selected
            else:
                bound_low = bound_high = radiance
                constrained = eligible
            lower = np.maximum(lower, np.where(constrained[..., None], bound_low, -np.inf))
            upper = np.minimum(upper, np.where(constrained[..., None], bound_high, np.inf))
            evidence_count += constrained
            weight = (
                np.ones(shape[:2])
                if p["weighting"] == "uniform"
                else np.minimum(
                    minimum.astype(np.float64) - p["clip_low"],
                    p["clip_high"] - maximum.astype(np.float64),
                )
            )
            weight = np.where(eligible, weight, 0)
            weighted += radiance * weight[..., None]
            mass += weight
            observed += eligible
            original = metadata(entry)
            sources.append(
                {
                    "label": group.labels[index],
                    "ticket": original.get("ticket"),
                    "spatial": original.get("spatial"),
                    "alignment_support": original.get("exposure_support"),
                    "time_seconds": float(seconds),
                    "clipped_pixels": int((domain & ~unclipped).sum()),
                    "masked_pixels": int((domain & unclipped & ~selected).sum()),
                    "zero_response_pixels": int((selected & unclipped & ~positive).sum()),
                    "eligible_pixels": int(eligible.sum()),
                }
            )
        finite = np.isfinite(lower) & np.isfinite(upper)
        lower, upper = np.where(finite, lower, 0), np.where(finite, upper, 0)
        span = lower - upper
        tolerance = p["absolute_tolerance"] + p["relative_tolerance"] * np.maximum(
            abs(lower), abs(upper)
        )
        motion = (evidence_count >= 2) & np.any(span > tolerance, axis=2)
        valid = domain & ~motion & (observed >= p["minimum_contributors"])
        measured = np.divide(
            weighted, mass[..., None], out=np.zeros_like(weighted), where=valid[..., None]
        )
        display = (
            np.where(valid[..., None], measured, reference_radiance)
            if p["fallback"] == "reference"
            else measured.copy()
        )
        if max(float(measured.max()), float(display.max())) > np.finfo(np.float32).max:
            raise ValidationError("Radiance exceeds float32 output range")
        fallback = ~valid if p["fallback"] == "reference" else np.zeros(shape[:2], bool)
        packets = []
        for index, eligible in enumerate(eligible_masks):
            used = eligible & valid
            sources[index].update(
                used_pixels=int(used.sum()),
                rejected_inconsistent_pixels=int((eligible & motion).sum()),
            )
            if p["emit_support_masks"]:
                packets.append(
                    ValuePacket(
                        np.where(used, 255, 0).astype(np.uint8),
                        copy.deepcopy(metadata(group.items[index])),
                    )
                )
        selected_count, valid_count = int(domain.sum()), int(valid.sum())
        coverage = valid_count / selected_count if selected_count else 0.0
        passed = valid_count > 0 and coverage >= p["minimum_coverage"]
        report = {
            "model": "motion_hdr_v1",
            "algorithm": "supported_weighted_linear_radiance",
            "reference_index": reference,
            "reference_label": group.labels[reference],
            "times_seconds": times.tolist(),
            "source_count": count,
            "sources": sources,
            "response": "assumed_linear_z_over_128" if response is None else model,
            "radiance_units": "response units per second; not absolute radiometry",
            "grid_policy": p["grid_policy"],
            "weighting": p["weighting"],
            "absolute_tolerance": p["absolute_tolerance"],
            "relative_tolerance": p["relative_tolerance"],
            "clip_low": p["clip_low"],
            "clip_high": p["clip_high"],
            "clipping_evidence": p["clipping_evidence"],
            "minimum_contributors": p["minimum_contributors"],
            "minimum_coverage": p["minimum_coverage"],
            "selected_pixels": selected_count,
            "measured_pixels": valid_count,
            "coverage": coverage,
            "inconsistent_pixels": int(motion.sum()),
            "insufficient_pixels": int((domain & (observed < p["minimum_contributors"])).sum()),
            "display_filled_pixels": int(fallback.sum()),
            "estimated_working_bytes": estimate,
            "motion_is_photometric_proxy": True,
            "reconstructed_pixels_are_measured": False,
        }
        context = copy.deepcopy(context)
        context["hdr_support"] = {
            "model": "motion_hdr_v1",
            "requires_valid_mask": True,
            "reference_index": reference,
            "sources": sources,
        }
        display_context = copy.deepcopy(context)
        display_context["hdr_support"]["review_only"] = True
        display_context["hdr_support"]["fallback"] = p["fallback"]
        names = ("image", "valid_mask", "motion_mask", "fallback_mask", "contributors", "report")
        payloads = {name: copy.deepcopy(context) for name in names}
        payloads["display"] = display_context
        if packets:
            payloads["support_masks"] = copy.deepcopy(context)
        if channels == 1:
            measured, display = measured[..., 0], display[..., 0]
        objects = []
        if self.generate_graphics and p["maximum_graphics_points"]:
            objects = rejected_boundaries(
                np.where(motion, 255, 0).astype(np.uint8), p["maximum_graphics_points"] - 1
            )
            objects.append(
                graphic_object(
                    "text",
                    [[8, 18]],
                    (70, 220, 140) if passed else (60, 90, 240),
                    text=f"Measured {coverage:.1%}; inconsistent {motion.sum()}",
                )
            )
        return Result(
            {
                "image": ImageArray(measured.astype(np.float32), encoding),
                "display": ImageArray(display.astype(np.float32), encoding),
                "valid_mask": np.where(valid, 255, 0).astype(np.uint8),
                "motion_mask": np.where(motion, 255, 0).astype(np.uint8),
                "fallback_mask": np.where(fallback, 255, 0).astype(np.uint8),
                "contributors": np.where(valid, observed, 0).astype(np.uint8),
                "support_masks": FlowGroup(tuple(packets), group.labels) if packets else NO_VALUE,
                "report": report,
            },
            payloads=payloads,
            passed=passed,
            score=coverage,
            fail_reasons=[] if passed else ["hdr_motion.insufficient_measured_support"],
            drawings={
                "HDR rejected support": GraphicDrawing(tuple(objects), "display", from_output=True)
            }
            if objects
            else {},
            message=f"{valid_count}/{selected_count} pixels measured; {motion.sum()} inconsistent",
        )
