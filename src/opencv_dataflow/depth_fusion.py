"""Fuse registered camera-Z images without averaging across depth discontinuities."""

import cv2
import numpy as np

from .cloud_projection import MAX_PIXELS
from .core import Port, Result
from .depth_smoothing import normalize_depth_confidence
from .geometry3d_common import Tool3D
from .images import DepthArray, ImageArray
from .parameter_validation import NumericRange, ValidationError
from .payloads import require_same_frame
from .selection import selection_mask
from .stereo3d import metric_depth
from .tickets import ticket_key

_VIEW_NAMES = ("a", "b", "c", "d")
_BAND_ROWS = 128


def _consensus(eligible, tolerance):
    # Each candidate is the subset in a closed interval starting at a measured Z.
    shape = eligible[0][1].shape
    best_count = np.zeros(shape, np.uint8)
    best_views = np.zeros(shape, np.uint8)
    ambiguous = np.zeros(shape, bool)
    for _, lower, _, lower_active in eligible:
        upper = lower.astype(np.float64) + tolerance
        count = np.zeros(shape, np.uint8)
        members = np.zeros(shape, np.uint8)
        for view, z, _, active in eligible:
            within = lower_active & active & (z >= lower) & (z <= upper)
            count += within
            members |= within.astype(np.uint8) * _view_bit(view)
        more = count > best_count
        tied = (count == best_count) & (members != best_views)
        ambiguous = (ambiguous | tied) & ~more
        best_count[more] = count[more]
        best_views[more] = members[more]
    best_views[ambiguous] = 0
    return best_views, ambiguous


def _view_bit(name):
    return 1 << _VIEW_NAMES.index(name)


def _fuse_band(views, selection, index, p, counts):
    consensus = p["surface_policy"] == "consensus"
    nearest = None if consensus else np.full(selection.shape, np.inf, dtype=np.float32)
    minimum = float(p["minimum_confidence"])
    tolerance = float(p["surface_tolerance_m"])
    eligible = []
    for name, depth, confidence in views:
        z = metric_depth(depth[index], p["units_per_sample"], p["input_units"])
        w = normalize_depth_confidence(confidence[index]) if confidence is not None else None
        active = np.isfinite(z) & selection
        if w is not None:
            active &= (w > 0) & (w >= minimum)
        z[~active] = np.inf
        if nearest is not None:
            np.minimum(nearest, z, out=nearest)
        counts[name]["eligible_samples"] += int(np.count_nonzero(active))
        eligible.append((name, z, w, active))

    if consensus:
        members, ambiguous = _consensus(eligible, tolerance)
    else:
        ambiguous = np.zeros(selection.shape, bool)
        bound = nearest.astype(np.float64) + tolerance
    rejected_views = np.zeros(selection.shape, np.uint8)
    total = np.zeros(selection.shape, np.float64)
    weight = np.zeros_like(total)
    support = np.zeros(selection.shape, np.uint8)
    conflict = np.zeros(selection.shape, bool)
    for name, z, w, active in eligible:
        close = (members & _view_bit(name)) != 0 if consensus else active & (z <= bound)
        samples = z[close].astype(np.float64)
        accepted = 1 if w is None else w[close]
        total[close] += samples * accepted
        weight[close] += accepted
        support[close] += 1
        counts[name]["surface_samples"] += len(samples)
        rejected = active & ~close
        conflict |= rejected
        counts[name]["ambiguous_samples"] += int(np.count_nonzero(active & ambiguous))
        rejected_views |= (rejected & ~ambiguous).astype(np.uint8) * _view_bit(name)
    return total, weight, support, conflict, ambiguous, rejected_views


def _validate_views(block, inputs):
    views = []
    for name in _VIEW_NAMES:
        depth, confidence = (inputs.get(f"{prefix}_{name}") for prefix in ("depth", "confidence"))
        if depth is None:
            if confidence is not None:
                raise ValidationError(f"Confidence {name.upper()} needs its matching depth")
            continue
        views.append((name, depth, confidence))
    primary = views[0][1]
    if primary.size > MAX_PIXELS:
        raise ValidationError("Depth grid exceeds the 8,388,608 pixel fusion limit")
    if any(depth.shape != primary.shape for _, depth, _ in views):
        raise ValidationError("Registered depth images must have matching dimensions")
    if any(w is not None and w.shape != depth.shape for _, depth, w in views):
        raise ValidationError("Confidence must be monochrome with matching depth dimensions")
    if block.parameters["alignment"] == "same_frame":
        require_same_frame(
            [value for name, value in inputs.items() if name != "region"],
            message="Depth fusion requires one common registered pixel frame",
        )
    tickets = {
        ticket_key(payload["ticket"])
        for payload in block.input_payloads.values()
        if "ticket" in payload
    }
    if len(tickets) > 1:
        raise ValidationError("Depth fusion inputs must refer to the same ticket")
    return views


class FuseRegisteredDepth(Tool3D):
    type_id, title, subtype = "3d.fuse_registered_depth", "Fuse registered depth views", "Depth"
    inputs = (
        *(
            Port(
                f"depth_{name}", "depth", required=name in ("a", "b"), required_when_connected=True
            )
            for name in _VIEW_NAMES
        ),
        *(
            Port(f"confidence_{name}", required=False, required_when_connected=True)
            for name in _VIEW_NAMES
        ),
        Port("mask", required=False, visual_type="region", required_when_connected=True),
        Port("region", "data", required=False, visual_type="region", required_when_connected=True),
    )
    outputs = (
        Port("depth", "depth"),
        Port("valid_mask", visual_type="region"),
        Port("confidence"),
        Port("support_count"),
        Port("conflict_mask", visual_type="region"),
        Port("ambiguous_mask", visual_type="region"),
        Port("rejected_views"),
        Port("report", "data"),
    )
    defaults = {
        "alignment": "same_frame",
        "input_units": "m",
        "units_per_sample": 0.001,
        "minimum_confidence": 0.1,
        "surface_policy": "nearest",
        "surface_tolerance_m": 0.02,
        "minimum_support": 1,
        "minimum_coverage": 0.0,
    }
    parameter_choices = {
        "alignment": ("same_frame", "pixel_grid"),
        "input_units": ("m", "mm"),
        "surface_policy": ("nearest", "consensus"),
    }
    parameter_ranges = {
        "units_per_sample": NumericRange(1e-9, 1000),
        "minimum_confidence": NumericRange(0, 1),
        "surface_tolerance_m": NumericRange(0, 100),
        "minimum_support": NumericRange(1, len(_VIEW_NAMES), whole=True),
        "minimum_coverage": NumericRange(0, 1),
    }
    parameter_help = {
        "alignment": "Check known frames, or assert independently loaded grids are registered.",
        "input_units": "Physical units for an untagged depth raster.",
        "units_per_sample": "Physical units per raw uint16 sample.",
        "minimum_confidence": "Discard lower confidence. Zero confidence is always invalid.",
        "surface_policy": "Use the nearest surface, or choose the largest agreeing group.",
        "surface_tolerance_m": "Maximum depth range of one agreeing surface group.",
        "minimum_support": "Require this many views in the chosen surface group.",
        "minimum_coverage": "Required fused fraction of selected pixels.",
    }
    input_help = {
        **{
            f"depth_{name}": "Positive axial Z registered to Depth A's pixel grid."
            for name in _VIEW_NAMES
        },
        **{
            f"confidence_{name}": "Float [0,1] or uint8/uint16 full scale. Omit for weight 1."
            for name in _VIEW_NAMES
        },
        "mask": "Optional uint8 selection on the common grid.",
        "region": "Optional region intersected with Mask. Outputs outside are invalid.",
    }
    payload_reference = "depth_a"
    parallel_safe = preview_safe = process_inputs_readonly = True

    def process(self, inputs):
        p = self.parameters
        views = _validate_views(self, inputs)
        minimum_support = int(p["minimum_support"])
        if minimum_support > len(views):
            raise ValidationError(
                "Minimum support exceeds connected depth views", "minimum_support"
            )
        primary = inputs["depth_a"]
        selection = selection_mask(inputs, primary, match_mask_frame=p["alignment"] == "same_frame")
        selected = np.ones(primary.shape, bool) if selection is None else selection != 0
        fused = np.full(primary.shape, np.nan, np.float32)
        confidence = np.zeros(primary.shape, np.float32)
        support_count = np.zeros(primary.shape, np.uint8)
        conflicts = np.zeros(primary.shape, bool)
        ambiguous = np.zeros(primary.shape, bool)
        rejected_views = np.zeros(primary.shape, np.uint8)
        counts = {
            name: {
                "view": name,
                "eligible_samples": 0,
                "surface_samples": 0,
                "ambiguous_samples": 0,
            }
            for name, _, _ in views
        }
        for row in range(0, primary.shape[0], _BAND_ROWS):
            index = slice(row, row + _BAND_ROWS)
            total, weight, support, conflict, ties, rejects = _fuse_band(
                views, selected[index], index, p, counts
            )
            usable = support >= minimum_support
            np.divide(total, weight, where=usable, out=fused[index])
            np.divide(weight, support, where=usable, out=confidence[index])
            support_count[index] = support
            conflicts[index] = conflict
            ambiguous[index] = ties
            rejected_views[index] = rejects

        valid = np.isfinite(fused)
        selected_count = int(np.count_nonzero(selected))
        fused_count = int(np.count_nonzero(valid))
        ambiguous_count = int(np.count_nonzero(ambiguous))
        positive_support = int(np.count_nonzero(support_count)) + ambiguous_count
        coverage = fused_count / selected_count if selected_count else 0.0
        valid_mask = valid.astype(np.uint8) * 255
        conflict_mask = conflicts.astype(np.uint8) * 255
        reasons = []
        if not fused_count:
            reasons.append("depth_fusion_no_supported_pixels")
        elif coverage < float(p["minimum_coverage"]):
            reasons.append("depth_fusion_coverage_below_minimum")
        report = {
            "model": "registered_depth_fusion_v1",
            "units": "m",
            "surface_policy": p["surface_policy"],
            "ambiguous_pixels": ambiguous_count,
            "rejected_view_bits": {name: _view_bit(name) for name, _, _ in views},
            "selected_pixels": selected_count,
            "fused_pixels": fused_count,
            "coverage": coverage,
            "conflicting_pixels": int(np.count_nonzero(conflicts)),
            "no_eligible_depth_pixels": selected_count - positive_support,
            "insufficient_support_pixels": positive_support - fused_count,
            "views": [
                {
                    **row,
                    "rejected_samples": (
                        row["eligible_samples"] - row["surface_samples"] - row["ambiguous_samples"]
                    ),
                    **(
                        {"farther_samples": row["eligible_samples"] - row["surface_samples"]}
                        if p["surface_policy"] == "nearest"
                        else {}
                    ),
                }
                for row in counts.values()
            ],
        }
        graphics = {}
        if self.generate_graphics:
            # Green shows output support and red marks pixels with incompatible surfaces.
            graphics["Fusion support"] = cv2.merge(
                (ambiguous.astype(np.uint8) * 255, valid_mask, conflict_mask)
            )
        return Result(
            {
                "depth": DepthArray(fused, "m"),
                "valid_mask": ImageArray(valid_mask, "gray"),
                "confidence": ImageArray(confidence, "gray"),
                "support_count": ImageArray(support_count, "gray"),
                "conflict_mask": ImageArray(conflict_mask, "gray"),
                "ambiguous_mask": ImageArray(ambiguous.astype(np.uint8) * 255, "gray"),
                "rejected_views": ImageArray(rejected_views, "gray"),
                "report": report,
            },
            graphics=graphics,
            passed=not reasons,
            score=coverage,
            message=f"{fused_count}/{selected_count} selected pixels fused",
            fail_reasons=reasons,
        )
