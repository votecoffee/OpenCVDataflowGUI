"""Conservative rejection of suspect samples on an aligned time-of-flight depth grid."""

import numpy as np

from .core import Port, Result
from .depth_smoothing import normalize_depth_confidence
from .geometry3d_common import Tool3D
from .parameter_validation import NumericRange, ValidationError
from .payloads import require_same_frame
from .selection import selection_mask
from .tickets import ticket_key

REASONS = ("unchanged", "low_confidence", "flying_pixel", "isolated_outlier", "invalid_input")
PALETTE = np.array(
    [[20, 20, 20], [0, 190, 255], [230, 80, 230], [30, 50, 255], [180, 180, 180]],
    dtype=np.uint8,
)


def _overlap(length, offset):
    return slice(max(0, -offset), min(length, length - offset))


def _outliers(values, trusted, targets, settings, units_per_meter):
    """Compare each sample to real neighbors, without padding or shifting the output."""
    radius = settings["radius"]
    tolerance = settings["agreement_m"] * units_per_meter
    separation = settings["minimum_separation_m"] * units_per_meter
    matches, nearer, farther = (np.zeros(values.shape, np.uint8) for _ in range(3))
    shallowest = np.full(values.shape, np.inf)
    deepest = np.full(values.shape, -np.inf)
    height, width = values.shape
    for dy in range(-min(radius, height - 1), min(radius, height - 1) + 1):
        for dx in range(-min(radius, width - 1), min(radius, width - 1) + 1):
            if dy == dx == 0:
                continue
            center = _overlap(height, dy), _overlap(width, dx)
            neighbor = _overlap(height, -dy), _overlap(width, -dx)
            sample = values[neighbor]
            eligible = targets[center] & trusted[neighbor]
            difference = sample - values[center]
            matches[center] += eligible & (np.abs(difference) <= tolerance)
            nearer[center] += eligible & (difference <= -separation)
            farther[center] += eligible & (difference >= separation)
            np.minimum(shallowest[center], sample, out=shallowest[center], where=eligible)
            np.maximum(deepest[center], sample, out=deepest[center], where=eligible)

    # Any agreeing neighbor protects a potential real feature. An isolated sample
    # between two supported ranges is a different case from a one-sided outlier.
    unsupported = targets & (matches == 0)
    edge_support = settings["edge_support"]
    flying = unsupported & (nearer >= edge_support) & (farther >= edge_support)
    dominant = np.maximum(nearer, farther) >= settings["outlier_support"]
    coherent = deepest - shallowest <= 2 * tolerance
    isolated = unsupported & dominant & coherent
    return flying, isolated


class CleanToFDepth(Tool3D):
    type_id = "3d.clean_tof_depth"
    title, subtype = "Clean time-of-flight depth", "Depth"
    payload_reference = "depth"
    parallel_safe = True
    inputs = (
        Port("depth", "depth"),
        Port("confidence", required=False),
        Port("mask", required=False),
        Port("region", "data", required=False),
    )
    outputs = (
        Port("depth", "depth"),
        Port("raw_depth", "depth"),
        Port("valid_mask", visual_type="region"),
        Port("rejected_mask", visual_type="region"),
        Port("reason_codes"),
        Port("report", "data"),
    )
    defaults = {
        "minimum_confidence": 0.25,
        "reject_flying_pixels": True,
        "reject_isolated_outliers": True,
        "radius": 2,
        "agreement_m": 0.025,
        "minimum_separation_m": 0.12,
        "edge_support": 2,
        "outlier_support": 6,
        "input_units": "m",
        "units_per_sample": 0.001,
    }
    parameter_choices = {"input_units": ("m", "mm")}
    parameter_ranges = {
        "minimum_confidence": NumericRange(0, 1),
        "radius": NumericRange(1, 4, whole=True),
        "agreement_m": NumericRange(0.000001, 100),
        "minimum_separation_m": NumericRange(0.000001, 1000),
        "edge_support": NumericRange(1, 40, whole=True),
        "outlier_support": NumericRange(1, 80, whole=True),
        "units_per_sample": NumericRange(1e-9, 1000),
    }
    parameter_help = {
        "minimum_confidence": "Only positive confidence at or above this value is trusted.",
        "reject_flying_pixels": "Reject samples bracketed by nearer and farther ranges.",
        "reject_isolated_outliers": "Reject isolated samples offset from a coherent local surface.",
        "radius": "Square neighborhood radius in pixels. Borders use real pixels only.",
        "agreement_m": "A trusted neighbor within this distance in meters protects the sample.",
        "minimum_separation_m": "Minimum distance to a conflicting neighbor, in meters.",
        "edge_support": "Minimum trusted neighbors on each side of a flying-pixel candidate.",
        "outlier_support": (
            "Minimum trusted neighbors on a coherent surface with a depth span of up to "
            "twice the agreement distance."
        ),
        "input_units": "Physical units for a depth array without explicit units.",
        "units_per_sample": "Physical units per uint16 count, in the chosen input units.",
    }
    input_help = {
        "confidence": "Aligned monochrome float in [0, 1] or full-scale uint8/uint16.",
        "mask": "Limit rejection to this mask. Neighbors outside it can contribute.",
        "region": "Limit rejection to this region, intersected with the mask if supplied.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not (p["reject_flying_pixels"] or p["reject_isolated_outliers"]):
            return
        if float(p["agreement_m"]) >= float(p["minimum_separation_m"]):
            raise ValidationError(
                "Minimum separation must exceed the agreement distance",
                "agreement_m",
                "minimum_separation_m",
            )
        capacity = (2 * int(p["radius"]) + 1) ** 2 - 1
        if p["reject_flying_pixels"] and 2 * int(p["edge_support"]) > capacity:
            raise ValidationError("Too few neighbors for edge support", "radius", "edge_support")
        if p["reject_isolated_outliers"] and int(p["outlier_support"]) > capacity:
            raise ValidationError(
                "Too few neighbors for outlier support", "radius", "outlier_support"
            )

    def _confidence_trust(self, source, confidence):
        if confidence.shape != source.shape:
            raise ValidationError("Confidence must be monochrome and match the depth dimensions")
        require_same_frame(
            (source, confidence), message="Confidence and depth use different coordinate frames"
        )
        depth_payload = self.input_payloads["depth"]
        confidence_payload = self.input_payloads["confidence"]
        for key in ("ticket", "recorded_rgbd", "optical_frame"):
            depth_value, confidence_value = depth_payload.get(key), confidence_payload.get(key)
            if depth_value and confidence_value:
                if key == "ticket":
                    depth_value, confidence_value = (
                        ticket_key(depth_value),
                        ticket_key(confidence_value),
                    )
                if depth_value != confidence_value:
                    raise ValidationError(f"Confidence and depth must refer to the same {key}")
        weights = normalize_depth_confidence(confidence)
        return (weights > 0) & (weights >= float(self.parameters["minimum_confidence"]))

    def process(self, inputs):
        source = inputs["depth"]
        if not 0 < source.size <= 4_194_304:
            raise ValidationError("Depth cleanup requires 1 to 4,194,304 pixels")
        p = self.parameters
        raw = np.asarray(source)
        units = getattr(source, "units", p["input_units"])
        if units not in ("m", "mm"):
            raise ValidationError("Depth units must be m or mm", "input_units")
        integer = raw.dtype == np.uint16
        meters_per_sample = (0.001 if units == "mm" else 1) * (
            float(p["units_per_sample"]) if integer else 1
        )
        selection = selection_mask(inputs, source)
        selected = np.ones(raw.shape, bool) if selection is None else selection != 0
        valid = np.isfinite(raw) & (raw > 0)
        candidates = selected & valid
        trusted = valid.copy()
        confidence = inputs.get("confidence")
        if confidence is not None:
            trusted &= self._confidence_trust(source, confidence)

        codes = np.zeros(raw.shape, np.uint8)
        codes[selected & ~valid] = 4
        codes[candidates & ~trusted] = 1
        if (
            candidates.any()
            and trusted.any()
            and (p["reject_flying_pixels"] or p["reject_isolated_outliers"])
        ):
            settings = {
                name: convert(p[name])
                for name, convert in (
                    ("radius", int),
                    ("edge_support", int),
                    ("outlier_support", int),
                    ("agreement_m", float),
                    ("minimum_separation_m", float),
                )
            }
            # Zero invalid neighbors before subtraction to avoid inf/NaN arithmetic.
            values = np.where(valid, raw, 0).astype(np.float64)
            flying, isolated = _outliers(
                values, trusted, candidates, settings, 1 / meters_per_sample
            )
            if p["reject_isolated_outliers"]:
                codes[isolated] = 3
            if p["reject_flying_pixels"]:
                codes[flying] = 2

        rejected = (codes > 0) & (codes < 4)
        cleaned = source.copy()
        cleaned[rejected] = 0 if integer else np.nan
        keep = valid & ~rejected
        selected_count = int(selected.sum())
        retained_count = int(np.count_nonzero(selected & keep))
        counts = np.bincount(codes.ravel(), minlength=len(REASONS))
        report = {
            "selected_samples": selected_count,
            "retained_samples": retained_count,
            "confidence_supplied": confidence is not None,
            "reason_counts": {name: int(counts[i]) for i, name in enumerate(REASONS) if i},
            "reason_codes": dict(enumerate(REASONS)),
            "units": units,
            "meters_per_sample": meters_per_sample,
        }
        return Result(
            {
                "depth": cleaned,
                "raw_depth": source,
                "valid_mask": keep.astype(np.uint8) * 255,
                "rejected_mask": rejected.astype(np.uint8) * 255,
                "reason_codes": codes,
                "report": report,
            },
            graphics={"Rejection reasons": PALETTE[codes]} if self.generate_graphics else {},
            passed=bool(retained_count),
            score=retained_count / selected_count if selected_count else 0.0,
            fail_reasons=[] if retained_count else ["no_trusted_depth_in_selection"],
            message=f"Kept {retained_count} of {selected_count} selected depth samples",
        )
