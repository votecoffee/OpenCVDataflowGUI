"""Native spatial phase unwrapping with independent reference cycles per component."""

import math
from collections import defaultdict

import cv2
import numpy as np

from .core import Port, Result
from .geometry3d_common import Tool3D
from .images import DepthArray, ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .selection import selection_mask
from .tickets import ticket_key

_MAX_PIXELS = 1_048_576
_MAX_COMPONENTS = 512
_MAX_REFERENCE_RAD = math.tau * (2**31 - 1)
_MAX_EDGE_RELIABILITY = 32 * math.pi**2
_PI32 = np.float32(math.pi)
_HALF_TURN_MARGIN = 8 * np.spacing(_PI32)


def _anchors(value):
    if not isinstance(value, (list, tuple)) or len(value) > _MAX_COMPONENTS:
        raise ValidationError("Use a list of at most 512 phase references", "references")
    refs = {}
    for record in value:
        if not isinstance(record, dict):
            raise ValidationError("Each reference needs a pixel and phase_rad", "references")
        pixel = record.get("pixel")
        phase = record.get("phase_rad")
        if (
            not isinstance(pixel, (list, tuple))
            or len(pixel) != 2
            or any(type(index) is not int or index < 0 for index in pixel)
            or type(phase) not in (int, float)
            or not -_MAX_REFERENCE_RAD <= phase <= _MAX_REFERENCE_RAD
        ):
            raise ValidationError(
                "Use whole [x,y] pixels and finite phase_rad in the supported range", "references"
            )
        key = tuple(pixel)
        if key in refs:
            raise ValidationError("Phase reference pixels must be unique", "references")
        refs[key] = float(phase)
    return refs


def _histogram_settings(p):
    threshold = np.float32(p["histogram_threshold"])
    bins = int(p["large_bins"])
    width = np.float32(_MAX_EDGE_RELIABILITY - float(threshold)) / np.float32(bins)
    # Match the float precision used to assign OpenCV's maximum-reliability-distance edge.
    largest = (np.float32(_MAX_EDGE_RELIABILITY) - threshold) / width
    if largest > bins:
        raise ValidationError(
            "Choose another histogram threshold or large bin count for stable native rounding",
            "histogram_threshold",
            "large_bins",
        )
    return float(threshold)


def _validate_phase_jumps(phase, valid):
    """Reject half-turn neighbor differences whose direction is numerically ambiguous."""
    horizontal = (np.s_[:, :-1], np.s_[:, 1:])
    vertical = (np.s_[:-1, :], np.s_[1:, :])
    diagonals = (
        (np.s_[:-1, :-1], np.s_[1:, 1:]),
        (np.s_[:-1, 1:], np.s_[1:, :-1]),
    )
    interior = (
        cv2.erode(
            valid.astype(np.uint8),
            np.ones((3, 3), np.uint8),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        != 0
    )
    for a, b in (horizontal, vertical, *diagonals):
        support = valid[a] & valid[b]
        if (a, b) in diagonals:
            support &= interior[a] | interior[b]
        delta = np.abs(phase[a] - phase[b])
        if np.any(support & (np.abs(delta - _PI32) <= _HALF_TURN_MARGIN)):
            raise ValidationError(
                "Mask adjacent phase samples with ambiguous half-turn differences"
            )


def _native_unwrap(phase, mask, p):
    namespace = getattr(cv2, "phase_unwrapping", None)
    implementation = getattr(namespace, "HistogramPhaseUnwrapping", None)
    if implementation is None:
        raise RuntimeError("Install an OpenCV build with HistogramPhaseUnwrapping")
    settings = implementation.Params()
    settings.height, settings.width = phase.shape
    settings.histThresh = _histogram_settings(p)
    settings.nbrOfSmallBins = int(p["small_bins"])
    settings.nbrOfLargeBins = int(p["large_bins"])
    # This native instance holds the histogram and pixel state for one capture.
    algorithm = implementation.create(settings)
    unwrapped = algorithm.unwrapPhaseMap(phase, shadowMask=mask)
    inverse = algorithm.getInverseReliabilityMap()
    return unwrapped, inverse


def _by_component(labels, refs):
    groups = defaultdict(list)
    h, w = labels.shape
    for pixel, phase in refs.items():
        x, y = pixel
        label = int(labels[y, x]) if x < w and y < h else 0
        if not label:
            raise ValidationError(f"Phase reference {pixel} lies outside valid phase", "references")
        groups[label].append((pixel, phase))
    return groups


def _references(phase, unwrapped, labels, count, groups, p):
    ids = labels.ravel()
    active = np.flatnonzero(ids)
    first = np.full(count, ids.size, np.intp)
    np.minimum.at(first, ids[active], active)
    counts = np.bincount(ids, minlength=count)
    turns = np.full(count, np.nan)
    rows = []
    for label in range(1, count):
        index = first[label]
        y, x = np.unravel_index(index, phase.shape)
        row = {"id": label, "pixels": int(counts[label]), "first_pixel": [int(x), int(y)]}
        if refs_on_region := groups.get(label):
            candidates = []
            for (px, py), wanted in refs_on_region:
                current = float(unwrapped[py, px])
                offset = round((wanted - current) / math.tau)
                residual = current + offset * math.tau - wanted
                candidates.append(
                    {
                        "pixel": [px, py],
                        "phase_rad": wanted,
                        "cycles": offset,
                        "residual_rad": residual,
                    }
                )
            if any(abs(c["residual_rad"]) > float(p["anchor_tolerance_rad"]) for c in candidates):
                status = "reference_out_of_tolerance"
            elif len({c["cycles"] for c in candidates}) > 1:
                status = "conflicting_reference_cycles"
            else:
                status = "anchored"
                turns[label] = candidates[0]["cycles"]
            row["references"] = candidates
        elif p["reference_mode"] == "required":
            status = "missing_reference"
        else:
            status = "relative"
            turns[label] = round((float(phase[y, x]) - float(unwrapped[y, x])) / math.tau)
        row.update(
            status=status, offset_cycles=int(turns[label]) if np.isfinite(turns[label]) else None
        )
        rows.append(row)
    return turns, rows


def _phase_graphic(values, valid, low, high):
    scaled = np.zeros(values.shape, np.uint8)
    span = high - low
    if span > 0:
        scaled[valid] = np.rint(np.clip((values[valid] - low) / span, 0, 1) * 255).astype(np.uint8)
    graphic = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    graphic[~valid] = 0
    return graphic


class UnwrapFringePhase(Tool3D):
    type_id, title, subtype = "3d.unwrap_phase", "Unwrap measured fringe phase", "Structured light"
    inputs = (
        Port("wrapped_phase"),
        Port("shadow_mask", required=False, required_when_connected=True, visual_type="region"),
        Port("region", "data", required=False, required_when_connected=True, visual_type="region"),
        Port("references", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("unwrapped_phase"),
        Port("inverse_reliability"),
        Port("component_labels"),
        Port("valid_mask", visual_type="region"),
        Port("report", "data"),
    )
    defaults = {
        "reference_mode": "relative",
        "alignment": "same_frame",
        "references": [],
        "anchor_tolerance_rad": 0.03,
        "histogram_threshold": 3 * math.pi**2,
        "small_bins": 10,
        "large_bins": 5,
    }
    parameter_choices = {
        "reference_mode": ("relative", "required"),
        "alignment": ("same_frame", "pixel_grid"),
    }
    parameter_ranges = {
        "anchor_tolerance_rad": NumericRange(1e-5, 3),
        "histogram_threshold": NumericRange(0.001, _MAX_EDGE_RELIABILITY - 0.001),
        "small_bins": NumericRange(1, 256, whole=True),
        "large_bins": NumericRange(1, 256, whole=True),
    }
    parameter_help = {
        "reference_mode": "Leave components relative or require an absolute reference in each.",
        "references": "List of records with a whole [x,y] pixel and known phase_rad.",
        "alignment": "Check known mask frames, or declare an independently loaded mask aligned.",
        "anchor_tolerance_rad": "Maximum residual after adding the best whole number of cycles.",
        "histogram_threshold": "Inverse-reliability split between fine and coarse histogram bins.",
        "small_bins": "Number of bins below the histogram threshold.",
        "large_bins": "Number of bins above the histogram threshold.",
    }
    input_help = {
        "wrapped_phase": "Single-channel float radians in [-π, π]. Nonfinite values are masked.",
        "shadow_mask": "Nonzero pixels are illuminated and eligible for unwrapping.",
        "region": "Optional selection intersected with Shadow Mask.",
        "references": "Wired list of known phases. Takes precedence over the References parameter.",
    }
    payload_reference = "wrapped_phase"
    flow_primary = ("unwrapped_phase",)
    parallel_safe = preview_safe = process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        _anchors(p["references"])
        _histogram_settings(p)

    def process(self, inputs):
        p = self.parameters
        image = inputs["wrapped_phase"]
        value = np.asarray(image)
        if (
            isinstance(image, DepthArray)
            or value.ndim != 2
            or not 0 < value.size <= _MAX_PIXELS
            or value.dtype not in (np.float32, np.float64)
            or image_format(image) != "gray"
        ):
            raise ValidationError(
                "Wrapped phase must be a float monochrome image up to 1,048,576 pixels"
            )
        tickets = {ticket_key(v["ticket"]) for v in self.input_payloads.values() if "ticket" in v}
        if len(tickets) > 1:
            raise ValidationError("Phase inputs must refer to the same ticket")
        selection = selection_mask(
            {"mask": inputs.get("shadow_mask"), "region": inputs.get("region")},
            image,
            match_mask_frame=p["alignment"] == "same_frame",
        )
        selected = np.ones(value.shape, bool) if selection is None else selection != 0
        eligible = selected & np.isfinite(value)
        if np.any(np.abs(value[eligible]) > _PI32):
            raise ValidationError("Selected wrapped phase must be in radians within [-π, π]")
        phase = np.where(eligible, value, 0).astype(np.float32)
        mask = eligible.astype(np.uint8) * 255
        count, labels = cv2.connectedComponentsWithAlgorithm(mask, 4, cv2.CV_32S, cv2.CCL_SAUF)
        if count - 1 > _MAX_COMPONENTS:
            raise ValidationError("Phase selection exceeds the 512 connected component limit")
        _validate_phase_jumps(phase, eligible)
        refs = _anchors(inputs.get("references", p["references"]))
        groups = _by_component(labels, refs)
        if count > 1:
            native, inverse = _native_unwrap(phase, mask, p)
            if not np.isfinite(native[eligible]).all() or not np.isfinite(inverse[eligible]).all():
                raise ValidationError("Native phase unwrapping returned nonfinite selected results")
        else:
            native = inverse = np.zeros(image.shape, np.float32)
        shifts, regions = _references(phase, native, labels, count, groups, p)
        values = native.astype(np.float64) + math.tau * shifts[labels]
        inverse = np.where(eligible, inverse, np.nan)
        valid = np.isfinite(values)
        valid_count, eligible_count = int(valid.sum()), int(eligible.sum())
        failures = sorted({r["status"] for r in regions} - {"anchored", "relative"})
        if not eligible_count:
            failures.append("no_phase_support")
        graphics = {}
        if self.capture_graphics and valid_count:
            low, high = float(values[valid].min()), float(values[valid].max())
            graphics["Unwrapped phase"] = _phase_graphic(values, valid, low, high)
        if self.capture_graphics and eligible_count:
            graphics["Inverse reliability"] = _phase_graphic(
                inverse, eligible, 0.0, _MAX_EDGE_RELIABILITY / 2
            )

        report = {
            "model": "histogram_phase_unwrap_v1",
            "units": "rad",
            "inverse_reliability_units": "rad²",
            "selected_pixels": int(selected.sum()),
            "eligible_pixels": eligible_count,
            "valid_pixels": valid_count,
            "components": regions,
        }
        phase_payload = dict(self.input_payloads["wrapped_phase"])
        phase_payload["phase_unwrap"] = {
            "units": "rad",
            "anchored_labels": [r["id"] for r in regions if r["status"] == "anchored"],
            "relative_labels": [r["id"] for r in regions if r["status"] == "relative"],
        }
        return Result(
            {
                "unwrapped_phase": ImageArray(values, "gray"),
                "inverse_reliability": ImageArray(inverse, "gray"),
                "component_labels": ImageArray(labels, "gray"),
                "valid_mask": ImageArray(valid.astype(np.uint8) * 255, "gray"),
                "report": report,
            },
            payloads={"unwrapped_phase": phase_payload},
            graphics=graphics,
            passed=not failures,
            score=valid_count / eligible_count if eligible_count else 0.0,
            fail_reasons=[f"phase_unwrap_{failure}" for failure in failures],
            message=f"{valid_count}/{eligible_count} phase pixels · {len(regions)} components",
        )
