"""Measure consecutive out-of-range samples on an ordered image path."""

from numbers import Real

import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .histogram import CHANNEL_CHOICES
from .parameter_validation import NumericRange, ValidationError
from .result_graphics import REJECTED


def _read_profile(profile, channel):
    error = "Use a profile with ordered distances, image points, and per-channel samples"
    if not isinstance(profile, dict):
        raise ValidationError(error)
    closed = profile.get("closed")
    raw_distances = profile.get("distances")
    channels = profile.get("channels")
    if (
        type(closed) is not bool
        or not isinstance(raw_distances, list)
        or not (1 if closed else 2) <= len(raw_distances) <= 8192
        or not isinstance(channels, list)
        or not 1 <= len(channels) <= 4
        or not all(isinstance(series, dict) for series in channels)
    ):
        raise ValidationError(error)

    series = channels[0] if channel == "auto" else next(
        (series for series in channels if series.get("channel") == channel), None
    )
    if series is None:
        raise ValidationError("Selected channel is unavailable in the profile", "channel")
    raw_values = series.get("values")
    if (
        not isinstance(raw_values, list)
        or len(raw_values) != len(raw_distances)
        or any(v is not None and (isinstance(v, bool) or not isinstance(v, Real))
               for v in raw_values)
    ):
        raise ValidationError(error)
    try:
        distances = np.asarray(raw_distances, dtype=np.float64)
        points = np.asarray(profile.get("points"), dtype=np.float64)
        values = np.asarray(raw_values, dtype=np.float64)
        length = float(profile.get("path_length", distances[-1] if not closed else np.nan))
    except (TypeError, ValueError, OverflowError):
        raise ValidationError(error) from None
    sampled = np.fromiter((value is not None for value in raw_values), dtype=bool)
    if (
        distances.ndim != 1 or distances[0] != 0
        or not np.all(np.diff(distances) > 0)
        or not 0 < length <= 20_000_000_000
        or (distances[-1] >= length if closed else distances[-1] != length)
        or points.shape != (len(distances), 2)
        or not np.all(np.abs(points) <= 1_000_000)
        or not np.isfinite(values[sampled]).all()
    ):
        raise ValidationError(error)
    return closed, length, distances, points, values, sampled


def sample_runs(failed, distances, length, closed):
    """Give each sample the arc-length cell between its neighboring midpoints."""
    changes = np.diff(np.r_[0, failed, 0])
    starts, stops = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    count = len(distances)
    if closed and failed.all():
        yield 0, count, 0.0, length, False, length
        return
    if closed and len(starts) > 1 and failed[0] and failed[-1]:
        starts = starts[1:]
        stops = np.r_[stops[1:-1], stops[0] + count]
    seam = (distances[-1] - length) / 2 if closed else 0.0
    edges = np.r_[seam, (distances[:-1] + distances[1:]) / 2, seam + length]
    for first, stop in zip(starts, stops, strict=True):
        start = float(edges[first])
        end = float(edges[stop]) if stop <= count else float(edges[stop - count] + length)
        wraps = bool(closed and (start < 0 or end >= length))
        yield (
            first, stop, start % length if wraps else start, end % length if wraps else end,
            wraps, end - start,
        )


class InspectPathContinuity(Block):
    type_id, title = "inspect.path_continuity", "Inspect path continuity"
    category, block_type, subtype = "Inspection", "Inspection", "Measurements"
    inputs = (Port("profile", "data"),)
    outputs = (
        Port("longest_gap", "number"), Port("gap_count", "number"),
        Port("sampled_fraction", "number"), Port("gaps", "data"),
    )
    payload_reference = "profile"
    process_inputs_readonly = True
    parallel_safe = True
    defaults = {
        "channel": "auto", "range_mode": "at_least",
        "minimum_value": 1.0, "maximum_value": 255.0,
        "maximum_gap": 0.0, "minimum_sampled_fraction": 1.0,
    }
    parameter_choices = {
        "channel": ("auto", *CHANNEL_CHOICES),
        "range_mode": ("at_least", "at_most", "between"),
    }
    parameter_ranges = {
        "minimum_value": NumericRange(), "maximum_value": NumericRange(),
        "maximum_gap": NumericRange(-1), "minimum_sampled_fraction": NumericRange(0, 1),
    }
    parameter_help = {
        "channel": "Auto uses the first channel in the path profile.",
        "range_mode": "Select the inclusive limits that define an acceptable sample.",
        "maximum_gap": "Maximum contiguous out-of-range length in path pixels. Negative disables.",
        "minimum_sampled_fraction": "Minimum fraction of path samples with values. "
        "Unavailable samples break a gap and are checked separately.",
    }
    input_help = {"profile": "Ordered samples and path length from Path intensity profile."}

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["range_mode"] == "between" and float(p["minimum_value"]) > float(p["maximum_value"]):
            raise ValidationError("Maximum value must be at least the minimum", "maximum_value")

    def process(self, inputs):
        p = self.parameters
        closed, length, distances, points, values, sampled = _read_profile(
            inputs["profile"], p["channel"]
        )
        failed = np.zeros_like(sampled)
        if p["range_mode"] != "at_most":
            failed |= values < float(p["minimum_value"])
        if p["range_mode"] != "at_least":
            failed |= values > float(p["maximum_value"])
        sampled_fraction = float(sampled.mean())
        runs = list(sample_runs(failed, distances, length, closed))
        gaps = [
            {"start_distance": start, "end_distance": end, "length": size,
             "sample_count": int(stop - first), "wraps_seam": wraps,
             "first_point": points[first].tolist(),
             "last_point": points[(stop - 1) % len(points)].tolist()}
            for first, stop, start, end, wraps, size in runs
        ]
        longest = max((gap["length"] for gap in gaps), default=0.0)
        reasons = []
        if not sampled_fraction or sampled_fraction < float(p["minimum_sampled_fraction"]):
            reasons.append({
                "code": "path_sampling_below_minimum",
                "message": f"{sampled_fraction:.1%} of the path samples have values",
            })
        limit = float(p["maximum_gap"])
        if limit >= 0 and longest > limit:
            reasons.append({
                "code": "path_gap_above_maximum",
                "message": f"Longest out-of-range gap is {longest:g} px (limit {limit:g} px)",
            })
        drawings = {}
        if runs and self.generate_graphics:
            objects = []
            for first, stop, _, _, _, _ in sorted(runs, key=lambda run: -run[-1])[:256]:
                coords = points[np.arange(first, stop) % len(points)]
                objects.append(graphic_object(
                    "polyline", coords, REJECTED, thickness=3,
                    closed=len(coords) == 1 or (closed and len(coords) == len(points)),
                ))
            drawings["Path gaps"] = GraphicDrawing(tuple(objects), source="profile")
        return Result(
            {"longest_gap": longest, "gap_count": len(gaps),
             "sampled_fraction": sampled_fraction, "gaps": gaps},
            drawings=drawings, passed=not reasons, fail_reasons=reasons,
            score=float(np.mean(sampled & ~failed)),
            message=f"{len(gaps)} path gaps, longest {longest:g} px",
        )
