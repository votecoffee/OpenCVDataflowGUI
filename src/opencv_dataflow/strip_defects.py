"""Report contiguous defects from a strip width inspection."""

from numbers import Real

import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .path_continuity import sample_runs
from .payloads import NO_VALUE
from .result_graphics import REJECTED
from .tickets import ticket_key

_COLORS = {
    "narrow": REJECTED, "wide": REJECTED, "out_of_range": REJECTED,
    "missing": (30, 210, 250), "unmeasurable": (220, 70, 210),
}


def _profile_data(profile):
    error = "Use a strip width profile with one width and status per column"
    if not isinstance(profile, dict):
        raise ValidationError(error)
    raw_widths, raw_status = profile.get("widths"), profile.get("status")
    if (
        not isinstance(raw_widths, list) or not isinstance(raw_status, list)
        or not 1 <= len(raw_widths) == len(raw_status) <= 8192
        or any(not isinstance(s, str) or s not in
               ("pass", "out_of_range", "missing", "unmeasurable") for s in raw_status)
        or any(value is not None and (isinstance(value, bool) or not isinstance(value, Real))
               for value in raw_widths)
        or profile.get("units") not in ("source px", "strip px")
    ):
        raise ValidationError(error)
    widths = np.asarray(raw_widths, dtype=np.float64)
    status = np.asarray(raw_status, dtype="<U12")
    measured = (status == "pass") | (status == "out_of_range")
    if (
        not np.isfinite(widths[measured]).all() or np.any(widths[measured] <= 0)
        or any(value is not None for value, keep in zip(raw_widths, measured, strict=True)
               if not keep)
    ):
        raise ValidationError(error)
    limits = profile.get("width_limits")
    if limits is not None:
        if not isinstance(limits, list) or len(limits) != 2:
            raise ValidationError("Width limits must contain a minimum and an optional maximum")
        low = NumericRange(0).check(limits[0], "Profile minimum width")
        high = None if limits[1] is None else NumericRange(low).check(
            limits[1], "Profile maximum width"
        )
        outside = status == "out_of_range"
        narrow = outside & (widths < low)
        status[narrow] = "narrow"
        if high is not None:
            status[outside & (widths > high)] = "wide"
    center = NumericRange(0, 2_100_000, whole=True).check(
        profile.get("center_row"), "Profile center row"
    )
    return widths, status, center


class GroupStripDefects(Block):
    type_id, title = "inspect.strip_defects", "Group strip defects"
    category, block_type, subtype = "Inspection", "Inspection", "Measurements"
    inputs = (
        Port("profile", "data"),
        Port("sampling", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("defects", "data"), Port("defect_count", "number"), Port("longest_defect", "number"),
    )
    payload_reference = "profile"
    process_inputs_readonly = True
    parallel_safe = True
    defaults = {"minimum_length": 0.0}
    parameter_ranges = {"minimum_length": NumericRange(0)}
    parameter_help = {
        "minimum_length": "Shortest run to include in the report and overlay. "
        "Uses source-path pixels with sampling, or strip pixels without it.",
    }
    input_help = {
        "profile": "Per-column profile from Inspect strip width.",
        "sampling": "Unwrap path strip sampling data for source-path lengths and closed loops.",
    }

    def _axis(self, count, center, sampling):
        profile_payload = self.input_payloads["profile"]
        spatial = profile_payload.get("spatial")
        if spatial is not None:
            width, height = spatial["image_size"]
            if width != count or center >= height:
                raise ValidationError("Width profile does not match its strip grid")
        if sampling is NO_VALUE:
            return np.arange(count) + 0.5, float(count), False, "strip px"

        if (
            not isinstance(sampling, dict) or sampling.get("model") != "path_strip_v1"
            or spatial is None or sampling.get("strip_spatial") != spatial
            or type(sampling.get("closed")) is not bool
        ):
            raise ValidationError("Sampling must describe the width profile's current strip grid")
        profile_ticket = profile_payload.get("ticket")
        sampling_ticket = self.input_payloads["sampling"].get("ticket")
        if (
            profile_ticket is not None and sampling_ticket is not None
            and ticket_key(profile_ticket) != ticket_key(sampling_ticket)
        ):
            raise ValidationError("Width profile and sampling must share one ticket")
        closed = sampling["closed"]
        if count == 1 and not closed:
            raise ValidationError("Open strip sampling needs at least two columns")
        length = NumericRange(np.finfo(float).tiny, 20_000_000_000).check(
            sampling.get("path_length"), "Sampling path length"
        )
        distances = np.linspace(0, length, count, endpoint=not closed)
        return distances, length, closed, "source px"

    def process(self, inputs):
        profile = inputs["profile"]
        widths, status, center = _profile_data(profile)
        count = len(status)
        distances, path_length, closed, units = self._axis(
            count, center, inputs.get("sampling", NO_VALUE)
        )
        minimum = float(self.parameters["minimum_length"])
        defects = []
        for kind in _COLORS:
            runs = sample_runs(status == kind, distances, path_length, closed)
            for first, stop, start, end, wraps, length in runs:
                if length < minimum and not np.isclose(length, minimum, rtol=1e-12, atol=0):
                    continue
                run_widths = widths[np.arange(first, stop) % count]
                measured = np.isfinite(run_widths)
                defects.append({
                    "kind": kind,
                    "first_column": int(first), "last_column": int((stop - 1) % count),
                    "column_count": int(stop - first),
                    "start_distance": start, "end_distance": end, "length": length,
                    "units": units, "wraps_seam": wraps,
                    "minimum_width": float(run_widths.min()) if measured.all() else None,
                    "maximum_width": float(run_widths.max()) if measured.all() else None,
                    "width_units": profile["units"],
                })
        defects.sort(key=lambda item: item["first_column"])
        longest = max((item["length"] for item in defects), default=0.0)
        drawings = {}
        if self.generate_graphics and defects:
            objects = []
            for item in sorted(defects, key=lambda item: -item["length"])[:128]:
                first, last = item["first_column"], item["last_column"]
                # The strip is displayed flat. A wrapped run occupies two separate intervals.
                intervals = ((first, count - 1), (0, last)) if last < first else ((first, last),)
                for a, b in intervals:
                    points = ((a, center),) if a == b else ((a, center), (b, center))
                    objects.append(graphic_object(
                        "polyline", points, _COLORS[item["kind"]], closed=a == b, thickness=3,
                    ))
            drawings["Defect runs"] = GraphicDrawing(tuple(objects), source="profile")
        return Result(
            {"defects": defects, "defect_count": len(defects), "longest_defect": longest},
            drawings=drawings,
            message=f"{len(defects)} defect runs · longest {longest:.3g} {units}",
        )
