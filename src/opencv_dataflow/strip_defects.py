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

_REPORT_MODEL = "strip_defect_report_v1"
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
        Port("inspection_report", "data", required=False),
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
        all_defects = []
        for kind in _COLORS:
            runs = sample_runs(status == kind, distances, path_length, closed)
            for first, stop, start, end, wraps, length in runs:
                run_widths = widths[np.arange(first, stop) % count]
                measured = np.isfinite(run_widths)
                all_defects.append({
                    "kind": kind,
                    "first_column": int(first), "last_column": int((stop - 1) % count),
                    "column_count": int(stop - first),
                    "start_distance": start, "end_distance": end, "length": length,
                    "units": units, "wraps_seam": wraps,
                    "minimum_width": float(run_widths.min()) if measured.all() else None,
                    "maximum_width": float(run_widths.max()) if measured.all() else None,
                    "width_units": profile["units"],
                })
        all_defects.sort(key=lambda item: item["first_column"])
        defects = [item for item in all_defects if item["length"] >= minimum
                   or np.isclose(item["length"], minimum, rtol=1e-12, atol=0)]
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
        report = {
            "model": _REPORT_MODEL, "units": units, "columns": count,
            "measured_count": int(np.count_nonzero(np.isfinite(widths))), "defects": all_defects,
        }
        return Result(
            {"defects": defects, "defect_count": len(defects), "longest_defect": longest,
             "inspection_report": report},
            drawings=drawings,
            message=f"{len(defects)} defect runs · longest {longest:.3g} {units}",
        )


class JudgeStripDefects(Block):
    type_id, title = "inspect.judge_strip_defects", "Judge strip defects"
    category, block_type, subtype = "Inspection", "Inspection", "Measurements"
    inputs = (Port("inspection_report", "data"),)
    outputs = (Port("summary", "data"), Port("passed", "data"))
    payload_reference = "inspection_report"
    process_inputs_readonly = True
    parallel_safe = True
    defaults = {"length_units": "source px", "limits": {}}
    parameter_choices = {"length_units": ("source px", "strip px")}
    parameter_help = {
        "length_units": "Required units for defect lengths. Match the Group strip defects report.",
        "limits": 'Per-kind limits, for example {"narrow": {"max_count": 1, "max_length": 12}}. '
        "Use narrow, wide, out_of_range, missing or unmeasurable. "
        "An unlisted kind is rejected. Set either or both inclusive limits for a listed kind.",
    }
    input_help = {
        "inspection_report": "Complete inspection report from Group strip defects. "
        "The group's minimum report length does not affect this input.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        rules = p["limits"]
        if not isinstance(rules, dict) or not rules.keys() <= _COLORS.keys():
            raise ValidationError("Limits must use known strip defect kinds", "limits")
        for kind, rule in rules.items():
            if (
                not isinstance(rule, dict) or not rule
                or not rule.keys() <= {"max_count", "max_length"}
            ):
                raise ValidationError(f"{kind} needs max_count, max_length or both", "limits")
            if "max_count" in rule:
                NumericRange(0, 8192, whole=True).check(
                    rule["max_count"], f"{kind} max_count", "limits"
                )
            if "max_length" in rule:
                if type(rule["max_length"]) not in (int, float):
                    raise ValidationError(f"{kind} max_length must be numeric", "limits")
                NumericRange(0).check(rule["max_length"], f"{kind} max_length", "limits")

    def process(self, inputs):
        report = inputs["inspection_report"]
        if not isinstance(report, dict) or report.get("model") != _REPORT_MODEL:
            raise ValidationError("Connect the complete Group strip defects inspection report")
        units = self.parameters["length_units"]
        if report.get("units") != units:
            raise ValidationError(f"Defect report length units must be {units}", "length_units")
        columns = NumericRange(1, 8192, whole=True).check(report.get("columns"), "Strip columns")
        measured = NumericRange(0, columns, whole=True).check(
            report.get("measured_count"), "Measured columns"
        )
        runs = report.get("defects")
        if not isinstance(runs, list) or len(runs) > columns:
            raise ValidationError("Defect report must contain a bounded list of runs")
        by_kind = {kind: {"count": 0, "longest_length": 0.0} for kind in _COLORS}
        measured_defect_columns, unavailable_columns = 0, 0
        for run in runs:
            if (
                not isinstance(run, dict) or not isinstance(run.get("kind"), str)
                or run["kind"] not in by_kind or run.get("units") != units
            ):
                raise ValidationError("Every run must have a known defect kind and matching units")
            kind = run["kind"]
            count = NumericRange(1, columns, whole=True).check(
                run.get("column_count"), "Defect column count"
            )
            length = NumericRange(np.finfo(float).tiny, 20_000_000_000).check(
                run.get("length"), "Defect length"
            )
            row = by_kind[kind]
            row["count"] += 1
            row["longest_length"] = max(row["longest_length"], length)
            if kind in ("missing", "unmeasurable"):
                unavailable_columns += count
            else:
                measured_defect_columns += count
        if measured_defect_columns > measured or unavailable_columns != columns - measured:
            raise ValidationError("Defect runs do not account for measured and unavailable columns")

        reasons = []
        if not measured:
            reasons.append({
                "code": "strip_defect_no_measurable_width",
                "message": "The strip has no measured widths",
            })
        for kind, row in by_kind.items():
            rule = self.parameters["limits"].get(kind, {"max_count": 0})
            count = row["count"]
            max_count = rule.get("max_count")
            if max_count is not None and count > int(max_count):
                reasons.append({
                    "code": "strip_defect_count_above_maximum",
                    "message": f"{kind}: run count {count} exceeds the limit of {max_count}",
                })
            max_length = rule.get("max_length")
            longest = row["longest_length"]
            if (
                max_length is not None and longest > float(max_length)
                and not np.isclose(longest, float(max_length), rtol=1e-12, atol=0)
            ):
                reasons.append({
                    "code": "strip_defect_length_above_maximum",
                    "message": f"{kind}: longest run {longest:.3g} {units} exceeds "
                               f"{max_length} {units}",
                })
        passed = not reasons
        summary = {
            "units": units, "columns": columns, "measured_count": measured,
            "defect_count": len(runs), "by_kind": by_kind,
        }
        return Result(
            {"summary": summary, "passed": passed}, passed=passed, fail_reasons=reasons,
            message=f"{len(runs)} defect runs · {'within limits' if passed else 'outside limits'}",
        )
