"""Bounded per-stream video clock diagnostics, with optional independent time references."""

import copy
import math

import cv2
import numpy as np

from .core import Port
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, frame
from .sequence import (
    SEQUENCE_CHOICES,
    SEQUENCE_DEFAULTS,
    SEQUENCE_HELP,
    SequenceBlock,
    SequenceStep,
)
from .tickets import ticket_key


def finite(value, limit=1e15):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and abs(value) <= limit
        and math.isfinite(value)
    )


def reference_values(value):
    if (
        not isinstance(value, list)
        or len(value) > 100000
        or any(not finite(n) or n < 0 for n in value)
    ):
        raise ValidationError(
            "Reference needs at most 100000 finite nonnegative times", "reference_timestamps_ms"
        )
    return value


def timing_chart(rows):
    chart = np.full((320, 640, 3), 24, np.uint8)
    stamps = [r[k] for r in rows for k in ("timestamp_ms", "reference_ms") if r[k] is not None]
    low, high = min(stamps, default=0), max(stamps, default=1)
    span = max(1.0, high - low)
    cv2.putText(
        chart,
        "Video position (ms) / decoded order",
        (18, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
    )
    cv2.line(chart, (36, 235), (620, 235), (130, 130, 130), 1)
    previous = None
    reference_previous = None
    for i, row in enumerate(rows):
        x = 36 + round(i * 584 / max(1, len(rows) - 1))
        if row["reference_ms"] is not None:
            reference_point = (x, 235 - round((row["reference_ms"] - low) * 180 / span))
            if reference_previous is not None:
                cv2.line(chart, reference_previous, reference_point, (230, 140, 80), 1)
            reference_previous = reference_point
        else:
            reference_previous = None
        value = row["timestamp_ms"]
        if value is None:
            previous = None
            continue
        point = (
            x,
            235 - round((value - low) * 180 / span),
        )
        color = (
            (40, 80, 240)
            if row["errors"]
            else (40, 190, 240)
            if row["estimated"]
            else (80, 220, 130)
        )
        if previous is not None:
            cv2.line(chart, previous, point, color, 1)
        cv2.circle(chart, point, 3, color, -1)
        previous = point
    cv2.putText(
        chart,
        f"Observed range {low:.6g} .. {high:.6g} ms",
        (36, 265),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (210, 210, 210),
        1,
    )
    cv2.putText(
        chart,
        "Green: decoder  Amber: estimate  Red: rejected  Blue: reference",
        (36, 297),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (210, 210, 210),
        1,
    )
    return chart


class VideoTiming(SequenceBlock):
    type_id, title = "sequence.video_timing", "Video timing monitor"
    subtype = "Video diagnostics"
    inputs = (
        Port("image"),
        Port("info", "data", required=False, required_when_connected=True),
        Port("reference", "data", required=False, required_when_connected=True),
        Port("reset", "boolean", required=False),
        Port("stream", "data", required=False),
    )
    data_outputs = (
        Port("image"),
        Port("report", "data"),
        Port("ok", "boolean"),
        Port("variable_rate", "boolean"),
        Port("chart", "image", required=False),
    )
    defaults = {
        **SEQUENCE_DEFAULTS,
        "warmup": 1,
        "history_size": 256,
        "timestamp_policy": "decoder_only",
        "cadence_policy": "allow_variable",
        "absolute_tolerance_ms": 1.0,
        "relative_tolerance_percent": 5.0,
        "maximum_interval_ms": 0.0,
        "latch_failures": True,
        "reference_timestamps_ms": [],
        "reference_tolerance_ms": 1.0,
    }
    parameter_choices = {
        **SEQUENCE_CHOICES,
        "timestamp_policy": ("decoder_only", "allow_estimates"),
        "cadence_policy": ("allow_variable", "require_constant", "match_nominal_fps"),
    }
    parameter_ranges = {
        "warmup": NumericRange(1, 10000, whole=True),
        "history_size": NumericRange(3, 4096, whole=True),
        "absolute_tolerance_ms": NumericRange(0, 1e9),
        "relative_tolerance_percent": NumericRange(0, 100),
        "maximum_interval_ms": NumericRange(0, 1e12),
        "reference_tolerance_ms": NumericRange(0, 1e12),
        "max_streams": NumericRange(1, 64, whole=True),
        "max_bytes_mb": NumericRange(0.001, 4096),
        "max_gap_ms": NumericRange(0, 1e12),
    }
    parameter_help = {
        **SEQUENCE_HELP,
        "history_size": "Bounded row/plot window. Session counters survive row eviction.",
        "timestamp_policy": "Require decoder properties, or explicitly allow labeled estimates.",
        "cadence_policy": "Allow VFR, require constant cadence, or match the reported FPS.",
        "absolute_tolerance_ms": "Cadence tolerance floor in media milliseconds per decoded frame.",
        "relative_tolerance_percent": "Additional relative cadence tolerance; not a drop detector.",
        "maximum_interval_ms": "Reject larger selected-frame media gaps. Zero disables this check.",
        "latch_failures": "Keep stream failures until Reset or a context reset.",
        "reference_timestamps_ms": "Independent full-clip times indexed by source frame index.",
        "reference_tolerance_ms": "Maximum absolute difference from the time reference.",
    }
    input_help = {
        "image": "Current image with video_frame/network_frame metadata; pixels remain unchanged.",
        "info": "Optional matching source Info; otherwise reads the image's timing payload.",
        "reference": "Optional list of independent times, overriding Reference timestamps ms.",
        "reset": "Wire paced Cycle start here to keep loop boundaries out of the timing history.",
    }
    preview_safe = False
    process_inputs_readonly = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = [n for n in self.exposed_outputs if n != "chart"]
        self.description = (
            self.description or "Observe video clocks without silently repairing them."
        )
        self.documentation = self.documentation or (
            "<p>Checks current raw/effective time, index order, cadence and optional independent "
            "references. Variable rate is valid by default. Reset explicitly across loops. "
            "See docs/video-timing.md for scope, labels, retention and acceptance limits.</p>"
        )

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        reference_values(p["reference_timestamps_ms"])
        if type(p["latch_failures"]) is not bool:
            raise ValidationError("Latch failures must be boolean", "latch_failures")
        for name, limits in self.parameter_ranges.items():
            if not finite(p[name]) or (limits.whole and type(p[name]) is not int):
                raise ValidationError(f"{name} must be a finite typed number", name)

    def info(self, inputs):
        payload = self.input_payloads.get("image", {})
        attached = payload.get("video_frame", payload.get("network_frame"))
        supplied = inputs.get("info", attached)
        if not isinstance(supplied, dict):
            raise ValidationError("Connect source Info or an image with video timing", "info")
        if "info" in inputs and isinstance(attached, dict):
            image_ticket = payload.get("ticket")
            info_ticket = self.input_payloads.get("info", {}).get("ticket")
            if (
                image_ticket is not None
                and info_ticket is not None
                and (ticket_key(image_ticket) != ticket_key(info_ticket))
            ):
                raise ValidationError("Connected Info belongs to another ticket", "info")
            for key in (
                "filename",
                "frame_index",
                "timestamp_ms",
                "timestamp_source",
                "decoder_timestamp_ms",
            ):
                if key in supplied and key in attached and supplied[key] != attached[key]:
                    raise ValidationError("Connected Info does not match this image", "info")
        return supplied

    def extra_context(self, inputs):
        info = self.info(inputs)
        return {
            "source": info.get("filename", info.get("source_name")),
            "backend": info.get("backend"),
            "reference": reference_values(
                inputs.get("reference", self.parameters["reference_timestamps_ms"])
            ),
        }

    def estimate_state_bytes(self, image):
        return max(image.nbytes, self.parameters["history_size"] * 1024)

    def step(self, inputs, state, mask):
        p, info = self.parameters, self.info(inputs)
        reference = reference_values(inputs.get("reference", p["reference_timestamps_ms"]))
        rows = state.setdefault("rows", [])
        counts = state.setdefault("counts", {"frames": 0, "rejected": 0, "estimated": 0})
        error_counts = state.setdefault("error_counts", {})
        previous = state.get("previous")
        index = info.get("frame_index")
        index = index if type(index) is int and 0 <= index <= 2**53 else None
        origin = info.get("timestamp_source", "unverified")
        if not isinstance(origin, str) or len(origin) > 80:
            origin = "unverified"
        estimated = origin in ("frame_rate_estimate", "configured_rate_estimate")
        effective = info.get("timestamp_ms")
        effective = float(effective) if finite(effective) else None
        raw = info.get("decoder_timestamp_ms", effective if origin == "decoder" else None)
        raw = float(raw) if finite(raw) else None
        stamp = effective if estimated and p["timestamp_policy"] == "allow_estimates" else raw
        errors, warnings = [], []
        if origin == "decoder":
            if effective is None:
                errors.append("missing_effective_timestamp")
            elif raw is not None and abs(effective - raw) > max(1e-7, math.ulp(raw) * 4):
                errors.append("decoder_effective_mismatch")
        if index is None:
            errors.append("missing_frame_index")
        if stamp is None:
            errors.append("missing_timestamp")
        elif stamp < 0:
            errors.append("negative_timestamp")
        if estimated:
            (errors if p["timestamp_policy"] == "decoder_only" else warnings).append(
                "estimated_timestamp"
            )
        if origin not in ("decoder", "frame_rate_estimate", "configured_rate_estimate"):
            errors.append("unverified_timestamp")
        delta = normalized = None
        if previous is not None:
            if (
                index is not None
                and previous["frame_index"] is not None
                and index <= previous["frame_index"]
            ):
                errors.append("frame_index_not_increasing")
            if stamp is not None and previous["timestamp_ms"] is not None:
                delta = stamp - previous["timestamp_ms"]
                if delta == 0:
                    errors.append("repeated_timestamp")
                elif delta < 0:
                    errors.append("backward_timestamp")
                elif (
                    index is not None
                    and previous["frame_index"] is not None
                    and index > previous["frame_index"]
                ):
                    normalized = delta / (index - previous["frame_index"])
                if p["maximum_interval_ms"] and delta > p["maximum_interval_ms"]:
                    errors.append("excessive_interval")
        clock_index = info.get("connection_frame_index", index)
        if raw == 0 and clock_index not in (None, 0):
            (
                warnings if estimated and p["timestamp_policy"] == "allow_estimates" else errors
            ).append("zero_time_after_first")
        fps = info.get("fps")
        fps = float(fps) if finite(fps, 100000) and fps >= 1e-9 else None
        if fps is None:
            (errors if p["cadence_policy"] == "match_nominal_fps" else warnings).append(
                "missing_nominal_fps"
            )
        if normalized is not None and fps:
            nominal = 1000 / fps
            tolerance = max(
                p["absolute_tolerance_ms"], nominal * p["relative_tolerance_percent"] / 100
            )
            if abs(normalized - nominal) > tolerance:
                (errors if p["cadence_policy"] == "match_nominal_fps" else warnings).append(
                    "nominal_rate_mismatch"
                )
        ref = ref_error = None
        if reference:
            if index is None or index >= len(reference):
                errors.append("reference_missing")
            else:
                ref = reference[index]
                if stamp is not None:
                    ref_error = stamp - ref
                    if abs(ref_error) > p["reference_tolerance_ms"]:
                        errors.append("reference_mismatch")
        row = {
            "frame_index": index,
            "timestamp_ms": stamp,
            "decoder_timestamp_ms": raw,
            "effective_timestamp_ms": effective,
            "timestamp_source": origin,
            "estimated": estimated,
            "delta_ms": delta,
            "interval_per_frame_ms": normalized,
            "fps": fps,
            "reference_ms": ref,
            "reference_error_ms": ref_error,
            "errors": errors,
            "warnings": warnings,
        }
        rows.append(row)
        del rows[: -p["history_size"]]
        intervals = [
            r["interval_per_frame_ms"] for r in rows if r["interval_per_frame_ms"] is not None
        ]
        median = float(np.median(intervals)) if intervals else None
        tolerance = max(
            p["absolute_tolerance_ms"], (median or 0) * p["relative_tolerance_percent"] / 100
        )
        variable = len(intervals) >= 2 and max(intervals) - min(intervals) > tolerance
        if variable and p["cadence_policy"] == "require_constant":
            errors.append("variable_cadence")
        counts["frames"] += 1
        counts["rejected"] += bool(errors)
        counts["estimated"] += estimated
        for code in errors:
            error_counts[code] = error_counts.get(code, 0) + 1
        state["previous"] = copy.deepcopy(row)
        ok = counts["rejected"] == 0 if p["latch_failures"] else not errors
        summary = {
            "model": "video_timing_v1",
            "ok": ok,
            "current_ok": not errors,
            "judgment_scope": "session" if p["latch_failures"] else "current",
            "counts": copy.deepcopy(counts),
            "error_counts": copy.deepcopy(error_counts),
            "cadence": "variable"
            if variable
            else "constant"
            if len(intervals) >= 2
            else "insufficient",
            "variable_rate": variable,
            "interval_per_frame_ms": {
                "minimum": min(intervals, default=None),
                "median": median,
                "maximum": max(intervals, default=None),
            },
            "history_rows": len(rows),
            "current": copy.deepcopy(row),
            "reference_supplied": bool(reference),
            "time_domain": "OpenCV video position; not exposure time",
        }
        report = {**summary, "rows": copy.deepcopy(rows)}
        chart = timing_chart(rows) if self.generate_graphics else NO_VALUE
        return SequenceStep(
            {
                "image": inputs["image"],
                "report": report,
                "ok": ok,
                "variable_rate": variable,
                "chart": chart,
            },
            details={"timing": summary},
            graphics={"Video timing": chart} if chart is not NO_VALUE else {},
            output_frames={"chart": frame([640, 320], f"{self.id}:timing-chart")}
            if chart is not NO_VALUE
            else {},
        )

    def process(self, inputs):
        result = super().process(inputs)
        timing = result.outputs["status"]["timing"]
        result.passed = timing["ok"]
        counts = timing["counts"]
        result.score = 1 - counts["rejected"] / counts["frames"]
        if not result.passed:
            codes = (
                timing["error_counts"]
                if self.parameters["latch_failures"]
                else timing["current"]["errors"]
            )
            result.fail_reasons = [
                {"code": f"video_timing.{c}", "message": c.replace("_", " ")} for c in codes
            ]
        if timing["current"]["errors"]:
            self.logger.warning(
                "Video timing: %s",
                ", ".join(timing["current"]["errors"]),
                parameters={
                    "frame_index": timing["current"]["frame_index"],
                    "errors": timing["current"]["errors"],
                },
            )
        result.message += (
            f" · {timing['cadence']} cadence · {counts['rejected']}/{counts['frames']} rejected"
        )
        if self.generate_graphics and result.outputs["image"] is not NO_VALUE:
            value = timing["current"]["timestamp_ms"]
            label = "missing" if value is None else f"{value:.6g} ms"
            color = (50, 200, 80) if result.passed else (40, 70, 240)
            result.drawings["Video timing"] = GraphicDrawing(
                (
                    graphic_object(
                        "text",
                        [(5, min(18, inputs["image"].shape[0] - 1))],
                        color,
                        text=f"t={label}; {timing['cadence']}",
                    ),
                )
            )
        return result
