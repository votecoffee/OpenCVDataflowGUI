"""A cancellable media clock and an explicitly paced recorded-video source."""

import copy
import math
import time

import cv2

from .core import Port
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .video_files import VideoFile


class MediaPacer:
    """Only copyable numbers live in state; clocks and waits are invocation-local services."""

    def __init__(self, now=time.monotonic, sleep=time.sleep):
        self.now, self.sleep = now, sleep

    def release(self, state, index, info, p, cancelled):
        if cancelled():
            return None
        loop = state.get("loop", False)
        fresh = "timestamp_ms" not in state
        reported = info["decoder_timestamp_ms"]
        native_fps = info["fps"]
        configured = p["fallback_fps"] or None
        policy = p["timing_policy"]
        use_configured = policy == "configured_fps" or native_fps is None
        fps = configured if use_configured else native_fps
        valid = reported is not None and 0 <= reported <= 1e15
        increasing = fresh or loop or (valid and reported > state["timestamp_ms"])
        if policy != "configured_fps" and valid and increasing:
            timestamp, origin = reported, "decoder"
        elif policy == "decoder_only" or fps is None:
            raise ValidationError(
                "Paced replay needs increasing decoder time or an allowed valid frame rate"
            )
        else:
            timestamp = (
                index * 1000 / fps
                if fresh or loop
                else (state["timestamp_ms"] + (index - state["index"]) * 1000 / fps)
            )
            origin = "configured_rate_estimate" if use_configured else "frame_rate_estimate"
        delta = 0.0
        interval_source = "first_frame"
        if not fresh:
            if loop:
                delta = 1000 * p["stride"] / fps if fps else state.get("last_interval_ms", 0)
                interval_source = (
                    "loop_configured_rate_estimate"
                    if fps and use_configured
                    else "loop_frame_rate_estimate"
                    if fps
                    else "loop_last_interval_estimate"
                )
                if delta <= 0:
                    raise ValidationError(
                        "A one-frame loop needs a valid frame rate for its boundary interval"
                    )
            else:
                delta = timestamp - state["timestamp_ms"]
                interval_source = "decoder_delta" if origin == "decoder" else origin
            if delta <= 0 or delta > p["maximum_gap_ms"]:
                raise ValidationError(
                    "Media interval is nonpositive or exceeds Maximum gap", "maximum_gap_ms"
                )
        started = self.now()
        resume = state.get("resume", False)
        deadline = started if fresh or resume else state["deadline"] + delta / (1000 * p["speed"])
        action = "first_frame" if fresh else "resumed" if resume else "loop" if loop else "paced"
        late = max(0.0, (started - deadline) * 1000)
        if p["late_policy"] == "rebase" and late > p["maximum_lateness_ms"]:
            deadline, action = started, "late_rebase"
        while True:
            if cancelled():
                return None
            remaining = deadline - self.now()
            if remaining <= 0:
                break
            self.sleep(min(remaining, 0.01))
        released = self.now()
        late = max(late, max(0, (released - deadline) * 1000))
        if p["late_policy"] == "rebase" and late > p["maximum_lateness_ms"]:
            deadline, action = released, "late_rebase"
        elif late > p["maximum_lateness_ms"] and action == "paced":
            action = "late_catch_up"
        cycle = state.get("cycle", 0) + int(loop)
        state.update(
            index=index,
            timestamp_ms=timestamp,
            deadline=deadline,
            last_interval_ms=delta or state.get("last_interval_ms", 0),
            cycle=cycle,
            loop=False,
            resume=False,
        )
        return {
            **info,
            "timestamp_ms": timestamp,
            "timestamp_source": origin,
            "playback": {
                "mode": "media_time",
                "speed": p["speed"],
                "action": action,
                "cycle": cycle,
                "cycle_start": bool(fresh or loop),
                "waited_ms": max(0, (released - started) * 1000),
                "lateness_ms": late,
                "interval_ms": delta,
                "interval_source": interval_source,
                "timing_policy": policy,
            },
        }


class PacedVideoFile(VideoFile):
    type_id, title = "input.paced_video_file", "Paced video file"
    outputs = (*VideoFile.outputs, Port("cycle_start", "boolean", required=False))
    defaults = {
        **VideoFile.defaults,
        "speed": 1.0,
        "timing_policy": "decoder_or_fps",
        "fallback_fps": 0.0,
        "late_policy": "rebase",
        "maximum_lateness_ms": 250.0,
        "maximum_gap_ms": 10000.0,
    }
    parameter_choices = {
        **VideoFile.parameter_choices,
        "timing_policy": ("decoder_or_fps", "decoder_only", "configured_fps"),
        "late_policy": ("rebase", "catch_up"),
    }
    parameter_ranges = {
        "speed": NumericRange(0.01, 100),
        "fallback_fps": NumericRange(0, 1000),
        "maximum_lateness_ms": NumericRange(0, 60000),
        "maximum_gap_ms": NumericRange(0.001, 3_600_000),
        "start_frame": NumericRange(0, 2**31 - 1, whole=True),
        "end_frame": NumericRange(-1, 2**31 - 1, whole=True),
        "stride": NumericRange(1, 100000, whole=True),
    }
    parameter_help = {
        **VideoFile.parameter_help,
        "speed": (
            "Wall-clock playback factor. Two runs twice as fast; no implicit frame dropping."
        ),
        "timing_policy": (
            "Prefer increasing decoder timestamps, require them, or use the configured frame rate."
        ),
        "fallback_fps": (
            "Configured rate; zero disables this fallback. Configured FPS policy "
            "requires a positive rate."
        ),
        "late_policy": (
            "Rebase after excessive delay to avoid bursts, or retain deadlines "
            "and catch up without dropping frames."
        ),
        "maximum_lateness_ms": "Delay allowed before the chosen late policy takes effect.",
        "maximum_gap_ms": (
            "Reject larger intervals before waiting; measured in media milliseconds, before Speed."
        ),
    }
    preview_safe = False
    requires_serial_execution = True

    def __init__(self, **kwargs):
        self._pacer = MediaPacer()
        self._playback = {}
        self._prepared_info = None
        super().__init__(**kwargs)
        self.exposed_outputs = [name for name in self.exposed_outputs if name != "cycle_start"]
        self.description = (
            self.description or "Replay recorded frames against a cancellable media-time clock."
        )
        self.documentation = self.documentation or (
            "<p>Frames retain their media time, index and ticket. "
            "Speed controls replay timing. Stop interrupts clock waits; "
            "Resume releases the pending "
            "frame immediately and starts a fresh wall-clock schedule. "
            "No implicit frame dropping. "
            "Expose Cycle start and wire it to sequence Reset "
            "to avoid tracking across loop boundaries. "
            "See docs/paced-video.md for timestamp, loop and cancellation contracts.</p>"
        )

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if type(p["frame_ticket"]) is not bool:
            raise ValidationError("Frame ticket must be boolean", "frame_ticket")
        if p["end_frame"] != -1 and p["end_frame"] <= p["start_frame"]:
            raise ValidationError("End must be later than Start", "end_frame", "start_frame")
        if p["timing_policy"] == "configured_fps" and float(p["fallback_fps"]) <= 0:
            raise ValidationError("Configured FPS policy needs a positive rate", "fallback_fps")
        if {
            "speed",
            "timing_policy",
            "fallback_fps",
            "late_policy",
            "maximum_lateness_ms",
            "maximum_gap_ms",
        } & self.recipe_bindings.keys():
            raise ValidationError("Playback controls are run settings and cannot be recipe-bound")

    def reset_runtime(self):
        super().reset_runtime()
        self._playback = {}
        self._prepared_info = None

    def close(self):
        super().close()
        self._playback["resume"] = True

    def configure_replay(self, repeat, *, rewind=False):
        super().configure_replay(repeat, rewind=rewind)
        self._playback["resume"] = True

    def snapshot_runtime(self):
        return {"video": super().snapshot_runtime(), "playback": copy.deepcopy(self._playback)}

    def restore_runtime(self, snapshot):
        super().restore_runtime(snapshot["video"])
        self._playback = copy.deepcopy(snapshot["playback"])
        self._prepared_info = None

    def open_frame(self, path, index):
        resume = self._playback.get("resume", False)
        result = super().open_frame(path, index)
        if not self._cancelled():
            self._playback["resume"] = resume
        return result

    def prepare_frame(self, index):
        self._prepared_info = None
        info = super().frame_info(index)
        # Scheduling observes the actual decoder property, before any legacy fallback.
        reported = self._capture.get(cv2.CAP_PROP_POS_MSEC)
        info["decoder_timestamp_ms"] = (
            float(reported) if math.isfinite(reported) and reported >= 0 else None
        )
        p = dict(self.parameters)
        for name, bounds in self.parameter_ranges.items():
            p[name] = bounds.check(p[name], name, name)
        self._prepared_info = self._pacer.release(self._playback, index, info, p, self._cancelled)
        return self._prepared_info is not None

    def frame_info(self, index):
        return copy.deepcopy(self._prepared_info)

    def process(self, inputs):
        result = super().process(inputs)
        result.outputs["cycle_start"] = (
            result.outputs["info"]["playback"]["cycle_start"]
            if result.outputs["image"] is not NO_VALUE
            else NO_VALUE
        )
        return result

    def finish_range(self):
        super().finish_range()
        self._playback["resume"] = False
        if not self.finite_source:
            self._playback["loop"] = True
