"""Media-time frame sampling with native seek, bounded bracketing and explicit provenance."""

import copy
import math

import cv2
import numpy as np

from .core import Block, Port, Result
from .images import COLOR_FORMATS, PIXEL_DEPTHS, format_image
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, frame
from .video_files import PATH_BASES, video_path


class TimestampUnavailable(ValidationError):
    pass


def number(value):
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(value)
    )


class TimeSampler:
    """Bound candidate storage; no native object escapes this invocation."""

    def __init__(self, block, path, target):
        self.block, self.p, self.path, self.target = block, block.parameters, path, target
        self.capture = None
        self.reads = self.decoded = 0
        self.ordinal = None
        self.previous_time = None
        self.size = None
        self.epsilon = max(1e-7, math.ulp(target) * 4)
        self.seek = {
            "requested_ms": target,
            "selection": self.p["selection"],
            "method_requested": self.p["seek_method"],
            "native_set_accepted": None,
            "native_first_timestamp_ms": None,
            "native_first_timestamp_source": None,
            "native_first_decoder_timestamp_ms": None,
            "native_first_frame_index": None,
            "anchor_ms": max(0, target - self.p["preroll_ms"]),
            "fallback_reason": None,
        }

    def close(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def open(self, linear=False):
        self.close()
        if self.block._cancelled():
            raise InterruptedError()
        self.capture = cv2.VideoCapture()
        if not self.capture.open(
            str(self.path),
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                self.p["open_timeout_ms"],
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                self.p["read_timeout_ms"],
                cv2.CAP_PROP_N_THREADS,
                self.p["decoder_threads"],
            ],
        ):
            raise ValidationError("FFmpeg cannot open the selected video", "path")
        if not self.capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0):
            raise ValidationError("FFmpeg cannot disable automatic video orientation")
        self.ordinal = 0 if linear else None
        self.previous_time = None
        self.size = None
        width, height = (
            self.capture.get(key) for key in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)
        )
        if all(number(n) and n >= 1 for n in (width, height)):
            self.guard_size(int(width), int(height))

    def guard_size(self, width, height):
        pixels = width * height
        if pixels > self.p["maximum_frame_pixels"]:
            raise ValidationError("Video exceeds Maximum frame pixels", "maximum_frame_pixels")
        channels = {"gray": 1, "bgra": 4, "rgba": 4}.get(self.p["color_format"], 3)
        item = {"uint16": 2, "float32": 4}.get(self.p["pixel_depth"], 1)
        # Native first sample plus two bracket arrays, output/conversion; not a native RSS cap.
        if pixels * (9 + channels * (item + 8)) > self.p["maximum_working_bytes"]:
            raise ValidationError("Video exceeds Maximum working bytes", "maximum_working_bytes")

    def read(self):
        if self.block._cancelled():
            raise InterruptedError()
        if self.reads >= self.p["maximum_read_frames"]:
            raise ValidationError("Seek exceeded Maximum read frames", "maximum_read_frames")
        self.reads += 1
        ok, image = self.capture.read()
        if not ok:
            return None
        self.decoded += 1
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValidationError("Video decoder must return uint8 BGR")
        height, width = image.shape[:2]
        if min(width, height) < 1:
            raise ValidationError("Video decoder returned an empty frame")
        self.guard_size(width, height)
        if self.size is not None and self.size != (width, height):
            raise ValidationError("Video dimensions changed while bracketing time")
        self.size = width, height
        pos = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
        native_index = (
            int(pos) - 1 if number(pos) and 1 <= pos <= 2**53 and pos == int(pos) else None
        )
        index = native_index if self.ordinal is None else self.ordinal
        raw, fps = (self.capture.get(key) for key in (cv2.CAP_PROP_POS_MSEC, cv2.CAP_PROP_FPS))
        fps = float(fps) if number(fps) and 0 < fps <= 100000 else None
        reported = float(raw) if number(raw) and 0 <= raw <= 1e15 else None
        valid = reported is not None and not (reported == 0 and index not in (None, 0))
        valid = valid and (self.previous_time is None or reported > self.previous_time)
        if valid:
            stamp, origin = reported, "decoder"
        elif (
            self.p["timestamp_policy"] == "decoder_or_fps" and fps is not None and index is not None
        ):
            stamp = (
                index * 1000 / fps
                if self.previous_time is None
                else self.previous_time + 1000 / fps
            )
            origin = "frame_rate_estimate"
        else:
            raise TimestampUnavailable(
                "Seek needs increasing decoder time or an allowed FPS estimate"
            )
        self.previous_time = stamp
        info = {
            "filename": self.path.name,
            "frame_index": index,
            "frame_index_source": "decoded_ordinal"
            if self.ordinal is not None
            else "decoder_property",
            "decoder_frame_index": native_index,
            "decoder_timestamp_ms": reported,
            "timestamp_ms": stamp,
            "timestamp_source": origin,
            "fps": fps,
            "backend": self.capture.getBackendName(),
        }
        if self.ordinal is not None:
            self.ordinal += 1
        return image, info

    def choose(self, current):
        previous = None
        policy = self.p["selection"]
        while current is not None:
            stamp = current[1]["timestamp_ms"]
            if stamp >= self.target - self.epsilon:
                if abs(stamp - self.target) <= self.epsilon or policy == "at_or_after":
                    return current
                if policy == "nearest":
                    if previous is None or (
                        self.target - previous[1]["timestamp_ms"]
                        > stamp - self.target + self.epsilon
                    ):
                        return current
                return previous
            previous = current
            current = self.read()
        return None if policy == "at_or_after" else previous

    def sample(self):
        method, selected = self.p["seek_method"], None
        try:
            if method != "scan_from_start":
                self.open()
                try:
                    accepted = self.capture.set(cv2.CAP_PROP_POS_MSEC, self.seek["anchor_ms"])
                except cv2.error:
                    accepted = False
                self.seek["native_set_accepted"] = bool(accepted)
                reason = "native_seek_rejected"
                if accepted:
                    try:
                        first = self.read()
                        reason = "native_seek_eof"
                        if first is not None:
                            self.seek.update(
                                native_first_timestamp_ms=first[1]["timestamp_ms"],
                                native_first_timestamp_source=first[1]["timestamp_source"],
                                native_first_decoder_timestamp_ms=first[1]["decoder_timestamp_ms"],
                                native_first_frame_index=first[1]["frame_index"],
                            )
                            reason = "native_seek_overshot"
                            if first[1]["timestamp_ms"] <= self.target + self.epsilon:
                                selected = self.choose(first)
                                self.seek["method_used"] = "native_bracket"
                                reason = "no_eligible_frame"
                    except TimestampUnavailable:
                        reason = "native_time_unavailable"
                if selected is None:
                    if method == "native_only":
                        raise ValidationError(f"Native time seek: {reason}", "seek_method")
                    self.seek["fallback_reason"] = reason
            if selected is None:
                self.open(linear=True)
                self.seek["method_used"] = "scan_from_start"
                selected = self.choose(self.read())
            if selected is None:
                raise ValidationError("No frame satisfies the requested time selection", "time_ms")
            image, info = selected
            error = info["timestamp_ms"] - self.target
            if abs(error) > self.p["maximum_error_ms"] + self.epsilon:
                raise ValidationError(
                    "Selected frame exceeds Maximum time error", "maximum_error_ms"
                )
            self.seek.update(
                error_ms=error,
                absolute_error_ms=abs(error),
                read_requests=self.reads,
                decoded_frames=self.decoded,
            )
            return image, {**info, "seek": self.seek}
        finally:
            self.close()


class VideoAtTime(Block):
    type_id, title, category = "input.video_at_time", "Video frame at time", "Input"
    block_type, subtype = "Input", "Video"
    inputs = (Port("time_ms", "number", required=False),)
    outputs = (Port("image"), Port("info", "data"))
    defaults = {
        "path": "",
        "path_base": "working_directory",
        "backend": "ffmpeg",
        "time_ms": 0.0,
        "selection": "nearest",
        "seek_method": "native_then_scan",
        "preroll_ms": 0.0,
        "maximum_error_ms": 1000.0,
        "timestamp_policy": "decoder_only",
        "frame_ticket": True,
        "open_timeout_ms": 5000,
        "read_timeout_ms": 5000,
        "decoder_threads": 1,
        "maximum_read_frames": 10000,
        "maximum_frame_pixels": 16_000_000,
        "maximum_working_bytes": 256 * 1024**2,
        "color_format": "bgr",
        "pixel_depth": "uint8",
    }
    parameter_choices = {
        "path_base": PATH_BASES,
        "backend": ("ffmpeg",),
        "selection": ("nearest", "at_or_before", "at_or_after"),
        "seek_method": ("native_then_scan", "native_only", "scan_from_start"),
        "timestamp_policy": ("decoder_only", "decoder_or_fps"),
        "color_format": COLOR_FORMATS,
        "pixel_depth": PIXEL_DEPTHS,
    }
    parameter_ranges = {
        "time_ms": NumericRange(0, 1e12),
        "preroll_ms": NumericRange(0, 1e12),
        "maximum_error_ms": NumericRange(0, 1e12),
        "open_timeout_ms": NumericRange(1, 60000, whole=True),
        "read_timeout_ms": NumericRange(1, 60000, whole=True),
        "decoder_threads": NumericRange(1, 16, whole=True),
        "maximum_read_frames": NumericRange(1, 1000000, whole=True),
        "maximum_frame_pixels": NumericRange(1, 64_000_000, whole=True),
        "maximum_working_bytes": NumericRange(1, 1024**3, whole=True),
    }
    parameter_help = {
        "path": "Local video file. Each request opens and closes its own decoder.",
        "time_ms": "Presentation time in milliseconds; a connected Time ms overrides this value.",
        "selection": "Bracket the requested time. Nearest ties prefer the earlier frame.",
        "seek_method": "Try native time positioning, require it, or decode from the beginning.",
        "preroll_ms": "Native seek anchor is max(0, requested time minus preroll).",
        "maximum_error_ms": "Maximum selected-time error; zero requires numerical equality.",
        "timestamp_policy": "Require decoder time or allow labeled native-FPS estimates.",
        "maximum_read_frames": "Read-request budget, including native probe, fallback and EOF.",
        "maximum_working_bytes": "Estimated candidate/conversion storage; excludes native memory.",
        "frame_ticket": "Auto tickets label filename/index/time; every request stays distinct.",
        "open_timeout_ms": "Native FFmpeg opening timeout; not a hard execution deadline.",
        "read_timeout_ms": "Native FFmpeg read timeout. Stop is cooperative between native calls.",
        "color_format": "Convert native uint8 BGR. Automatic orientation is disabled.",
        "pixel_depth": "Output conversion, not recovery of encoded high-bit-depth samples.",
    }
    input_help = {"time_ms": "Finite nonnegative milliseconds, independent of playback wall time."}
    output_help = {
        "info": "Requested/actual time, signed error, selected index, method and work counts."
    }
    preview_safe = False
    requires_serial_execution = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prepared = None
        self.description = self.description or "Sample a video at a requested presentation time."
        self.documentation = self.documentation or (
            "<p>Nearest/earlier/later selection reports actual position and time error. "
            "Each request is independent and receives a fresh ticket/pixel frame. "
            "See docs/video-time-seeking.md for native bracketing, fallback and limits.</p>"
        )

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not isinstance(p["path"], str) or not p["path"].strip() or len(p["path"]) > 4096:
            raise ValidationError("Select a video file", "path")
        for name, bounds in self.parameter_ranges.items():
            if not number(p[name]) or (bounds.whole and type(p[name]) is not int):
                raise ValidationError(
                    f"{name} must be a finite {'integer' if bounds.whole else 'number'}", name
                )
        if type(p["frame_ticket"]) is not bool:
            raise ValidationError("Frame ticket must be boolean", "frame_ticket")
        if p["pixel_depth"] == "uint16" and p["color_format"] in ("hsv", "lab"):
            raise ValidationError("HSV/Lab need uint8 or float32", "pixel_depth", "color_format")

    def close(self):
        self._prepared = None

    def issue_source_ticket(self):
        self._prepared = None
        target = self.last_inputs.get("time_ms", self.parameters["time_ms"])
        if not number(target) or not 0 <= target <= 1e12:
            raise ValidationError("Time ms input must be a finite number in 0..1e12", "time_ms")
        path = video_path(self)
        if not path.is_file():
            raise FileNotFoundError(f"Video file does not exist: {path}")
        before = path.stat()
        try:
            image, info = TimeSampler(self, path, float(target)).sample()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ino,
            ):
                raise ValidationError("Video file changed during time selection", "path")
            image = format_image(
                image, self.parameters["color_format"], self.parameters["pixel_depth"]
            )
            if self._cancelled():
                return None
            ticket = super().issue_source_ticket()
            if self.parameters["frame_ticket"] and self.ticket_settings["mode"] == "auto_number":
                label = (
                    f"#{info['frame_index']:06d}"
                    if info["frame_index"] is not None
                    else f"@{info['timestamp_ms']:.6f}ms"
                )
                ticket.update(
                    kind="string",
                    value=path.name + label,
                    session=f"{ticket['session']}:seek:{ticket['sequence']}",
                )
            info["seek"]["target_source"] = (
                "input" if "time_ms" in self.last_inputs else "parameter"
            )
            self._prepared = image, info
            return ticket
        except InterruptedError:
            return None

    def process(self, inputs):
        if self._prepared is None:
            return Result(
                {"image": NO_VALUE, "info": NO_VALUE}, message="No selected frame emitted"
            )
        image, info = self._prepared
        self._prepared = None
        ticket = self._source_ticket
        context = {
            "video_frame": info,
            "spatial": frame(
                [image.shape[1], image.shape[0]],
                f"{self.id}:seek:{ticket['session']}:{ticket['sequence']}",
            ),
        }
        request_ticket = self.input_payloads.get("time_ms", {}).get("ticket")
        if request_ticket is not None:
            context["video_request_ticket"] = copy.deepcopy(request_ticket)
        return Result(
            {"image": image, "info": info},
            payloads={"image": context, "info": context},
            message=f"{info['filename']} at {info['timestamp_ms']:.6g} ms",
        )
