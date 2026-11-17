"""Ticketed live network video, bounded reconnects and explicit stream-segment identity."""

import copy
import time
import uuid
from urllib.parse import urlsplit

from .core import Block, Port, Result
from .images import COLOR_FORMATS, PIXEL_DEPTHS, format_image
from .network_capture import CaptureFailure, CaptureWorker
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, frame


class NetworkVideo(Block):
    type_id, title, category = "input.network_video", "Network video", "Input"
    block_type, subtype = "Input", "Video"
    outputs = (Port("image"), Port("info", "data"), Port("stream_start", "boolean", required=False))
    defaults = {
        "url": "",
        "source_name": "stream",
        "backend": "ffmpeg",
        "rtsp_transport": "tcp",
        "startup_timeout_ms": 5000,
        "open_timeout_ms": 3000,
        "read_timeout_ms": 1000,
        "reconnect_attempts": 2,
        "reconnect_delay_ms": 250,
        "decoder_threads": 1,
        "maximum_frame_pixels": 16_000_000,
        "maximum_frame_bytes": 48_000_000,
        "color_format": "bgr",
        "pixel_depth": "uint8",
    }
    parameter_choices = {
        "backend": ("ffmpeg",),
        "rtsp_transport": ("tcp", "udp"),
        "color_format": COLOR_FORMATS,
        "pixel_depth": PIXEL_DEPTHS,
    }
    parameter_ranges = {
        "startup_timeout_ms": NumericRange(100, 60000, whole=True),
        "open_timeout_ms": NumericRange(50, 60000, whole=True),
        "read_timeout_ms": NumericRange(50, 60000, whole=True),
        "reconnect_attempts": NumericRange(0, 20, whole=True),
        "reconnect_delay_ms": NumericRange(0, 60000, whole=True),
        "decoder_threads": NumericRange(1, 16, whole=True),
        "maximum_frame_pixels": NumericRange(1, 64_000_000, whole=True),
        "maximum_frame_bytes": NumericRange(3, 192_000_000, whole=True),
    }
    parameter_help = {
        "url": "HTTP, HTTPS or RTSP stream. Saved as plain project text; avoid embedded secrets.",
        "source_name": "Public result label, without the URL or credentials.",
        "backend": "Explicit FFmpeg in an owned decoder process; no automatic backend fallback.",
        "rtsp_transport": "TCP or UDP for RTSP. Ignored for HTTP/HTTPS.",
        "startup_timeout_ms": "Worker initialization deadline, separate from stream opening.",
        "open_timeout_ms": "Per-connection opening deadline, checked by parent and FFmpeg.",
        "read_timeout_ms": "Per-frame read deadline, checked by parent and FFmpeg.",
        "reconnect_attempts": "Retries after an initial open/read failure in each activation.",
        "reconnect_delay_ms": "Cancellable wait between connection attempts.",
        "maximum_frame_pixels": "Reject larger decoded frames. Native allocations excluded.",
        "maximum_frame_bytes": "Shared BGR slot. Native and converted buffers excluded.",
        "color_format": "Convert decoded uint8 BGR; automatic display rotation is disabled.",
        "pixel_depth": "Output conversion, not recovery of encoded high-bit-depth samples.",
    }
    output_help = {
        "info": "Frame/connection indices and decoder time. Host receipt is not exposure time.",
        "stream_start": "True on first frame of each connection. Wire to sequence Reset if needed.",
    }
    preview_safe = False
    requires_serial_execution = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._worker = None
        self._prepared = None
        self.reset_runtime()
        self.exposed_outputs = [p for p in self.exposed_outputs if p != "stream_start"]
        self.description = self.description or "Read a live stream with bounded recovery and Stop."
        self.documentation = self.documentation or (
            "<p>One frame per activation; no application read-ahead. Reconnection and Stop/Resume "
            "open a new segment and publish Stream start. See docs/network-video.md for deadlines, "
            "timestamp provenance, buffering, credentials and resource limits.</p>"
        )

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for name in self.parameter_ranges:
            if type(p[name]) is not int:
                raise ValidationError(f"{name} must be a JSON integer", name)
        url = p["url"]
        try:
            if (
                not isinstance(url, str)
                or not 1 <= len(url) <= 4096
                or len(url.encode("utf8")) > 4096
                or any(ord(c) <= 32 for c in url)
            ):
                raise ValueError()
            parsed = urlsplit(url)
            if parsed.scheme.lower() not in ("http", "https", "rtsp") or not parsed.hostname:
                raise ValueError()
            if parsed.fragment or parsed.port == 0:
                raise ValueError()
        except ValueError, UnicodeError:
            raise ValidationError(
                "Select an HTTP, HTTPS or RTSP URL without a fragment", "url"
            ) from None
        name = p["source_name"]
        if not isinstance(name, str) or not 1 <= len(name) <= 80 or any(ord(c) < 32 for c in name):
            raise ValidationError("Source name needs 1..80 printable characters", "source_name")
        if p["pixel_depth"] == "uint16" and p["color_format"] in ("hsv", "lab"):
            raise ValidationError("HSV/Lab need uint8 or float32", "pixel_depth", "color_format")
        if self.recipe_bindings:
            raise ValidationError("Live acquisition settings cannot be recipe-bound")

    def close(self):
        if self._worker is not None:
            self._worker.close()
            self._worker = None
        self._prepared = None

    def reset_runtime(self):
        self.close()
        self._stream = {"settings": None, "next": 0, "connection": 0, "session": uuid.uuid4().hex}

    def snapshot_runtime(self):
        return copy.deepcopy(self._stream)

    def copy_runtime_from(self, other):
        state = other.snapshot_runtime()
        if self._stream != state:
            self.close()
            self._stream = state

    def issue_source_ticket(self):
        p = self.parameters
        self._prepared = None
        if self._stream["settings"] != p:
            self.reset_runtime()
            self._stream["settings"] = copy.deepcopy(p)
        try:
            for attempt in range(p["reconnect_attempts"] + 1):
                if self._cancelled():
                    raise InterruptedError()
                try:
                    if self._worker is None:
                        self._worker = CaptureWorker()
                        self._worker.open(p, self._cancelled)
                        self._stream["connection"] += 1
                    raw, info = self._worker.read(p, self._cancelled)
                    image = format_image(raw, p["color_format"], p["pixel_depth"])
                    if self._cancelled():
                        raise InterruptedError()
                    info.update(
                        source_name=p["source_name"],
                        frame_index=self._stream["next"],
                        connection=self._stream["connection"],
                        retries=attempt,
                        stream_start=info["connection_frame_index"] == 0,
                    )
                    ticket = super().issue_source_ticket()
                    self._prepared = image, info
                    self._stream["next"] += 1
                    return ticket
                except CaptureFailure as exc:
                    self.close()
                    if not exc.retryable or attempt == p["reconnect_attempts"]:
                        raise
                    self.logger.warning(
                        "Network video retry %s after %s",
                        attempt + 1,
                        exc.code,
                        parameters={"retry": attempt + 1, "code": exc.code},
                    )
                    end = time.monotonic() + p["reconnect_delay_ms"] / 1000
                    while time.monotonic() < end:
                        if self._cancelled():
                            raise InterruptedError()
                        time.sleep(min(0.01, max(0, end - time.monotonic())))
        except InterruptedError:
            self.close()
            return None
        except Exception:
            self.close()
            raise

    def process(self, inputs):
        if self._prepared is None:
            return Result(
                {p.name: NO_VALUE for p in self.outputs}, message="No network frame emitted"
            )
        image, info = self._prepared
        self._prepared = None
        grid = f"{self.id}:network:{self._stream['session']}:{info['connection']}"
        return Result(
            {"image": image, "info": info, "stream_start": info["stream_start"]},
            payloads={
                "image": {
                    "network_frame": info,
                    "spatial": frame([image.shape[1], image.shape[0]], grid),
                }
            },
            message=(
                f"{info['source_name']} frame {info['frame_index']} "
                f"/ connection {info['connection']}"
            ),
        )
