"""Bounded immutable video streams decoded by the native FFmpeg IStreamReader API."""

import copy
import hashlib
import io
import math
import os
from pathlib import Path

import cv2
import numpy as np

from .core import Block, Port, Result
from .images import COLOR_FORMATS, PIXEL_DEPTHS, format_image
from .parameter_validation import NumericRange, ValidationError
from .payloads import FlowGroup, ValuePacket, frame
from .tickets import TicketClock

MIB = 1024**2


class BoundedVideoReader(io.BufferedIOBase):
    """Native callbacks return bytes/positions, recording failures outside the bridge."""

    def __init__(self, data, chunk, maximum_reads, maximum_calls, cancelled=lambda: False):
        super().__init__()
        self.data, self.chunk = data, chunk
        self.maximum_reads, self.maximum_calls = maximum_reads, maximum_calls
        self.cancelled = cancelled
        self.position = self.calls = self.bytes_read = self.short_reads = self.seeks = 0
        self.failure = ""

    def permit(self):
        if self.closed or self.failure:
            return False
        self.calls += 1
        if self.calls > self.maximum_calls:
            self.failure = "Video stream exceeded Maximum IO calls"
        elif self.cancelled():
            self.failure = "Video stream decoding cancelled"
        return not self.failure

    def read(self, size=-1):
        try:
            if not self.permit():
                return b""
            wanted = self.chunk if size < 0 else size
            size = min(wanted, self.chunk, len(self.data) - self.position)
            if self.bytes_read + size > self.maximum_reads:
                self.failure = "Video stream exceeded Maximum read bytes"
                return b""
            result = self.data[self.position : self.position + size]
            self.position += size
            self.bytes_read += size
            self.short_reads += int(0 < size < wanted)
            return result
        except Exception as exc:
            # Do not raise through the SDK's Python-to-IStreamReader callback bridge.
            self.failure = f"Video stream read callback failed: {type(exc).__name__}"
            return b""

    def seek(self, offset, whence=io.SEEK_SET):
        try:
            if not self.permit():
                return -1
            self.seeks += 1
            if whence not in (io.SEEK_SET, io.SEEK_CUR, io.SEEK_END):
                self.failure = "Video stream requested an unsupported seek origin"
                return -1
            start = (0, self.position, len(self.data))[whence]
            position = start + offset
            if not 0 <= position <= len(self.data):
                self.failure = "Video stream requested a seek outside the encoded buffer"
                return -1
            self.position = position
            return position
        except Exception as exc:
            self.failure = f"Video stream seek callback failed: {type(exc).__name__}"
            return -1

    def check(self):
        if self.failure:
            raise ValidationError(self.failure)

    def close(self):
        self.data = b""
        super().close()

    def describe(self):
        return {
            "calls": self.calls,
            "bytes_read": self.bytes_read,
            "short_reads": self.short_reads,
            "seeks": self.seeks,
            "chunk_bytes": self.chunk,
            "closed": self.closed,
        }


def encoded_bytes(value, maximum):
    if isinstance(value, np.ndarray):
        if (
            value.dtype != np.uint8
            or value.ndim not in (1, 2)
            or (value.ndim == 2 and min(value.shape) != 1)
        ):
            raise ValidationError("Encoded video must be bytes or a uint8 vector")
        size = value.nbytes
    elif isinstance(value, (bytes, bytearray)):
        size = len(value)
    else:
        raise ValidationError("Encoded video must be bytes or a uint8 vector")
    if not 1 <= size <= maximum:
        raise ValidationError("Encoded video exceeds Maximum encoded bytes or is empty")
    return value.tobytes() if isinstance(value, np.ndarray) else bytes(value)


def timestamp(capture, index, fps, policy):
    value = capture.get(cv2.CAP_PROP_POS_MSEC)
    if math.isfinite(value) and value >= 0 and not (index and value == 0):
        return value, "decoder"
    if policy == "decoder_or_fps" and fps is not None:
        return index * 1000.0 / fps, "frame_rate_estimate"
    if policy == "allow_missing":
        return None, "unavailable"
    raise ValidationError("Video presentation timestamp unavailable under Timestamp policy")


class DecodeVideo(Block):
    type_id, title, category = "cv.decode_video", "Decode encoded video", "Image processing"
    block_type, subtype = "Image processing", "Video decoding"
    inputs = (Port("encoded", "any", required=False),)
    outputs = (
        Port("frames", "any"),
        Port("image"),
        Port("frame_count", "integer"),
        Port("info", "data"),
    )
    defaults = {
        "path": "",
        "path_base": "working_directory",
        "backend": "ffmpeg",
        "start_frame": 0,
        "frame_count": 16,
        "stride": 1,
        "preview_frame": 0,
        "short_clip": "allow",
        "timestamp_policy": "decoder_or_fps",
        "color_format": "bgr",
        "pixel_depth": "uint8",
        "orientation_auto": False,
        "decoder_threads": 1,
        "open_timeout_ms": 5000,
        "read_timeout_ms": 5000,
        "maximum_encoded_bytes": 64 * MIB,
        "maximum_frame_pixels": 16_000_000,
        "maximum_decoded_bytes": 256 * MIB,
        "maximum_scan_frames": 4096,
        "read_chunk_bytes": 4096,
        "maximum_read_bytes": 256 * MIB,
        "maximum_io_calls": 100000,
    }
    parameter_choices = {
        "path_base": ("working_directory", "project"),
        "backend": ("ffmpeg",),
        "short_clip": ("allow", "error"),
        "timestamp_policy": ("decoder_or_fps", "decoder_only", "allow_missing"),
        "color_format": COLOR_FORMATS,
        "pixel_depth": PIXEL_DEPTHS,
    }
    parameter_ranges = {
        "start_frame": NumericRange(0, 99999, whole=True),
        "frame_count": NumericRange(1, 64, whole=True),
        "stride": NumericRange(1, 4096, whole=True),
        "preview_frame": NumericRange(0, 63, whole=True),
        "decoder_threads": NumericRange(1, 16, whole=True),
        "open_timeout_ms": NumericRange(1, 60000, whole=True),
        "read_timeout_ms": NumericRange(1, 60000, whole=True),
        "maximum_encoded_bytes": NumericRange(1, 256 * MIB, whole=True),
        "maximum_frame_pixels": NumericRange(1, 64_000_000, whole=True),
        "maximum_decoded_bytes": NumericRange(1, 512 * MIB, whole=True),
        "maximum_scan_frames": NumericRange(1, 100000, whole=True),
        "read_chunk_bytes": NumericRange(1, MIB, whole=True),
        "maximum_read_bytes": NumericRange(1, 1024 * MIB, whole=True),
        "maximum_io_calls": NumericRange(1, 1000000, whole=True),
    }
    parameter_help = {
        "path": "Optional file snapshot, used only when Encoded is unconnected.",
        "backend": "Explicit FFmpeg IStreamReader; no fallback to files or other backends.",
        "frame_count": "Maximum selected frames in the output group (1..64).",
        "start_frame": "Zero-based ordinal; decoding scans from the start, without frame seeks.",
        "stride": "Select every Nth decoded frame, starting at Start frame.",
        "preview_frame": "Position inside the selected group, not the original frame index.",
        "short_clip": "Allow fewer frames, or reject if the requested count is unavailable.",
        "timestamp_policy": "Presentation time; labeled FPS fallback. Not camera acquisition time.",
        "color_format": "Native uint8 BGR converted using normal image format rules.",
        "pixel_depth": "Output conversion, not recovery of encoded high-bit-depth video samples.",
        "orientation_auto": "Native orientation flag; rotated geometry must pass size checks.",
        "maximum_decoded_bytes": (
            "Selected outputs plus estimated conversion buffers; excludes native decoder internals "
            "and runtime snapshots. Not a hard RSS bound."
        ),
        "maximum_scan_frames": "Cap selected and skipped decoded frames per activation.",
        "read_chunk_bytes": "Maximum bytes per native read callback; supports short reads.",
        "maximum_read_bytes": "Cumulative bytes served, including decoder rereads after seeks.",
        "maximum_io_calls": "Combined read/seek callback limit; counters do not retain a call log.",
        "open_timeout_ms": "Native FFmpeg opening timeout; not a hard execution deadline.",
        "read_timeout_ms": "Native FFmpeg read timeout; cancellation is cooperative.",
    }
    input_help = {"encoded": "Immutable snapshot of bytes/bytearray or a uint8 vector."}
    output_help = {
        "frames": "Ordered group with fresh per-frame tickets, presentation times and pixel grids.",
        "image": "Selected preview; carries exactly that frame's payload and ticket.",
    }
    preview_safe = parallel_safe = process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for name in self.parameter_ranges:
            if type(p[name]) is not int:
                raise ValidationError(f"{name} must be a JSON integer", name)
        if not isinstance(p["path"], str) or len(p["path"]) > 4096:
            raise ValidationError("Path must be text up to 4096 characters", "path")
        if type(p["orientation_auto"]) is not bool:
            raise ValidationError("Orientation auto must be boolean", "orientation_auto")
        if p["pixel_depth"] == "uint16" and p["color_format"] in ("hsv", "lab"):
            raise ValidationError(
                "HSV and Lab require uint8 or float32 output", "pixel_depth", "color_format"
            )
        last = p["start_frame"] + (p["frame_count"] - 1) * p["stride"]
        if last >= p["maximum_scan_frames"]:
            raise ValidationError(
                "Requested range exceeds Maximum scan frames",
                "maximum_scan_frames",
                "start_frame",
                "frame_count",
                "stride",
            )
        if p["preview_frame"] >= p["frame_count"]:
            raise ValidationError("Preview frame must be below Frame count", "preview_frame")

    def load_bytes(self, inputs):
        p = self.parameters
        if "encoded" in inputs:
            return encoded_bytes(inputs["encoded"], p["maximum_encoded_bytes"]), "buffer"
        if not p["path"].strip():
            raise ValidationError("Connect Encoded or select a video Path", "path")
        path = Path(p["path"])
        if p["path_base"] == "project" and not path.is_absolute():
            if self.project_dir is None:
                raise ValidationError("Save/open the project to resolve its video Path", "path")
            path = self.project_dir / path
        with path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            if not 1 <= size <= p["maximum_encoded_bytes"]:
                raise ValidationError(
                    "Video file exceeds Maximum encoded bytes or is empty", "maximum_encoded_bytes"
                )
            data = stream.read(size + 1)
            if len(data) != size:
                raise ValidationError("Video file size changed during the snapshot", "path")
        return encoded_bytes(data, p["maximum_encoded_bytes"]), path.name

    def process(self, inputs):
        p = self.parameters
        data, label = self.load_bytes(inputs)
        digest = hashlib.sha256(data).hexdigest()
        reader = BoundedVideoReader(
            data,
            p["read_chunk_bytes"],
            p["maximum_read_bytes"],
            p["maximum_io_calls"],
            self._cancelled,
        )
        capture, packets, indexes, times = None, [], [], []
        clock = TicketClock()
        context = copy.deepcopy(self.input_payloads.get("encoded", {}))
        parent = context.pop("ticket", None)
        for key in ("spatial", "pyramid", "raw_sensor", "alpha", "video_frame", "processing_time"):
            context.pop(key, None)
        try:
            capture = cv2.VideoCapture()
            opened = capture.open(
                reader,
                cv2.CAP_FFMPEG,
                [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                    p["open_timeout_ms"],
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                    p["read_timeout_ms"],
                    cv2.CAP_PROP_N_THREADS,
                    p["decoder_threads"],
                ],
            )
            reader.check()
            if not opened:
                raise ValidationError("FFmpeg cannot open this encoded video stream")
            if not capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, int(p["orientation_auto"])):
                raise ValidationError("FFmpeg cannot apply Orientation auto", "orientation_auto")
            width, height = (
                capture.get(key) for key in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)
            )
            if any(not math.isfinite(n) or n < 1 or n != int(n) for n in (width, height)):
                raise ValidationError("Video header has no usable pixel dimensions")
            width, height = int(width), int(height)
            pixels = width * height
            if pixels > p["maximum_frame_pixels"]:
                raise ValidationError(
                    "Video header exceeds Maximum frame pixels", "maximum_frame_pixels"
                )
            # Upper bound for supported output formats plus a conversion buffer and native BGR Mat.
            item_bytes = {
                "uint8": 1,
                "uint16": 2,
                "float32": 4,
                "unchanged": 1,
            }[p["pixel_depth"]]
            channels = {
                "gray": 1,
                "rgb": 3,
                "bgr": 3,
                "rgba": 4,
                "bgra": 4,
                "hsv": 3,
                "lab": 3,
                "unchanged": 3,
            }[p["color_format"]]
            estimate = pixels * (p["frame_count"] * channels * item_bytes + 3 + channels * 8)
            if estimate > p["maximum_decoded_bytes"]:
                raise ValidationError(
                    "Video outputs exceed Maximum decoded bytes",
                    "maximum_decoded_bytes",
                    "frame_count",
                )
            fps = capture.get(cv2.CAP_PROP_FPS)
            fps = fps if math.isfinite(fps) and fps > 0 else None
            last = p["start_frame"] + (p["frame_count"] - 1) * p["stride"]
            scanned = 0
            for index in range(last + 1):
                if self._cancelled():
                    raise ValidationError("Video stream decoding cancelled")
                ok, raw = capture.read()
                reader.check()
                if not ok:
                    break
                scanned += 1
                if raw.dtype != np.uint8 or raw.shape != (height, width, 3):
                    raise ValidationError("Decoded video changed its uint8 BGR header layout")
                if index < p["start_frame"] or (index - p["start_frame"]) % p["stride"]:
                    continue
                value, origin = timestamp(capture, index, fps, p["timestamp_policy"])
                info = {
                    "frame_index": index,
                    "timestamp_ms": value,
                    "timestamp_source": origin,
                    "fps": fps,
                    "backend": capture.getBackendName(),
                    "buffer_sha256": digest,
                }
                ticket = clock.issue(self.id)
                ticket.update(kind="string", value=f"{label}#{index:06d}")
                payload = {
                    **copy.deepcopy(context),
                    "ticket": ticket,
                    "spatial": frame([width, height], f"{self.id}:{clock.session}:image"),
                    "video_frame": info,
                }
                if parent is not None:
                    payload["video_source_ticket"] = copy.deepcopy(parent)
                image = format_image(raw, p["color_format"], p["pixel_depth"])
                image.metadata = copy.deepcopy(payload)
                packets.append(ValuePacket(image, payload))
                indexes.append(index)
                times.append(info)
            if not packets or p["short_clip"] == "error" and len(packets) != p["frame_count"]:
                raise ValidationError("Video ended or decoding failed before the requested frames")
            if p["preview_frame"] >= len(packets):
                raise ValidationError("Preview frame is outside the decoded group", "preview_frame")
        finally:
            try:
                if capture is not None:
                    capture.release()
            finally:
                reader.close()
        group = FlowGroup(tuple(packets), tuple(f"frame_{i:06d}" for i in indexes))
        preview = packets[p["preview_frame"]]
        info = {
            "source": label,
            "buffer_sha256": digest,
            "encoded_bytes": len(data),
            "backend": "FFMPEG",
            "selected_indexes": indexes,
            "frames": times,
            "scanned_frames": scanned,
            "requested_count": p["frame_count"],
            "selection_satisfied": len(packets) == p["frame_count"],
            "stop_reason": "selection_complete"
            if len(packets) == p["frame_count"]
            else "native_end_or_decode_failure",
            "workspace_estimate_bytes": estimate,
            "reader": reader.describe(),
        }
        return Result(
            {"frames": group, "image": preview.value, "frame_count": len(packets), "info": info},
            payloads={"image": preview.payload},
            message=f"Decoded {len(packets)} video frames via bounded FFmpeg stream",
        )
