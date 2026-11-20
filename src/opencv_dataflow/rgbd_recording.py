"""Timestamp-paired playback of lossless RGB-D recordings."""

import copy
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .cloud_projection import MAX_PIXELS
from .core import Port, Result
from .depth_smoothing import normalize_depth_confidence
from .images import DepthArray, ImageArray
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, frame
from .replay import ReplaySource

_MAX_MANIFEST_BYTES = 8 * 1024**2
_MAX_PNG_BYTES = 64 * 1024**2
_MAX_RECORDS = 20_000
_PNG_HEADER = b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR"


@dataclass(frozen=True)
class Sample:
    timestamp_ns: int
    image: str
    confidence: str | None = None


def _bounded_read(path, limit):
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if not 0 < len(data) <= limit:
        raise ValidationError(f"{path.name} is empty or exceeds the recording size limit")
    return data


def _png_path(value):
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 1024
        or "\x00" in value
        or Path(value).anchor
        or ":" in value
        or ".." in Path(value).parts
        or Path(value).suffix.lower() != ".png"
    ):
        raise ValidationError("Frame paths must be relative PNG files inside the recording folder")
    return value


def _decode(directory, name, kind):
    path = (directory / name).resolve()
    if not path.is_relative_to(directory):
        raise ValidationError("Frame paths must stay inside the recording folder")
    data = _bounded_read(path, _MAX_PNG_BYTES)
    if len(data) < 33 or not data.startswith(_PNG_HEADER):
        raise ValidationError(f"Invalid PNG header in {name}")
    width, height = struct.unpack_from(">II", data, len(_PNG_HEADER))
    if not width or not height or width * height > MAX_PIXELS:
        raise ValidationError(f"{name} exceeds the 8,388,608 pixel recording limit")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.shape[:2] != (height, width):
        raise ValidationError(f"Could not decode recorded frame {name}")
    if kind == "color":
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype not in (np.uint8, np.uint16):
            raise ValidationError("Color frames must be three-channel uint8/uint16 PNGs")
    elif image.ndim != 2 or image.dtype not in (
        (np.uint16,) if kind == "depth" else (np.uint8, np.uint16)
    ):
        raise ValidationError(
            f"Invalid {kind} encoding. Use uint16 depth and uint8/uint16 confidence"
        )
    return image


def _samples(records, kind):
    if not isinstance(records, list) or not 1 <= len(records) <= _MAX_RECORDS:
        raise ValidationError(f"{kind} stream must contain 1..20,000 records")
    previous = -1
    samples = []
    for record in records:
        if not isinstance(record, dict):
            raise ValidationError(f"Invalid {kind} frame record")
        stamp = record.get("timestamp_ns")
        if type(stamp) is not int or not previous < stamp <= 2**63 - 1:
            raise ValidationError(
                f"{kind} timestamps must be nonnegative, strictly increasing integers"
            )
        confidence = (
            _png_path(record["confidence"]) if kind == "depth" and "confidence" in record else None
        )
        samples.append(Sample(stamp, _png_path(record.get("path")), confidence))
        previous = stamp
    return samples


def _pair(color, depth, maximum_ns):
    """Match the earliest feasible pair in FIFO order, with no reuse."""
    pairs = []
    i = j = 0
    while i < len(color) and j < len(depth):
        delta = depth[j].timestamp_ns - color[i].timestamp_ns
        if abs(delta) <= maximum_ns:
            pairs.append((color[i], depth[j]))
            i += 1
            j += 1
        elif delta > 0:
            i += 1
        else:
            j += 1
    return tuple(pairs)


def _read_manifest(path, maximum_ns):
    manifest = json.loads(_bounded_read(path, _MAX_MANIFEST_BYTES))
    if not isinstance(manifest, dict) or manifest.get("format") != "opencv_dataflow_rgbd_v1":
        raise ValidationError("Choose an opencv_dataflow_rgbd_v1 manifest")
    labels = [manifest.get(name) for name in ("clock", "color_frame", "depth_frame")]
    if any(not isinstance(name, str) or not name.strip() or len(name) > 128 for name in labels):
        raise ValidationError("Name the recording clock and both optical frames")
    scale = manifest.get("depth_m_per_sample")
    if type(scale) not in (int, float) or not 1e-9 <= scale <= 1:
        raise ValidationError("Depth scale must be 1e-9..1 meter per sample")
    color = _samples(manifest.get("color"), "color")
    depth = _samples(manifest.get("depth"), "depth")
    pairs = _pair(color, depth, maximum_ns)
    if not pairs:
        raise ValidationError("No recorded color and depth frames meet the time tolerance")
    clock, color_frame, depth_frame = labels
    return pairs, {
        "clock": clock,
        "color_frame": color_frame,
        "depth_frame": depth_frame,
        "depth_m_per_sample": scale,
        "paired_frames": len(pairs),
        "unmatched_color": len(color) - len(pairs),
        "unmatched_depth": len(depth) - len(pairs),
    }


class RecordedRGBD(ReplaySource):
    type_id, title, category = "input.rgbd_recording", "Recorded RGB-D", "Input"
    block_type, subtype = "Input", "Depth and RGB-D"
    outputs = (
        Port("image"),
        Port("depth", "depth"),
        Port("confidence", required=False),
        Port("info", "data"),
    )
    flow_primary = ("image", "depth")
    project_path_parameters = ("path",)
    defaults = {"path": "", "repeat": "stop", "maximum_skew_ms": 5.0, "display_max_m": 5.0}
    parameter_choices = {"repeat": ("stop", "loop")}
    parameter_ranges = {
        "maximum_skew_ms": NumericRange(0, 1000),
        "display_max_m": NumericRange(0.001, 1000),
    }
    parameter_help = {
        "path": "Manifest JSON relative to the project. Frame paths are relative to the manifest.",
        "repeat": "Stop after the last pair or loop from the first. Run All uses stop.",
        "maximum_skew_ms": "Inclusive timestamp tolerance. Zero requires exact timestamp equality.",
        "display_max_m": "Upper preview bound. Physical depth values are preserved.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_runtime()

    def reset_runtime(self):
        self._sequence = {
            "settings": None,
            "cursor": 0,
            "finished": False,
            "selected": None,
            "selected_index": None,
        }
        self._manifest = None

    def snapshot_runtime(self):
        return copy.deepcopy(self._sequence)

    def restore_runtime(self, snapshot):
        if snapshot["settings"] != self._sequence["settings"]:
            self._manifest = None
        self._sequence = copy.deepcopy(snapshot)

    def copy_runtime_from(self, other):
        self.restore_runtime(other.snapshot_runtime())
        # The manifest cache is read-only after loading and contains no decoded frames.
        self._manifest = other._manifest

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not isinstance(p["path"], str):
            raise ValidationError("Choose a recording manifest path", "path")
        if "repeat" in self.recipe_bindings:
            raise ValidationError("Repeat is a run setting and cannot be recipe-bound", "repeat")

    @property
    def exhausted(self):
        return self.finite_source and self._sequence["finished"] and self._settings_match()

    def _settings_match(self):
        return self._sequence["settings"] == (
            str(self.project_dir),
            {**self.parameters, **self.last_recipe_parameters},
        )

    def issue_source_ticket(self):
        if self._sequence["settings"] != (str(self.project_dir), self.parameters):
            self.reset_runtime()
            self._sequence["settings"] = (str(self.project_dir), dict(self.parameters))
        state = self._sequence
        state["selected"] = state["selected_index"] = None
        if state["finished"] or self._cancelled():
            return None
        try:
            if self._manifest is None:
                path = Path(self.parameters["path"])
                if not path.is_absolute():
                    if self.project_dir is None:
                        raise ValidationError(
                            "Save or open the project before using a relative path"
                        )
                    path = self.project_dir / path
                path = path.resolve()
                if not self.parameters["path"].strip() or path.suffix.lower() != ".json":
                    raise ValidationError("Choose a replay manifest JSON file", "path")
                tolerance = round(float(self.parameters["maximum_skew_ms"]) * 1_000_000)
                self._manifest = (path, *_read_manifest(path, tolerance))
            _, pairs, _ = self._manifest
            if self._cancelled():
                return None
            ticket = super().issue_source_ticket()
            state["selected_index"] = state["cursor"]
            state["selected"] = pairs[state["cursor"]][0].image
            state["cursor"] += 1
            if self.finite_source:
                state["finished"] = state["cursor"] == len(pairs)
            else:
                state["cursor"] %= len(pairs)
            return ticket
        except Exception:
            state["finished"] = True
            raise

    def process(self, inputs):
        index = self._sequence["selected_index"]
        if index is None:
            return Result(
                {port.name: NO_VALUE for port in self.outputs}, message="RGB-D recording complete"
            )
        try:
            return self._read_pair(index)
        except Exception:
            self._sequence["finished"] = True
            raise

    def _read_pair(self, index):
        path, pairs, details = self._manifest
        color, depth = pairs[index]
        color_pixels = _decode(path.parent, color.image, "color")
        depth_pixels = _decode(path.parent, depth.image, "depth")
        depth_m = depth_pixels.astype(np.float32) * details["depth_m_per_sample"]
        depth_m[depth_pixels == 0] = np.nan
        confidence = NO_VALUE
        if depth.confidence is not None:
            raw = _decode(path.parent, depth.confidence, "confidence")
            if raw.shape != depth_pixels.shape:
                raise ValidationError("Recorded confidence must match its depth raster")
            normalized = normalize_depth_confidence(raw)
            normalized[depth_pixels == 0] = 0
            confidence = ImageArray(normalized, "gray")
        info = {
            **details,
            "manifest": path.name,
            "pair_index": index,
            "color_path": color.image,
            "depth_path": depth.image,
            "color_timestamp_ns": color.timestamp_ns,
            "depth_timestamp_ns": depth.timestamp_ns,
            "skew_ns": depth.timestamp_ns - color.timestamp_ns,
            "has_confidence": confidence is not NO_VALUE,
        }
        payloads = {}
        for name, array, stamp in (
            ("image", color_pixels, color.timestamp_ns),
            ("depth", depth_pixels, depth.timestamp_ns),
        ):
            payloads[name] = {
                "spatial": frame(array.shape[:2][::-1], f"{self.id}:{name}"),
                "recorded_rgbd": {
                    "clock": details["clock"],
                    "timestamp_ns": stamp,
                    "pair_index": index,
                },
                "optical_frame": details["color_frame" if name == "image" else "depth_frame"],
            }
        if confidence is not NO_VALUE:
            payloads["confidence"] = payloads["depth"]
        return Result(
            {
                "image": ImageArray(color_pixels, "bgr"),
                "depth": DepthArray(depth_m, "m", 0, float(self.parameters["display_max_m"])),
                "confidence": confidence,
                "info": info,
            },
            payloads=payloads,
            message=f"Pair {index + 1}/{len(pairs)} · {info['skew_ns'] / 1_000_000:+.3f} ms",
        )
