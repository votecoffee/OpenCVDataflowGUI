"""Continuous centered stabilization with bounded look-ahead and packet provenance."""

import copy
import hashlib
import json
import threading
import time
from uuid import uuid4

import numpy as np

from .buffered_stream import BufferedStreamBlock
from .control_flow import byte_size
from .core import Port, Result
from .images import ImageArray, image_format
from .offline_stabilization import StabilizeRecordedSequence
from .outcomes import failed_input, source_info
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, ValuePacket, check_spatial, frame
from .processing import gray_image, image_value
from .selection import selection_mask
from .stabilization import (
    _estimate_pose,
    _features,
    _pose_matrix,
    _transform_payload,
    _warp_correction,
)

_PRIMARY = ("image", "coverage", "transform", "report")


class StabilizeStream(BufferedStreamBlock):
    type_id, title = "sequence.stabilize_stream", "Stabilize continuous stream"
    category, block_type, subtype = "Sequence", "Sequence", "Temporal processing"
    sequence_memory = True
    process_inputs_readonly = True
    payload_reference = "image"
    inputs = (
        Port("image", required=False, required_when_connected=True),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
        Port("reset", "data", required=False),
        Port("flush", "data", required=False),
    )
    outputs = (
        Port("image"),
        Port("coverage", visual_type="region"),
        Port("transform", "data"),
        Port("report", "data"),
        Port("ready", "data"),
        Port("buffered", "number"),
        Port("status", "data"),
    )
    flow_primary = _PRIMARY
    defaults = {
        **StabilizeRecordedSequence.defaults,
        "max_bytes_mb": 128.0,
        "max_gap_ms": 0.0,
    }
    parameter_choices = StabilizeRecordedSequence.parameter_choices
    parameter_ranges = {
        **StabilizeRecordedSequence.parameter_ranges,
        "max_bytes_mb": NumericRange(0.01, 1024),
        "max_gap_ms": NumericRange(0, 86_400_000),
    }
    parameter_help = {
        **StabilizeRecordedSequence.parameter_help,
        "radius_frames": "Look-ahead in frames. Retain at most 2 × radius + 1 source frames.",
        "max_bytes_mb": "State budget; native work buffers and GUI snapshots use extra memory.",
        "max_gap_ms": "Start a new segment after this arrival gap. Zero disables the gap limit.",
        "min_coverage_fraction": "Reject the affected frame, retaining its ticket in Report.",
    }
    input_help = {
        "image": "One ordered camera stream. Each output keeps its original ticket and timestamp.",
        "mask": "Optional static-background selection for tracking.",
        "region": "Optional static-background region in the raw image grid.",
        "reset": "Boolean pulse discarding buffered history before accepting this input.",
        "flush": "Boolean pulse emitting the tail. Finite playback flushes automatically.",
    }

    def __init__(self, **kwargs):
        self._memory_lock = threading.RLock()
        super().__init__(**kwargs)
        self.reset_runtime()

    def reset_runtime(self):
        with self._memory_lock:
            self.runtime = {
                "epoch": uuid4().hex,
                "segment": -1,
                "received": 0,
                "cursor": 0,
                "entries": [],
                "context": None,
                "gray": None,
                "points": None,
                "last_pose": None,
                "last_time": None,
                "last_video_index": None,
            }
            self.end_of_input = False

    def close(self):
        self.reset_runtime()

    def snapshot_runtime(self):
        with self._memory_lock:
            return copy.deepcopy(self.runtime)

    def copy_runtime_from(self, other):
        snapshot = other.snapshot_runtime()
        with self._memory_lock:
            self.runtime = snapshot

    @property
    def pending_output(self):
        with self._memory_lock:
            return self.runtime["cursor"] < self.runtime["received"]

    def now(self):
        return time.monotonic()

    @property
    def pending_tickets(self):
        with self._memory_lock:
            return tuple(
                copy.deepcopy(entry["payload"]["ticket"])
                for entry in self.runtime["entries"]
                if entry["index"] >= self.runtime["cursor"] and "ticket" in entry["payload"]
            )

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if int(p["min_inliers"]) > int(p["max_features"]):
            raise ValidationError("Minimum inliers exceeds maximum features", "min_inliers")

    def _preflight(self, image, payload):
        image_value(image, (np.uint8, np.uint16, np.float32, np.float64))
        height, width = image.shape[:2]
        if min(height, width) < 8 or max(height, width) >= 32767 or height * width > 16_777_216:
            raise ValidationError("Use 8..32766 pixels per side and at most 16,777,216 pixels")
        if image_format(image) not in ("gray", "bgr", "rgb", "bgra", "rgba"):
            raise ValidationError("Use grayscale or RGB/BGR image encoding")
        check_spatial(payload, [width, height])
        radius = int(self.parameters["radius_frames"])
        metadata_bytes = len(json.dumps(payload, allow_nan=False).encode("utf-8"))
        required = (2 * radius + 1) * (image.nbytes + metadata_bytes + 4096)
        required += 2 * width * height + int(self.parameters["max_features"]) * 16 + 8192
        if required > float(self.parameters["max_bytes_mb"]) * 1024**2:
            raise ValidationError("Look-ahead would exceed the stored-state memory budget")
        signature = {
            "shape": list(image.shape),
            "dtype": image.dtype.name,
            "encoding": image_format(image),
            "spatial": {
                key: payload.get("spatial", {}).get(key)
                for key in ("root_id", "root_size", "image_size")
            },
            "pyramid": {
                key: payload.get("pyramid", {}).get(key) for key in ("level", "largest_size")
            },
            "recipe_ref": payload.get("recipe_ref"),
            "parameters": self.parameters,
            "source": payload.get("ticket", {}).get("source"),
            "recording": payload.get("video_frame", {}).get("filename"),
        }
        return hashlib.sha256(
            json.dumps(signature, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    def _packet_payload(self, original, passed, reasons=()):
        payload = copy.deepcopy(original)
        payload["outcome"] = {
            "model": "tool_outcome_v1",
            "passed": passed,
            "chain_passed": passed and not failed_input(ValuePacket(None, original)),
            "source": source_info(self),
            "score": None,
            "fail_reasons": [*original.get("outcome", {}).get("fail_reasons", []), *reasons],
        }
        return payload

    def _emit_one(self, emissions):
        state = self.runtime
        p = state["parameters"]
        index = state["cursor"]
        entries = state["entries"]
        radius = int(p["radius_frames"])
        target_entry = next(entry for entry in entries if entry["index"] == index)
        window = [entry for entry in entries if abs(entry["index"] - index) <= radius]
        weights = np.exp(
            -0.5
            * (np.array([entry["index"] - index for entry in window]) / float(p["sigma_frames"]))
            ** 2
        )
        target = np.average([entry["pose"] for entry in window], axis=0, weights=weights)
        image, observed = target_entry["image"], target_entry["pose"]
        size = image.shape[1::-1]
        center = (np.asarray(size) - 1) / 2
        target_pose = _pose_matrix(target, center)
        correction = target_pose @ np.linalg.inv(_pose_matrix(observed, center))
        output, valid, coverage = _warp_correction(image, correction, p)
        accepted = output is not None
        reasons = (
            []
            if accepted
            else [
                {
                    "code": "stream_stabilization_insufficient_coverage",
                    "message": "Corrected frame falls below the configured coverage fraction",
                }
            ]
        )
        payload = self._packet_payload(target_entry["payload"], accepted, reasons)
        report = {
            "model": "stream_stabilization_v1",
            **target_entry["details"],
            "index": index,
            "segment": state["segment"],
            "window": [window[0]["index"], window[-1]["index"] + 1],
            "look_ahead_frames": window[-1]["index"] - index,
            "observed_pose": observed.tolist(),
            "smoothed_pose": target.tolist(),
            "coverage_fraction": coverage,
            "status": "complete" if accepted else "insufficient_coverage",
        }
        if accepted:
            spatial = frame(size, f"{self.id}:stream:{state['epoch']}:{state['segment']}")
            spatial["to_root"] = np.linalg.inv(target_pose).tolist()
            transform = _transform_payload(correction, size, payload, spatial)
            payload["spatial"] = spatial
            payload.pop("pyramid", None)
            for name, value in (
                ("image", ImageArray(output, image_format(image))),
                ("coverage", valid),
                ("transform", transform),
            ):
                emissions[name].append(ValuePacket(value, copy.deepcopy(payload)))
        emissions["report"].append(ValuePacket(report, payload))
        state["cursor"] += 1
        state["entries"] = [
            entry for entry in entries if entry["index"] >= state["cursor"] - radius
        ]

    def _drain(self, emissions):
        while self.pending_output and not self._cancelled():
            self._emit_one(emissions)
        if not self.pending_output:
            self.runtime.update(entries=[], context=None, gray=None, points=None, last_pose=None)

    def _start_segment(self, signature):
        state = self.runtime
        state.update(
            entries=[],
            cursor=state["received"],
            context=signature,
            parameters=copy.deepcopy(self.parameters),
            gray=None,
            points=None,
            last_pose=None,
        )
        state["segment"] += 1

    def process(self, inputs):
        with self._memory_lock:
            try:
                return self._process(inputs)
            except Exception:
                self.reset_runtime()
                raise

    def _process(self, inputs):
        for name in ("reset", "flush"):
            if type(inputs.get(name, False)) is not bool:
                raise ValidationError(f"{name.title()} input must be boolean")
        emissions = {name: [] for name in _PRIMARY}
        discarded = 0
        if inputs.get("reset", False):
            discarded = self.runtime["received"] - self.runtime["cursor"]
            end = self.end_of_input
            self.reset_runtime()
            self.end_of_input = end
        if "image" in inputs and not self._cancelled():
            image = inputs["image"]
            payload = self.input_payloads["image"]
            signature = self._preflight(image, payload)
            gray = gray_image(image, (np.uint8,), auto_depth=True)
            mask = selection_mask(inputs, image)
            state = self.runtime
            now = self.now()
            gap = float(self.parameters["max_gap_ms"]) / 1000
            video_index = payload.get("video_frame", {}).get("frame_index")
            looped = (
                video_index is not None
                and state["last_video_index"] is not None
                and video_index <= state["last_video_index"]
            )
            changed = (
                signature != state["context"]
                or looped
                or (gap and state["last_time"] is not None and now - state["last_time"] > gap)
            )
            if changed:
                self._drain(emissions)
                if self.pending_output:
                    return self._result(emissions, discarded)
                self._start_segment(signature)
            observed = np.zeros(4)
            details = {"tracked": 0, "inliers": 0, "inlier_ratio": 0.0, "reason": "initial_frame"}
            if state["gray"] is not None:
                observed, _, reason = _estimate_pose(
                    state["gray"],
                    gray,
                    state["points"],
                    mask,
                    (np.asarray(image.shape[1::-1]) - 1) / 2,
                    state["last_pose"],
                    self.parameters,
                    details,
                )
                details["reason"] = reason or "tracked"
                if reason is not None:
                    self._drain(emissions)
                    if self.pending_output:
                        return self._result(emissions, discarded)
                    self._start_segment(signature)
                    observed = np.zeros(4)
            state["entries"].append(
                {
                    "index": state["received"],
                    "image": image.copy(),
                    "payload": copy.deepcopy(payload),
                    "pose": observed,
                    "details": details,
                }
            )
            state["received"] += 1
            state.update(
                gray=gray.copy(),
                points=_features(gray, mask, self.parameters, "max_features"),
                last_pose=observed,
                last_time=now,
                last_video_index=video_index,
            )
            if byte_size(state) > float(self.parameters["max_bytes_mb"]) * 1024**2:
                raise ValidationError("Stored-state memory budget exceeded")
            radius = int(self.parameters["radius_frames"])
            while state["cursor"] + radius < state["received"] and not self._cancelled():
                self._emit_one(emissions)
        if inputs.get("flush", False) or self.end_of_input:
            self._drain(emissions)
        return self._result(emissions, discarded)

    def _result(self, emissions, discarded):
        state = self.runtime
        buffered = state["received"] - state["cursor"]
        ready = bool(emissions["image"])
        status = {
            "model": "buffered_stream_status_v1",
            "ready": ready,
            "buffered": buffered,
            "received": state["received"],
            "stored_frames": len(state["entries"]),
            "stored_bytes": byte_size(state),
            "reset_discarded": discarded,
            "end_of_input": bool(self.end_of_input),
        }
        failed = any(packet.value["status"] != "complete" for packet in emissions["report"])
        outputs = {
            name: packets[-1].value if packets else NO_VALUE for name, packets in emissions.items()
        }
        outputs.update(ready=ready, buffered=buffered, status=status)
        payloads = {name: packets[-1].payload for name, packets in emissions.items() if packets}
        if "image" in self.input_payloads:
            accepted = self._packet_payload(self.input_payloads["image"], True)
            payloads.update(ready=accepted, buffered=accepted, status=accepted)
        return Result(
            outputs,
            emissions=emissions,
            payloads=payloads,
            passed=not failed,
            message=f"Emitted {len(emissions['image'])} frames · {buffered} buffered",
        )
