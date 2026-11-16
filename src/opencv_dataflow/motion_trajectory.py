"""Versioned, content-bound affine motion records and stateless clip replay."""

import copy
import hashlib
import json
import re

import numpy as np

from .core import Block, Port, Result
from .flow_files import endpoint, same_endpoint
from .images import ImageArray
from .offline_stabilization import _frames
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, FlowGroup, ValuePacket, check_spatial, validate_payload
from .processing import image_value
from .stabilization import _transform_payload, _warp_correction
from .tickets import ticket_key

MODEL = "image_motion_trajectory_v1"
MAX_BYTES = 1024**2
_PRIMARY = ("corrected", "coverage", "transforms", "image")


def checked_spatial(context, size):
    try:
        check_spatial(context, size)
        validate_payload(context)
    except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
        raise ValidationError("Invalid trajectory coordinate geometry: " + str(exc)) from exc


def trajectory_frames(group):
    try:
        return _frames(group, min_frames=1)
    except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
        raise ValidationError("Invalid trajectory frame group: " + str(exc)) from exc


def finite_json(value):
    try:
        raw = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValidationError("Trajectory must contain finite JSON values") from exc
    if len(raw) > MAX_BYTES:
        raise ValidationError("Trajectory exceeds 1 MiB")
    return raw


def sha(value):
    return hashlib.sha256(finite_json(value)).hexdigest()


def affine(value):
    try:
        matrix = np.asarray(value, np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("Expected a finite 3x3 matrix")
        if (
            not np.allclose(matrix[2], [0, 0, 1], atol=1e-12, rtol=0)
            or np.max(np.abs(matrix)) > 1e6
        ):
            raise ValueError("Use bounded source-to-target affine coordinates")
        singular = np.linalg.svd(matrix[:2, :2], compute_uv=False)
        if singular.min() < 1e-6 or singular.max() > 1e6:
            raise ValueError("Affine scale is singular or outside 1e-6..1e6")
        return matrix
    except (TypeError, ValueError, OverflowError, np.linalg.LinAlgError) as exc:
        raise ValidationError("Invalid trajectory affine transform: " + str(exc)) from exc


def source_schema(value):
    if not isinstance(value, dict) or value.keys() - {"label", "image", "spatial", "pyramid"}:
        raise ValidationError("Invalid trajectory source record")
    signature = value.get("image", {})
    shape = signature.get("shape") if isinstance(signature, dict) else None
    if (
        not isinstance(shape, list)
        or len(shape) not in (2, 3)
        or any(type(n) is not int or not 1 <= n < 32767 for n in shape)
        or min(shape[:2]) < 8
        or shape[0] * shape[1] > 2_100_000
        or (len(shape) == 3 and shape[2] not in (3, 4))
    ):
        raise ValidationError("Trajectory sources require bounded 8..32766-pixel gray/color grids")
    encoding, depth = signature.get("encoding"), signature.get("dtype")
    channels = {"gray": 1, "rgb": 3, "bgr": 3, "rgba": 4, "bgra": 4}
    if depth not in ("uint8", "uint16", "float32", "float64") or channels.get(encoding) != (
        shape[2] if len(shape) == 3 else 1
    ):
        raise ValidationError("Invalid trajectory source depth/encoding")
    if not isinstance(signature.get("sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", signature["sha256"]
    ):
        raise ValidationError("Invalid trajectory image fingerprint")
    if not isinstance(value.get("label"), str) or len(value["label"]) > 256:
        raise ValidationError("Trajectory labels require at most 256 characters")
    checked_spatial(value, shape[1::-1])
    return shape[1::-1]


def media_time(context):
    video = context.get("video_frame", {})
    if not isinstance(video, dict):
        raise ValidationError("Media timing requires video_frame metadata")
    index, timestamp = video.get("frame_index"), video.get("timestamp_ms")
    if (
        type(index) is not int
        or not 0 <= index <= 10**12
        or isinstance(timestamp, bool)
        or not isinstance(timestamp, (int, float))
        or not np.isfinite(timestamp)
        or not 0 <= timestamp <= 1e12
    ):
        raise ValidationError(
            "Media timing requires video_frame frame_index and finite timestamp_ms"
        )
    return {"media_index": index, "timestamp_ms": float(timestamp)}


def clip_identity(records):
    return sha([{"image": r["source"]["image"], "time": r["time"]} for r in records])


def seal_trajectory(value):
    output = copy.deepcopy(value)
    output["clip_id"] = clip_identity(output["frames"])
    output["trajectory_id"] = sha({k: v for k, v in output.items() if k != "trajectory_id"})
    validate_trajectory(output)
    return output


def validate_trajectory(value):
    finite_json(value)
    if (
        not isinstance(value, dict)
        or value.get("model") != MODEL
        or value.get("direction") != "source_to_target"
        or value.get("interpolation") not in ("nearest", "linear")
    ):
        raise ValidationError("Choose a supported source-to-target motion trajectory")
    records, clock = value.get("frames"), value.get("clock")
    if not isinstance(records, list) or not 1 <= len(records) <= 64 or not isinstance(clock, dict):
        raise ValidationError("Trajectory requires 1..64 timed frame records")
    mode, period = clock.get("mode"), clock.get("frame_period_ms")
    if mode not in ("media", "indexed") or (
        mode == "indexed"
        and (
            isinstance(period, bool)
            or not isinstance(period, (int, float))
            or not 0.001 <= period <= 1e6
        )
    ):
        raise ValidationError("Invalid trajectory clock; indexed timing needs a nominal period")
    last_time, last_index, segment, first_layout = -1.0, -1, 0, None
    target_sizes = {}
    if mode == "media" and period is not None:
        raise ValidationError("Media timing cannot declare a nominal frame period")
    for i, record in enumerate(records):
        if (
            not isinstance(record, dict)
            or type(record.get("index")) is not int
            or record["index"] != i
        ):
            raise ValidationError("Trajectory indices must be contiguous and ordered")
        size = source_schema(record.get("source"))
        layout = {k: record["source"]["image"][k] for k in ("shape", "dtype", "encoding")}
        if first_layout is not None and layout != first_layout:
            raise ValidationError("Trajectory frame dimensions/depth/encoding must match")
        first_layout = layout
        time = record.get("time", {})
        timestamp = time.get("timestamp_ms") if isinstance(time, dict) else None
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not 0 <= timestamp <= 1e12
            or timestamp <= last_time
        ):
            raise ValidationError("Trajectory frame timestamps must strictly increase")
        if mode == "media":
            index = time.get("media_index")
            if type(index) is not int or not last_index < index <= 10**12:
                raise ValidationError("Trajectory media indices must strictly increase")
            last_index = index
        elif time.get("media_index") is not None or not np.isclose(
            timestamp, i * period, atol=1e-8, rtol=0
        ):
            raise ValidationError("Indexed timestamps must match their declared nominal period")
        last_time = timestamp
        if type(record.get("valid")) is not bool or type(record.get("segment")) is not int:
            raise ValidationError("Trajectory frames require explicit validity and segment")
        if not isinstance(record.get("reason"), str) or len(record["reason"]) > 512:
            raise ValidationError(
                "Trajectory frames require a reason string of at most 512 characters"
            )
        if record["segment"] not in (segment, segment + 1) or (i == 0 and record["segment"] != 0):
            raise ValidationError("Trajectory segments must begin at zero and advance contiguously")
        segment = record["segment"]
        if record["valid"]:
            affine(record.get("source_to_target"))
            target = record.get("target_spatial")
            if not isinstance(target, dict):
                raise ValidationError("Valid trajectory frames require target geometry")
            checked_spatial({"spatial": target}, size)
            if segment in target_sizes and target["root_size"] != target_sizes[segment]:
                raise ValidationError(
                    "A trajectory segment must use one consistent target root size"
                )
            target_sizes[segment] = target["root_size"]
        elif (
            record.get("source_to_target") is not None
            or record.get("target_spatial") is not None
            or not isinstance(record.get("reason"), str)
            or not 1 <= len(record["reason"]) <= 512
        ):
            raise ValidationError(
                "Invalid trajectory frames require a reason and no invented transform"
            )
    if value.get("clip_id") != clip_identity(records) or value.get("trajectory_id") != sha(
        {k: v for k, v in value.items() if k != "trajectory_id"}
    ):
        raise ValidationError("Trajectory identity differs from its content; create a new revision")
    return records


def capture_trajectory(
    group, transforms, *, timestamp_mode="indexed", frame_period_ms=40.0, interpolation="linear"
):
    images, contexts, _ = trajectory_frames(group)
    if not isinstance(transforms, FlowGroup) or len(transforms.items) != len(images):
        raise ValidationError("Provide one transform packet per source frame in matching order")
    rows, previous_root, segment = [], None, -1
    for i, (image, context, packet) in enumerate(
        zip(images, contexts, transforms.items, strict=True)
    ):
        image_value(image, (np.uint8, np.uint16, np.float32, np.float64))
        source = endpoint(image, context, group.labels[i])
        source_schema(source)
        value = packet.value
        if not isinstance(value, dict) or type(value.get("valid", True)) is not bool:
            raise ValidationError(
                "Transform entries must be stabilization records or explicit invalid placeholders"
            )
        if context.get("ticket") is not None and packet.payload.get("ticket") is not None:
            if ticket_key(context["ticket"]) != ticket_key(packet.payload["ticket"]):
                raise ValidationError("Transform packet ticket differs from its source frame")
        target, correction = None, None
        valid = value.get("valid", True)
        if valid:
            size = list(image.shape[1::-1])
            if value.get("model") != "image_stabilization_v1" or any(
                value.get(k) != size for k in ("source_image_size", "target_image_size")
            ):
                raise ValidationError(
                    "Use same-size image_stabilization_v1 source-to-target transforms"
                )
            correction = affine(value.get("source_to_target"))
            inverse = affine(value.get("target_to_source"))
            if not np.allclose(inverse @ correction, np.eye(3), atol=1e-8, rtol=0):
                raise ValidationError("Transform directions/inverses disagree")
            if value.get("source_spatial") is not None and value["source_spatial"] != context.get(
                "spatial"
            ):
                raise ValidationError("Transform source grid differs from the frame")
            if value.get("source_pyramid") != context.get("pyramid"):
                raise ValidationError("Transform pyramid differs from its source frame")
            target = copy.deepcopy(value.get("target_spatial"))
            if not isinstance(target, dict):
                raise ValidationError("Transform target spatial frame is missing")
            checked_spatial({"spatial": target}, size)
            root = target.get("root_id")
            if segment < 0 or root != previous_root:
                segment += 1
            previous_root = root
            target["root_id"] = None
        else:
            segment = max(segment, 0)
        rows.append(
            {
                "index": i,
                "source": source,
                "time": media_time(context)
                if timestamp_mode == "media"
                else {"media_index": None, "timestamp_ms": i * frame_period_ms},
                "valid": valid,
                "segment": segment,
                "reason": "" if valid else value.get("reason", ""),
                "source_to_target": correction.tolist() if valid else None,
                "target_spatial": target,
            }
        )
    return seal_trajectory(
        {
            "model": MODEL,
            "direction": "source_to_target",
            "interpolation": interpolation,
            "clock": {
                "mode": timestamp_mode,
                "frame_period_ms": frame_period_ms if timestamp_mode == "indexed" else None,
            },
            "frames": rows,
        }
    )


class CaptureMotionTrajectory(Block):
    type_id, title = "sequence.capture_trajectory", "Capture motion trajectory"
    category = block_type = "Sequence"
    subtype = "Recorded motion"
    parallel_safe = process_inputs_readonly = True
    inputs = (Port("frames", "any"), Port("transforms", "any"))
    outputs = (Port("trajectory", "data"), Port("info", "data"))
    defaults = {"timestamp_mode": "indexed", "frame_period_ms": 40.0, "interpolation": "linear"}
    parameter_choices = {
        "timestamp_mode": ("indexed", "media"),
        "interpolation": ("nearest", "linear"),
    }
    parameter_ranges = {"frame_period_ms": NumericRange(0.001, 1e6)}
    parameter_help = {
        "timestamp_mode": "Nominal indexed times, or require each video_frame index/timestamp_ms.",
        "frame_period_ms": "Nominal indexed frame period; does not assert exposure timing.",
        "interpolation": "Saved sampling policy; match the original stabilization interpolation.",
    }
    input_help = {
        "frames": "One to 64 original source frames in clip order, with current metadata.",
        "transforms": "Matching stabilization transforms, or valid=false/reason placeholders.",
    }

    def process(self, inputs):
        trajectory = capture_trajectory(inputs["frames"], inputs["transforms"], **self.parameters)
        return Result(
            {
                "trajectory": trajectory,
                "info": {
                    "clip_id": trajectory["clip_id"],
                    "trajectory_id": trajectory["trajectory_id"],
                    "frames": len(trajectory["frames"]),
                },
            },
            message="Captured content-bound motion trajectory",
        )


class ReplayMotionTrajectory(Block):
    type_id, title = "sequence.replay_trajectory", "Replay motion trajectory"
    category = block_type = "Sequence"
    subtype = "Recorded motion"
    parallel_safe = process_inputs_readonly = True
    payload_reference = "frames"
    inputs = (Port("frames", "any"), Port("trajectory", "data"))
    outputs = (*tuple(Port(n, "any") for n in _PRIMARY[:3]), Port("image"), Port("report", "data"))
    flow_primary = ("corrected",)
    defaults = {
        "interpolation": "recorded",
        "min_coverage_fraction": 0.65,
        "invalid_frames": "reject",
        "preview_frame": 0,
    }
    parameter_choices = {
        "interpolation": ("recorded", "nearest", "linear"),
        "invalid_frames": ("reject", "fail_partial", "skip"),
    }
    parameter_ranges = {
        "min_coverage_fraction": NumericRange(0, 1),
        "preview_frame": NumericRange(0, 63, whole=True),
    }
    parameter_help = {
        "interpolation": "Use saved interpolation or explicitly override sampling.",
        "min_coverage_fraction": "Minimum valid support per frame; zero coverage always fails.",
        "invalid_frames": "Reject all, emit failed partial, or skip invalid/low-coverage frames.",
        "preview_frame": "Original ordinal to show as Image; omitted frames emit no preview.",
    }
    input_help = {
        "frames": "Complete recorded clip in original order; exact pixels/types/grids are checked.",
        "trajectory": "Versioned trajectory; replay does not fit motion or restore old tickets.",
    }

    @staticmethod
    def _reject(report, reason):
        report["status"] = reason
        return Result(
            {**dict.fromkeys(_PRIMARY, NO_VALUE), "report": report},
            passed=False,
            fail_reasons=["trajectory_" + reason],
            message=reason.replace("_", " "),
        )

    def process(self, inputs):
        trajectory = inputs["trajectory"]
        records = validate_trajectory(trajectory)
        group = inputs["frames"]
        images, contexts, encoding = trajectory_frames(group)
        if len(records) != len(images):
            raise ValidationError("Replay needs the complete clip in recorded order")
        if self.parameters["preview_frame"] >= len(images):
            raise ValidationError("Preview ordinal is outside this clip", "preview_frame")
        for image, context, record in zip(images, contexts, records, strict=True):
            image_value(image, (np.uint8, np.uint16, np.float32, np.float64))
            same_endpoint(record["source"], image, context, "trajectory")
            if trajectory["clock"]["mode"] == "media":
                actual = media_time(context)
                if actual["media_index"] != record["time"]["media_index"] or not np.isclose(
                    actual["timestamp_ms"], record["time"]["timestamp_ms"], atol=1e-6, rtol=0
                ):
                    raise ValidationError("Clip frame index/timestamp differs from the trajectory")
        report = {
            "model": "motion_trajectory_replay_v1",
            "trajectory_id": trajectory["trajectory_id"],
            "clip_id": trajectory["clip_id"],
            "timestamp_mode": trajectory["clock"]["mode"],
            "frames": [],
            "omitted_indices": [],
            "source_binding": "exact_pixels_type_encoding_and_coordinate_geometry",
            "interpolation": trajectory["interpolation"]
            if self.parameters["interpolation"] == "recorded"
            else self.parameters["interpolation"],
        }
        policy = self.parameters["invalid_frames"]
        corrected, coverage, transforms, labels = [], [], [], []
        preview = NO_VALUE
        preview_payload = {}
        for i, (image, context, record) in enumerate(zip(images, contexts, records, strict=True)):
            if self._cancelled():
                return self._reject(report, "cancelled")
            row = {
                "index": i,
                "time": record["time"],
                "source_ticket": context.get("ticket"),
                "valid": record["valid"],
                "reason": record["reason"],
            }
            output, valid, covered = None, None, 0.0
            if record["valid"]:
                correction = affine(record["source_to_target"])
                output, valid, covered = _warp_correction(
                    image,
                    correction,
                    {
                        "interpolation": report["interpolation"],
                        "min_coverage_fraction": self.parameters["min_coverage_fraction"],
                    },
                )
                if output is None:
                    row.update(valid=False, reason="insufficient_coverage")
            row["coverage_fraction"] = covered
            report["frames"].append(row)
            if not row["valid"]:
                report["omitted_indices"].append(i)
                if policy == "reject":
                    return self._reject(report, "invalid_frames")
                continue
            target = copy.deepcopy(record["target_spatial"])
            target["root_id"] = (
                "trajectory:" + trajectory["trajectory_id"] + ":" + str(record["segment"])
            )
            payload = copy.deepcopy(context)
            payload.pop("pyramid", None)
            payload.update(
                spatial=target,
                motion_trajectory={
                    "trajectory_id": trajectory["trajectory_id"],
                    "clip_id": trajectory["clip_id"],
                    "index": i,
                    "time": record["time"],
                    "segment": record["segment"],
                },
            )
            transform = _transform_payload(correction, list(image.shape[1::-1]), context, target)
            corrected.append(ValuePacket(ImageArray(output, encoding), payload))
            coverage.append(ValuePacket(valid, copy.deepcopy(payload)))
            transforms.append(ValuePacket(transform, copy.deepcopy(payload)))
            labels.append(group.labels[i])
            if i == self.parameters["preview_frame"]:
                preview, preview_payload = corrected[-1].value, payload
        if not corrected:
            return self._reject(report, "no_valid_frames")
        report.update(
            status="partial" if report["omitted_indices"] else "complete",
            emitted_count=len(corrected),
        )
        failed = bool(report["omitted_indices"]) and policy == "fail_partial"
        return Result(
            {
                "corrected": FlowGroup(tuple(corrected), tuple(labels)),
                "coverage": FlowGroup(tuple(coverage), tuple(labels)),
                "transforms": FlowGroup(tuple(transforms), tuple(labels)),
                "image": preview,
                "report": report,
            },
            payloads={"image": preview_payload},
            passed=not failed,
            fail_reasons=["trajectory_invalid_frames"] if failed else [],
            message=f"Replayed {len(corrected)}/{len(records)} frames without motion fitting",
        )
