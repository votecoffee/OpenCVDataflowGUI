"""Original jittered clips and sealed, independently known motion trajectories."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.motion_trajectory import capture_trajectory
from opencv_dataflow.motion_trajectory_io import trajectory_bytes
from opencv_dataflow.payloads import FlowGroup, ValuePacket, frame
from opencv_dataflow.stabilization import _transform_payload

HERE = Path(__file__).resolve().parent
SHIFTS = ((0, 0), (2, -1), (-1, 2), (3, 1), (0, -2))
CASES = {
    "replay_files": {},
    "replay_video": {"video": True},
    "highdepth": {"depth": True},
    "invalid_reject": {"invalid": True, "expected_state": "fail"},
    "invalid_partial": {
        "invalid": True,
        "invalid_frames": "fail_partial",
        "expected_state": "fail",
    },
    "invalid_skip": {"invalid": True, "invalid_frames": "skip"},
    "changed_pixels": {"changed": True, "expected_state": "error"},
    "reordered_frames": {"order": [0, 2, 1, 3, 4], "expected_state": "error"},
    "native_fit_capture": {"video": True, "native": True},
    "time_guard": {"video": True, "changed_time": True, "expected_state": "error"},
}


def images():
    base = cv2.GaussianBlur(
        np.random.default_rng(167).integers(20, 220, (112, 160), np.uint8), (5, 5), 0.9
    )
    for x in range(15, 150, 23):
        cv2.circle(base, (x, 20 + x % 70), 6, 230, 2)
    return [np.roll(base, (dy, dx), axis=(0, 1)) for dx, dy in SHIFTS]


def known_trajectory(values, *, media=False, invalid=False):
    packets, transforms = [], []
    for i, (image, (dx, dy)) in enumerate(zip(values, SHIFTS, strict=True)):
        context = {"spatial": frame([160, 112], "capture:image")}
        if media:
            context["video_frame"] = {"frame_index": i, "timestamp_ms": i * 40.0}
        packets.append(ValuePacket(ImageArray(image, "gray"), context))
        matrix = np.array([[1.0, 0, -dx], [0, 1.0, -dy], [0, 0, 1.0]])
        value = _transform_payload(matrix, [160, 112], context, frame([160, 112], "known-canvas"))
        if invalid and i == 2:
            value = {"valid": False, "reason": "reference occluded"}
        transforms.append(ValuePacket(value, context))
    labels = tuple(f"frame_{i:03d}" for i in range(len(values)))
    return capture_trajectory(
        FlowGroup(tuple(packets), labels),
        FlowGroup(tuple(transforms), labels),
        timestamp_mode="media" if media else "indexed",
        interpolation="nearest",
    )


def build():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    values = images()
    for depth in (False, True):
        for i, value in enumerate(values):
            image = value.astype(np.uint16) * 257 if depth else value
            ok, encoded = cv2.imencode(".png", image)
            assert ok
            (assets / f"frame_{i:03d}{'_16' if depth else ''}.png").write_bytes(encoded.tobytes())
    changed = values[1].copy()
    changed[20, 20] ^= 1
    ok, encoded = cv2.imencode(".png", changed)
    assert ok
    (assets / "changed.png").write_bytes(encoded.tobytes())
    video_path = assets / "jitter.avi"
    writer = cv2.VideoWriter(
        str(video_path), cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*"FFV1"), 25, (160, 112), False
    )
    assert writer.isOpened()
    try:
        for value in values:
            writer.write(value)
    finally:
        writer.release()
    records = {
        "known": known_trajectory(values),
        "media": known_trajectory(values, media=True),
        "depth16": known_trajectory([v.astype(np.uint16) * 257 for v in values]),
        "invalid": known_trajectory(values, invalid=True),
    }
    for name, record in records.items():
        (assets / (name + ".json")).write_bytes(trajectory_bytes(record))
    registry = builtin_registry()
    truth = {
        "creator": "OpenCVDataflowGUI contributors",
        "license": "MIT",
        "shifts": [list(v) for v in SHIFTS],
        "image_size": [160, 112],
        "cases": {},
    }
    for case, settings in CASES.items():
        graph = Graph()
        order = settings.get("order", list(range(5)))
        is_video = settings.get("video", False)
        if is_video:
            source = graph.add(
                registry.create(
                    "input.video_file",
                    block_id="capture",
                    parameters={
                        "path": "assets/jitter.avi",
                        "end_frame": 5,
                        "path_base": "project",
                        "color_format": "gray",
                        "pixel_depth": "unchanged",
                        "repeat": "stop",
                    },
                )
            )
        else:
            paths = [
                f"assets/frame_{i:03d}{'_16' if settings.get('depth') else ''}.png" for i in order
            ]
            if settings.get("changed"):
                paths[1] = "assets/changed.png"
            source = graph.add(
                registry.create(
                    "input.file",
                    block_id="capture",
                    parameters={
                        "source_mode": "files",
                        "paths": paths,
                        "path_base": "project",
                        "color_format": "gray",
                        "pixel_depth": "unchanged",
                        "repeat": "stop",
                    },
                )
            )
        source.position = [0, 0]
        batch = graph.add(registry.create("flow.batch", block_id="frames", parameters={"size": 5}))
        batch.position = [430, 0]
        if settings.get("changed_time"):
            changed_time = graph.add(
                registry.create(
                    "core.attach_payload",
                    block_id="changed_time",
                    parameters={
                        "payload": {"video_frame": {"frame_index": 0, "timestamp_ms": 1.0}}
                    },
                )
            )
            changed_time.position = [0, 350]
            graph.connect("capture", "image", changed_time.id, "value")
            graph.connect(changed_time.id, "value", "frames", "item_1")
        else:
            graph.connect("capture", "image", "frames", "item_1")
        if settings.get("native"):
            fit = graph.add(
                registry.create(
                    "sequence.stabilize_batch",
                    block_id="fit",
                    parameters={
                        "motion_model": "translation",
                        "min_inliers": 6,
                        "radius_frames": 1,
                        "sigma_frames": 1.0,
                        "interpolation": "nearest",
                    },
                )
            )
            fit.position = [430, 430]
            graph.connect("frames", "group", "fit", "frames")
            model = graph.add(
                registry.create(
                    "sequence.capture_trajectory",
                    block_id="trajectory",
                    parameters={"timestamp_mode": "media", "interpolation": "nearest"},
                )
            )
            model.position = [890, 430]
            graph.connect("frames", "group", model.id, "frames")
            graph.connect("fit", "transforms", model.id, "transforms")
            record_name = None
        else:
            record_name = (
                "invalid"
                if settings.get("invalid")
                else "depth16"
                if settings.get("depth")
                else "media"
                if is_video
                else "known"
            )
            raw = (assets / (record_name + ".json")).read_bytes()
            model = graph.add(
                registry.create(
                    "sequence.load_trajectory",
                    block_id="trajectory",
                    parameters={
                        "path": "assets/" + record_name + ".json",
                        "expected_sha256": hashlib.sha256(raw).hexdigest(),
                    },
                )
            )
            model.position = [430, 430]
        replay = graph.add(
            registry.create(
                "sequence.replay_trajectory",
                block_id="replay",
                parameters={
                    "invalid_frames": settings.get("invalid_frames", "reject"),
                    "preview_frame": 1,
                },
            )
        )
        replay.position = [1320, 0]
        graph.connect("frames", "group", "replay", "frames")
        graph.connect("trajectory", "trajectory", "replay", "trajectory")
        serial = graph.add(registry.create("flow.unbatch", block_id="serial"))
        serial.position = [1750, 0]
        graph.connect("replay", "corrected", "serial", "group")
        note = graph.add(registry.create("layout.comment", block_id="attribution"))
        note.name, note.position, note.node_size = (
            "Original sealed motion replay",
            [0, -440],
            "large",
        )
        note.description = (
            "Original MIT frames, FFV1 clip and motion projects by OpenCVDataflowGUI "
            "contributors; no customer or third-party images. The five frames have known integer "
            "shifts (0,0), (2,-1), (-1,2), (3,1), (0,-2). Frozen source-to-target corrections undo "
            "these shifts on one 160x112 canvas. Content/depth/encoding/grid fingerprints and "
            "media indices/timestamps bind replay to its source clip without reviving capture "
            "tickets. Invalid geometry is rejected, failed partial or explicitly skipped. Native "
            "fit/capture is a separate workflow; loaded trajectories perform no feature fitting. "
            "Changed pixels, reordered frames and changed media time deliberately reject before "
            "warp. FFV1 and PNG store the same original pixel values. Coverage masks mark invalid "
            "borders. These fixtures qualify recorded pixel motion, not physical camera identity."
        )
        graph.save(HERE / (case + ".json"))
        graph.close()
        truth["cases"][case] = {
            "order": order,
            "video": is_video,
            "record": record_name,
            "expected_state": settings.get("expected_state", "pass"),
            "invalid_frames": settings.get("invalid_frames", "reject"),
        }
    truth["asset_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(assets.iterdir())
        if p.is_file()
    }
    (HERE / "truth.json").write_bytes((json.dumps(truth, indent=2) + "\n").encode())


if __name__ == "__main__":
    build()
