"""Original MIT color frames, lossless FFV1 clip, annotations and bounded-decoder graphs."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
CASES = (
    "selected_frames",
    "group_inspection",
    "short_clip",
    "read_limit",
    "decoded_limit",
    "malformed",
)


def build():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    path = assets / "color-motion.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*"FFV1"), 10, (64, 48)
    )
    if not writer.isOpened():
        raise RuntimeError("FFmpeg/FFV1 fixture writer unavailable")
    annotations = []
    try:
        y, x = np.indices((48, 64))
        for index in range(8):
            image = np.dstack(((x + index) % 40, y % 35, (x + y) % 45)).astype(np.uint8)
            left, top = 8 + index * 3, 15
            image[top : top + 12, left : left + 14] = (220, 235, 250)
            writer.write(image)
            ok, encoded = cv2.imencode(".png", image)
            assert ok
            name = f"frame_{index:02d}.png"
            (assets / name).write_bytes(encoded.tobytes())
            annotations.append(
                {
                    "index": index,
                    "timestamp_ms": index * 100,
                    "bounds": [left, top, 14, 12],
                    "file": name,
                    "bgr_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                }
            )
    finally:
        writer.release()
    (assets / "not-video.bin").write_bytes(b"Original deliberate invalid video fixture.\n")
    registry = builtin_registry()
    for case in CASES:
        graph = Graph()
        decoder = graph.add(
            registry.create(
                "cv.decode_video",
                block_id="decode",
                parameters={
                    "path": "assets/not-video.bin"
                    if case == "malformed"
                    else "assets/color-motion.avi",
                    "path_base": "project",
                    "frame_count": 16
                    if case == "short_clip"
                    else 4
                    if case == "selected_frames"
                    else 8,
                    "start_frame": 1 if case == "selected_frames" else 0,
                    "stride": 2 if case == "selected_frames" else 1,
                    "preview_frame": 1 if case == "selected_frames" else 0,
                    "color_format": "rgb" if case == "selected_frames" else "bgr",
                    "pixel_depth": "uint16" if case == "selected_frames" else "uint8",
                    "read_chunk_bytes": 257,
                    "maximum_read_bytes": 1 if case == "read_limit" else 256 * 1024**2,
                    "maximum_decoded_bytes": 1 if case == "decoded_limit" else 256 * 1024**2,
                },
            )
        )
        decoder.position = [0, 0]
        if case == "group_inspection":
            for kind, name, position, parameters in (
                ("flow.unbatch", "unbatch", [360, 0], {}),
                ("cv.gray", "gray", [720, 0], {}),
                ("cv.threshold", "threshold", [1080, 0], {"value": 128}),
                ("inspect.contours", "parts", [1440, 0], {"min_area": 20, "expected_count": 1}),
            ):
                block = graph.add(registry.create(kind, block_id=name, parameters=parameters))
                block.position = position
            for source, output, target, input in (
                ("decode", "frames", "unbatch", "group"),
                ("unbatch", "value", "gray", "image"),
                ("gray", "image", "threshold", "image"),
                ("threshold", "image", "parts", "image"),
            ):
                graph.connect(source, output, target, input)
        note = graph.add(registry.create("layout.comment", block_id="scope"))
        note.position, note.node_size = [360, 360], "large"
        note.name = "Original clip and timestamp scope"
        note.description = (
            "Frames, lossless FFV1 video, annotations and project: OpenCVDataflowGUI contributors, "
            "MIT. Original synthetic moving part; no customer or third-party media. "
            "Ten FPS, eight 64x48 BGR frames, known pixel boxes and presentation times. "
            "Native FFmpeg reads a bounded immutable buffer in chunks of at most 257 bytes. "
            "Presentation time is not camera acquisition time. Each frame has its own fresh "
            "ticket; Unbatch dispatches individual images in one graph activation. "
            "Read-limit, decoded-limit and malformed cases deliberately produce no result. "
            "Short-clip mode explicitly reports an unsatisfied requested frame count."
        )
        graph.save(HERE / f"{case}.json")
        graph.close()
    truth = {
        "creator": "OpenCVDataflowGUI contributors",
        "license": "MIT",
        "codec": "FFV1",
        "fps": 10,
        "size": [64, 48],
        "cases": list(CASES),
        "annotations": annotations,
        "asset_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(assets.iterdir())
            if p.is_file()
        },
    }
    (HERE / "truth.json").write_bytes((json.dumps(truth, indent=2) + "\n").encode())


if __name__ == "__main__":
    build()
