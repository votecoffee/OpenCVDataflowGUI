"""Original MIT media-time fixture, independent annotations and inspection graphs."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
CASES = {
    "native_exact": {"time_ms": 250, "seek_method": "native_only", "index": 5},
    "nearest": {"time_ms": 175, "selection": "nearest", "index": 3},
    "at_or_before": {"time_ms": 175, "selection": "at_or_before", "index": 3},
    "at_or_after": {"time_ms": 175, "selection": "at_or_after", "index": 4},
    "error_guard": {"time_ms": 225, "maximum_error_ms": 10, "index": None},
}


def images():
    for index in range(8):
        image = np.full((96, 160, 3), 20, np.uint8)
        image[30:54, 18 + 8 * index : 46 + 8 * index] = [150, 220, 250]
        yield image


def build():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    path = assets / "time_samples.avi"
    registry = builtin_registry()
    frames = list(images())
    if not path.exists():
        writer = registry.create(
            "output.video",
            parameters={
                "path": str(path),
                "backend": "ffmpeg",
                "codec": "FFV1",
                "fps": 20,
            },
        )
        try:
            for image in frames:
                assert writer.execute({"image": image}) is not None, writer.error
        finally:
            writer.close()
    decoder = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    try:
        for index, expected in enumerate(frames):
            ok, actual = decoder.read()
            if not ok or not np.array_equal(expected, actual):
                raise ValueError("Existing fixture differs; use a fresh output folder")
            if abs(decoder.get(cv2.CAP_PROP_POS_MSEC) - index * 50) > 1e-6:
                raise ValueError("Fixture time differs from independently specified rate")
        if decoder.read()[0]:
            raise ValueError("Existing fixture contains additional frames")
    finally:
        decoder.release()
    for name, options in CASES.items():
        graph = Graph()
        try:
            source = graph.add(
                registry.create(
                    "input.video_at_time",
                    block_id="video",
                    parameters={
                        "path": "assets/time_samples.avi",
                        "path_base": "project",
                        **{k: v for k, v in options.items() if k != "index"},
                    },
                )
            )
            gray = graph.add(registry.create("cv.gray", block_id="gray"))
            threshold = graph.add(registry.create("cv.threshold", block_id="threshold"))
            blobs = graph.add(
                registry.create(
                    "inspect.blobs",
                    block_id="blobs",
                    parameters={
                        "min_area": 40,
                    },
                )
            )
            blobs.exposed_outputs.append("result_graphics")
            overlay = graph.add(registry.create("cv.overlay_result_graphics", block_id="overlay"))
            film = graph.add(registry.create("output.filmstrip_image", block_id="filmstrip"))
            for block, position in zip(
                (source, gray, threshold, blobs, overlay, film),
                ([0, 0], [380, 200], [760, 200], [1140, 200], [1520, 0], [1900, 0]),
                strict=True,
            ):
                block.position = position
            source.node_size = "large"
            for wire in (
                ("video", "image", "gray", "image"),
                ("gray", "image", "threshold", "image"),
                ("threshold", "image", "blobs", "image"),
                ("video", "image", "overlay", "image"),
                ("blobs", "result_graphics", "overlay", "graphics_1"),
                ("overlay", "image", "filmstrip", "image"),
            ):
                graph.connect(*wire)
            note = graph.add(registry.create("layout.comment", block_id="attribution"))
            note.name = "Original media-time fixture - MIT"
            note.position, note.node_size = [0, -500], "large"
            note.description = (
                "Original MIT synthetic clip, generator, annotations and projects by "
                "OpenCVDataflowGUI contributors. No customer or third-party images. "
                "Eight lossless 20 FPS frames contain a moving 28 by 24 pixel part. "
                "Native exact requests 250 ms (frame 5). At 175 ms nearest/earlier selects "
                "150 ms (frame 3); later selects 200 ms (frame 4). Native positioning "
                "overshoots to 200 ms on this fixture, so bounded scanning verifies the bracket. "
                "The error-guard project deliberately rejects 225 ms because its nearest "
                "frame is 25 ms away and only 10 ms error is allowed. Each Run once is an "
                "independent sample with a fresh ticket and pixel frame; it does not advance "
                "a playback cursor. See truth.json for original frame hashes and geometry."
            )
            graph.save(HERE / f"{name}.json")
        finally:
            graph.close()
    (HERE / "truth.json").write_bytes(
        (
            json.dumps(
                {
                    "license": "MIT - original repository fixtures",
                    "image_size": [160, 96],
                    "fps": 20,
                    "frames": [
                        {
                            "index": i,
                            "timestamp_ms": i * 50,
                            "center": [31.5 + 8 * i, 41.5],
                            "area": 672,
                            "sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                        }
                        for i, image in enumerate(frames)
                    ],
                    "cases": CASES,
                    "asset_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()},
                },
                indent=2,
            )
            + "\n"
        ).encode()
    )


if __name__ == "__main__":
    build()
