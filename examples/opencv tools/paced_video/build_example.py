"""Original lossless motion clip and scheduled detector/loop-reset inspection graphs."""

import hashlib
import json
from pathlib import Path

import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
W, H, FRAMES, FPS = 160, 96, 8, 20
CASES = {
    "native_paced": {"stride": 1, "speed": 1, "start": 0, "end": 8, "runs": 8},
    "strided_fast": {"stride": 2, "speed": 2, "start": 0, "end": 8, "runs": 4},
    "loop_reset": {"stride": 1, "speed": 2, "start": 2, "end": 5, "runs": 7, "repeat": "loop"},
}


def frames():
    y, x = np.mgrid[:H, :W]
    background = (20 + (x // 8 + y // 8) % 2 * 12).astype(np.uint8)
    for index in range(FRAMES):
        image = np.repeat(background[..., None], 3, axis=2)
        image[30:54, 18 + 8 * index : 46 + 8 * index] = [150, 220, 250]
        yield image


def build():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    path = assets / "moving_part.avi"
    registry = builtin_registry()
    # Preserve a checked-in clip; regenerate in a fresh folder to avoid overwriting user files.
    images = list(frames())
    if not path.exists():
        writer = registry.create(
            "output.video",
            parameters={"path": str(path), "backend": "ffmpeg", "codec": "FFV1", "fps": FPS},
        )
        try:
            for image in images:
                assert writer.execute({"image": image}) is not None, writer.error
        finally:
            writer.close()
    readback = registry.create(
        "input.video_file", parameters={"path": str(path), "backend": "ffmpeg"}
    )
    try:
        for expected in images:
            result = readback.execute({})
            if result is None or not np.array_equal(result.outputs["image"], expected):
                raise ValueError(
                    "Existing clip differs from original fixture; use a fresh output folder"
                )
        if not readback.exhausted:
            raise ValueError("Existing clip contains additional frames")
    finally:
        readback.close()
    for name, options in CASES.items():
        graph = Graph()
        try:
            source = graph.add(
                registry.create(
                    "input.paced_video_file",
                    block_id="video",
                    parameters={
                        "path": "assets/moving_part.avi",
                        "path_base": "project",
                        "backend": "ffmpeg",
                        "speed": options["speed"],
                        "stride": options["stride"],
                        "start_frame": options["start"],
                        "end_frame": options["end"],
                        "repeat": options.get("repeat", "stop"),
                    },
                )
            )
            source.exposed_outputs.append("cycle_start")
            gray = graph.add(registry.create("cv.gray", block_id="gray"))
            threshold = graph.add(registry.create("cv.threshold", block_id="threshold"))
            blobs = graph.add(
                registry.create("inspect.blobs", block_id="blobs", parameters={"min_area": 40})
            )
            track = graph.add(registry.create("sequence.track_image", block_id="track"))
            track.exposed_outputs.append("result_graphics")
            overlay = graph.add(registry.create("cv.overlay_result_graphics", block_id="overlay"))
            filmstrip = graph.add(registry.create("output.filmstrip_image", block_id="filmstrip"))
            for block, position in zip(
                (source, gray, threshold, blobs, track, overlay, filmstrip),
                ([0, 0], [380, 200], [760, 200], [1140, 200], [1520, 0], [1900, 0], [2280, 0]),
                strict=True,
            ):
                block.position = position
            source.node_size, track.node_size = "large", "large"
            for a, b, c, d in (
                ("video", "image", "gray", "image"),
                ("gray", "image", "threshold", "image"),
                ("threshold", "image", "blobs", "image"),
                ("blobs", "collection", "track", "detections"),
                ("video", "image", "track", "image"),
                ("video", "cycle_start", "track", "reset"),
                ("video", "image", "overlay", "image"),
                ("track", "result_graphics", "overlay", "graphics_1"),
                ("overlay", "image", "filmstrip", "image"),
            ):
                graph.connect(a, b, c, d)
            note = graph.add(registry.create("layout.comment", block_id="attribution"))
            note.name, note.position, note.node_size = (
                "Original paced-video fixtures - MIT",
                [0, -500],
                "large",
            )
            note.description = (
                "Original MIT clip, generator, annotations and projects by OpenCVDataflowGUI "
                "contributors. No customer or third-party images. Eight lossless frames at "
                "twenty FPS show a moving part. Playback speed affects release timing, "
                "preserving original media timestamps and pixels. The strided project emits "
                "0,2,4,6 at speed two. Run All consumes finite projects; Continuous loops "
                "recorded ranges. The loop-reset project uses frames 2,3,4 repeatedly. Its "
                "Cycle start output resets image-track memory at each wrap so no false motion "
                "or timestamp regression crosses clips. Stop interrupts pacing waits; Resume "
                "keeps the pending frame. Scheduling is cooperative, not hard real time. "
                "truth.json contains original pixel hashes and intended timing/positions."
            )
            graph.save(HERE / (name + ".json"))
        finally:
            graph.close()
    (HERE / "truth.json").write_bytes(
        (
            json.dumps(
                {
                    "license": "MIT - original repository fixtures",
                    "image_size": [W, H],
                    "fps": FPS,
                    "frames": [
                        {
                            "index": i,
                            "timestamp_ms": i * 1000 / FPS,
                            "center": [31.5 + 8 * i, 41.5],
                            "sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                        }
                        for i, image in enumerate(images)
                    ],
                    "cases": CASES,
                    "asset_sha256": {
                        "moving_part.avi": hashlib.sha256(path.read_bytes()).hexdigest()
                    },
                },
                indent=2,
            )
            + "\n"
        ).encode()
    )


if __name__ == "__main__":
    build()
