"""Generate an original lossless jittered clip and continuous stabilization graph."""

from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def main():
    folder = Path(__file__).resolve().parent
    assets = folder / "assets"
    assets.mkdir(exist_ok=True)
    base = np.random.default_rng(314).integers(0, 256, (112, 160), np.uint8)
    base = cv2.GaussianBlur(base, (5, 5), 0.9)
    writer = cv2.VideoWriter(
        str(assets / "inspection-jitter.avi"), cv2.VideoWriter_fourcc(*"FFV1"), 30, (160, 112)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV needs FFV1 recording support to regenerate this example")
    try:
        for i in range(144):
            x = 0.12 * i + 3.5 * np.sin(i * 1.7)
            y = 1.5 * np.sin(i * 1.3)
            image = cv2.warpAffine(
                base,
                np.array([[1.0, 0.0, x], [0.0, 1.0, y]]),
                (160, 112),
                borderMode=cv2.BORDER_REFLECT_101,
            )
            writer.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    finally:
        writer.release()
    registry = builtin_registry()
    graph = Graph()
    source = graph.add(
        registry.create(
            "input.video_file",
            block_id="source",
            parameters={
                "path": "assets/inspection-jitter.avi",
                "path_base": "project",
                "color_format": "gray",
                "repeat": "stop",
            },
        )
    )
    source.name = "144-frame inspection clip"
    smooth = graph.add(
        registry.create(
            "sequence.stabilize_stream",
            block_id="smooth",
            parameters={
                "motion_model": "translation",
                "radius_frames": 4,
                "sigma_frames": 2.0,
            },
        )
    )
    smooth.position = [330.0, 0.0]
    smooth.exposed_outputs = ["image", "coverage", "transform", "report", "ready", "buffered"]
    graph.connect(source.id, "image", smooth.id, "image")
    note = graph.add(registry.create("layout.comment", block_id="fixture_note"))
    note.name = "Original synthetic fixture"
    note.description = (
        "Original synthetic clip, generator and project under this repository's MIT license. "
        "Run all 144 frames. The stabilizer waits for four future frames, then emits in order "
        "with each original ticket and timestamp. The final tail flushes automatically. "
        "Stored images remain bounded by the configured smoothing window."
    )
    note.position = [0.0, -270.0]
    note.node_size = "large"
    graph.save(folder / "continuous_camera_stabilization.json")
    graph.close()


if __name__ == "__main__":
    main()
