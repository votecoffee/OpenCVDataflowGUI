"""Original MIT annotated part motion; actual blob detections drive every saved graph."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
W, H, FRAMES = 256, 128, 20
CASES = {"oriented_gaps": 10, "expire_reacquire": 2}


def build():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    truth = []
    for index in range(FRAMES):
        center = [40 + 7 * index, 64]
        angle = 80 + 4 * index
        scale = float(np.exp(0.015 * index))
        missing = index in (6, 7, 12, 13, 14, 15)
        image = np.zeros((H, W), np.uint8)
        if not missing:
            cv2.ellipse(
                image,
                tuple(center),
                tuple(np.rint(np.array([20, 9]) * scale).astype(int)),
                -angle,
                0,
                360,
                255,
                -1,
                cv2.LINE_8,
            )
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        (assets / f"frame_{index:02d}.png").write_bytes(encoded.tobytes())
        truth.append(
            {
                "frame": index,
                "timestamp_ms": index * 100,
                "center": center,
                "angle_degrees": (angle + 90) % 180 - 90,
                "scale": scale,
                "visible": not missing,
            }
        )
    registry = builtin_registry()
    for name, limit in CASES.items():
        graph = Graph()
        try:
            paths = [f"assets/frame_{i:02d}.png" for i in range(FRAMES)]
            source = graph.add(
                registry.create(
                    "input.file",
                    block_id="capture",
                    parameters={
                        "path": paths[0],
                        "paths": paths,
                        "source_mode": "files",
                        "path_base": "project",
                        "color_format": "gray",
                        "pixel_depth": "unchanged",
                        "repeat": "stop",
                    },
                )
            )
            blobs = graph.add(
                registry.create("inspect.blobs", block_id="detect", parameters={"min_area": 40})
            )
            tracker = graph.add(
                registry.create(
                    "sequence.track_image",
                    block_id="track",
                    parameters={
                        "clock": "nominal",
                        "nominal_interval_ms": 100,
                        "angle_period": "180",
                        "maximum_missed_frames": limit,
                        "maximum_unobserved_ms": 2000,
                        "position_noise": 0.6,
                        "scale_noise": 0.1,
                        "angle_noise": 2,
                        "association_distance": 30,
                    },
                )
            )
            tracker.exposed_outputs.append("result_graphics")
            overlay = graph.add(registry.create("cv.overlay_result_graphics", block_id="overlay"))
            final = graph.add(registry.create("output.filmstrip_image", block_id="filmstrip"))
            for block, position in zip(
                (source, blobs, tracker, overlay, final),
                ([0, 0], [380, 160], [760, 0], [1140, 0], [1520, 0]),
                strict=True,
            ):
                block.position = position
            tracker.node_size = "large"
            graph.connect("capture", "image", "detect", "image")
            graph.connect("capture", "image", "track", "image")
            graph.connect("detect", "collection", "track", "detections")
            graph.connect("capture", "image", "overlay", "image")
            graph.connect("track", "result_graphics", "overlay", "graphics_1")
            graph.connect("overlay", "image", "filmstrip", "image")
            note = graph.add(registry.create("layout.comment", block_id="attribution"))
            note.name, note.position, note.node_size = (
                "Annotated original part motion - MIT",
                [0, -400],
                "large",
            )
            note.description = (
                "Original MIT fixtures and projects by OpenCVDataflowGUI contributors. "
                "No customer or external images. Run All consumes twenty frames. Actual OpenCV "
                "blob measurements drive the image tracker; independent truth.json "
                "records intended "
                "motion, scale, angle and hidden frames. Green is observed and amber predicted. "
                "First detection is tentative; predictions are estimates, not fresh observations. "
                "The four-frame gap retains identity in oriented_gaps and expires it in "
                "expire_reacquire. Reappearance gets a fresh ID after expiry. Reset to replay. "
                "These are controlled synthetic tracker tests, not production accuracy evidence."
            )
            graph.save(HERE / (name + ".json"))
        finally:
            graph.close()
    manifest = {
        "license": "MIT - original repository fixtures",
        "image_size": [W, H],
        "frames": truth,
        "cases": CASES,
        "asset_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(assets.glob("*.png"))
        },
    }
    (HERE / "truth.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode())


if __name__ == "__main__":
    build()
