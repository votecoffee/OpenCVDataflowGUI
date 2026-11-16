"""Original annotated entering/lost parts; native blobs gated by detector cadence."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
W, H, FRAMES = 224, 112, 16
CASES = {
    "periodic": {"period": 4, "age": 15},
    "every_frame": {"period": 1, "age": 15},
    "freshness_limit": {"period": 4, "age": 2},
}


def build():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    texture = np.random.default_rng(21).integers(80, 240, (32, 40), np.uint8)
    truth = []
    for index in range(FRAMES):
        image = np.zeros((H, W), np.uint8)
        objects = []
        if index < 9:
            x = 20 + 3 * index
            image[30:62, x : x + 40] = texture
            objects.append({"label": "A", "center": [x + 19.5, 45.5], "size": [39, 31]})
        if index >= 5:
            image[76:100, 160:190] = texture[:24, :30]
            objects.append({"label": "B", "center": [174.5, 87.5], "size": [29, 23]})
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        (assets / f"frame_{index:02d}.png").write_bytes(encoded.tobytes())
        truth.append({"frame": index, "timestamp_ms": index * 100, "objects": objects})
    registry = builtin_registry()
    for name, options in CASES.items():
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
                        "repeat": "stop",
                        "path_base": "project",
                        "color_format": "gray",
                        "pixel_depth": "unchanged",
                    },
                )
            )
            cadence = graph.add(
                registry.create(
                    "flow.detection_cadence",
                    block_id="cadence",
                    parameters={"period": options["period"]},
                )
            )
            detector = graph.add(
                registry.create("inspect.blobs", block_id="detector", parameters={"min_area": 40})
            )
            tracker = graph.add(
                registry.create(
                    "sequence.track_refreshed",
                    block_id="track",
                    parameters={
                        "clock": "nominal",
                        "nominal_interval_ms": 100,
                        "angle_period": "180",
                        "maximum_missed_frames": 2,
                        "maximum_detector_age_frames": options["age"],
                    },
                )
            )
            tracker.exposed_outputs.append("result_graphics")
            overlay = graph.add(registry.create("cv.overlay_result_graphics", block_id="overlay"))
            final = graph.add(registry.create("output.filmstrip_image", block_id="filmstrip"))
            for block, position in zip(
                (source, cadence, detector, tracker, overlay, final),
                ([0, 0], [380, 200], [760, 200], [1140, 0], [1520, 0], [1900, 0]),
                strict=True,
            ):
                block.position = position
            tracker.node_size = "large"
            for a, b, c, d in (
                ("capture", "image", "cadence", "image"),
                ("cadence", "image", "detector", "image"),
                ("capture", "image", "track", "image"),
                ("cadence", "refresh", "track", "refresh"),
                ("detector", "collection", "track", "detections"),
                ("capture", "image", "overlay", "image"),
                ("track", "result_graphics", "overlay", "graphics_1"),
                ("overlay", "image", "filmstrip", "image"),
            ):
                graph.connect(a, b, c, d)
            note = graph.add(registry.create("layout.comment", block_id="attribution"))
            note.name, note.position, note.node_size = (
                "Original annotated detector-refresh fixtures - MIT",
                [0, -420],
                "large",
            )
            note.description = (
                "Original MIT images, annotations and projects by OpenCVDataflowGUI "
                "contributors; no customer or external images. Run All consumes sixteen "
                "frames. Native blob detection runs only on cadence images; local image "
                "evidence updates existing identities between refreshes. A new lower-right "
                "part enters at frame 5, discovered at frame 8 for period 4 or frame 5 for "
                "period 1. The moving upper part disappears at frame 9; lost local support "
                "must predict then expire, without learning replacement background. Blue "
                "means local evidence, green detector evidence and amber prediction. "
                "Freshness limit deliberately expires even successful local tracks after two "
                "acquisitions without a detector match. truth.json records independent "
                "intended motion and hashes. Detector calls are measured; this makes no "
                "runtime speed claim. These synthetic tests do not qualify production tracking."
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
                    "frames": truth,
                    "cases": CASES,
                    "asset_sha256": {
                        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(assets.glob("*.png"))
                    },
                },
                indent=2,
            )
            + "\n"
        ).encode()
    )


if __name__ == "__main__":
    build()
