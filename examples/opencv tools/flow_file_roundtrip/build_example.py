"""Original MIT motion fields: analytic truth, computed reference and safe round trips."""

import hashlib
import json
import struct
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
CASES = ("roundtrip", "computed_motion", "bare_file", "direction_mismatch", "wrong_image")


def build():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    rng = np.random.default_rng(154)
    source = cv2.GaussianBlur(rng.integers(0, 256, (96, 128), np.uint8), (3, 3), 0.6)
    cv2.rectangle(source, (19, 17), (41, 43), 235, 2)
    cv2.circle(source, (85, 52), 14, 25, 3)
    target = np.zeros_like(source)
    target[:-2, 3:] = source[2:, :-3]
    for name, image in (("source", source), ("target", target)):
        ok, data = cv2.imencode(".png", image)
        assert ok
        (assets / f"{name}.png").write_bytes(data.tobytes())
    exact = np.full((*source.shape, 2), [3, -2], np.float32)
    computed = cv2.calcOpticalFlowFarneback(source, target, None, 0.5, 3, 25, 5, 7, 1.5, 0)
    for name, flow in (("exact", exact), ("computed", computed)):
        data = struct.pack("<4sii", b"PIEH", 128, 96) + flow.astype("<f4").tobytes()
        (assets / f"{name}.flo").write_bytes(data)
    error = np.linalg.norm(computed[16:-16, 16:-16] - exact[16:-16, 16:-16], axis=2)
    registry = builtin_registry()
    for case in CASES:
        graph = Graph()

        def add(kind, name, position, **parameters):
            block = graph.add(registry.create(kind, block_id=name, parameters=parameters))
            block.position = list(position)
            return block

        for row, name in enumerate(("source", "target")):
            add(
                "input.file",
                name,
                (0, row * 320),
                path=f"assets/{name}.png",
                path_base="project",
                color_format="gray",
            )
        add(
            "input.flow_file",
            "load",
            (0, 640),
            path="assets/computed.flo" if case == "computed_motion" else "assets/exact.flo",
        )
        if case == "bare_file":
            flow_id = "load"
        else:
            add(
                "output.flow_file",
                "save",
                (360, 240),
                path="output/motion.flo",
                metadata_mode="require_images",
                unique_name=True,
                create_directories=True,
                source_label="before",
                target_label="after",
            )
            graph.connect("load", "flow", "save", "flow")
            for name in ("source", "target"):
                graph.connect(name, "image", "save", name)
            add("input.flow_file", "reload", (720, 240), metadata_mode="verify_images")
            graph.connect("save", "path", "reload", "path")
            for name in ("source", "target"):
                graph.connect(name, "image", "reload", name)
            flow_id = "reload"
        warp = add(
            "cv.warp_flow",
            "warp",
            (1080, 0),
            direction=("target_to_source" if case == "direction_mismatch" else "source_to_target"),
            interpolation="nearest",
            consistency="off",
        )
        warp.exposed_inputs.append("target")
        graph.connect("target" if case == "wrong_image" else "source", "image", "warp", "image")
        graph.connect("target", "image", "warp", "target")
        graph.connect(flow_id, "flow", "warp", "flow")
        note = add("layout.comment", "scope", (360, 700))
        note.name = "Original fixtures; displacement and association contracts"
        note.node_size = "large"
        note.description = (
            "Original synthetic images, fields and project: OpenCVDataflowGUI contributors, MIT. "
            "No third-party/customer images. Analytic truth is dx=3, dy=-2 local pixels. Computed "
            "field uses Farneback; its error is measured against independent analytic truth. "
            "Roundtrip writes a fresh UUID-named .flo plus SHA-bound image-pair manifest into "
            "the ignored output folder, reloads with image verification and warps the source. "
            "Bare file has no direction/pair provenance. Direction mismatch and wrong image "
            "deliberately stop Warp as invalid, with no image emission. Fingerprints establish "
            "pixel/grid identity, not camera identity, acquisition timing, visibility "
            "or motion accuracy."
        )
        graph.save(HERE / f"{case}.json")
        graph.close()
    truth = {
        "creator": "OpenCVDataflowGUI contributors",
        "license": "MIT",
        "cases": list(CASES),
        "translation": [3, -2],
        "exact_covered_pixels": 94 * 125,
        "computed_interior_median_endpoint_error": float(np.median(error)),
        "computed_interior_p95_endpoint_error": float(np.percentile(error, 95)),
        "asset_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(assets.iterdir())
        },
    }
    (HERE / "truth.json").write_bytes((json.dumps(truth, indent=2) + "\n").encode())


if __name__ == "__main__":
    build()
