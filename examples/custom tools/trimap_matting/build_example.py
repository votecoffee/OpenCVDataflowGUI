"""Build a synthetic matting job with independent analytic alpha ground truth."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def fixture():
    y, x = np.indices((108, 144), dtype=np.float64)
    radius = np.hypot(x - 72, (y - 54) * 1.12)
    alpha = np.clip(np.minimum((43.5 - radius) / 5, (radius - 19.5) / 5), 0, 1)
    for center in (44, 100):
        hole = np.hypot(x - center, y - 54)
        alpha *= np.clip((hole - 4) / 2, 0, 1)
    texture = (np.sin(x / 24) + np.cos(y / 18))[..., None]
    background = np.array([30.0, 66.0, 88.0]) + texture
    foreground = np.array([199.0, 146.0, 53.0]) + texture * [1.5, 1.0, 0.5]
    image = np.rint(alpha[..., None] * foreground + (1 - alpha[..., None]) * background)
    trimap = np.full(alpha.shape, 128, np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    for label, known in ((0, alpha == 0), (255, alpha == 1)):
        trimap[cv2.erode(known.astype(np.uint8), kernel, iterations=2) != 0] = label
    truth = np.rint(alpha * 65535).astype(np.uint16)
    return {"composite": image.astype(np.uint8), "trimap": trimap, "truth": truth}


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name, image in fixture().items():
        ok, data = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode {name}")
        (assets / f"{name}.png").write_bytes(data.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Estimate the gasket's soft edge", 0, -245)
        note.node_size = "large"
        note.description = (
            "Run Once. Black and white trimap pixels fix the background and foreground. "
            "The gray band is solved. Compare the matte with the supplied 16-bit truth "
            "image. The fixture is original MIT-licensed artwork."
        )
        add(
            "input.file",
            "image",
            "Composite gasket",
            0,
            0,
            path="assets/composite.png",
            path_base="project",
            color_format="bgr",
        )
        add(
            "input.file",
            "trimap",
            "Background / unknown / foreground",
            0,
            230,
            path="assets/trimap.png",
            path_base="project",
            color_format="gray",
        )
        add("cv.trimap_matting", "matte", "Estimate soft alpha", 320, 85)
        graph.connect("image", "image", "matte", "image")
        graph.connect("trimap", "image", "matte", "trimap")
        add("output.filmstrip_image", "filmstrip", "Review estimated alpha", 650, 85)
        graph.connect("matte", "alpha", "filmstrip", "image")
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise RuntimeError(validation["issues"])
        graph.save(output / "trimap_matting.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
