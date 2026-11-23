"""Build a textured 16-bit target for comparing smoothing strength."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

SHAPE = (91, 163)
RECTANGLE = (14, 7, 136, 72)
EDGE = 83


def images():
    y, x = np.indices(SHAPE)
    clean = np.where(x < EDGE, 14500, 42400).astype(np.uint16)
    clean[19:71, 120:125] = 48200
    texture = np.where((x // 4 + y // 6) % 2 == 0, 1250, -1250)
    noise = np.random.default_rng(7731).normal(0, 215, SHAPE)
    textured = np.rint(clean.astype(float) + texture + noise).astype(np.uint16)
    return {"clean": clean, "textured": textured}


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name, image in images().items():
        ok, png = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode {name}")
        (assets / f"{name}.png").write_bytes(png.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Remove texture and preserve edges", 0, -220)
        note.node_size = "large"
        note.description = (
            "Run Once. Compare Gaussian blur and two L0 weights on the same textured 16-bit "
            "signal. The weaker L0 setting retains a thin stripe. The rectangle branch "
            "keeps source pixels outside the region. Original MIT assets."
        )
        add(
            "input.file",
            "textured",
            "Textured 16-bit signal",
            0,
            40,
            path="assets/textured.png",
            path_base="project",
            pixel_depth="unchanged",
            color_format="gray",
        )
        region = add(
            "region.shape", "region", "Inspection rectangle", 225, 395, rectangle=list(RECTANGLE)
        )
        region.exposed_outputs = ("region",)
        graph.connect("textured", "image", "region", "image")
        add(
            "cv.gaussian_blur",
            "gaussian",
            "Gaussian baseline",
            495,
            -55,
            kernel=11,
            sigma=2.4,
        )
        for index, (key, name, weight) in enumerate(
            (
                ("moderate", "L0 with stripe detail", 0.001),
                ("strong", "L0 with stronger smoothing", 0.02),
                ("selected", "L0 within a rectangle", 0.001),
            )
        ):
            add("cv.l0_smoothing", key, name, 495, 145 + index * 202, lambda_weight=weight)
        for key in ("gaussian", "moderate", "strong", "selected"):
            graph.connect("textured", "image", key, "image")
        graph.connect("region", "region", "selected", "region")
        check = validate_graph(graph)
        if not check["valid"]:
            raise RuntimeError(check["issues"])
        graph.save(output / "l0_smoothing.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
