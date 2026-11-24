"""Build an original textured-part comparison for local histogram smoothing."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

SHAPE = (104, 160)
RECTANGLE = (23, 24, 95, 69)


def images():
    clean = np.full((*SHAPE, 3), (54, 66, 75), dtype=np.uint8)
    clean[15:90, 56:138] = (114, 150, 177)
    clean[15:36, 115:138] = (54, 66, 75)
    cv2.circle(clean, (105, 63), 10, (54, 66, 75), -1)
    y, x = np.indices(SHAPE)
    lighting = np.where((x + 2 * y) % 5 == 0, 24, 0)
    variation = (3 * x + y) % 7 - 3
    textured = (clean.astype(np.int16) + (lighting + variation)[..., None]).astype(np.uint8)
    return {"clean": clean, "textured": textured}


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name, image in images().items():
        ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode {name}")
        (assets / f"{name}.png").write_bytes(encoded.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Smooth texture and inspect the part edge", 0, -250)
        note.node_size = "large"
        note.description = (
            "Run Once. Compare the straight left edge of the part in the local-histogram "
            "and Gaussian outputs. The region branch writes only inside the rectangle. "
            "Original synthetic assets are MIT-licensed."
        )
        add(
            "input.file", "source", "Textured part", 0, 135,
            path="assets/textured.png", path_base="project", color_format="unchanged",
        )
        region = add("region.shape", "region", "Selected rectangle", 260, 390,
                     rectangle=list(RECTANGLE))
        region.exposed_outputs = ("region",)
        add("cv.gaussian_blur", "gaussian", "Gaussian 9 × 9", 555, -45, kernel=9, sigma=2)
        add("cv.oil_painting", "smooth", "Oil painting 9 × 9", 555, 145)
        add("cv.oil_painting", "selected", "Oil painting inside rectangle", 555, 390)
        add("output.filmstrip_image", "filmstrip", "Review the full result", 860, 145)
        for key in ("region", "gaussian", "smooth", "selected"):
            graph.connect("source", "image", key, "image")
        graph.connect("region", "region", "selected", "region")
        graph.connect("smooth", "image", "filmstrip", "image")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise RuntimeError(checked["issues"])
        graph.save(output / "oil_painting.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
