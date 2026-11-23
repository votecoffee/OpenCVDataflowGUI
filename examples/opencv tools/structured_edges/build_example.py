"""Build a color edge target and its reference boundary mask."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    color = np.full((138, 182, 3), [35, 50, 69], dtype=np.uint8)
    edges = np.zeros(color.shape[:2], np.uint8)
    cv2.rectangle(color, (30, 25), (125, 101), (206, 155, 85), -1)
    cv2.circle(color, (151, 64), 17, (93, 199, 215), -1)
    cv2.rectangle(edges, (30, 25), (125, 101), 255, 1)
    cv2.circle(edges, (151, 64), 17, 255, 1)
    for name, image in (
        ("color_target", cv2.cvtColor(color, cv2.COLOR_RGB2BGR)),
        ("reference_boundaries", edges),
    ):
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
        note = add("layout.comment", "note", "Structured forest edge responses", 0, -205)
        note.node_size = "large"
        note.description = (
            "Prepare the OpenCV 5.0.0 reference model once, then Run Once. "
            "Compare detected edges with the independently drawn rectangle and circle boundaries. "
            "The images and graph are original MIT assets. The local reference forest comes "
            "from opencv_extra for the Structured Forests method by Dollár and Zitnick."
        )
        add(
            "input.file",
            "source",
            "Color shape target",
            0,
            60,
            path="assets/color_target.png",
            color_format="rgb",
            path_base="project",
        )
        add(
            "input.file",
            "reference",
            "Reference boundaries",
            0,
            290,
            path="assets/reference_boundaries.png",
            color_format="gray",
            path_base="project",
        )
        add(
            "cv.structured_edges",
            "detect",
            "Detect and thin edges",
            280,
            60,
            model_path=".local/models/model.yml.gz",
        )
        add("cv.normalize", "view", "View continuous thinned response", 580, 60)
        graph.connect("source", "image", "detect", "image")
        graph.connect("detect", "thinned", "view", "image")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise RuntimeError(checked["issues"])
        graph.save(output / "structured_edges.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
