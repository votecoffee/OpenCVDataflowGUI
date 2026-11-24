"""Create an original shape plate and an Edge Drawing inspection graph."""

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
    image = np.full((240, 360, 3), (24, 31, 38), np.uint8)
    cv2.rectangle(image, (20, 28), (158, 61), (194, 211, 238), -1, cv2.LINE_AA)
    cv2.circle(image, (85, 154), 34, (185, 201, 226), -1, cv2.LINE_AA)
    cv2.ellipse(image, (270, 115), (53, 30), 27, 0, 360, (198, 215, 239), -1, cv2.LINE_AA)
    ok, data = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise OSError("Could not encode shape plate")
    (assets / "plate.png").write_bytes(data.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Trace paths and fit their geometry", 0, -220)
        note.node_size = "large"
        note.description = (
            "Run Once. The original MIT shape plate has a rectangle, a circle of radius 34, "
            "and an ellipse with 106 × 60 pixel axes rotated 27°. Edge Drawing returns "
            "ordered paths and fitted shapes. Filter for lines at least 80 pixels long."
        )
        add(
            "input.file", "image", "Known geometry plate", 0, 60,
            path="assets/plate.png", path_base="project", color_format="bgr",
        )
        edge = add("inspect.edge_drawing", "edge", "Paths, lines and round fits", 290, 65)
        edge.node_size = "large"
        add("collection.filter", "long", "Keep lines ≥ 80 px", 600, 50, metric="length", minimum=80.0)
        add("collection.draw", "review", "Review the long sides", 890, 135, labels="none")
        graph.connect("image", "image", "edge", "image")
        graph.connect("edge", "lines", "long", "collection")
        graph.connect("long", "collection", "review", "collection")
        graph.connect("image", "image", "review", "image")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise RuntimeError(checked["issues"])
        graph.save(output / "edge_drawing.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
