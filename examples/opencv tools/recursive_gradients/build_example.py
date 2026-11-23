"""Build a step target for comparing signed Deriche and Paillou responses."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def image():
    target = np.full((116, 154), 40, np.uint8)
    target[14:64, 36:76] = 161
    target[79:100, 88:142] = 181
    return target


def build(output):
    asset = output / "assets"
    asset.mkdir(parents=True, exist_ok=True)
    ok, png = cv2.imencode(".png", image(), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise OSError("Could not encode the step target")
    (asset / "steps.png").write_bytes(png.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Compare signed gradient responses", 0, -230)
        note.node_size = "large"
        note.description = (
            "Run Once. Both filters expose signed dx and dy and a magnitude preview. "
            "Rising edges are positive and falling edges are negative. The lower branch "
            "publishes Deriche results inside the inspection rectangle. Original MIT assets."
        )
        add(
            "input.file",
            "steps",
            "Known step edges",
            0,
            40,
            path="assets/steps.png",
            color_format="gray",
            path_base="project",
        )
        region = add(
            "region.shape",
            "region",
            "Inspection rectangle",
            220,
            390,
            rectangle=[21, 19, 70, 39],
        )
        region.exposed_outputs = ("region",)
        graph.connect("steps", "image", "region", "image")
        for index, (key, family, title) in enumerate(
            (
                ("deriche", "deriche", "OpenCV Deriche"),
                ("paillou", "paillou", "Paillou recurrence"),
                ("selected", "deriche", "Deriche in region"),
            )
        ):
            add(f"cv.{family}_gradient", key, title, 485, -45 + index * 205)
            graph.connect("steps", "image", key, "image")
        graph.connect("region", "region", "selected", "region")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise RuntimeError(checked["issues"])
        graph.save(output / "recursive_gradients.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
