"""Create a labeled synthetic scene and a graph that measures HFS segments."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def make_images():
    truth = np.zeros((97, 145), np.uint8)
    truth[24:72, 36:72] = 1
    truth[24:72, 90:126] = 2
    colors = np.array([[40, 60, 80], [30, 215, 40], [210, 45, 25]], np.int16)
    rng = np.random.default_rng(71)
    image = colors[truth] + rng.integers(-4, 5, size=(*truth.shape, 1), dtype=np.int16)
    return image.astype(np.uint8), truth


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name, image in zip(("scene", "truth"), make_images(), strict=True):
        ok, data = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode {name}")
        (assets / f"{name}.png").write_bytes(data.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, position, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, list(position)
        return block

    try:
        note = add("layout.comment", "note", "Hierarchical segmentation", (0, -230))
        note.node_size = "large"
        note.description = (
            "Run Once. The HFS block segments two colored panels and their background. "
            "Inspect mean colors, label IDs and boundaries. Measure label regions lists "
            "each segment's size and position. Original generated images under MIT."
        )
        add(
            "input.file", "scene", "Two colored panels", (0, 30),
            path="assets/scene.png", path_base="project", color_format="unchanged",
        )
        add("cv.hfs_segments", "segments", "Separate the panels and background", (320, 30))
        add("inspect.measure_label_regions", "measure", "Measure each region", (655, 245))
        add("output.filmstrip_image", "filmstrip", "Review region colors", (655, 30))
        graph.connect("scene", "image", "segments", "image")
        graph.connect("segments", "labels", "measure", "labels")
        graph.connect("segments", "image", "filmstrip", "image")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise RuntimeError(checked["issues"])
        graph.save(output / "hfs_segmentation.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
