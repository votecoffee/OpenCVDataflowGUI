"""Build a Fast Hough line example with known endpoints and offsets."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def image():
    target = np.zeros((104, 148), np.uint8)
    for start, end in (((36, 0), (36, 103)), ((0, 48), (147, 48)), ((0, 25), (147, 82))):
        cv2.line(target, start, end, 255, 1)
    cv2.line(target, (100, 11), (117, 17), 180, 1)
    cv2.line(target, (125, 91), (140, 90), 180, 1)
    return target


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    valid, png = cv2.imencode(".png", image(), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not valid:
        raise OSError("Could not encode the Fast Hough edge target")
    (assets / "edge_lines.png").write_bytes(png.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Inspect Fast Hough lines and peaks", 0, -230)
        note.node_size = "large"
        note.description = (
            "Run Once. Three known lines cross this edge map. See their overlays in the "
            "Fast Hough block. View the bin grid in the Normalize block. A filter and "
            "overlay retain the vertical line. Original MIT assets."
        )
        add(
            "input.file",
            "source",
            "Known edge lines",
            0,
            60,
            path="assets/edge_lines.png",
            color_format="gray",
            path_base="project",
        )
        detect = add(
            "inspect.fast_hough_lines",
            "detect",
            "Find three line peaks",
            280,
            60,
            minimum_lines=3,
        )
        add("cv.normalize", "display", "Scale accumulator for display", 590, -120)
        add(
            "collection.filter",
            "vertical",
            "Keep near-vertical lines",
            590,
            115,
            metric="angle",
            minimum=85,
            use_maximum=True,
            maximum=95,
        )
        add(
            "collection.draw",
            "draw",
            "Review vertical line",
            895,
            115,
            draw_bounds=False,
            labels="none",
        )
        detect.exposed_outputs = ("image", "accumulator", "collection")
        graph.connect("source", "image", "detect", "image")
        graph.connect("detect", "accumulator", "display", "image")
        graph.connect("detect", "collection", "vertical", "collection")
        graph.connect("vertical", "collection", "draw", "collection")
        graph.connect("detect", "image", "draw", "image")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise RuntimeError(checked["issues"])
        graph.save(output / "fast_hough_lines.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
