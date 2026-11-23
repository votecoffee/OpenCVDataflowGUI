"""Generate a straight-edge target and a Fast Line Detector project."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def target_image():
    target = np.full((144, 176), 35, np.uint8)
    cv2.rectangle(target, (16, 25), (123, 49), 195, -1)
    cv2.rectangle(target, (141, 19), (156, 112), 212, -1)
    cv2.fillConvexPoly(target, np.array([[24, 77], [70, 123], [86, 107], [40, 61]]), 173)
    return target


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", target_image(), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise OSError("Could not encode the straight-edge target")
    (assets / "target.png").write_bytes(encoded.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(type_id, key, name, x, y, **parameters):
        block = graph.add(registry.create(type_id, block_id=key, parameters=parameters))
        block.name, block.position = name, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Measure straight edges", 0, -270)
        note.node_size = "large"
        note.description = (
            "Compare detection on intensity samples and a separate Canny edge map. "
            "Result arrows show endpoint order. Sort the intensity branch by length. "
            "The synthetic straight-edge target is an original MIT-licensed asset."
        )
        add(
            "input.file",
            "target",
            "Straight-edge target",
            0,
            40,
            path="assets/target.png",
            path_base="project",
            color_format="gray",
        )
        add("cv.canny", "edges", "Precompute Canny edges", 255, 230, low=50.0, high=50.0)
        add("inspect.fast_line_segments", "intensity", "Detect on intensity image", 520, 40)
        add(
            "inspect.fast_line_segments",
            "edge_map",
            "Detect using edge map",
            520,
            230,
            canny_aperture="0",
        )
        add("collection.sort", "sorted", "Longest segments first", 805, 40, metric="length")
        graph.connect("target", "image", "intensity", "image")
        graph.connect("target", "image", "edges", "image")
        graph.connect("edges", "image", "edge_map", "image")
        graph.connect("intensity", "collection", "sorted", "collection")
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise RuntimeError(validation["issues"])
        graph.save(output / "fast_line_detection.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
