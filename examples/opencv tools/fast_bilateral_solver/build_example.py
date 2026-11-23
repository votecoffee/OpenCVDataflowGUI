"""Build the sparse confidence and edge-preservation example."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def images():
    guide = np.empty((96, 144), np.uint8)
    guide[:, :48], guide[:, 48:96], guide[:, 96:] = 32, 144, 232
    signal, confidence = np.zeros_like(guide), np.zeros_like(guide)
    signal[:, 96:] = 83
    anchors = ((9, 10), (33, 42), (16, 74), (54, 15), (78, 40), (88, 70))
    for index, (x, y) in enumerate(anchors):
        signal[y : y + 5, x : x + 5] = 53 if index < 3 else 187
        confidence[y : y + 5, x : x + 5] = 255
    return {"signal": signal, "guide": guide, "confidence": confidence}


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name, raster in images().items():
        ok, png = cv2.imencode(".png", raster, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode {name}")
        (assets / f"{name}.png").write_bytes(png.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, block_id, name, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=block_id, parameters=parameters))
        block.name, block.position = name, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Recover a sparsely measured signal", 0, -230)
        note.node_size = "large"
        note.description = (
            "Run Once. Six measured patches supply confidence values in the first two bands. "
            "The guide keeps their edges separate. The unobserved third band keeps its input "
            "and stays black in the Supported output. Original MIT assets."
        )
        for index, (key, title) in enumerate(
            (
                ("signal", "Sparse measurements"),
                ("guide", "Edge guide"),
                ("confidence", "Trusted sample pixels"),
            )
        ):
            add(
                "input.file",
                key,
                title,
                0,
                index * 205 + 35,
                path=f"assets/{key}.png",
                path_base="project",
                color_format="gray",
            )
        block = add(
            "cv.fast_bilateral_solver",
            "solver",
            "Reconstruct measured bands",
            385,
            235,
            alignment="pixel_grid",
            lambda_weight=32.0,
            max_tolerance=1e-6,
        )
        block.exposed_outputs = ("image", "supported", "report")
        for key, port in (("signal", "image"), ("guide", "guide"), ("confidence", "confidence")):
            graph.connect(key, "image", "solver", port)
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise ValueError(validation["issues"])
        graph.save(output / "sparse_signal.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
