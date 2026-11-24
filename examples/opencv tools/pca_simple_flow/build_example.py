"""Build a two-frame translation fixture for PCAFlow and SimpleFlow."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

FILENAMES = ("00_reference.png", "01_shifted.png")


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    seed = np.random.default_rng(9084)
    pixels = cv2.GaussianBlur(seed.integers(15, 240, (96, 128), np.uint8), (0, 0), 1.2)
    for name, image in zip(FILENAMES, (pixels, np.roll(pixels, (-2, 3), (0, 1))), strict=True):
        ok, data = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode {name}")
        (assets / name).write_bytes(data.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Measure a known image translation", 0, -255)
        note.node_size = "large"
        note.description = (
            "Run All. The second image moves 3 px right and 2 px up. Compare each optical "
            "flow preview and consistency mask. The synthetic texture is an original MIT asset."
        )
        add(
            "input.file",
            "frames",
            "Reference and shifted view",
            0,
            115,
            source_mode="files",
            paths=[f"assets/{name}" for name in FILENAMES],
            color_format="bgr",
            path_base="project",
            repeat="stop",
        )
        for key, title, y in (("pcaflow", "PCAFlow", 0), ("simpleflow", "SimpleFlow", 240)):
            node = add(
                f"sequence.{key}",
                key,
                title,
                305,
                y,
                max_gap_ms=0,
                consistency_tolerance=0.25,
            )
            node.exposed_outputs = ["flow", "visualization", "consistency_mask", "status"]
            node.preview_output = "visualization"
            graph.connect("frames", "image", key, "image")
        add("output.filmstrip_image", "filmstrip", "Review SimpleFlow", 620, 240)
        graph.connect("simpleflow", "visualization", "filmstrip", "image")
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise RuntimeError(validation["issues"])
        graph.save(output / "pca_simple_flow.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
