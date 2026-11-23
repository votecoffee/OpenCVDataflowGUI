"""Rebuild the original Dense RLOF translated texture and saved graph."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

FILENAMES = ("00_reference.png", "01_translated.png")
SHIFT = (3, 2)
INTERIOR = (24, 24, 72, 48)


def images():
    rng = np.random.default_rng(240)
    texture = rng.integers(8, 248, (96, 120, 3), np.uint8)
    reference = cv2.GaussianBlur(texture, (0, 0), 1)
    dx, dy = SHIFT
    shifted = cv2.warpAffine(
        reference,
        np.float32([[1, 0, dx], [0, 1, dy]]),
        (reference.shape[1], reference.shape[0]),
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return zip(FILENAMES, (reference, shifted), strict=True)


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name, image in images():
        ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode {name}")
        (assets / name).write_bytes(encoded.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(type_id, block_id, name, x, y, **parameters):
        block = graph.add(registry.create(type_id, block_id=block_id, parameters=parameters))
        block.name, block.position = name, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Compare local support windows", 0, -240)
        note.node_size = "large"
        note.description = (
            "Run All replays two color images. The second moves 3 px right and 2 px down. "
            "Both branches use geodesic interpolation and the same interior region. "
            "Select a flow block to inspect motion and round-trip error. Original MIT assets."
        )
        add(
            "input.file",
            "frames",
            "Translated texture",
            0,
            60,
            source_mode="files",
            paths=[f"assets/{name}" for name in FILENAMES],
            path_base="project",
            repeat="stop",
        )
        add("region.shape", "interior", "Interior measurement", 0, 350, rectangle=list(INTERIOR))
        graph.connect("frames", "image", "interior", "image")
        for index, support in enumerate(("fixed", "cross")):
            flow = add(
                "sequence.dense_rlof",
                support,
                f"{support.title()} support",
                385,
                index * 300,
                support=support,
                interpolation="geodesic",
                max_gap_ms=0,
                seed_consistency_threshold=0.75,
                consistency_tolerance=0.1,
            )
            flow.exposed_outputs = (
                "flow",
                "visualization",
                "consistency_mask",
                "consistency_error",
                "status",
            )
            flow.preview_output = "visualization"
            graph.connect("frames", "image", support, "image")
            graph.connect("interior", "region", support, "region")
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise RuntimeError(validation["issues"])
        graph.save(output / "dense_rlof.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
