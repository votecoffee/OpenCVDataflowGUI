"""Build an original DeepFlow sequence with independently rendered occlusion."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

SHIFT = (5, -2)
OCCLUDER = (66, 29, 34, 41)
FILENAMES = ("00_start.png", "01_shifted.png", "02_occluded.png")


def images():
    rng = np.random.default_rng(5528)
    background = cv2.GaussianBlur(rng.integers(31, 68, (96, 132), np.uint8), (0, 0), 1.4)
    patch = cv2.GaussianBlur(rng.integers(70, 245, (58, 68), np.uint8), (0, 0), 1.2)
    cv2.circle(patch, (18, 25), 9, 72, -1)
    cv2.rectangle(patch, (34, 19), (56, 37), 205, -1)
    frames = []
    dx, dy = SHIFT
    for index in range(len(FILENAMES)):
        frame = background.copy()
        x, y = 25 + dx * index, 23 + dy * index
        frame[y : y + patch.shape[0], x : x + patch.shape[1]] = patch
        frames.append(frame)
    x, y, width, height = OCCLUDER
    frames[-1][y : y + height, x : x + width] = 235
    return zip(FILENAMES, frames, strict=True)


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name, image in images():
        ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode {name}")
        (assets / name).write_bytes(encoded.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(type_id, key, name, x, y, **parameters):
        block = graph.add(registry.create(type_id, block_id=key, parameters=parameters))
        block.name, block.position = name, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Track a partly hidden patch", 0, -245)
        note.node_size = "large"
        note.description = (
            "Run All plays three frames. The patch moves 5 px right and 2 px up between frames. "
            "The final image adds an occluding rectangle. Compare the consistency mask to the "
            "hidden area described in the README. The synthetic images are original MIT assets."
        )
        add(
            "input.file",
            "frames",
            "Three-frame replay",
            0,
            50,
            source_mode="files",
            paths=[f"assets/{name}" for name in FILENAMES],
            path_base="project",
            color_format="gray",
            repeat="stop",
        )
        flow = add(
            "sequence.deepflow",
            "flow",
            "DeepFlow with consistency",
            340,
            50,
            max_gap_ms=0,
            consistency_tolerance=0.05,
        )
        flow.exposed_outputs = (
            "flow",
            "visualization",
            "consistency_mask",
            "consistency_error",
            "status",
        )
        flow.preview_output = "visualization"
        graph.connect("frames", "image", "flow", "image")
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise RuntimeError(validation["issues"])
        graph.save(output / "deepflow.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
