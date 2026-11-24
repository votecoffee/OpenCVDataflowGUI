"""Build a two-frame replay with a known change in motion at a color boundary."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

FILENAMES = ("00_reference.png", "01_shifted.png")


def fixture_images():
    rng = np.random.default_rng(4814)
    panels = []
    for low, high in ((28, 125), (135, 252)):
        pixels = rng.integers(low, high, (112, 88), np.uint8)
        texture = cv2.GaussianBlur(pixels, (0, 0), 1)
        panels.append(cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR))
    return np.hstack(panels), np.hstack(
        (np.roll(panels[0], 4, axis=0), np.roll(panels[1], -3, axis=0))
    )


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name, image in zip(FILENAMES, fixture_images(), strict=True):
        ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode {name}")
        (assets / name).write_bytes(encoded.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Two motions, one image pair", 0, -245)
        note.node_size = "large"
        note.description = (
            "Run All plays an original MIT two-panel texture. Left moves 4 px down. Right "
            "moves 3 px up. Compare the change in direction at the center of each preview."
        )
        add(
            "input.file",
            "frames",
            "Two-panel replay",
            0,
            120,
            source_mode="files",
            paths=[f"assets/{name}" for name in FILENAMES],
            color_format="bgr",
            path_base="project",
            repeat="stop",
        )
        for key, title, y, settings in (
            ("default", "Default grid and smoothing", 0, {}),
            (
                "coarse",
                "Coarse grid without smoothing",
                245,
                {"grid_step": 12, "neighbors": 32, "sigma": 0.07, "post_smoothing": False},
            ),
        ):
            flow = add(
                "sequence.sparse_to_dense",
                key,
                title,
                305,
                y,
                max_gap_ms=0,
                consistency_tolerance=0.25,
                **settings,
            )
            flow.exposed_outputs = ["flow", "visualization", "consistency_mask", "status"]
            flow.preview_output = "visualization"
            graph.connect("frames", "image", key, "image")
        add("output.filmstrip_image", "filmstrip", "Review default flow", 620, 0)
        graph.connect("default", "visualization", "filmstrip", "image")
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise RuntimeError(validation["issues"])
        graph.save(output / "sparse_to_dense_flow.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
