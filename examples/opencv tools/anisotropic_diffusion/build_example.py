"""Build a registered step target with seeded noise for diffusion comparison."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

SHAPE = (96, 160)
EDGE = 81
RECTANGLE = (19, 10, 122, 67)


def images():
    clean = np.full(SHAPE, 57, np.uint8)
    clean[:, EDGE:] = 178
    noise = np.random.default_rng(2067).normal(0, 11.5, SHAPE)
    noisy = np.rint(np.clip(clean.astype(np.float32) + noise, 1, 254)).astype(np.uint8)
    return {"clean": clean, "noisy": noisy}


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name, image in images().items():
        ok, png = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode {name}")
        (assets / f"{name}.png").write_bytes(png.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Preserve a known boundary", 0, -230)
        note.node_size = "large"
        note.description = (
            "Run Once to compare one and ten passes. The higher-sensitivity branch shows how "
            "larger K smooths across the step. The region branch retains the input outside "
            "the rectangle. Original MIT assets."
        )
        add(
            "input.file",
            "noisy",
            "Noisy step target",
            0,
            40,
            path="assets/noisy.png",
            path_base="project",
            color_format="gray",
        )
        region = add(
            "region.shape", "region", "Inspection rectangle", 235, 430, rectangle=list(RECTANGLE)
        )
        region.exposed_outputs = ("region",)
        graph.connect("noisy", "image", "region", "image")
        for index, (key, title, sensitivity, iterations) in enumerate(
            (
                ("early", "One diffusion pass", 0.12, 1),
                ("refined", "Ten edge-preserving passes", 0.12, 10),
                ("permissive", "Ten passes with larger K", 0.30, 10),
                ("selected", "Ten passes in region", 0.12, 10),
            )
        ):
            add(
                "cv.anisotropic_diffusion",
                key,
                title,
                505,
                -55 + index * 200,
                edge_sensitivity=sensitivity,
                iterations=iterations,
            )
            graph.connect("noisy", "image", key, "image")
        graph.connect("region", "region", "selected", "region")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise ValueError(checked["issues"])
        graph.save(output / "anisotropic_diffusion.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
