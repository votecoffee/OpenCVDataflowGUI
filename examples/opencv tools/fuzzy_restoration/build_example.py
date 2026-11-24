"""Build a noisy intensity ramp and a copy with one known hole."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HOLE = [36, 25, 7, 7]


def make_images():
    y, x = np.indices((72, 96))
    clean = np.rint(48 + 0.9 * x + 0.6 * y).astype(np.uint8)
    rng = np.random.default_rng(121)
    noisy = np.rint(clean + rng.normal(0, 7, clean.shape)).clip(0, 255).astype(np.uint8)
    damaged = clean.copy()
    x, y, w, h = HOLE
    damaged[y : y + h, x : x + w] = 0
    return {"clean": clean, "noisy": noisy, "damaged": damaged}


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name, image in make_images().items():
        ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode {name}")
        (assets / f"{name}.png").write_bytes(encoded.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add(
            "layout.comment", "note", "Compare fuzzy smoothing and three repair methods", 0, -265
        )
        note.node_size = "large"
        note.description = (
            "Run Once. Smoothing reduces noise on the upper ramp. Three fill algorithms repair "
            "the lower ramp inside the marked rectangle. Compare their images with "
            "assets/clean.png. The filmstrip shows the iterative result. Original MIT assets."
        )
        for name, y in (("noisy", 30), ("damaged", 300)):
            add(
                "input.file",
                name,
                name.capitalize() + " ramp",
                0,
                y,
                path=f"assets/{name}.png",
                path_base="project",
                color_format="unchanged",
            )
        add("cv.fuzzy_filter", "smooth", "Smooth the noisy ramp", 360, 30, radius=3)
        graph.connect("noisy", "image", "smooth", "image")
        region = add("region.shape", "area", "Known hole", 365, 435, rectangle=HOLE)
        region.exposed_outputs = ("region",)
        graph.connect("damaged", "image", "area", "image")
        for key, label, y, radius in (
            ("one_step", "Repair at radius 4", 235, 4),
            ("multi_step", "Find a supported radius", 430, 2),
            ("iterative", "Reuse completed estimates", 625, 2),
        ):
            add(
                "cv.fuzzy_inpaint",
                key,
                label,
                710,
                y,
                algorithm=key,
                radius=radius,
                maximum_radius=8,
            )
            graph.connect("damaged", "image", key, "image")
            graph.connect("area", "region", key, "region")
        add("output.filmstrip_image", "filmstrip", "Review iterative repair", 1030, 625)
        graph.connect("iterative", "image", "filmstrip", "image")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise RuntimeError(checked["issues"])
        graph.save(output / "fuzzy_restoration.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
