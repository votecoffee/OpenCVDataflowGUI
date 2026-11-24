"""Build a grayscale step-and-detail fixture with seeded Gaussian noise."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def fixture():
    height, width = 80, 112
    y, x = np.indices((height, width))
    clean = np.where(x < width // 2, 66.0, 158.0) + 6 * np.cos(y / 7)
    clean[(y >= 25) & (y < 55) & (x >= 22) & (x < 44)] += 35
    clean[(y >= 28) & (y < 51) & (x >= 78) & (x < 98)] -= 28
    clean = np.rint(clean).astype(np.uint8)
    noise = np.random.default_rng(4921).normal(0, 13, clean.shape)
    noisy = np.clip(np.rint(clean + noise), 0, 255).astype(np.uint8)
    return {"clean": clean, "noisy": noisy}


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name, image in fixture().items():
        ok, data = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode {name}")
        (assets / f"{name}.png").write_bytes(data.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Denoise a known step edge", 0, -245)
        note.node_size = "large"
        note.description = (
            "Run Once. The first stage groups similar patches and hard-thresholds their "
            "coefficients. The Wiener stage refines the estimate. Compare Basic and Image "
            "with assets/clean.png. Original MIT-licensed synthetic fixture."
        )
        add(
            "input.file",
            "noisy",
            "Noisy grayscale",
            0,
            85,
            path="assets/noisy.png",
            path_base="project",
            color_format="gray",
        )
        add("cv.bm3d_denoise", "denoise", "Two-stage BM3D", 315, 85, noise_sigma=13.0)
        graph.connect("noisy", "image", "denoise", "image")
        add("output.filmstrip_image", "filmstrip", "Review the denoised image", 625, 85)
        graph.connect("denoise", "image", "filmstrip", "image")
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise RuntimeError(validation["issues"])
        graph.save(output / "bm3d_denoising.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
