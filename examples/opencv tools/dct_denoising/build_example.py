"""Generate a seeded noisy inspection target and its DCT denoising project."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def images():
    clean = np.full((88, 112, 3), (60, 92, 128), np.uint8)
    clean[:, 56:] = (172, 182, 196)
    cv2.circle(clean, (27, 56), 14, (159, 149, 133), -1)
    cv2.rectangle(clean, (73, 43), (98, 70), (90, 139, 161), -1)
    cv2.line(clean, (76, 66), (94, 47), (199, 208, 227), 2)
    rng = np.random.default_rng(88419)
    noisy = np.rint(clean + rng.normal(0, 12, size=clean.shape)).clip(0, 255).astype(np.uint8)
    return {"clean": clean, "noisy": noisy}


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name, image in images().items():
        ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode {name}")
        (assets / f"{name}.png").write_bytes(encoded.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(type_id, key, name, x, y, **parameters):
        block = graph.add(registry.create(type_id, block_id=key, parameters=parameters))
        block.name, block.position = name, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Clean up sensor noise", 0, -185)
        note.node_size = "large"
        note.description = (
            "Compare the noisy target with full-frame and circle-limited DCT denoising. "
            "Inspect fine edges and flat patches at full zoom. "
            "Both synthetic image assets are original and MIT-licensed."
        )
        add(
            "input.file",
            "noisy",
            "Simulated sensor image",
            0,
            135,
            path="assets/noisy.png",
            path_base="project",
            pixel_depth="unchanged",
            color_format="unchanged",
        )
        region = add(
            "region.shape",
            "area",
            "Circular inspection area",
            280,
            260,
            shape="circle",
            center=[27, 56],
            radius=22,
        )
        region.exposed_outputs = ("region",)
        add("cv.dct_denoise", "full", "Whole-frame denoising", 540, 0, noise_sigma=12.0)
        add("cv.dct_denoise", "selected", "Denoise the inspection area", 540, 215, noise_sigma=12.0)
        graph.connect("noisy", "image", "area", "image")
        graph.connect("noisy", "image", "full", "image")
        graph.connect("noisy", "image", "selected", "image")
        graph.connect("area", "region", "selected", "region")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise RuntimeError(checked["issues"])
        graph.save(output / "dct_denoising.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
