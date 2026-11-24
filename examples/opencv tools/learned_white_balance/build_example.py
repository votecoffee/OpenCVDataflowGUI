"""Build separate illuminated charts with a known neutral target."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

ILLUMINANTS = (
    ("warm", 112, (0.72, 1.00, 1.32)),
    ("cool", 812, (1.28, 1.00, 0.78)),
    ("green_deficit", 586, (1.06, 0.84, 0.98)),
)


def chart(seed, illuminant):
    rng = np.random.default_rng(seed)
    y, x = np.indices((114, 160))
    palette = rng.integers(30, 166, (6, 8, 3))
    scene = palette[y // 19, x // 20] * (0.83 + 0.19 * np.sin(x / 25 + y / 16))[..., None]
    scene += rng.normal(0, 2.5, scene.shape)
    scene[16:97, 14:98] = np.linspace(35, 165, 84)[None, :, None]
    return np.rint(np.clip(scene * illuminant, 0, 255)).astype(np.uint8)


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name, seed, illuminant in ILLUMINANTS:
        ok, encoded = cv2.imencode(".png", chart(seed, illuminant), [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode {name}")
        (assets / f"{name}.png").write_bytes(encoded.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Check a learned illuminant estimate", 0, -240)
        note.node_size = "large"
        note.description = (
            "Choose Run All to process three different synthetic illuminants. Compare the gray "
            "ramp in Source and Learned White Balance. The warm and cool ramps improve, while "
            "the green-deficit ramp gets worse. Original MIT-licensed charts."
        )
        add(
            "input.file", "source", "Three illuminated charts", 0, 125,
            source_mode="files", paths=[f"assets/{name}.png" for name, *_ in ILLUMINANTS],
            path_base="project", color_format="unchanged",
        )
        add("cv.learned_white_balance", "learned", "Learned white balance", 315, 125)
        add("output.filmstrip_image", "filmstrip", "Review corrected charts", 650, 125)
        graph.connect("source", "image", "learned", "image")
        graph.connect("learned", "image", "filmstrip", "image")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise RuntimeError(checked["issues"])
        graph.save(output / "learned_white_balance.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
