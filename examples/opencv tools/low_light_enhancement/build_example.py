"""Generate a shadow/bright-target fixture and a low-light comparison graph."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def make_image():
    y, x = np.indices((120, 176), dtype=float)
    level = 12 + 0.05 * x + 0.03 * y + 3 * np.sin(0.16 * x) * np.cos(0.2 * y)
    level += 160 * np.clip((x - 89) / 32, 0, 1)
    level[21:61, 12:42] = 14
    level[21:61, 42:72] = 22
    level[73:108, 15:74] += 12 * ((x[73:108, 15:74] // 7) % 2)
    level[16:54, 128:163] = 225
    level[67:108, 128:163] = 245
    noise = np.random.default_rng(73).normal(0, 1.25, size=level.shape)
    color = (level + noise)[..., None] * np.array([0.93, 1, 1.045])
    return np.rint(color).clip(0, 255).astype(np.uint8)


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(".png", make_image(), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise OSError("Could not encode the low-light fixture")
    (assets / "shadow_chart.png").write_bytes(data.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, position, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, list(position)
        return block

    try:
        note = add("layout.comment", "note", "Compare three shadow enhancements", (0, -230))
        note.node_size = "large"
        note.description = (
            "Run Once. Compare automatic and explicit BIMEF with ordinary gamma correction. "
            "Inspect the paired dark patches on the left and the bright patches on the right. "
            "The filmstrip shows the automatic result. Original MIT-licensed chart."
        )
        add(
            "input.file", "source", "Noisy shadow chart", (0, 180),
            path="assets/shadow_chart.png", path_base="project", color_format="unchanged",
        )
        add("cv.bimef", "automatic", "Automatic exposure", (350, 30))
        add("cv.bimef", "explicit", "Fourfold exposure", (350, 215), exposure_mode="explicit")
        add("cv.gamma", "gamma", "Global gamma baseline", (350, 400), gamma=1.8)
        add("output.filmstrip_image", "filmstrip", "Review automatic enhancement", (685, 30))
        for key in ("automatic", "explicit", "gamma"):
            graph.connect("source", "image", key, "image")
        graph.connect("automatic", "image", "filmstrip", "image")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise RuntimeError(checked["issues"])
        graph.save(output / "low_light_enhancement.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
