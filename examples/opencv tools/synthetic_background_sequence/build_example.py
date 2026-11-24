"""Recreate the original synthetic assets and tracking validation project."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    y, x = np.indices((128, 176))
    background = np.uint8(36 + 9 * np.sin(x / 7) + 10 * np.cos(y / 12) + 12 * (x // 12 % 2 == 0))
    target = np.full((24, 30, 3), (175, 211, 232), np.uint8)
    cv2.rectangle(target, (1, 1), (28, 22), (223, 235, 250), 2)
    cv2.line(target, (6, 18), (22, 5), (160, 184, 201), 3)
    for name, pixels in (("background", background), ("target", target)):
        ok, data = cv2.imencode(".png", pixels)
        if not ok:
            raise OSError(f"Could not encode {name}")
        (assets / f"{name}.png").write_bytes(data.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, name, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = name, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Check foreground across moving frames", 0, -205)
        note.node_size = "large"
        note.description = (
            "Choose Run all for 24 frames. OpenCV moves an opaque rectangle across a rippling "
            "background and supplies its exact mask. Compare the bright-pixel threshold with "
            "that mask. The expected IoU is 1.0 for every ticket. Original MIT fixtures."
        )
        source = add(
            "input.synthetic_background_sequence", "source", "Generate frame and reference mask",
            0, 25, background_path="assets/background.png", object_path="assets/target.png",
            amplitude=1.6, wavelength=25.0, wave_speed=0.25, object_speed=5.0, fps=12.0,
        )
        source.node_size = "large"
        threshold = add("cv.threshold", "threshold", "Segment bright pixels", 310, -15, value=130.0)
        compare = add(
            "inspect.mask_boundary", "compare", "Check exact mask and boundary", 580, 120,
            tolerance=0.0, check_area=True, area_basis="raw", tickets="require_match",
        )
        filmstrip = add("output.filmstrip_image", "filmstrip", "Review source frames", 880, 125)
        graph.connect(source.id, "image", threshold.id, "image")
        graph.connect(source.id, "image", compare.id, "image")
        graph.connect(source.id, "foreground_mask", compare.id, "reference")
        graph.connect(threshold.id, "image", compare.id, "sample")
        graph.connect(compare.id, "image", filmstrip.id, "image")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise RuntimeError(checked["issues"])
        graph.save(output / "synthetic_background_sequence.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
