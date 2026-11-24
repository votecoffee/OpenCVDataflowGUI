"""Generate a printed-text chart and independent glyph bounds."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

SHAPE = (270, 430)
WORDS = (
    ("INLET", (32, 61), (27, 32, 38)),
    ("OUTLET", (64, 154), (216, 206, 195)),
    ("VALVE", (178, 245), (126, 133, 140)),
)


def chart():
    image = np.full((*SHAPE, 3), (214, 220, 226), np.uint8)
    image[90:180] = (44, 53, 61)
    image[180:] = (195, 201, 205)
    words = []
    for text, origin, color in WORDS:
        support = np.zeros(SHAPE, np.uint8)
        settings = (cv2.FONT_HERSHEY_SIMPLEX, 1.6)
        cv2.putText(support, text, origin, *settings, 255, 3, cv2.LINE_AA)
        cv2.putText(image, text, origin, *settings, color, 3, cv2.LINE_AA)
        bounds = cv2.boundingRect((support >= 128).astype(np.uint8))
        words.append({"text": text, "bounds": list(bounds)})
    return image, words


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    image, words = chart()
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise OSError("Could not encode the text chart")
    (assets / "text_chart.png").write_bytes(encoded.tobytes())
    (output / "truth.json").write_text(
        json.dumps({"words": words}, indent=2) + "\n", encoding="utf-8"
    )

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Find words before OCR", 0, -245)
        note.node_size = "large"
        note.description = (
            "Choose Run Once. Inspect the group boxes for three labels at different contrasts. "
            "The component branch shows individual candidates. Original MIT-licensed chart. "
            "Detection uses OpenCV's bundled Apache-2.0 NM1/NM2 classifiers."
        )
        add(
            "input.file",
            "source",
            "Contrast chart",
            0,
            140,
            path="assets/text_chart.png",
            path_base="project",
            color_format="unchanged",
        )
        add("cv.text_extremal_regions", "groups", "Group horizontal text", 350, 30, minimum_count=3)
        add("cv.text_extremal_regions", "parts", "Detect components", 350, 255, mode="components")
        add("output.filmstrip_image", "filmstrip", "Review grouped words", 715, 30)
        for key in ("groups", "parts"):
            graph.connect("source", "image", key, "image")
        graph.connect("groups", "image", "filmstrip", "image")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise RuntimeError(checked["issues"])
        graph.save(output / "native_text_regions.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
