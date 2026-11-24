"""Generate three printed labels with independent word geometry."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

LANGUAGE_SHA256 = "7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2"
LABELS = (("PUMP 7", 30, 22), ("VALVE 24", 52, 44), ("LOT 305", 110, 105))


def chart():
    image = np.full((255, 480, 3), 249, np.uint8)
    image[85:170] = 212
    image[170:] = 238
    words = []
    for index, (label, x, intensity) in enumerate(LABELS):
        settings = ((x, 62 + index * 85), cv2.FONT_HERSHEY_SIMPLEX, 1.55)
        cv2.putText(image, label, *settings, (intensity,) * 3, 3, cv2.LINE_AA)
        support = np.zeros(image.shape[:2], np.uint8)
        cv2.putText(support, label, *settings, 255, 3, cv2.LINE_AA)
        columns = np.flatnonzero((support >= 128).any(axis=0))
        left_end = columns[np.argmax(np.diff(columns))]
        for text, raster, offset in zip(
            label.split(),
            (support[:, : left_end + 1], support[:, left_end + 1 :]),
            (0, left_end + 1),
            strict=True,
        ):
            u, v, w, h = cv2.boundingRect((raster >= 128).astype(np.uint8))
            words.append({"text": text, "bounds": [int(u + offset), v, w, h], "crop_index": index})
    return image, words


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    image, words = chart()
    ok, data = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise OSError("Could not encode the OCR label chart")
    (assets / "labels.png").write_bytes(data.tobytes())
    (output / "truth.json").write_text(
        json.dumps({"words": words}, indent=2) + "\n", encoding="utf-8"
    )

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, name, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = name, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Read three printed equipment labels", 0, -245)
        note.node_size = "large"
        note.description = (
            "Run Once. The text detector finds three lines. Tesseract reads six words and "
            "numbers, then draws their boxes on the source image. Original MIT-licensed chart. "
            "English language data comes from tessdata_fast under Apache 2.0."
        )
        add(
            "input.file",
            "source",
            "Label chart",
            0,
            120,
            path="assets/labels.png",
            path_base="project",
            color_format="unchanged",
        )
        add("cv.text_extremal_regions", "detect", "Find label lines", 280, 120, minimum_count=3)
        add(
            "cv.tesseract_ocr",
            "ocr",
            "Read words and numbers",
            575,
            120,
            layout="line",
            tessdata_path="assets/tessdata",
            language_sha256={"eng": LANGUAGE_SHA256},
        )
        add("output.filmstrip_image", "filmstrip", "Review source-grid word boxes", 885, 120)
        graph.connect("source", "image", "detect", "image")
        graph.connect("source", "image", "ocr", "image")
        graph.connect("detect", "regions", "ocr", "regions")
        graph.connect("ocr", "image", "filmstrip", "image")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise RuntimeError(checked["issues"])
        graph.save(output / "tesseract_ocr.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
