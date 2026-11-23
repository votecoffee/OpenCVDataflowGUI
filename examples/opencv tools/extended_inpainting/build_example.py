"""Generate a scratched periodic surface and its extended inpainting project."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

DEFECT = (30, 22, 8, 7)


def images():
    y, x = np.indices((64, 72))
    signal = np.rint(110 + 27 * np.sin(2 * np.pi * x / 12) + 20 * np.cos(2 * np.pi * y / 12))
    clean = np.stack((signal - 23, signal + 20, signal + 56), axis=2).astype(np.uint8)
    damaged = clean.copy()
    left, top, width, height = DEFECT
    damaged[top : top + height, left : left + width] = (246, 23, 210)
    return {"clean": clean, "damaged": damaged}


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
        note = add("layout.comment", "note", "Repair a known surface defect", 0, -220)
        note.node_size = "large"
        note.description = (
            "One rectangle marks the damaged samples. Compare two frequency profiles with "
            "patch matching. The Repair Mask output identifies the estimated pixels. "
            "The synthetic source and clean reference are original MIT-licensed assets."
        )
        add(
            "input.file",
            "damaged",
            "Damaged surface",
            0,
            125,
            path="assets/damaged.png",
            path_base="project",
            pixel_depth="unchanged",
            color_format="unchanged",
        )
        area = add(
            "region.shape", "area", "Known damage rectangle", 245, 305, rectangle=list(DEFECT)
        )
        area.exposed_outputs = ("mask", "region")
        graph.connect("damaged", "image", "area", "image")
        for method, name, connector, y in (
            ("fsr_fast", "Frequency reconstruction, Fast", "mask", -35),
            ("fsr_best", "Frequency reconstruction, Best", "region", 150),
            ("shift_map", "Reconstruct from repeating patches", "mask", 335),
        ):
            add("cv.extended_inpaint", method, name, 555, y, method=method)
            graph.connect("damaged", "image", method, "image")
            graph.connect("area", connector, method, connector)
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise RuntimeError(checked["issues"])
        graph.save(output / "extended_inpainting.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
