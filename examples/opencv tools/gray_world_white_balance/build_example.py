"""Generate synthetic cast charts and save runnable gray-world projects."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

LEVELS = (32, 48, 64, 84)


def chart():
    image = np.zeros((120, 168, 3), np.uint8)
    for index, v in enumerate(LEVELS):
        y = 8 + index * 26
        image[y : y + 26, 8:64] = (v, 3 * v // 2, 2 * v)
    for y, color in (
        (8, (32, 80, 232)),
        (34, (18, 255, 184)),
        (60, (205, 64, 25)),
        (86, (42, 119, 231)),
    ):
        image[y : y + 26, 80:160] = color
    return image


def make_graph(output, depth):
    registry = builtin_registry()
    graph = Graph()

    def add(type_id, key, label, x, y, **parameters):
        block = graph.add(registry.create(type_id, block_id=key, parameters=parameters))
        block.name, block.position = label, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Gray-world white balance", 0, -220)
        note.node_size = "large"
        note.description = (
            "The left patches should be neutral. The right patches are intentionally saturated. "
            "Compare gains estimated from all pixels, a saturation limit "
            "and an explicit rectangle. "
            "This original synthetic chart is covered by the repository's MIT license."
        )
        add(
            "input.file",
            "input",
            f"Color cast chart ({depth})",
            0,
            125,
            path=f"assets/chart_{depth}.png",
            path_base="project",
            pixel_depth="unchanged",
            color_format="unchanged",
        )
        shape = add(
            "region.shape", "roi", "First neutral patch", 245, 340, rectangle=[8, 8, 56, 26]
        )
        shape.exposed_outputs = ("region",)
        graph.connect("input", "image", "roi", "image")

        for key, title, y, threshold in (
            ("all_pixels", "All pixels", -20, 1.0),
            ("saturation_filtered", "Saturation limit 0.60", 160, 0.60),
            ("region_filtered", "Neutral region only", 340, 1.0),
        ):
            add("cv.gray_world_white_balance", key, title, 525, y, saturation_threshold=threshold)
            graph.connect("input", "image", key, "image")
        graph.connect("roi", "region", "region_filtered", "region")
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise RuntimeError(validation["issues"])
        graph.save(output / f"gray_world_{depth}.json")
    finally:
        graph.close()


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    image = chart()
    for depth, chart_image in (("uint8", image), ("uint16", image.astype(np.uint16) * 257)):
        ok, png = cv2.imencode(".png", chart_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode {depth} chart")
        (assets / f"chart_{depth}.png").write_bytes(png.tobytes())
        make_graph(output, depth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
