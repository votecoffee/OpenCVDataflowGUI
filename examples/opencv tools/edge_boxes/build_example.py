"""Build a bounded EdgeBoxes example with Canny and Sobel inputs."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def image():
    rgb = np.full((240, 340, 3), [35, 50, 69], np.uint8)
    cv2.rectangle(rgb, (43, 44), (135, 112), (226, 187, 108), -1)
    cv2.rectangle(rgb, (215, 112), (289, 188), (92, 205, 220), -1)
    cv2.line(rgb, (32, 182), (103, 201), (90, 115, 130), 1)
    cv2.circle(rgb, (170, 120), 8, (86, 102, 115), 1)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    valid, png = cv2.imencode(".png", image(), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not valid:
        raise OSError("Could not encode the proposal target")
    (assets / "parts.png").write_bytes(png.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Rank object proposals", 0, -240)
        note.node_size = "large"
        note.description = (
            "Run Once. Canny supplies edge strength, and the two Sobel gradients supply angles "
            "through Cartesian to polar fields. EdgeBoxes ranks candidate rectangles. "
            "Filter on Strength before drawing the final boxes. Original MIT assets."
        )
        add(
            "input.file",
            "source",
            "Two parts",
            0,
            60,
            path="assets/parts.png",
            color_format="bgr",
            path_base="project",
        )
        add("cv.convert", "gray", "Grayscale", 230, 60, color_format="gray")
        add("cv.gaussian_blur", "blur", "Smooth source", 445, 60, kernel=5, sigma=1.0)
        add("cv.canny", "edges", "Canny strengths", 680, -155, low=45, high=100)
        add("cv.gradient", "dx", "Horizontal derivative", 680, 60, dx=1, dy=0)
        add("cv.gradient", "dy", "Vertical derivative", 680, 275, dx=0, dy=1)
        add("cv.cart_to_polar", "polar", "Gradient angle", 920, 175, angle_units="degrees")
        add(
            "inspect.edge_boxes",
            "propose",
            "EdgeBoxes proposals",
            1180,
            35,
            beta=0.65,
            max_boxes=12,
            min_box_area=1100,
            preview_count=2,
        )
        add(
            "collection.filter",
            "filter",
            "Keep high scores",
            1420,
            35,
            metric="strength",
            minimum=0.35,
        )
        add(
            "collection.draw",
            "draw",
            "Review strong proposals",
            1650,
            35,
            draw_bounds=False,
            draw_centers=False,
        )
        graph.connect("source", "image", "gray", "image")
        graph.connect("gray", "image", "blur", "image")
        for block in ("edges", "dx", "dy"):
            graph.connect("blur", "image", block, "image")
        graph.connect("dx", "gradient", "polar", "x")
        graph.connect("dy", "gradient", "polar", "y")
        graph.connect("edges", "image", "propose", "edge_map")
        graph.connect("polar", "angle", "propose", "orientation")
        graph.connect("source", "image", "propose", "image")
        graph.connect("propose", "collection", "filter", "collection")
        graph.connect("filter", "collection", "draw", "collection")
        graph.connect("propose", "image", "draw", "image")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise RuntimeError(checked["issues"])
        graph.save(output / "edge_boxes.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
