"""Build a contour ellipse inspection example with known geometry."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def image():
    target = np.zeros((360, 460), np.uint8)
    ellipses = (((90, 85), (53, 31), 27), ((248, 167), (66, 37), 117), ((345, 248), (49, 24), 153))
    for center, axes, angle in ellipses:
        cv2.ellipse(target, center, axes, angle, 0, 360, 220, -1)
    cv2.line(target, (8, 155), (79, 166), 170, 2)
    cv2.rectangle(target, (338, 48), (390, 88), 160, 2)
    cv2.line(target, (335, 165), (390, 191), 180, 2)
    cv2.ellipse(target, (175, 67), (21, 12), 24, 25, 185, 150, 2)
    return target


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    ok, png = cv2.imencode(".png", image(), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise OSError("Could not encode the ellipse target")
    (assets / "ellipses.png").write_bytes(png.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Detect and measure rotated ellipses", 0, -230)
        note.node_size = "large"
        note.description = (
            "Run Once. The target contains three ellipses with known size and angle, "
            "plus straight edges and a partial arc. Review the detections, keep elongated "
            "ellipses by aspect ratio, or use the highest score as a fitted region. "
            "Original MIT assets."
        )
        add(
            "input.file",
            "source",
            "Ellipse target",
            0,
            60,
            path="assets/ellipses.png",
            color_format="gray",
            path_base="project",
        )
        add(
            "inspect.find_ellipses",
            "detect",
            "Find three rotated ellipses",
            270,
            60,
            minimum_count=3,
        )
        add(
            "collection.filter",
            "filter",
            "Keep elongated ellipses",
            565,
            60,
            metric="rotated_aspect_ratio",
            minimum=1.85,
        )
        add(
            "collection.draw",
            "draw",
            "Draw elongated ellipses",
            865,
            60,
            draw_bounds=False,
            labels="none",
        )
        add("region.fitted_shape", "region", "Region from strongest fit", 565, 330, margin=2)
        graph.connect("source", "image", "detect", "image")
        graph.connect("detect", "collection", "filter", "collection")
        graph.connect("filter", "collection", "draw", "collection")
        graph.connect("detect", "image", "draw", "image")
        graph.connect("detect", "image", "region", "image")
        graph.connect("detect", "best_fit", "region", "fit")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise RuntimeError(checked["issues"])
        graph.save(output / "contour_ellipses.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
