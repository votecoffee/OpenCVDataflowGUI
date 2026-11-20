"""Build a depth-step graph for lifting known pixel positions into camera space."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

WIDTH, HEIGHT = 80, 56
PIXELS = [[25, 14], [55, 14], [24.6, 41.2], [57, 40], [0, 55], [80, 14], [25, 14]]


def build(output):
    asset = output / "assets/depth_step_mm.png"
    project = output / "sparse_depth_backprojection.json"
    if asset.exists() or project.exists():
        raise FileExistsError("Use a new output directory to preserve the existing fixture")
    asset.parent.mkdir(parents=True, exist_ok=True)
    depth = np.full((HEIGHT, WIDTH), 1000, np.uint16)
    depth[:, WIDTH // 2 :] = 1600
    depth[38:43, 54:60] = 0
    ok, png = cv2.imencode(".png", depth)
    if not ok:
        raise RuntimeError("Could not encode the depth-step fixture")
    asset.write_bytes(png.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = position
        return block

    try:
        note = add("layout.comment", "notes", [0, -245])
        note.name, note.node_size = "From image landmarks to metric points", "large"
        note.description = (
            "Original synthetic depth step by OpenCVDataflowGUI contributors (MIT). "
            "Recover camera XYZ from seven requests. The report preserves the duplicated "
            "point, the missing depth and the request outside the frame."
        )
        add(
            "input.depth_file",
            "depth",
            [0, 0],
            path="assets/depth_step_mm.png",
            path_base="project",
            display_max=2.0,
        )
        add(
            "3d.camera",
            "camera",
            [0, 235],
            fx=100.0,
            fy=80.0,
            cx=40.0,
            cy=28.0,
            width=WIDTH,
            height=HEIGHT,
            frame="step_camera",
        )
        add(
            "3d.backproject_depth_pixels",
            "sparse",
            [390, 80],
            pixel_points=PIXELS,
            minimum_valid_fraction=0.5,
        )
        graph.connect("depth", "depth", "sparse", "depth")
        graph.connect("camera", "camera", "sparse", "camera")
        check = validate_graph(graph)
        if not check["valid"]:
            raise ValueError(check["issues"])
        graph.save(project)
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
