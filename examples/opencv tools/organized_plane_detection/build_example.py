"""Build a depth scene with a frontal plane, tilted plane and a curved occluder."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

WIDTH, HEIGHT, FOCAL = 168, 100, 145


def build(output):
    project = output / "organized_plane_detection.json"
    assets = output / "assets"
    if project.exists() or assets.exists():
        raise FileExistsError("Choose an empty output directory")
    assets.mkdir(parents=True)
    y, x = np.mgrid[:HEIGHT, :WIDTH]
    cx, cy = (WIDTH - 1) / 2, (HEIGHT - 1) / 2
    ray_x, ray_y = (x - cx) / FOCAL, (y - cy) / FOCAL
    depth = np.where(x < 84, 0.9, 1.12 / (1 - 0.29 * ray_x + 0.12 * ray_y))
    radius = ((x - 83) / 12) ** 2 + ((y - 42) / 9) ** 2
    depth[radius < 1] = 0.64 + 0.07 * radius[radius < 1]
    depth[38:47, 24:31] = 0
    depth[66:75, 138:145] = 0
    samples = np.rint(depth * 1000).astype(np.uint16)
    ok, png = cv2.imencode(".png", samples)
    if not ok:
        raise ValueError("Could not encode the analytic scene")
    (assets / "plane_scene.png").write_bytes(png.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = position
        return block

    try:
        note = add("layout.comment", "note", [0, -225])
        note.name, note.node_size = "Label planar surfaces and leave foreground unmatched", "large"
        note.description = (
            "Original analytic depth scene by OpenCVDataflowGUI contributors (MIT). "
            "OpenCV finds the frontal and tilted surfaces. The curved foreground "
            "remains unmatched, and the two missing-depth patches keep their own label."
        )
        add(
            "input.depth_file",
            "depth",
            [0, 0],
            path="assets/plane_scene.png",
            path_base="project",
            display_min=0.6,
            display_max=1.4,
        )
        add(
            "3d.camera",
            "camera",
            [0, 240],
            width=WIDTH,
            height=HEIGHT,
            fx=FOCAL,
            fy=FOCAL,
            cx=cx,
            cy=cy,
            frame="plane_camera",
        )
        planes = add(
            "3d.organized_planes",
            "planes",
            [380, 40],
            block_size=16,
            minimum_plane_pixels=900,
            threshold_m=0.005,
            minimum_planes=2,
        )
        planes.exposed_inputs = ["depth", "camera"]
        graph.connect("depth", "depth", "planes", "depth")
        graph.connect("camera", "camera", "planes", "camera")
        report = validate_graph(graph)
        if not report["valid"]:
            raise ValueError(report["issues"])
        graph.save(project)
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
