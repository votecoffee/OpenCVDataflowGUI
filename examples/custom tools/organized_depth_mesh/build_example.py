"""Build an organized depth mesh example with a step and an occlusion."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

SIZE = (42, 28)


def build(output):
    path = output / "organized_depth_mesh.json"
    assets = output / "assets"
    if path.exists() or assets.exists():
        raise FileExistsError("Choose an empty output directory")
    assets.mkdir(parents=True)
    width, height = SIZE
    y, x = np.mgrid[:height, :width]
    depth = np.where(x < width // 2, 800, 1040).astype(np.uint16)
    depth[((x - 10) / 4) ** 2 + ((y - 14) / 6) ** 2 <= 1] = 0
    ok, data = cv2.imencode(".png", depth)
    if not ok:
        raise ValueError("Could not encode the depth fixture")
    (assets / "step_depth.png").write_bytes(data.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.position = position
        return block

    try:
        note = add("layout.comment", "note", [0, -240])
        note.name, note.node_size = "Keep gaps at depth discontinuities", "large"
        note.description = (
            "Original analytic depth scan by OpenCVDataflowGUI contributors (MIT). "
            "Both branches reconstruct the same 240 mm step and missing oval. "
            "A tight depth-jump limit keeps the two surfaces apart."
        )
        add(
            "input.depth_file",
            "depth",
            [0, 0],
            path="assets/step_depth.png",
            path_base="project",
            display_min=0.75,
            display_max=1.1,
        )
        add(
            "3d.camera",
            "camera",
            [0, 240],
            width=width,
            height=height,
            fx=62,
            fy=64,
            cx=20.5,
            cy=13.5,
            frame="step_camera",
        )
        for key, title, y, limit in (
            ("strict", "Split at the step", 0, 0.02),
            ("loose", "Allow the artificial bridge", 240, 0.3),
        ):
            block = add("3d.organized_mesh", key, [380, y], max_depth_jump_m=limit)
            block.name = title
            block.exposed_inputs = ["depth", "camera"]
            graph.connect("depth", "depth", key, "depth")
            graph.connect("camera", "camera", key, "camera")
        report = validate_graph(graph)
        if not report["valid"]:
            raise ValueError(report["issues"])
        graph.save(path)
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
