"""Build an original pinhole scene with a tilted plane, clutter, and known labels."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

WIDTH, HEIGHT = 160, 112
INTRINSICS = {"fx": 149.0, "fy": 157.0, "cx": 78.4, "cy": 54.8}
NORMAL = np.array([0.22, -0.16, -1.0])
NORMAL /= np.linalg.norm(NORMAL)
OFFSET = 1.44


def fixture():
    labels = np.ones((HEIGHT, WIDTH), np.uint8)
    labels[19:65, 26:66] = 2
    labels[54:95, 108:140] = 2
    labels[11:36, 100:122] = 3
    labels[70:92, 55:80] = 4
    labels[35:42, 40:50] = 0
    labels[:2, :] = 0
    offsets = np.array([0, 0, 0.065, -0.07, 0.008])[labels]
    y, x = np.mgrid[:HEIGHT, :WIDTH]
    nx, ny, nz = NORMAL
    rays = (
        nx * (x - INTRINSICS["cx"]) / INTRINSICS["fx"]
        + ny * (y - INTRINSICS["cy"]) / INTRINSICS["fy"]
        + nz
    )
    rng = np.random.default_rng(1523)
    depth = (offsets - OFFSET) / rays + rng.uniform(-0.0013, 0.0013, labels.shape)
    depth[labels == 0] = 0
    return np.rint(1000 * depth).astype(np.uint16), labels


def build(output):
    assets = output / "assets"
    if assets.exists() or (output / "depth_plane.json").exists():
        raise FileExistsError("Choose an empty destination. Existing fixtures are preserved.")
    assets.mkdir(parents=True)
    for image, name in zip(fixture(), ("depth_mm.png", "truth_labels.png"), strict=True):
        ok, data = cv2.imencode(".png", image)
        if not ok:
            raise ValueError(f"Could not encode {name}")
        (assets / name).write_bytes(data.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, pos, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = pos
        return block

    try:
        add(
            "input.depth_file",
            "depth",
            [0, 0],
            path="assets/depth_mm.png",
            path_base="project",
            units="mm",
            units_per_sample=1.0,
            display_min=950,
            display_max=1750,
        )
        add(
            "3d.camera",
            "camera",
            [0, 285],
            **INTRINSICS,
            width=WIDTH,
            height=HEIGHT,
            frame="depth_camera",
        )
        separate = add("3d.separate_depth_plane", "separate", [350, 0])
        separate.preview_output = "foreground_mask"
        add("3d.depth_cloud", "objects", [720, 0], sample_step=1).name = "Foreground point cloud"
        comment = add("layout.comment", "credit", [0, -310])
        comment.name, comment.node_size = "Synthetic support plane and parts", "large"
        comment.description = (
            "Original depth and truth labels by OpenCVDataflowGUI contributors (MIT). "
            "A sloped support plane has two raised objects, a recess, a thin raised patch "
            "below the foreground cutoff, and invalid samples."
        )
        graph.connect("depth", "depth", "separate", "depth")
        graph.connect("camera", "camera", "separate", "camera")
        graph.connect("separate", "foreground_depth", "objects", "depth")
        graph.connect("camera", "camera", "objects", "camera")
        check = validate_graph(graph)
        if not check["valid"]:
            raise ValueError(check["issues"])
        graph.save(output / "depth_plane.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
