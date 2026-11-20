"""Build an original tilted-depth fixture with an unobserved interior strip."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

WIDTH, HEIGHT = 96, 72
FOCAL, CX, CY = 100.0, 47.5, 35.5
YAW = 0.08
TRANSLATION = (0.04, -0.01, 0.035)
WORLD_Z, SLOPE_X, SLOPE_Y = 1.02, 0.14, -0.09


def synthetic_depth():
    v, u = np.mgrid[:HEIGHT, :WIDTH]
    rays = np.stack(((u - CX) / FOCAL, (v - CY) / FOCAL, np.ones_like(u)), axis=-1)
    sine, cosine = np.sin(YAW), np.cos(YAW)
    rotation = np.array([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]])
    normal = np.array([-SLOPE_X, -SLOPE_Y, 1])
    depth = (WORLD_Z - normal @ TRANSLATION) / (rays @ rotation.T @ normal)
    depth[:, 44:52] = 0
    return np.rint(depth * 1000).astype(np.uint16)


def build(output):
    assets = output / "assets"
    if assets.exists() or (output / "tsdf_mesh.json").exists():
        raise FileExistsError("Choose an empty destination. Existing fixtures are preserved.")
    assets.mkdir(parents=True)
    ok, encoded = cv2.imencode(".png", synthetic_depth())
    if not ok:
        raise ValueError("Could not encode generated millimeter depth")
    (assets / "tilted_surface_mm.png").write_bytes(encoded.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = position
        return block

    try:
        add(
            "input.depth_file",
            "depth",
            [0, 0],
            path="assets/tilted_surface_mm.png",
            path_base="project",
            units="mm",
            units_per_sample=1.0,
            display_max=1500,
        )
        add(
            "3d.camera",
            "camera",
            [0, 270],
            fx=FOCAL,
            fy=FOCAL,
            cx=CX,
            cy=CY,
            width=WIDTH,
            height=HEIGHT,
        )
        add(
            "3d.pose",
            "pose",
            [0, 525],
            source_frame="camera",
            target_frame="world",
            rvec=[0.0, YAW, 0.0],
            translation=list(TRANSLATION),
        )
        add(
            "3d.tsdf_mesh",
            "surface",
            [400, 235],
            volume_origin_m=[-0.48, -0.38, 0.72],
            resolution=[60, 50, 40],
            voxel_size_m=0.02,
        )
        comment = add("layout.comment", "credit", [0, -270])
        comment.name, comment.node_size = "Synthetic posed depth and measured gap", "large"
        comment.description = (
            "Original depth by OpenCVDataflowGUI contributors (MIT). The depth camera "
            "views a known tilted plane in world coordinates. An eight-pixel strip "
            "has no observations. The extracted mesh preserves the gap."
        )
        graph.connect("depth", "depth", "surface", "depth")
        graph.connect("camera", "camera", "surface", "camera")
        graph.connect("pose", "transform", "surface", "poses")
        check = validate_graph(graph)
        if not check["valid"]:
            raise ValueError(check["issues"])
        graph.save(output / "tsdf_mesh.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
