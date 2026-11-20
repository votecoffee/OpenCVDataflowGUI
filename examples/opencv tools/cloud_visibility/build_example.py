"""Create a synthetic cloud, target depth, and independently labeled visibility cells."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud

WIDTH, HEIGHT = 80, 64
INTRINSICS = {"fx": 64.0, "fy": 64.0, "cx": 39.5, "cy": 31.5}
FRAME = "inspection_camera"
FAR_COLOR, NEAR_COLOR = (30, 210, 140), (245, 120, 35)


def layered_scene():
    near = np.zeros((HEIGHT, WIDTH), bool)
    far = np.zeros_like(near)
    far[12:56, 10:67] = True
    near[22:42, 31:59] = True
    covered = near | far
    depth = np.where(near, 900, np.where(far, 1600, 0)).astype(np.uint16)
    observed = depth.copy()
    # A closer actual surface, an invalid sensor patch, and a farther measured surface.
    observed[25:33, 43:49] = 650
    observed[45:50, 17:26] = 0
    observed[34:39, 50:59] = 1100
    labels = np.zeros_like(observed, dtype=np.uint8)
    labels[covered] = 1
    labels[25:33, 43:49] = 2
    labels[45:50, 17:26] = 3
    labels[34:39, 50:59] = 4
    labels[near & far] |= 16

    xyz, rgb = [], []
    for layer, z, color in [(far, 1.6, FAR_COLOR), (near, 0.9, NEAR_COLOR)]:
        y, x = np.nonzero(layer)
        xyz.append(np.column_stack(((x - 39.5) * z / 64.0, (y - 31.5) * z / 64.0, y * 0 + z)))
        rgb.append(np.tile(color, (len(x), 1)))
    cloud = PointCloud(np.vstack(xyz), "m", FRAME, colors=np.vstack(rgb))
    return cloud, depth, observed, labels


def build(output):
    assets = output / "assets"
    if assets.exists() or (output / "cloud_visibility.json").exists():
        raise FileExistsError("Choose an empty destination. Existing fixtures are preserved.")
    assets.mkdir(parents=True)
    cloud, expected, observed, labels = layered_scene()
    write_cloud(assets / "overlap.ply", cloud, binary=True)
    for values, name in (
        (expected, "expected_nearest_mm.png"),
        (observed, "observed_mm.png"),
        (labels, "expected_visibility.png"),
    ):
        ok, encoded = cv2.imencode(".png", values)
        if not ok:
            raise ValueError(f"Could not encode {name}")
        (assets / name).write_bytes(encoded.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, pos, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = pos
        return block

    try:
        source = add("input.cloud", "cloud", [0, 0], path="assets/overlap.ply", frame=FRAME)
        reference = add(
            "input.depth_file",
            "reference",
            [0, 275],
            path="assets/observed_mm.png",
            path_base="project",
            units="mm",
            units_per_sample=1.0,
            display_min=0,
            display_max=2000,
        )
        for capture in (source, reference):
            capture.ticket_settings = {"mode": "fixed", "value": "synthetic-overlap"}
        add("3d.camera", "camera", [0, 550], **INTRINSICS, width=WIDTH, height=HEIGHT, frame=FRAME)
        render = add("3d.cloud_project", "project", [385, 220], maximum_z_m=2.0)
        render.preview_output = "color"
        comment = add("layout.comment", "credit", [0, -280])
        comment.name, comment.node_size = "Synthetic camera visibility control", "large"
        comment.description = (
            "Original fixtures by OpenCVDataflowGUI contributors (MIT). Two modeled surfaces "
            "overlap. Observed depth adds one closer occluder, one hole, and one farther return."
        )
        graph.connect("cloud", "cloud", "project", "cloud")
        graph.connect("camera", "camera", "project", "camera")
        graph.connect("reference", "depth", "project", "reference_depth")
        check = validate_graph(graph)
        if not check["valid"]:
            raise ValueError(check["issues"])
        graph.save(output / "cloud_visibility.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
