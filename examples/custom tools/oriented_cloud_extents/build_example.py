"""Build a rotated calibration-box measurement example."""

import argparse
from itertools import product
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud

RVEC = [0.21, -0.39, 0.61]
CENTER = [128, -52, 480]
FRAME = "measurement_station"


def solid(dimensions):
    grid = np.asarray(list(product(np.linspace(-1, 1, 9), repeat=3)))
    shell = grid[np.any(np.abs(grid) == 1, axis=1)]
    local = shell * np.asarray(dimensions) / 2
    rotation = cv2.Rodrigues(np.array(RVEC))[0]
    xyz = local @ rotation.T + CENTER
    colors = np.where(shell[:, :1] > 0, [205, 162, 88], [78, 141, 222])
    return PointCloud(xyz, "mm", FRAME, colors=colors)


def build(output):
    target = output / "oriented_cloud_extents.json"
    assets = output / "assets"
    if target.exists() or assets.exists():
        raise FileExistsError("Choose an empty output directory")
    assets.mkdir(parents=True)
    write_cloud(assets / "rotated_cuboid.ply", solid([82, 34, 13]), binary=True)
    write_cloud(assets / "rotated_cube.ply", solid([36, 36, 36]), binary=True)
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **parameters):
        node = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        node.position = position
        return node

    try:
        note = add("layout.comment", "note", [0, -230])
        note.name, note.node_size = "Measure a rotated cuboid", "large"
        note.description = (
            "Original analytic cloud by OpenCVDataflowGUI contributors (MIT). "
            "Both branches measure the same 82 × 34 × 13 mm cuboid. "
            "Swap the source to the supplied cube to see PCA orientation ambiguity "
            "while the reference axes still give a valid measurement."
        )
        add("input.cloud", "source", [0, 0], path="assets/rotated_cuboid.ply")
        add(
            "3d.pose",
            "fixture_axes",
            [0, 245],
            rvec=RVEC,
            source_frame="fixture_axes",
            target_frame=FRAME,
            units="mm",
        )
        pca = add("3d.cloud_oriented_extents", "pca", [420, 0], box_frame="pca_box")
        pca.name = "Fit principal axes"
        pca.exposed_inputs = ["cloud"]
        reference = add(
            "3d.cloud_oriented_extents",
            "reference",
            [420, 245],
            orientation="reference",
            box_frame="fixture_box",
        )
        reference.name = "Measure on known fixture axes"
        for key in ("pca", "reference"):
            graph.connect("source", "cloud", key, "cloud")
        graph.connect("fixture_axes", "transform", "reference", "reference_pose")
        report = validate_graph(graph)
        if not report["valid"]:
            raise ValueError(report["issues"])
        graph.save(target)
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
