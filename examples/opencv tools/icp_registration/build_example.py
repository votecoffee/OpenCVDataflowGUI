"""Build an ICP graph with curved partial scans and an ambiguous planar control."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud

SOURCE, TARGET = "source_scan", "target_scan"
RVEC = [0.004, -0.003, 0.006]
SHIFT = [0.001, -0.002, 0.004]


def fixtures():
    grid = np.linspace(-0.1, 0.1, 24)
    x, y = (field.ravel() for field in np.meshgrid(grid, grid))
    z = 0.8 + 2 * x**2 + 3 * y**2 + 4 * x**2 * y
    points = np.column_stack((x, y, z))
    normals = np.column_stack((-4 * x - 8 * x * y, -6 * y - 4 * x**2, np.ones(len(x))))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    rotation = cv2.Rodrigues(np.array(RVEC))[0]
    for name, xyz, unit_normals in (
        ("curved", points, normals),
        ("plane", np.column_stack((x, y, np.full(len(x), 0.8))), np.tile([0, 0, 1], (len(x), 1))),
    ):
        first = x < 0.04 if name == "curved" else slice(None)
        second = x > -0.065 if name == "curved" else slice(None)
        yield f"{name}_source", PointCloud(xyz[first], frame=SOURCE, normals=unit_normals[first])
        yield (
            f"{name}_target",
            PointCloud(
                xyz[second] @ rotation.T + SHIFT,
                frame=TARGET,
                normals=unit_normals[second] @ rotation.T,
            ),
        )


def build(output):
    project = output / "icp_registration.json"
    assets = output / "assets"
    if assets.exists() or project.exists():
        raise FileExistsError("Choose an empty output directory to preserve the example")
    assets.mkdir(parents=True)
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = position
        return block

    try:
        note = add("layout.comment", "notes", [0, -285])
        note.name, note.node_size = "Check the support for an ICP pose", "large"
        note.description = (
            "Original synthetic fixture by OpenCVDataflowGUI contributors (MIT). "
            "Two curved crops share 288 points. The plane has full overlap but "
            "cannot constrain all six pose axes."
        )
        for i, (key, cloud) in enumerate(fixtures()):
            write_cloud(assets / f"{key}.ply", cloud, binary=True)
            node = add("input.cloud", key, [0, i * 220], path=f"assets/{key}.ply")
            node.name = f"Load {key.replace('_', ' ')}"
            node.ticket_settings = {"mode": "fixed", "value": "icp-overlap-fixture"}
        seed = add(
            "3d.pose",
            "seed",
            [0, 880],
            source_frame=SOURCE,
            target_frame=TARGET,
            rvec=[0.0043, -0.003, 0.0058],
            translation=[0.0017, -0.0023, 0.0033],
        )
        seed.name = "Define approximate scan pose"
        for i, (key, name, surface, overlap) in enumerate(
            (
                ("align", "Accept partial curved overlap", "curved", 0.6),
                ("strict", "Require 90% overlap", "curved", 0.9),
                ("plane", "Reject unconstrained plane fit", "plane", 0.9),
            )
        ):
            node = add(
                "3d.icp",
                key,
                [550, 110 + i * 270],
                quality_checks=True,
                pyramid_levels=1,
                correspondence_radius=0.003,
                minimum_source_overlap=overlap,
                minimum_target_overlap=overlap,
                maximum_inlier_rmse=0.001,
            )
            node.name = name
            graph.connect(f"{surface}_source", "cloud", key, "source")
            graph.connect(f"{surface}_target", "cloud", key, "target")
            graph.connect(seed.id, "transform", key, "initial_pose")
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise ValueError(validation["issues"])
        graph.save(project)
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
