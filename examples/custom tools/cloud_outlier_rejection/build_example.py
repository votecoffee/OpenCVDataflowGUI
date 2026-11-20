"""Build an edged metric surface with six labeled noise points."""

import argparse
from pathlib import Path

import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud


def fixture():
    x, y = np.meshgrid(np.arange(-35, 36, 5), np.arange(-25, 26, 5))
    base = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 500)))
    y, z = np.meshgrid(np.arange(-25, 26, 5), [505, 510, 515])
    flange = np.column_stack((np.full(y.size, 35), y.ravel(), z.ravel()))
    noise = [
        [-90, -70, 558],
        [90, 65, 550],
        [90, -65, 460],
        [-90, 65, 450],
        [3, 4, 600],
        [8, 9, 430],
    ]
    xyz = np.vstack((base, flange, noise))
    colors = np.vstack(
        (
            np.tile([45, 145, 245], (len(base), 1)),
            np.tile([115, 230, 80], (len(flange), 1)),
            np.tile([240, 75, 55], (len(noise), 1)),
        )
    )
    normals = np.tile([0, 0, 1], (len(xyz), 1))
    normals[len(base) : len(base) + len(flange)] = [1, 0, 0]
    return PointCloud(xyz, "mm", "folded_panel", colors=colors, normals=normals)


def build(output):
    asset = output / "assets/noisy_folded_panel.ply"
    project = output / "cloud_outlier_rejection.json"
    if asset.exists() or project.exists():
        raise FileExistsError("Use an empty output directory to preserve the example")
    asset.parent.mkdir(parents=True, exist_ok=True)
    write_cloud(asset, fixture(), binary=False)

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = position
        return block

    try:
        note = add("layout.comment", "note", [0, -300])
        note.name, note.node_size = "Recover a folded surface", "large"
        note.description = (
            "Original synthetic cloud by OpenCVDataflowGUI contributors (MIT). "
            "A base and raised flange contain 198 measured points. Compare which rows "
            "the two density rules remove, including the panel corners and flange tip."
        )
        add(
            "input.cloud",
            "source",
            [0, 60],
            path="assets/noisy_folded_panel.ply",
            units="mm",
            frame="folded_panel",
        )
        add("3d.cloud_outlier", "radius", [425, 0], radius_m=0.008, minimum_neighbors=3)
        add(
            "3d.cloud_outlier",
            "statistical",
            [425, 300],
            method="statistical",
            statistical_neighbors=6,
            standard_deviations=1.5,
        )
        for key in ("radius", "statistical"):
            graph.connect("source", "cloud", key, "cloud")
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
