"""Build an anisotropic voxel fixture with retained and synthetic samples."""

import argparse
from pathlib import Path

import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud

POINTS = [
    [-9, -18, 2],
    [-1, -2, 4],
    [1, 1, 1],
    [4, 9, 6],
    [9, 19, 26],
    [2, 20, 5],
    [10, 18, 19],
    [13, 16, 18],
    [19, 19, 28],
    [2, 6, 30],
    [-1, 1, 5],
]


def build(output):
    asset = output / "assets/voxel_groups.ply"
    project = output / "voxel_representatives.json"
    if asset.exists() or project.exists():
        raise FileExistsError("Use a new output directory to preserve the fixture")
    asset.parent.mkdir(parents=True, exist_ok=True)
    rows = np.arange(len(POINTS))
    colors = np.column_stack((70 + rows * 12, 210 - rows * 15, 20 + rows * 8))
    normals = np.eye(3)[rows % 3]
    cloud = PointCloud(POINTS, "mm", "voxel_fixture", colors=colors, normals=normals)
    write_cloud(asset, cloud, binary=False)

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.position = position
        return block

    try:
        note = add("layout.comment", "note", [0, -280])
        note.name, note.node_size = "Retain measured voxel representatives", "large"
        note.description = (
            "Original synthetic colored scan by the OpenCVDataflowGUI contributors (MIT). "
            "Use 10 × 20 × 30 mm voxels and compare the measured representative with the "
            "geometric centroid for the same groups."
        )
        add(
            "input.cloud",
            "source",
            [0, 70],
            path="assets/voxel_groups.ply",
            units="mm",
            frame="voxel_fixture",
        )
        for row, mode in enumerate(("representative", "centroid")):
            node = add(
                "3d.cloud_voxel",
                mode,
                [405, row * 300],
                mode=mode,
                spacing="per_axis",
                voxel_size_x=10,
                voxel_size_y=20,
                voxel_size_z=30,
            )
            node.name = "Nearest measured row" if mode == "representative" else "Voxel centroids"
            graph.connect("source", "cloud", mode, "cloud")
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
