"""Build a metric scan with three parts, two stray points and an overlong edge."""

import argparse
from pathlib import Path

import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud


def fixture():
    x, y = np.meshgrid([0, 5, 10], [0, 5, 10])
    pad = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    edge_x = np.arange(0, 40, 5)
    groups = (
        pad + [0, 0, 500],
        pad + [35, 0, 500],
        pad + [0, 35, 505],
        np.array([[35, 35, 500], [40, 35, 500]]),
        np.column_stack((edge_x, np.full(edge_x.size, 65), np.full(edge_x.size, 500))),
    )
    palette = ([45, 145, 245], [115, 230, 80], [205, 90, 210], [240, 75, 55], [250, 220, 70])
    # Interleave the sensor rows to make source identity independently testable.
    scan = [
        (group[row], color)
        for row in range(9)
        for group, color in zip(groups, palette, strict=True)
        if row < len(group)
    ]
    points, colors = zip(*scan, strict=True)
    normals = np.tile([0, 0, 1], (len(points), 1))
    return PointCloud(points, "mm", "parts_tray", colors=colors, normals=normals)


def build(output):
    asset = output / "assets/tray_parts.ply"
    project = output / "cloud_clustering.json"
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
        note = add("layout.comment", "note", [0, -270])
        note.name, note.node_size = "Separate measured parts", "large"
        note.description = (
            "Original synthetic cloud by OpenCVDataflowGUI contributors (MIT). Three "
            "pads on a tray are separated by known clearances. A two-point stray group and a "
            "long edge fail the count and size limits."
        )
        add(
            "input.cloud",
            "source",
            [0, 50],
            path="assets/tray_parts.ply",
            units="mm",
            frame="parts_tray",
        )
        add(
            "3d.cloud_cluster",
            "cluster",
            [410, 50],
            radius_m=0.006,
            minimum_points=4,
            maximum_extent_m=0.03,
        )
        graph.connect("source", "cloud", "cluster", "cloud")
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
