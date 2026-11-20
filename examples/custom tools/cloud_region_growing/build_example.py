"""Build two measured 25-row faces sharing an orthogonal seam."""

import argparse
from pathlib import Path

import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud


def fixture():
    offset, y = np.meshgrid(np.arange(0, 25, 5), np.arange(0, 25, 5))
    floor = np.column_stack((offset.ravel(), y.ravel(), np.full(y.size, 500)))
    wall = np.column_stack((np.full(y.size, 20), y.ravel(), 500 + offset.ravel()))
    points = np.stack((floor, wall), axis=1).reshape(-1, 3)
    normals = np.tile([[0, 0, 1], [1, 0, 0]], (len(floor), 1))
    colors = np.tile([[45, 145, 245], [115, 230, 80]], (len(floor), 1))
    return PointCloud(points, "mm", "fold_fixture", colors=colors, normals=normals)


def build(output):
    asset = output / "assets/touching_faces.ply"
    project = output / "cloud_region_growing.json"
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
        note.name, note.node_size = "Separate two touching surfaces", "large"
        note.description = (
            "Original synthetic cloud by OpenCVDataflowGUI contributors (MIT). "
            "Two faces share one seam. Compare joining both, splitting by normal direction, "
            "and rejecting seam measurements by curvature."
        )
        add(
            "input.cloud",
            "source",
            [0, 75],
            path="assets/touching_faces.ply",
            units="mm",
            frame="fold_fixture",
        )
        cases = (
            ("join", "Join both faces", 90, 1 / 3),
            ("split", "Separate faces", 20, 1 / 3),
            ("trim", "Leave the seam unassigned", 20, 0.05),
        )
        for row, (key, name, angle, curvature) in enumerate(cases):
            block = add(
                "3d.cloud_region_grow",
                key,
                [440, row * 290 - 160],
                radius_m=0.006,
                normal_angle_deg=angle,
                maximum_curvature=curvature,
            )
            block.name = name
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
