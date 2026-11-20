"""Build a colored metric cloud and a saved octree query graph."""

import argparse
from pathlib import Path

import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud


def build(output):
    asset = output / "assets/bounded_cloud.ply"
    project = output / "cloud_octree_query.json"
    if asset.exists() or project.exists():
        raise FileExistsError("Use a new output directory to preserve the example")
    asset.parent.mkdir(parents=True, exist_ok=True)
    x, y = np.meshgrid(np.arange(-200, 201, 50), np.arange(-200, 201, 50))
    xy = np.column_stack((x.ravel(), y.ravel()))
    points = np.vstack(
        (
            np.column_stack((xy, np.full(len(xy), 1000))),
            np.column_stack((xy, np.full(len(xy), 1125))),
            [[0, 0, 1000], [250, 250, 1500], [300, 0, 1000]],
        )
    )
    colors = np.vstack(
        (
            np.tile([30, 180, 235], (len(xy), 1)),
            np.tile([65, 210, 90], (len(xy), 1)),
            [[230, 70, 180], [250, 200, 55], [220, 70, 60]],
        )
    )
    cloud = PointCloud(points, "mm", "scan", colors=colors)
    write_cloud(asset, cloud, binary=False)

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = position
        return block

    try:
        note = add("layout.comment", "note", [0, -290])
        note.name, note.node_size = "Query a bounded point cloud", "large"
        note.description = (
            "Original synthetic cloud by the OpenCVDataflowGUI contributors (MIT). "
            "Inspect both reports for original PLY row IDs and metric distances. "
            "The two coincident points have different colors and distinct row IDs."
        )
        add(
            "input.cloud",
            "source",
            [0, 60],
            path="assets/bounded_cloud.ply",
            units="mm",
            frame="scan",
        )
        bound = {
            "bounds": "explicit",
            "minimum_corner_m": [-0.25, -0.25, 1.0],
            "cube_side_m": 0.5,
            "query_units": "m",
        }
        a = add(
            "3d.cloud_octree_query",
            "knn",
            [400, 0],
            **bound,
            search="knn",
            max_neighbors=5,
            query_points=[[0, 0, 1], [0.24, 0.24, 1.45]],
        )
        a.name = "Five nearest neighbors"
        b = add(
            "3d.cloud_octree_query",
            "radius",
            [400, 270],
            **bound,
            search="radius",
            max_neighbors=12,
            radius_m=0.125,
            subdivision="resolution",
            resolution_m=0.01,
            query_points=[[0, 0, 1], [0.25, 0.25, 1.375]],
        )
        b.name = "Neighbors within 125 mm"
        for key in ("knn", "radius"):
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
