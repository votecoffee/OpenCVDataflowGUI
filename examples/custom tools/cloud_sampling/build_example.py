"""Build a density-biased colored cloud and two comparable sampling graphs."""

import argparse
from pathlib import Path

import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud


def make_cloud():
    x, y = np.meshgrid(np.arange(-32, 33, 4), np.arange(-32, 33, 4))
    dense = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 1050)))
    x, y = np.meshgrid(np.arange(-400, 401, 160), np.arange(-400, 401, 160))
    plate = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 1000)))
    points = np.vstack((dense, plate))
    colors = np.vstack(
        (np.tile([50, 200, 95], (len(dense), 1)), np.tile([230, 120, 35], (len(plate), 1)))
    )
    normals = np.tile([0, 0, 1], (len(points), 1))
    return PointCloud(points, "mm", "sampling_fixture", colors=colors, normals=normals)


def build(output):
    asset = output / "assets/dense_patch_on_plate.ply"
    project = output / "cloud_sampling.json"
    if asset.exists() or project.exists():
        raise FileExistsError("Choose an empty output directory to preserve existing files")
    asset.parent.mkdir(parents=True, exist_ok=True)
    write_cloud(asset, make_cloud(), binary=False)

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = position
        return block

    try:
        note = add("layout.comment", "note", [0, -300])
        note.name, note.node_size = "Compare equal sampling budgets", "large"
        note.description = (
            "Original synthetic colored cloud by OpenCVDataflowGUI contributors (MIT). "
            "A dense raised patch and a coarser surrounding plate share a frame. "
            "Compare the two 20-point outputs and their original row IDs."
        )
        add(
            "input.cloud",
            "source",
            [0, 60],
            path="assets/dense_patch_on_plate.ply",
            units="mm",
            frame="sampling_fixture",
        )
        for y, mode in enumerate(("random", "farthest")):
            add(
                "3d.cloud_sample",
                mode,
                [410, 300 * y],
                method=mode,
                max_points=20,
                seed=71,
            ).name = "Random sample" if mode == "random" else "Farthest-point sample"
            graph.connect("source", "cloud", mode, "cloud")
        result = validate_graph(graph)
        if not result["valid"]:
            raise ValueError(result["issues"])
        graph.save(project)
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
