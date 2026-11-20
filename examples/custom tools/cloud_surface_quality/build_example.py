"""Build one scan with a plane, a sphere, a line and one isolated row."""

import argparse
from pathlib import Path

import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud


def fixture():
    x, y = np.meshgrid(np.arange(-20, 21, 5), np.arange(-20, 21, 5))
    plane = np.column_stack((x.ravel() - 70, y.ravel(), np.full(x.size, 450)))
    sample = np.arange(232)
    z = 1 - (2 * sample + 1) / len(sample)
    angle = np.pi * (3 - np.sqrt(5)) * sample
    xy = np.sqrt(1 - z**2)
    sphere = [60, -10, 430] + 50 * np.column_stack((xy * np.cos(angle), xy * np.sin(angle), z))
    line_x = np.arange(-15, 16, 5)
    line = np.column_stack((line_x, np.full(line_x.size, 85), np.full(line_x.size, 500)))
    isolated = [[-105, 80, 530]]
    groups = (plane, sphere, line, isolated)
    colors = np.repeat(
        [[50, 145, 245], [90, 215, 100], [230, 160, 65], [230, 55, 85]],
        [len(group) for group in groups],
        axis=0,
    )
    order = np.random.default_rng(84).permutation(len(colors))
    return PointCloud(np.vstack(groups)[order], "mm", "surface_fixture", colors=colors[order])


def build(output):
    project = output / "cloud_surface_quality.json"
    asset = output / "assets/surface_scan.ply"
    if project.exists() or asset.exists():
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
        note.name, note.node_size = "Review local surface fits", "large"
        note.description = (
            "Original synthetic cloud by OpenCVDataflowGUI contributors (MIT). "
            "Compare plane and sphere curvature, then pass the fitted normals and "
            "aligned curvature into region growing. A line and isolated point test rejection."
        )
        add(
            "input.cloud",
            "source",
            [0, 45],
            path="assets/surface_scan.ply",
            units="mm",
            frame="surface_fixture",
        )
        add(
            "3d.cloud_surface_quality",
            "quality",
            [410, 45],
            radius_m=0.022,
            minimum_neighbors=5,
            orientation="away_from_viewpoint",
            viewpoint_m=[0.060, -0.010, 0.430],
        )
        add(
            "3d.cloud_region_grow",
            "regions",
            [830, 45],
            radius_m=0.022,
            normal_angle_deg=35,
            maximum_curvature=0.05,
        )
        graph.connect("source", "cloud", "quality", "cloud")
        graph.connect("quality", "cloud", "regions", "cloud")
        graph.connect("quality", "curvature", "regions", "curvature")
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
