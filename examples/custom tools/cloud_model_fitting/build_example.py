"""Build two interleaved scans with known metric geometry and labeled clutter."""

import argparse
from pathlib import Path

import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud


def colored_scan(support, clutter, frame, rng):
    xyz = np.vstack((support, clutter))
    colors = np.array([[50, 145, 235]] * len(support) + [[230, 160, 65]] * len(clutter))
    order = rng.permutation(len(xyz))
    return PointCloud(xyz[order], "mm", frame, colors=colors[order])


def plane_scan():
    rng = np.random.default_rng(41)
    x, y = np.meshgrid(np.arange(-40, 41, 10), np.arange(-30, 31, 10))
    z = 600 + 0.15 * x.ravel() - 0.20 * y.ravel()
    support = np.column_stack((x.ravel(), y.ravel(), z + rng.uniform(-0.1, 0.1, z.size)))
    xy = rng.uniform(-40, 40, (17, 2))
    offsets = rng.uniform(30, 50, len(xy)) * np.where(np.arange(len(xy)) % 2, -1, 1)
    clutter = np.column_stack((xy, 600 + 0.15 * xy[:, 0] - 0.20 * xy[:, 1] + offsets))
    return colored_scan(support, clutter, "tilted_fixture", rng)


def sphere_scan():
    rng = np.random.default_rng(42)
    i = np.arange(72)
    z = 1 - (2 * i + 1) / len(i)
    theta = np.pi * (3 - np.sqrt(5)) * i
    xy = np.sqrt(1 - z * z)
    directions = np.column_stack((xy * np.cos(theta), xy * np.sin(theta), z))
    center = np.array([-33, 55, 430])
    support = center + directions * (38 + rng.uniform(-0.1, 0.1, len(i)))[:, None]
    radii = np.where(np.arange(20) % 2, 18, 63)
    clutter = center + directions[::3][:20] * radii[:, None]
    return colored_scan(support, clutter, "ball_fixture", rng)


def build(output):
    project = output / "cloud_model_fitting.json"
    scans = {"plane": plane_scan(), "sphere": sphere_scan()}
    if project.exists() or any((output / f"assets/{kind}.ply").exists() for kind in scans):
        raise FileExistsError("Use an empty output directory to preserve the example")
    (output / "assets").mkdir(parents=True, exist_ok=True)
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = position
        return block

    try:
        note = add("layout.comment", "note", [0, -300])
        note.name, note.node_size = "Fit a metric plane and sphere", "large"
        note.description = (
            "Original synthetic clouds by OpenCVDataflowGUI contributors (MIT). "
            "Recover two known surfaces from interleaved scan rows with clutter. "
            "View the metric fit and compare its inlier and outlier indices."
        )
        for row, (kind, cloud) in enumerate(scans.items()):
            path = f"assets/{kind}.ply"
            write_cloud(output / path, cloud, binary=False)
            source = add(
                "input.cloud",
                f"{kind}_source",
                [0, 40 + row * 280],
                path=path,
                units=cloud.units,
                frame=cloud.frame,
            )
            source.name = f"Load {kind} scan"
            fit = add(
                "3d.cloud_model_fit",
                kind,
                [435, 40 + row * 280],
                model=kind,
                inlier_tolerance_m=0.0005,
                minimum_sphere_radius_m=0.03,
                maximum_sphere_radius_m=0.046,
            )
            fit.name = f"Fit {kind} through clutter"
            graph.connect(source.id, "cloud", fit.id, "cloud")
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
