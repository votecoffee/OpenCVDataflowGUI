"""Build independently cropped, noisy scans of a known asymmetric surface."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud

ROTATION_VECTOR = [0.19, -0.28, 0.31]
TRANSLATION = [0.125, -0.06, 0.09]


def cloud_arrays():
    rng = np.random.default_rng(351)
    x, y = rng.uniform([-0.33, -0.21], [0.33, 0.21], (3200, 2)).T
    z = (
        0.8
        + 0.08 * np.exp(-((x + 0.16) ** 2 * 35 + (y - 0.08) ** 2 * 150))
        + 0.045 * np.sin(13 * x + 9 * y) * np.cos(4 * y)
        + 0.11 * (x > 0.07)
    )
    points = np.column_stack((x, y, z))
    source = points[(x < 0.15) & (rng.random(len(x)) > 0.05)]
    target = points[(x > -0.13) & (rng.random(len(x)) > 0.055)]
    source += rng.normal(0, 0.00015, source.shape)
    rotation = Rotation.from_rotvec(ROTATION_VECTOR).as_matrix()
    target = target @ rotation.T + TRANSLATION + rng.normal(0, 0.00015, target.shape)
    source = np.vstack((source, rng.uniform([-0.32, -0.2, 1.05], [-0.2, 0.18, 1.09], (90, 3))))
    target = np.vstack((target, rng.uniform([0.22, -0.2, 0.65], [0.34, 0.15, 0.69], (110, 3))))
    return tuple(cloud[rng.permutation(len(cloud))] for cloud in (source, target))


def save_graph(path):
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, pos, **settings):
        node = graph.add(registry.create(kind, block_id=key, parameters=settings))
        node.position = pos
        return node

    try:
        notes = add("layout.comment", "notes", [0, -250])
        notes.name, notes.node_size = "Register two partial scans", "large"
        notes.description = (
            "Original synthetic surface scans by OpenCVDataflowGUI contributors (MIT). "
            "Each crop has independent noise and clutter. Local correspondence estimates "
            "one source-to-target pose and tests geometric overlap in both directions."
        )
        for i, (key, frame) in enumerate((("source", "source_scan"), ("target", "target_scan"))):
            block = add("input.cloud", key, [0, 240 * i], path=f"assets/{key}.ply", frame=frame)
            block.name = f"Load {key} scan"
            block.ticket_settings = {"mode": "fixed", "value": "partial-surface-registration"}
        aligned = add(
            "3d.register_cloud_features",
            "align",
            [415, 115],
            minimum_overlap_fraction=0.4,
            max_rms_m=0.0015,
        )
        aligned.preview_output = "Registered views"
        viewer = add("3d.cloud_render", "view", [830, 115])
        viewer.name = "Render aligned source"
        graph.connect("source", "cloud", aligned.id, "source")
        graph.connect("target", "cloud", aligned.id, "target")
        graph.connect(aligned.id, "cloud", viewer.id, "cloud")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise ValueError(checked["issues"])
        graph.save(path)
    finally:
        graph.close()


def build(output):
    assets = output / "assets"
    project = output / "cloud_feature_registration.json"
    if assets.exists() or project.exists():
        raise FileExistsError("Choose a new output directory to preserve the example")
    assets.mkdir(parents=True)
    for name, points in zip(("source", "target"), cloud_arrays(), strict=True):
        write_cloud(assets / f"{name}.ply", PointCloud(points, "m", f"{name}_scan"), binary=True)
    save_graph(project)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
