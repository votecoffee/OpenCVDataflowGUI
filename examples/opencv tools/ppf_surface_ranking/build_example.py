"""Build a native PPF project with two occluded copies and oriented clutter."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud

POSES = (
    ([0.08, -0.17, 0.15], [-0.13, 0, 0.70]),
    ([-0.1, 0.1, 0.3], [0.15, 0.035, 0.76]),
)


def fixtures():
    grid = np.meshgrid(np.linspace(-0.09, 0.09, 27), np.linspace(-0.07, 0.07, 23))
    x, y = (field.ravel() for field in grid)
    u, v = x + 0.03, y - 0.015
    bump = 0.04 * np.exp(-400 * u**2 - 800 * v**2)
    z = 2 * x**2 + 4 * y**2 + 10 * x**2 * y + bump
    xyz = np.column_stack((x, y, z))
    normals = np.column_stack(
        (
            -4 * x - 20 * x * y + 800 * u * bump,
            -8 * y - 10 * x**2 + 1600 * v * bump,
            np.ones(len(x)),
        )
    )
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    rng = np.random.default_rng(126)
    scene, directions = [], []
    visible = (x < 0.055, (x > -0.055) & (y < 0.055))
    for keep, (rvec, translation) in zip(visible, POSES, strict=True):
        rotation, _ = cv2.Rodrigues(np.array(rvec))
        samples = xyz[keep] @ rotation.T + translation
        scene.append(samples + rng.normal(0, 0.00008, samples.shape))
        directions.append(normals[keep] @ rotation.T)
    clutter = rng.uniform([-0.28, -0.12, 0.65], [0.28, 0.13, 0.85], (200, 3))
    clutter_normals = rng.normal(size=clutter.shape)
    clutter_normals /= np.linalg.norm(clutter_normals, axis=1, keepdims=True)
    return (
        PointCloud(xyz, frame="panel", normals=normals),
        PointCloud(
            np.vstack([*scene, clutter]),
            frame="station",
            normals=np.vstack([*directions, clutter_normals]),
        ),
    )


def build(output):
    project = output / "ppf_surface_ranking.json"
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
        note.name, note.node_size = "Rank two partly occluded surface matches", "large"
        note.description = (
            "Original synthetic fixture by OpenCVDataflowGUI contributors (MIT). "
            "Two differently cropped copies of a curved panel appear among 200 clutter points."
        )
        for i, (key, cloud) in enumerate(zip(("model", "scene"), fixtures(), strict=True)):
            write_cloud(assets / f"{key}.ply", cloud, binary=True)
            node = add("input.cloud", key, [0, i * 285], path=f"assets/{key}.ply")
            node.name = f"Load oriented {key}"
            node.ticket_settings = {"mode": "fixed", "value": "ppf-ranking-fixture"}
        add(
            "3d.ppf_train",
            "train",
            [365, 0],
            relative_sampling_step=0.08,
            relative_distance_step=0.04,
        )
        graph.connect("model", "cloud", "train", "cloud")
        for i, (key, name, ranking) in enumerate(
            (
                ("votes", "Review native vote order", "votes"),
                ("verify", "Refine and rank on scene support", "refined_geometry"),
            )
        ):
            node = add(
                "3d.ppf_match",
                key,
                [715, i * 285],
                ranking=ranking,
                scene_sample_step=0.25,
                scene_distance=0.05,
                max_matches=5,
                correspondence_radius=0.004,
                refinement_distance=0.012,
                maximum_inlier_rmse=0.002,
                minimum_model_overlap=0.4,
            )
            node.name = name
            graph.connect("train", "model", key, "model")
            graph.connect("scene", "cloud", key, "scene")
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
