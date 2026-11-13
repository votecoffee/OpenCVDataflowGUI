"""Generate an asymmetric oriented surface and a cluttered two-part scene."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from opencv_simple_3d_locate import save_recipe, train


def surface(seed=7, count=900):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 2 * np.pi, count)
    v = np.arccos(rng.uniform(-0.97, 0.97, count))

    def points(u, v):
        direction = np.column_stack((np.sin(v) * np.cos(u), np.sin(v) * np.sin(u), np.cos(v)))
        bump = 0.4 * np.exp(-np.sum((direction - [0.5, 0.7, 0.5]) ** 2, axis=1) / 0.2)
        radius = 1 + 0.2 * np.sin(3 * u) * np.sin(v) ** 2 + 0.13 * np.sin(2 * v) * np.cos(u) + bump
        return np.column_stack(
            (
                22 * radius * np.sin(v) * np.cos(u),
                15 * radius * np.sin(v) * np.sin(u),
                10 * radius * np.cos(v),
            )
        )

    xyz = points(u, v)
    du = points(u + 1e-4, v) - points(u - 1e-4, v)
    dv = points(u, v + 1e-4) - points(u, v - 1e-4)
    normals = np.cross(dv, du)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    return np.ascontiguousarray(np.c_[xyz, normals], np.float32)


def scene(seed=19):
    rng = np.random.default_rng(seed)
    parts, truth = [], []
    for index, (rotation_vector, translation) in enumerate(
        (([0.2, -0.4, 0.3], [-40, 5, 110]), ([0.7, 0.1, -0.2], [40, -10, 125]))
    ):
        data = surface(seed + index + 10)
        data = data[data[:, 0] < np.quantile(data[:, 0], 0.85)]
        rotation = cv2.Rodrigues(np.asarray(rotation_vector, float))[0]
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation, translation
        points = data[:, :3] @ rotation.T + translation
        points += rng.normal(0, 0.04, points.shape)
        parts.append(np.c_[points, data[:, 3:] @ rotation.T])
        truth.append(pose)
    clutter = rng.uniform([-70, -30, 85], [75, 30, 150], (120, 3))
    normals = rng.normal(size=clutter.shape)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    data = np.concatenate([*parts, np.c_[clutter, normals]])
    return np.ascontiguousarray(data[rng.permutation(len(data))], np.float32), truth


def generate(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    reference = surface()
    data, truth = scene()
    recipe = train(reference, units="mm", frame="part")
    save_recipe(folder / "part.s3l", recipe)
    np.savez_compressed(folder / "model.npz", points_normals=reference, units="mm", frame="part")
    np.savez_compressed(folder / "scene.npz", points_normals=data, units="mm", frame="camera")
    (folder / "truth.json").write_text(
        json.dumps([pose.tolist() for pose in truth], indent=2) + "\n", encoding="utf-8"
    )
    return recipe, data, truth


def host_project(folder):
    from opencv_simple_3d_locate import Locator
    from opencv_simple_3d_locate.graphics import overlay

    from opencv_dataflow.blocks import builtin_registry
    from opencv_dataflow.cloud_io import write_cloud
    from opencv_dataflow.core import Graph
    from opencv_dataflow.pointclouds import PointCloud

    folder = Path(folder)
    assets = folder / "assets"
    recipe, data, truth = generate(assets)
    write_cloud(
        assets / "model.ply",
        PointCloud(recipe.reference[:, :3], "mm", "part", normals=recipe.reference[:, 3:]),
        binary=True,
    )
    write_cloud(
        assets / "scene.ply",
        PointCloud(data[:, :3], "mm", "camera", normals=data[:, 3:]),
        binary=True,
    )
    import hashlib

    pin = hashlib.sha256((assets / "part.s3l").read_bytes()).hexdigest()
    graph = Graph()
    registry = builtin_registry()
    try:
        source = graph.add(
            registry.create(
                "input.cloud",
                block_id="scene",
                parameters={"path": "assets/scene.ply", "units": "mm", "frame": "camera"},
            )
        )
        locate = graph.add(
            registry.create(
                "3d.simple_locate",
                block_id="locate",
                parameters={"recipe_path": "assets/part.s3l", "sha256": pin},
            )
        )
        source.position, locate.position = (0.0, 0.0), (370.0, 0.0)
        locate.preview_output = "fit"
        locate.parameters["render_overlay"] = True  # Secondary documentation image.
        graph.connect(source.id, "cloud", locate.id, "scene")
        graph.save(folder / "project.json")
        graph.run()
        if locate.state != "pass" or locate.last_result.outputs["count"] != 2:
            raise RuntimeError(f"Example failed. {locate.error}")
        report = locate.last_result.outputs["report"]
        (folder / "expected.json").write_bytes(
            (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
        )
    finally:
        graph.close()
    result = Locator(recipe).locate(data, units="mm", frame="camera")
    if not cv2.imwrite(str(folder / "result.png"), overlay(recipe, result)):
        raise OSError("Could not write example render.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--host-project", action="store_true")
    args = parser.parse_args()
    (host_project if args.host_project else generate)(args.folder)
