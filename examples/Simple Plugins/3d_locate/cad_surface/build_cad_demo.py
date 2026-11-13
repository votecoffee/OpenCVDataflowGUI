"""Generate a stepped CAD part and independently sampled, occluded instances."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from opencv_simple_3d_locate import TriangleMesh, read_stl, save_recipe, train_cad
from opencv_simple_3d_locate.model import transformed


def stepped_mesh():
    occupied = {(x, y, 0) for x in range(6) for y in range(4)}
    occupied |= {(x, y, z) for x in range(2) for y in range(4) for z in (1, 2)}
    occupied |= {(x, y, 1) for x in (4, 5) for y in range(2)}
    occupied.add((3, 3, 1))
    triangles = []
    for cell in sorted(occupied):
        for axis in range(3):
            for sign in (-1, 1):
                neighbor = list(cell)
                neighbor[axis] += sign
                if tuple(neighbor) in occupied:
                    continue
                free = [index for index in range(3) if index != axis]
                quad = np.tile(np.asarray(cell, float), (4, 1))
                quad[:, axis] += sign == 1
                quad[:, free] += [[0, 0], [1, 0], [1, 1], [0, 1]]
                normal = np.cross(quad[1] - quad[0], quad[2] - quad[0])
                if normal[axis] * sign < 0:
                    quad = quad[[0, 3, 2, 1]]
                triangles.extend([quad[[0, 1, 2]], quad[[0, 2, 3]]])
    return TriangleMesh(np.asarray(triangles) * 10)


def write_stl(path, mesh):
    records = np.zeros(
        len(mesh.triangles),
        dtype=[("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")],
    )
    records["normal"], records["vertices"] = mesh.normals, mesh.triangles
    # A binary file can have a header that begins with solid.
    header = b"solid Simple3DLocate generated stepped part".ljust(80, b" ")
    Path(path).write_bytes(header + len(records).to_bytes(4, "little") + records.tobytes())


def cad_scene(mesh, seed=153):
    rng = np.random.default_rng(seed)
    parts, truth = [], []
    for index, (rotation, translation) in enumerate(
        (([2.3, -0.4, 0.3], [-70, -15, 160]), ([2.6, 0.3, -0.6], [50, 0, 150]))
    ):
        data = mesh.sample(8000, seed + index + 20)
        # Missing one spatial slice simulates partial occlusion.
        data = data[data[:, 0] <= 55]
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = cv2.Rodrigues(np.asarray(rotation))[0], translation
        data = transformed(data, pose)
        data[:, :3] += rng.normal(0, 0.04, data[:, :3].shape)
        parts.append(data)
        truth.append(pose)
    normals = rng.normal(size=(200, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    clutter = np.c_[rng.uniform([-100, -80, 80], [130, 50, 190], (200, 3)), normals]
    data = np.asarray(np.concatenate([*parts, clutter]), np.float32)
    return data[rng.permutation(len(data))], truth


def generate(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    mesh = stepped_mesh()
    write_stl(folder / "part.stl", mesh)
    # Exercise the file-import route used by a user teaching CAD.
    recipe = train_cad(read_stl(folder / "part.stl"), units="mm", frame="part")
    save_recipe(folder / "part.s3l", recipe)
    data, truth = cad_scene(mesh)
    np.savez_compressed(folder / "scene.npz", points_normals=data, units="mm", frame="camera")
    (folder / "truth.json").write_bytes(
        (json.dumps([pose.tolist() for pose in truth], indent=2) + "\n").encode("utf-8")
    )
    return recipe, data, truth


def host_project(folder, *, optimize=False):
    from opencv_simple_3d_locate import ValidationScene, compare_profiles, profile_recipe
    from opencv_simple_3d_locate.graphics import training_overlay

    from opencv_dataflow.blocks import builtin_registry
    from opencv_dataflow.cloud_io import write_cloud
    from opencv_dataflow.core import Graph
    from opencv_dataflow.pointclouds import PointCloud

    folder = Path(folder)
    assets = folder / "assets"
    recipe, data, truth = generate(assets)
    recipe_file = "part.s3l"
    if optimize:
        rng = np.random.default_rng(512)
        normals = rng.normal(size=(900, 3))
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        negative = np.c_[35 * normals, normals].astype(np.float32)
        np.savez_compressed(
            assets / "negative.npz", points_normals=negative, units="mm", frame="camera"
        )
        validation = {
            "version": 1,
            "scenes": [
                {
                    "name": "CAD instances",
                    "cloud": "assets/scene.npz",
                    "truth": [p.tolist() for p in truth],
                },
                {"name": "Sphere negative", "cloud": "assets/negative.npz", "truth": []},
            ],
        }
        (folder / "validation.json").write_bytes(
            (json.dumps(validation, indent=2) + "\n").encode("utf-8")
        )
        comparison = compare_profiles(
            recipe,
            [
                ValidationScene("CAD instances", data, "mm", "camera", tuple(truth)),
                ValidationScene("Sphere negative", negative, "mm", "camera", ()),
            ],
            progress=print,
        )
        (folder / "comparison.json").write_bytes(
            (json.dumps(comparison, indent=2, allow_nan=False) + "\n").encode("utf-8")
        )
        if comparison["recommended"] is None:
            raise RuntimeError("No execution plan passed CAD example validation.")
        recipe = profile_recipe(recipe, comparison["recommended"])
        recipe_file = "optimized.s3l"
        save_recipe(assets / recipe_file, recipe)
    with TemporaryDirectory(dir=assets, prefix=".cloud-") as scratch:
        temporary = Path(scratch) / "scene.ply"
        write_cloud(
            temporary, PointCloud(data[:, :3], "mm", "camera", normals=data[:, 3:]), binary=True
        )
        temporary.replace(assets / "scene.ply")
    digest = hashlib.sha256((assets / recipe_file).read_bytes()).hexdigest()
    graph, registry = Graph(), builtin_registry()
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
                parameters={"recipe_path": f"assets/{recipe_file}", "sha256": digest},
            )
        )
        source.position, locate.position = (0.0, 0.0), (370.0, 0.0)
        locate.preview_output = "overlay"
        graph.connect(source.id, "cloud", locate.id, "scene")
        graph.save(folder / "project.json")
        timings = {}
        for label in ("first_run", "cached_run"):
            start = time.perf_counter()
            graph.run()
            timings[label] = time.perf_counter() - start
            if locate.state != "pass" or locate.last_result.outputs["count"] != 2:
                raise RuntimeError(f"CAD example failed. {locate.error}")
        for name, report in (
            ("expected.json", locate.last_result.outputs["report"]),
            (
                "timing.json",
                {
                    "seconds": timings,
                    "scope": "Synthetic file-load plus locator runs. No cycle-time guarantee.",
                },
            ),
        ):
            (folder / name).write_bytes(
                (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
            )
        if not cv2.imwrite(str(folder / "result.png"), locate.last_result.outputs["overlay"]):
            raise OSError("Could not write CAD result render.")
        if optimize and not cv2.imwrite(
            str(folder / "model_analysis.png"),
            training_overlay(recipe, locate._locator, yaw=-25, pitch=25),
        ):
            raise OSError("Could not write training geometry render.")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--host-project", action="store_true")
    parser.add_argument("--optimize-profiles", action="store_true")
    args = parser.parse_args()
    if args.host_project:
        host_project(args.folder, optimize=args.optimize_profiles)
    else:
        generate(args.folder)
