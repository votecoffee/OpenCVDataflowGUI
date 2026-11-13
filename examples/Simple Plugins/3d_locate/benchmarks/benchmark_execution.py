"""Reproducible comparisons on curved and CAD geometry, with independent truth."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from build_cad_demo import cad_scene, stepped_mesh
from build_demo import scene, surface
from opencv_simple_3d_locate import (
    ValidationScene,
    compare_profiles,
    save_recipe,
    train,
    train_cad,
)
from opencv_simple_3d_locate.model import transformed


def benchmark(folder, repeats=2):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name in ("curved", "cad"):
        output = folder / name
        output.mkdir(exist_ok=True)
        mesh = stepped_mesh() if name == "cad" else None
        recipe = (
            train(surface(), units="mm", frame="part")
            if mesh is None
            else train_cad(mesh, units="mm", frame="part")
        )
        save_recipe(output / "part.s3l", recipe)
        manifest, cases = {"version": 1, "scenes": []}, []
        for index, seed in enumerate((19, 27, 71)):
            data, truth = scene(seed) if mesh is None else cad_scene(mesh, seed)
            # Change the entire scene frame to test larger 3D rotations and offsets.
            coordinate = np.eye(4)
            coordinate[:3, :3] = cv2.Rodrigues(
                np.array([0.17 * index, -0.3 * index, 0.21 * index])
            )[0]
            coordinate[:3, 3] = [13 * index, -11 * index, 7 * index]
            data, truth = transformed(data, coordinate), tuple(coordinate @ pose for pose in truth)
            filename = f"positive_{index}.npz"
            np.savez_compressed(
                output / filename, points_normals=data, units="mm", frame="test_camera"
            )
            label = f"{name}_independent_{index}"
            cases.append(ValidationScene(label, data, "mm", "test_camera", truth))
            manifest["scenes"].append(
                {"name": label, "cloud": filename, "truth": [p.tolist() for p in truth]}
            )
        rng = np.random.default_rng(512)
        normals = rng.normal(size=(900, 3))
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        negative = np.c_[35 * normals, normals].astype(np.float32)
        np.savez_compressed(
            output / "negative.npz", points_normals=negative, units="mm", frame="test_camera"
        )
        cases.append(ValidationScene(f"{name}_sphere_negative", negative, "mm", "test_camera", ()))
        manifest["scenes"].append(
            {"name": f"{name}_sphere_negative", "cloud": "negative.npz", "truth": []}
        )
        (output / "validation.json").write_bytes(
            (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
        )
        report = compare_profiles(
            recipe,
            cases,
            repeats=repeats,
            progress=lambda message: print(f"{name}: {message}", flush=True),
        )
        (output / "comparison.json").write_bytes(
            (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
        )
        summaries.append(
            {
                "geometry": name,
                "recommended": report["recommended"],
                "profiles": [
                    {
                        "profile": row["profile"],
                        "compatible": row["compatible"],
                        "mean_scene_median_seconds": row["median_scene_seconds"],
                        "missed": sum(run["missed"] for s in row["scenes"] for run in s["runs"]),
                        "extra": sum(run["extra"] for s in row["scenes"] for run in s["runs"]),
                    }
                    for row in report["profiles"]
                ],
            }
        )
    summary = {
        "models": summaries,
        "positive_scenes": 6,
        "negative_scenes": 2,
        "expected_instances_per_repeat": 12,
        "scope": "Generated fused surfaces with missing slices, noise and clutter. "
        "Real sensor qualification remains open.",
    }
    (folder / "summary.json").write_bytes(
        (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode("utf-8")
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    benchmark(args.folder, args.repeats)
