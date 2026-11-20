"""Build a known three-camera chain with disjoint training and validation captures."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

FRAMES = ("camera_a", "camera_b", "camera_c")
SIZE = (1280, 960)


def optics():
    return [
        np.array([[676 + 22 * i, 0, 640], [0, 681 + 27 * i, 480], [0, 0, 1.0]]) for i in range(3)
    ]


def transforms():
    rotations = ((0.0, 0.0, 0.0), (0.018, 0.064, -0.014), (-0.036, -0.055, 0.021))
    offsets = ((0.0, 0.0, 0.0), (-0.17, 0.013, 0.024), (0.19, 0.041, -0.018))
    results = []
    for rv, offset in zip(rotations, offsets, strict=True):
        pose = np.eye(4)
        pose[:3, :3] = cv2.Rodrigues(np.asarray(rv))[0]
        pose[:3, 3] = offset
        results.append(pose)
    return results


def observations(label, captures, seed, lens="pinhole"):
    grid = np.array([(x * 0.05, y * 0.05, 0) for y in range(-3, 4) for x in range(-4, 5)])
    rng = np.random.default_rng(seed)
    visibility = np.ones((3, captures), dtype=int)
    visibility[0, captures // 2 :] = 0
    visibility[2, : captures // 2] = 0
    pixels = [[] for _ in FRAMES]
    geometry = list(zip(optics(), transforms(), strict=True))
    for f in range(captures):
        rotation = cv2.Rodrigues(rng.uniform(-0.42, 0.42, 3))[0]
        offset = rng.uniform([-0.22, -0.13, 0.8], [0.21, 0.12, 1.55])
        scene = grid @ rotation.T + offset
        for c, (k, pose) in enumerate(geometry):
            if not visibility[c, f]:
                pixels[c].append(None)
                continue
            xyz = scene @ pose[:3, :3].T + pose[:3, 3]
            rays = xyz[:, :2] / xyz[:, 2:]
            if lens == "fisheye":
                radius = np.linalg.norm(rays, axis=1, keepdims=True)
                rays *= np.arctan(radius) / np.where(radius > 0, radius, 1)
            uv = rays @ k[:2, :2] + k[:2, 2] + rng.normal(0, 0.02, rays.shape)
            pixels[c].append(uv.round(5).tolist())
    return {
        "model": "multicamera_observations_v1",
        "units": "m",
        "camera_frames": list(FRAMES),
        "image_sizes": [list(SIZE) for _ in FRAMES],
        "capture_ids": [f"{label}-{i + 1:02}" for i in range(captures)],
        "object_points": grid.round(5).tolist(),
        "visibility": visibility.tolist(),
        "image_points": pixels,
    }


def save_graph(path):
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **parameters):
        node = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        node.position = position
        return node

    try:
        notes = add("layout.comment", "notes", [0, -235])
        notes.name, notes.node_size = "Calibrate three cameras with missing views", "large"
        notes.description = (
            "Original synthetic rig and target captures by OpenCVDataflowGUI contributors (MIT). "
            "Camera B bridges two sets of captures. Cameras A and C never see the target "
            "in the same training capture. The validation set uses independent target poses."
        )
        for i, (key, name) in enumerate(
            (("train", "Training captures"), ("holdout", "Independent validation captures"))
        ):
            reader = add(
                "input.multicamera_observations", key, [0, 265 * i], path=f"assets/{key}.json"
            )
            reader.name = name
            reader.ticket_settings = {"mode": "fixed", "value": "synthetic-rig-calibration"}
        fit = add(
            "3d.calibrate_multicamera", "fit", [330, 0], max_rms_px=0.08, max_view_rms_px=0.12
        )
        checked = add(
            "3d.validate_multicamera",
            "holdout_check",
            [710, 220],
            max_rms_px=0.09,
            max_view_rms_px=0.18,
            max_anchor_rms_px=0.12,
        )
        graph.connect("train", "observations", fit.id, "observations")
        graph.connect(fit.id, "rig", checked.id, "rig")
        graph.connect("holdout", "observations", checked.id, "observations")
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise ValueError(validation["issues"])
        graph.save(path)
    finally:
        graph.close()


def build(output):
    assets = output / "assets"
    path = output / "multicamera_rig.json"
    if assets.exists() or path.exists():
        raise FileExistsError("Use a new output directory to preserve the example")
    assets.mkdir(parents=True)
    for key, captures, seed in (("train", 20, 85147), ("holdout", 6, 15198)):
        record = observations(key, captures, seed)
        (assets / f"{key}.json").write_text(
            json.dumps(record, separators=(",", ":")) + "\n", encoding="utf-8"
        )
    save_graph(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
