"""Build independent metric outer-cylinder and partial-bore scan fixtures."""

import argparse
import json
from pathlib import Path

import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud

CASES = {
    "shaft": {
        "radius_m": 0.021,
        "arc_deg": 140,
        "length_m": 0.080,
        "units": "mm",
        "direction": [0.38, -0.47, 0.78],
        "center_m": [0.087, -0.051, 0.52],
        "label": "outer",
        "seed": 541,
    },
    "bore": {
        "radius_m": 0.0125,
        "arc_deg": 104,
        "length_m": 0.053,
        "units": "m",
        "direction": [-0.64, 0.48, 0.60],
        "center_m": [-0.063, 0.095, 0.35],
        "label": "bore",
        "seed": 793,
    },
}


def make_scan(key, *, arc=None, length=None, normals=False, clutter=True):
    case = CASES[key]
    rng = np.random.default_rng(case["seed"])
    axis = np.array(case["direction"], np.float64)
    axis /= np.linalg.norm(axis)
    u = np.cross(axis, [0, 1, 0])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    origin = np.asarray(case["center_m"])
    radius = case["radius_m"]
    arc_deg = case["arc_deg"] if arc is None else arc
    length_m = case["length_m"] if length is None else length
    # Independently synthesize the cylindrical wall and scan noise in metric space.
    angles = np.deg2rad(np.linspace(145, 145 + arc_deg, 30))
    heights = np.linspace(-length_m / 2, length_m / 2, 23)
    theta, height = np.meshgrid(angles, heights)
    theta = theta.ravel()
    direction = np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v
    noisy_radius = radius + rng.normal(0, 0.000075, len(theta))
    wall = origin + height.ravel()[:, None] * axis + noisy_radius[:, None] * direction
    rows = np.arange(len(wall))
    # Drop repeatable scan lines and random scan returns.
    visible = (rows % 29 != 8) & (rng.uniform(size=len(rows)) > 0.12)
    wall = wall[visible]
    direction = direction[visible]
    good = np.ones(len(wall), bool)
    points = wall
    if clutter:
        # A ring of end-face returns and isolated inner/outer clutter compete with the wall.
        count = 155
        angles = rng.uniform(-np.pi, np.pi, count)
        h = rng.uniform(-0.7, 0.7, count) * length_m
        h[:35] = length_m / 2
        radial = np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v
        offset = rng.choice([-1, 1], count) * rng.uniform(0.13, 0.55, count)
        distractor = origin + h[:, None] * axis + (radius * (1 + offset))[:, None] * radial
        points = np.vstack((wall, distractor))
        good = np.r_[good, np.zeros(count, bool)]
        nuisance = rng.normal(size=(count, 3))
        nuisance /= np.linalg.norm(nuisance, axis=1)[:, None]
        direction = np.vstack((direction, nuisance))
    if key == "bore":
        direction *= -1
    order = rng.permutation(len(points))
    multiplier = 1000 if case["units"] == "mm" else 1
    colors = np.array([[235, 159, 81], [99, 162, 223]], np.uint8)[good.astype(int)]
    cloud = PointCloud(
        points[order] * multiplier,
        case["units"],
        f"{key}_fixture",
        colors=colors[order],
        normals=direction[order] if normals else None,
    )
    truth = {
        "axis": axis.tolist(),
        "center_m": origin.tolist(),
        "radius_m": radius,
        "diameter_m": 2 * radius,
        "input_points": len(good),
        "wall_points": int(good.sum()),
        "arc_deg": arc_deg,
        "axial_span_m": length_m,
    }
    return cloud, truth, good[order]


def build(output):
    path = output / "cloud_cylinder.json"
    if path.exists():
        raise FileExistsError("Use an empty output directory")
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    graph, registry = Graph(), builtin_registry()
    truth = {}

    def add(kind, key, name, pos, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.name, block.position = name, pos
        return block

    try:
        note = add("layout.comment", "note", "Measure shaft and bore diameters", [0, -290])
        note.node_size = "large"
        note.description = (
            "Original synthetic scans by OpenCVDataflowGUI contributors (MIT). "
            "Recover a 42 mm shaft and a 25 mm bore from partial cylindrical walls. "
            "Green points support the fit. Orange points are outside the radial tolerance."
        )
        for i, key in enumerate(CASES):
            cloud, expected, _ = make_scan(key, normals=key == "bore")
            truth[key] = expected
            write_cloud(assets / f"{key}.ply", cloud, binary=False)
            add(
                "input.cloud",
                f"{key}_scan",
                f"Load partial {key} scan",
                [0, 260 * i],
                path=f"assets/{key}.ply",
                units=cloud.units,
                frame=cloud.frame,
            )
            diameter = expected["diameter_m"]
            fit = add(
                "3d.cloud_cylinder",
                key,
                f"Measure {key} wall",
                [452, 260 * i],
                surface=CASES[key]["label"],
                inlier_tolerance_m=0.00035,
                minimum_diameter_m=diameter - 0.0004,
                maximum_diameter_m=diameter + 0.0004,
            )
            fit.exposed_outputs = [
                "model",
                "axis_pose",
                "diameter_m",
                "inliers",
                "outliers",
                "report",
            ]
            graph.connect(f"{key}_scan", "cloud", key, "cloud")
        (assets / "truth.json").write_text(
            json.dumps(truth, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        report = validate_graph(graph)
        if not report["valid"]:
            raise ValueError(report["issues"])
        graph.save(path)
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
