"""Generate ordered observations of a slightly warped target with a known baseline."""

import argparse
import json
from pathlib import Path

import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.target_calibration import OBSERVATIONS_MODEL

ANCHOR = 6
BASELINE = 121.2
CAMERA = np.array([[860, 0, 480], [0, 850, 360], [0, 0, 1]], dtype=float)


def _rotation(roll, pitch, yaw):
    cx, cy, cz = np.cos([roll, pitch, yaw])
    sx, sy, sz = np.sin([roll, pitch, yaw])
    return (
        np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    )


def fixture():
    u, v = np.meshgrid(np.linspace(0, 1, 7), np.linspace(0, 1, 5))
    u, v = u.ravel(), v.ravel()
    nominal = np.column_stack((120 * u, 80 * v, np.zeros(len(u))))
    offsets = np.column_stack(
        (
            0.95 * np.sin(2 * np.pi * u) * np.sin(np.pi * v),
            0.8 * np.sin(2 * np.pi * v) * np.sin(np.pi * u),
            1.5 * np.sin(np.pi * u) * np.sin(np.pi * v),
        )
    )
    actual = nominal * (BASELINE / 120) + offsets
    actual[-1, 2] = 0
    rng = np.random.default_rng(498132)
    views = []
    for index in range(16):
        rotation = _rotation(*rng.uniform([-0.60, -0.58, -0.34], [0.62, 0.59, 0.36]))
        translation = rng.uniform([-128, -95, 420], [129, 96, 695])
        x, y, z = ((actual - [60, 40, 0]) @ rotation.T + translation).T
        x, y = x / z, y / z
        r2 = x**2 + y**2
        radial = 1 - 0.11 * r2 + 0.03 * r2**2
        distorted_x = x * radial + 0.0016 * x * y - 0.0007 * (r2 + 2 * x**2)
        distorted_y = y * radial + 0.0008 * (r2 + 2 * y**2) - 0.0014 * x * y
        pixels = np.column_stack((860 * distorted_x + 480, 850 * distorted_y + 360))
        pixels += rng.normal(0, 0.025, pixels.shape)
        holdout = index in (3, 10, 15)
        views.append(
            {
                "id": f"{'holdout' if holdout else 'fit'}-{index:02}",
                "role": "holdout" if holdout else "fit",
                "image_points": pixels.round(7).tolist(),
            }
        )
    return {
        "model": OBSERVATIONS_MODEL,
        "target_id": "warped-panel",
        "image_size": [960, 720],
        "units": "mm",
        "object_points": nominal.tolist(),
        "views": views,
    }, actual


def build(output):
    path = output / "assets/observations.json"
    project = output / "refine_calibration_target.json"
    if path.exists() or project.exists():
        raise FileExistsError("Use an empty output directory to preserve the example")
    path.parent.mkdir(parents=True, exist_ok=True)
    observations, _ = fixture()
    path.write_text(
        json.dumps(observations, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8"
    )
    registry = builtin_registry()
    graph = Graph()
    try:
        note = graph.add(registry.create("layout.comment", block_id="note"))
        note.name = "Recover repeatable target imperfections"
        note.position, note.node_size = [0, -245], "large"
        note.description = (
            "Synthetic observations by OpenCVDataflowGUI contributors (MIT). These two "
            "branches use the same measured 121.2 mm anchor baseline, fitting views and "
            "held-out points. Compare released target geometry with the nominal flat control."
        )
        source = graph.add(
            registry.create(
                "input.target_calibration_observations",
                block_id="views",
                parameters={"path": "assets/observations.json"},
            )
        )
        source.position = [0, 80]
        for index, (key, mode) in enumerate(
            (("release", "Release target"), ("nominal", "Fixed target"))
        ):
            fit = graph.add(
                registry.create(
                    "calibration.refine_target",
                    block_id=key,
                    parameters={
                        "fit_mode": mode,
                        "fixed_point_index": ANCHOR,
                        "measured_anchor_distance": BASELINE,
                        "fix_k3": True,
                        "max_fit_rms_pixels": 2.0,
                        "max_holdout_rms_pixels": 0.20,
                    },
                )
            )
            fit.name, fit.position = mode, [440, index * 235]
            graph.connect(source.id, "observations", fit.id, "observations")
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
