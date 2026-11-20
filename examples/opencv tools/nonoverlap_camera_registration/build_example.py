"""Generate a fixed camera pair and disjoint observed regions of one rigid target."""

import argparse
import json
from pathlib import Path

import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

K1 = np.array([[850.0, 0, 480], [0, 870.0, 360], [0, 0, 1.0]])
K2 = np.array([[820.0, 0, 400], [0, 800.0, 300], [0, 0, 1.0]])
BROWN = np.array([-0.13, 0.04, 0.001, -0.0015, 0.0])
FISHEYE = np.array([0.09, -0.035, 0.005, -0.0004])
SIZES = ((960, 720), (800, 600))
FRAMES = ("station_a", "station_b")
SHIFT = np.array([-1190.0, 45.0, 8.0])


def euler(a, b, c):
    ca, cb, cc = np.cos([a, b, c])
    sa, sb, sc = np.sin([a, b, c])
    return (
        np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
        @ np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
        @ np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    )


ROTATION = euler(0.03, -0.045, 0.021)


def project(xyz, intrinsics, lens):
    x, y = (xyz[:, :2] / xyz[:, 2:]).T
    r2 = x**2 + y**2
    if lens == "fisheye":
        radius = np.sqrt(r2)
        theta = np.arctan(radius)
        k1, k2, k3, k4 = FISHEYE
        factor = np.ones_like(radius)
        warped = theta * (1 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8)
        np.divide(warped, radius, out=factor, where=radius > 0)
        u, v = x * factor, y * factor
    else:
        k1, k2, p1, p2, k3 = BROWN
        factor = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        u = x * factor + 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
        v = y * factor + p1 * (r2 + 2 * y**2) + 2 * p2 * x * y
    return np.column_stack(
        (intrinsics[0, 0] * u + intrinsics[0, 2], intrinsics[1, 1] * v + intrinsics[1, 2])
    )


def observations(second_lens, prefix, count, seed, drift=0.0):
    x, y = np.meshgrid(np.linspace(-145, 145, 6), np.linspace(-115, 115, 5))
    left = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    x, y = np.meshgrid(np.linspace(-140, 140, 7), np.linspace(-105, 105, 4))
    right = np.column_stack((x.ravel() + 1200, y.ravel(), 0.08 * x.ravel()))
    object_points = np.vstack((left, right))
    rng = np.random.default_rng(seed)
    images = [[], []]
    for i in range(count):
        angles = rng.uniform([-0.18, -0.19, -0.08], [0.18, 0.18, 0.08])
        lift = rng.uniform([-48, -40, 640], [50, 42, 980])
        a = object_points @ euler(*angles).T + lift
        b = a @ ROTATION.T + SHIFT + [drift, 0, 0]
        for c, (xyz, k, size, lens) in enumerate(
            ((a, K1, SIZES[0], "pinhole"), (b, K2, SIZES[1], second_lens))
        ):
            if not (xyz[:, 2] > 0).all():
                raise ValueError("Fixture target must be in front of both cameras")
            xy = project(xyz, k, lens)
            valid = (xy >= 0).all(axis=1) & (xy < np.asarray(size)).all(axis=1)
            panel = slice(0, len(left)) if c == 0 else slice(len(left), None)
            hidden = slice(len(left), None) if c == 0 else slice(0, len(left))
            if not valid[panel].all() or valid[hidden].any():
                raise ValueError("Fixture must have disjoint visible target regions")
            tracked = (
                range(0, len(left) - i % 3)
                if c == 0
                else range(len(left) + i % 4, len(object_points))
            )
            recorded = xy[list(tracked)] + rng.normal(0, 0.017, (len(tracked), 2))
            obs = [None] * len(object_points)
            for row, pixel in zip(tracked, recorded.round(7), strict=True):
                obs[row] = pixel.tolist()
            images[c].append(obs)
    return {
        "model": "multicamera_observations_v1",
        "units": "mm",
        "camera_frames": list(FRAMES),
        "image_sizes": [list(size) for size in SIZES],
        "capture_ids": [f"{prefix}-{i:02}" for i in range(count)],
        "object_points": object_points.tolist(),
        "visibility": [[1] * count, [1] * count],
        "image_points": images,
    }


def build(output):
    saved = output / "nonoverlap_camera_registration.json"
    records = {
        "fit_pinhole": observations("pinhole", "fit", 8, 827),
        "held_pinhole": observations("pinhole", "holdout", 4, 822),
        "fit_mixed": observations("fisheye", "fit", 8, 827),
        "held_mixed": observations("fisheye", "holdout", 4, 822),
        "drift_mixed": observations("fisheye", "drift", 4, 822, drift=9.0),
    }
    paths = {key: output / f"assets/{key}.json" for key in records}
    if saved.exists() or any(path.exists() for path in paths.values()):
        raise FileExistsError("Use an empty output directory to preserve the example")
    (output / "assets").mkdir(parents=True, exist_ok=True)
    for key, record in records.items():
        paths[key].write_text(
            json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8"
        )

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, name, position, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = name, position
        return block

    try:
        note = add(
            "layout.comment", "note", "Register cameras across disjoint target panels", [0, -260]
        )
        note.node_size = "large"
        note.description = (
            "Synthetic observations by OpenCVDataflowGUI contributors (MIT). "
            "Each camera sees its own panel on one rigid target. Compare the fixed "
            "pinhole and mixed-lens fits on new poses. The drifted holdout changes "
            "one camera by 9 mm and intentionally fails cross-projection."
        )
        for i, key in enumerate(records):
            add(
                "input.multicamera_observations",
                key,
                key.replace("_", " ").capitalize(),
                [0, i * 220],
                path=f"assets/{key}.json",
            )

        for i, (key, k, lens, index) in enumerate(
            (
                ("cam_a", K1, "pinhole", 0),
                ("cam_b", K2, "pinhole", 1),
                ("cam_fish", K2, "fisheye", 1),
            )
        ):
            add(
                "3d.camera",
                key,
                f"{FRAMES[index]} {lens}",
                [370, i * 220],
                fx=k[0, 0],
                fy=k[1, 1],
                cx=k[0, 2],
                cy=k[1, 2],
                width=SIZES[index][0],
                height=SIZES[index][1],
                frame=FRAMES[index],
                units="mm",
                lens=lens,
                distortion=(FISHEYE if lens == "fisheye" else BROWN).tolist(),
            )
        add(
            "3d.pose",
            "seed",
            "Rough mixed-pair pose",
            [370, 660],
            source_frame=FRAMES[0],
            target_frame=FRAMES[1],
            units="m",
            translation=[-1.18, 0.055, 0.018],
        )
        branches = (
            ("pinhole", "fit_pinhole", "held_pinhole", "cam_b"),
            ("mixed", "fit_mixed", "held_mixed", "cam_fish"),
            ("drift", "fit_mixed", "drift_mixed", "cam_fish"),
        )
        for i, (key, fit, held, second) in enumerate(branches):
            add(
                "3d.register_cameras",
                key,
                key.capitalize(),
                [840, i * 290],
                max_holdout_rms_px=0.25,
                max_view_rms_px=0.5,
            )
            graph.connect(fit, "observations", key, "observations")
            graph.connect(held, "observations", key, "holdout")
            graph.connect("cam_a", "camera", key, "first_camera")
            graph.connect(second, "camera", key, "second_camera")
            if second == "cam_fish":
                graph.connect("seed", "transform", key, "initial_pose")
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise ValueError(validation["issues"])
        graph.save(saved)
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
