"""Build wide-angle pose projects from independent synthetic camera equations."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

SIZE = (640, 480)
ANGLES = (0.074, -0.089, 0.05)
TRANSLATION = np.array([8.0, -3.0, 79.0])
LENSES = {
    "fisheye": {"fx": 218.0, "fy": 214.0, "distortion": [-0.07, 0.013, 0.001, -0.0006]},
    "omnidir": {
        "fx": 330.0,
        "fy": 326.0,
        "distortion": [-0.045, 0.012, 0.001, -0.0006],
        "xi": 0.88,
    },
}
CORRUPT = (1, 14, 23, 27, 39, 48, 51, 60)
OUTSIDE_RAY = 9


def rotation():
    ax, ay, az = ANGLES
    sx, sy, sz = np.sin(ANGLES)
    cx, cy, cz = np.cos((ax, ay, az))
    return (
        np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    )


def project(lens, points, r, t):
    model = LENSES[lens]
    xyz = points @ r.T + t
    spatial = xyz[:, :2]
    coeff = model["distortion"]
    if lens == "fisheye":
        radius = np.linalg.norm(spatial, axis=1)
        theta = np.arctan2(radius, xyz[:, 2])
        theta_prime = theta * np.polynomial.polynomial.polyval(theta**2, [1, *coeff])
        uv = (
            spatial
            * np.divide(theta_prime, radius, out=np.zeros_like(radius), where=radius != 0)[:, None]
        )
    else:
        x, y = (spatial / (xyz[:, 2] + model["xi"] * np.linalg.norm(xyz, axis=1))[:, None]).T
        k1, k2, p1, p2 = coeff
        r2 = x * x + y * y
        radial = 1 + r2 * (k1 + r2 * k2)
        uv = np.column_stack(
            (
                x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y,
            )
        )
    return uv * [model["fx"], model["fy"]] + (np.array(SIZE) - 1) / 2


def test_scene(lens):
    xx, yy = np.meshgrid(np.linspace(-145, 145, 9), np.linspace(-104, 104, 7))
    xyz = np.column_stack(
        (xx.ravel(), yy.ravel(), 11 * np.cos(xx.ravel() / 26) + 8 * np.sin(yy.ravel() / 40))
    )
    r, t = rotation(), TRANSLATION
    uv = project(lens, xyz, r, t)
    rng = np.random.default_rng(471 if lens == "fisheye" else 472)
    measured = uv + rng.normal(0, 0.06, uv.shape)
    offsets = np.column_stack(
        (np.linspace(-16, 22, len(CORRUPT)), np.linspace(26, -22, len(CORRUPT)))
    )
    measured[list(CORRUPT)] += offsets
    measured[OUTSIDE_RAY] = [5, 5]
    held = np.array([[-133, 95, -8], [-2, 3, 7], [111, 94, 5], [88, -96, -7], [-114, -87, 1]])
    assert (measured >= 0).all() and (measured < np.asarray(SIZE)).all()
    return {
        "rotation": r.tolist(),
        "translation": t.tolist(),
        "xyz": xyz.tolist(),
        "measured": measured.tolist(),
        "pixels_truth": uv.tolist(),
        "accepted": np.setdiff1d(np.arange(len(xyz)), [*CORRUPT, OUTSIDE_RAY]).tolist(),
        "out_of_domain_row": OUTSIDE_RAY,
        "held_xyz": held.tolist(),
        "held_pixels": project(lens, held, r, t).tolist(),
    }


def png(path, uv):
    image = np.full((SIZE[1], SIZE[0], 3), (48, 41, 33), np.uint8)
    for index, (x, y) in enumerate(np.rint(uv).astype(int), 1):
        cv2.circle(image, (int(x), int(y)), 4, (200, 205, 210), -1, cv2.LINE_AA)
        cv2.putText(
            image,
            str(index),
            (int(x) + 7, int(y) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (187, 180, 172),
            1,
            cv2.LINE_AA,
        )
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise OSError(path)
    path.write_bytes(encoded.tobytes())


def save_graph(output, lens, truth):
    registry, graph = builtin_registry(), Graph()

    def add(type_id, key, label, x, y, **params):
        block = graph.add(registry.create(type_id, block_id=key, parameters=params))
        block.name, block.position = label, [x, y]
        return block

    try:
        note = add(
            "layout.comment", "note", "Estimate a pose in the original wide-angle image", 0, -255
        )
        note.node_size = "large"
        note.description = (
            "Original synthetic target by OpenCVDataflowGUI contributors (MIT). "
            "Known XYZ and simulated point detections include gross mismatches and a ray "
            "outside the forward domain. Held-out target points verify the fitted pose."
        )
        add(
            "3d.camera",
            "camera",
            "Calibrated " + lens + " camera",
            0,
            0,
            lens=lens,
            width=SIZE[0],
            height=SIZE[1],
            cx=319.5,
            cy=239.5,
            frame="wide_camera",
            units="mm",
            **LENSES[lens],
        )
        add(
            "3d.points",
            "xyz",
            "Known XYZ landmarks",
            0,
            190,
            points=truth["xyz"],
            units="mm",
            frame="fixture",
        )
        add(
            "ml.vector_data",
            "xy",
            "Recorded simulated XY detections",
            0,
            370,
            values=truth["measured"],
        )
        add(
            "input.file",
            "view",
            "View target with labeled landmarks",
            300,
            380,
            path=f"assets/{lens}.png",
            path_base="project",
        )
        add(
            "3d.wide_pnp",
            "fit",
            "Fit and check the raw lens reprojection",
            665,
            35,
            source_frame="fixture",
            inlier_threshold_px=0.65,
            max_rms_px=0.24,
            minimum_inliers=48,
        )
        add(
            "3d.points",
            "held",
            "Five withheld landmarks",
            655,
            430,
            points=truth["held_xyz"],
            units="mm",
            frame="fixture",
        )
        add("3d.project", "project_check", "Reproject withheld landmarks", 1060, 110)
        for a, port, b, terminal in (
            ("camera", "camera", "fit", "camera"),
            ("xyz", "points", "fit", "object_points"),
            ("xy", "vectors", "fit", "image_points"),
            ("view", "image", "fit", "image"),
            ("fit", "pose", "project_check", "transform"),
            ("camera", "camera", "project_check", "camera"),
            ("held", "points", "project_check", "points"),
        ):
            graph.connect(a, port, b, terminal)
        contract = validate_graph(graph)
        if not contract["valid"]:
            raise ValueError(contract["issues"])
        graph.save(output / f"{lens}_object_pose.json")
    finally:
        graph.close()


def build(output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty output directory")
    assets = output / "assets"
    assets.mkdir(parents=True)
    truth = {}
    for lens in LENSES:
        truth[lens] = test_scene(lens)
        png(assets / f"{lens}.png", truth[lens]["pixels_truth"])
        save_graph(output, lens, truth[lens])
    (assets / "truth.json").write_text(
        json.dumps(truth, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
