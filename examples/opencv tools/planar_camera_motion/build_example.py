"""Create a translated and rotated camera pair viewing one textured plane."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

SIZE = (640, 480)
K1 = np.array([[640.0, 0, 321.0], [0, 652.0, 237.0], [0, 0, 1]])
K2 = np.array([[655.0, 0, 326.0], [0, 646.0, 235.0], [0, 0, 1]])
TRANSLATION = np.array([0.115, -0.03, 0.11])
DISTANCE = 1.45
NORMAL = np.array([0.18, -0.13, 1.0])
NORMAL /= np.linalg.norm(NORMAL)
D1 = (-0.096, 0.017, 0.0012, -0.0019, 0.0)
D2 = (-0.124, 0.029, -0.0021, 0.0014, 0.0)


def rotation_matrix():
    sx, sy, sz = np.sin([0.04, -0.09, 0.027])
    cx, cy, cz = np.cos([0.04, -0.09, 0.027])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def brown_projection(xyz, k, distortion):
    x, y = np.moveaxis(xyz[:, :2] / xyz[:, 2:3], -1, 0)
    k1, k2, p1, p2, k3 = distortion
    rr = x * x + y * y
    factor = 1 + rr * (k1 + rr * (k2 + k3 * rr))
    return np.column_stack(
        (
            k[0, 0] * (x * factor + 2 * p1 * x * y + p2 * (rr + 2 * x * x)) + k[0, 2],
            k[1, 1] * (y * factor + p1 * (rr + 2 * y * y) + 2 * p2 * x * y) + k[1, 2],
        )
    )


def analytic_pairs(*, translation=TRANSLATION, use_distortion=True):
    rays = np.array(
        [[x, y, 1] for x in np.linspace(-0.38, 0.38, 12) for y in np.linspace(-0.25, 0.25, 9)]
    )
    xyz = rays * (DISTANCE / (rays @ NORMAL))[:, None]
    other = xyz @ rotation_matrix().T + translation
    left = brown_projection(xyz, K1, D1 if use_distortion else (0,) * 5)
    right = brown_projection(other, K2, D2 if use_distortion else (0,) * 5)
    return left, right


def pattern():
    rng = np.random.default_rng(541877)
    h, w = SIZE[1], SIZE[0]
    noise = rng.integers(20, 235, (120, 160), dtype=np.uint8)
    gray = cv2.resize(noise, SIZE, interpolation=cv2.INTER_CUBIC)
    for idx in range(210):
        x, y = [int(v) for v in rng.integers([24, 24], [w - 24, h - 24])]
        size, shade = [int(v) for v in rng.integers([2, 20], [12, 240])]
        if idx % 2:
            cv2.circle(gray, (x, y), size, shade, 2, cv2.LINE_AA)
        else:
            cv2.rectangle(gray, (x, y), (x + size, y + size), shade, 2, cv2.LINE_AA)
    return gray


def write_assets(out):
    out.mkdir(parents=True, exist_ok=True)
    reference = pattern()
    rotation = rotation_matrix()
    h = K2 @ (rotation + np.outer(TRANSLATION / DISTANCE, NORMAL)) @ np.linalg.inv(K1)
    target = cv2.warpPerspective(
        reference, h, SIZE, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=22
    )
    for name, image in (("source", reference), ("target", target)):
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise ValueError(f"Could not encode {name}")
        (out / f"{name}.png").write_bytes(encoded.tobytes())
    truth = {
        "source_frame": "source",
        "target_frame": "target",
        "rotation": rotation.tolist(),
        "translation_m": TRANSLATION.tolist(),
        "source_normal": NORMAL.tolist(),
        "source_plane_distance_m": DISTANCE,
        "source_to_target_image_homography": h.tolist(),
    }
    (out / "truth.json").write_text(
        json.dumps(truth, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def build(directory):
    path = directory / "planar_camera_motion.json"
    if path.exists():
        raise FileExistsError("Use an empty output directory")
    write_assets(directory / "assets")
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, name, pos, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.name, block.position = name, pos
        if block.ticket_source:
            block.ticket_settings = {"mode": "fixed", "value": "planar-camera-pair"}
        return block

    try:
        note = add("layout.comment", "note", "Recover camera motion from a plane", [0, -270])
        note.node_size = "large"
        note.description = (
            "Original synthetic texture and geometry by OpenCVDataflowGUI contributors (MIT). "
            "SIFT matches two calibrated views. Compare the two unscaled alternatives "
            "with the single motion selected using a measured plane normal and distance."
        )
        for i, (side, k) in enumerate((("source", K1), ("target", K2))):
            add(
                "input.file",
                side,
                f"{side.title()} image",
                [0, 400 * i],
                path=f"assets/{side}.png",
                path_base="project",
                color_format="gray",
            )
            add(
                "3d.camera",
                f"{side}_camera",
                f"{side.title()} intrinsics",
                [0, 200 + 400 * i],
                fx=k[0, 0],
                fy=k[1, 1],
                cx=k[0, 2],
                cy=k[1, 2],
                width=SIZE[0],
                height=SIZE[1],
                frame=side,
            )
            add(
                "inspect.sift",
                f"{side}_features",
                f"Detect {side} SIFT features",
                [345, 400 * i],
                max_features=550,
                contrast_threshold=0.028,
            )
            graph.connect(side, "image", f"{side}_features", "image")
        add(
            "inspect.match_sift",
            "matches",
            "Match source to target",
            [670, 170],
            ratio=0.72,
            max_distance=300,
        )
        graph.connect("source_features", "features", "matches", "query")
        graph.connect("target_features", "features", "matches", "train")
        for i, (key, label) in enumerate(
            (
                ("ambiguous", "Image-only alternatives"),
                ("metric", "Resolve motion with known plane"),
            )
        ):
            anchor = {"plane_distance_m": DISTANCE, "source_normal": NORMAL.tolist()} if i else {}
            add(
                "3d.planar_camera_motion",
                key,
                label,
                [1070, 395 * i],
                ransac_pixels=1.5,
                maximum_rms_px=1.0,
                **anchor,
            )
            graph.connect("matches", "matches", key, "matches")
            for side in ("source", "target"):
                graph.connect(f"{side}_camera", "camera", key, f"{side}_camera")
                graph.connect(side, "image", key, f"{side}_image")
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
