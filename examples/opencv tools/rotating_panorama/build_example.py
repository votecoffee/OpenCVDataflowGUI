"""Original pinhole rotating views, known focal/yaw and independently projected landmarks."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
SIZE = [480, 360]
FOCAL = 420.0
YAW = {"left": -36.0, "center": 0.0, "right": 36.0}
CASES = {
    "unordered": {"order": ["left", "right", "center"]},
    "reordered": {"order": ["center", "right", "left"]},
    "disconnected": {"order": ["left", "right", "center", "unrelated"], "passed": False},
    "allow_partial": {"order": ["left", "right", "center", "unrelated"], "policy": "allow_partial"},
    "fail_partial": {
        "order": ["left", "right", "center", "unrelated"],
        "policy": "fail_partial",
        "passed": False,
    },
    "uint16_plane": {
        "order": ["left", "right", "center"],
        "depth": "uint16",
        "projection": "plane",
    },
    "masked": {"order": ["left", "right", "center"], "masks": True},
    "distorted": {"order": ["left", "center", "distorted"], "passed": False},
}


def yaw_matrix(degrees):
    a = np.deg2rad(degrees)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def controls():
    rng = np.random.default_rng(104)
    texture = cv2.GaussianBlur(rng.integers(30, 226, (640, 1400, 3), np.uint8), (3, 3), 0)
    for _ in range(140):
        x, y = rng.integers([10, 10], [1390, 630])
        r = int(rng.integers(3, 18))
        cv2.circle(texture, (int(x), int(y)), r, tuple(map(int, rng.integers(10, 246, 3))), 2)
    w, h = SIZE
    y, x = np.indices((h, w), np.float64)
    rays = np.stack([(x - w / 2) / FOCAL, (y - h / 2) / FOCAL, np.ones_like(x)], axis=2)
    images = {}
    landmarks = {}
    world = np.array(
        [
            [(u - 700) / 280, (v - 320) / 280, 1]
            for u in range(360, 1041, 80)
            for v in (220, 320, 420)
        ],
        np.float64,
    )
    for label, yaw in YAW.items():
        R = yaw_matrix(yaw)
        q = rays @ R.T
        mx = (280 * q[..., 0] / q[..., 2] + 700).astype(np.float32)
        my = (280 * q[..., 1] / q[..., 2] + 320).astype(np.float32)
        images[label] = cv2.remap(texture, mx, my, cv2.INTER_LINEAR)
        local = world @ R
        points = local[:, :2] / local[:, 2, None] * FOCAL + [w / 2, h / 2]
        visible = (
            (local[:, 2] > 0) & (points > 15).all(axis=1) & (points < [w - 16, h - 16]).all(axis=1)
        )
        landmarks[label] = {
            "pixels": points[visible].tolist(),
            "world_rays": world[visible].tolist(),
        }
    images["unrelated"] = np.random.default_rng(22).integers(0, 256, (h, w, 3), np.uint8)
    images["distorted"] = cv2.warpAffine(
        images["right"], np.float32([[1.4, 0, -w * 0.2], [0, 0.7, h * 0.15]]), (w, h)
    )
    return images, landmarks


def build():
    assets = HERE / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    images, landmarks = controls()
    hashes = {}
    for label, image in images.items():
        variants = {label: image}
        if label in YAW:
            variants[label + "_uint16"] = image.astype(np.uint16) * 257
        for name, pixels in variants.items():
            ok, data = cv2.imencode(".png", pixels)
            assert ok
            (assets / f"{name}.png").write_bytes(data.tobytes())
            hashes[name + ".png"] = hashlib.sha256(data).hexdigest()
    registry = builtin_registry()
    for case, options in CASES.items():
        graph = Graph()
        try:

            def add(kind, key, parameters=None, position=(0, 0)):
                b = graph.add(registry.create(kind, block_id=key, parameters=parameters or {}))
                b.position = list(position)
                return b

            order = options["order"]
            for i, label in enumerate(order):
                asset = label + ("_uint16" if options.get("depth") == "uint16" else "")
                source = add(
                    "input.file",
                    label,
                    {
                        "path": f"assets/{asset}.png",
                        "path_base": "project",
                        "pixel_depth": options.get("depth", "uint8"),
                    },
                    (0, 300 * i),
                )
                source.name = label + " camera view"
            add("flow.batch", "batch", {"size": len(order)}, (760, 0))
            for i, label in enumerate(order):
                graph.connect(label, "image", "batch", f"item_{i + 1}")
            stitch = add(
                "cv.stitch_rotating_panorama",
                "panorama",
                {
                    "anchor_label": "center",
                    "projection_scale": 280.0,
                    "projection": options.get("projection", "spherical"),
                    "component_policy": options.get("policy", "reject"),
                },
                (1140, 0),
            )
            stitch.exposed_outputs.extend(["diagnostic", "contributors", "result_graphics"])
            stitch.node_size = "large"
            graph.connect("batch", "group", "panorama", "views")
            if options.get("masks"):
                add("flow.batch", "mask_batch", {"size": len(order)}, (760, 450))
                for i, label in enumerate(order):
                    key = "mask_" + label
                    add(
                        "region.shape",
                        key,
                        {"rectangle": [8, 8, SIZE[0] - 16, SIZE[1] - 16]},
                        (380, i * 300),
                    )
                    graph.connect(label, "image", key, "image")
                    graph.connect(key, "mask", "mask_batch", f"item_{i + 1}")
                graph.connect("mask_batch", "group", "panorama", "masks")
            add(
                "cv.convert",
                "corner_pixels",
                {"color_format": "gray", "pixel_depth": "uint8"},
                (380, -350),
            )
            graph.connect(order[0], "image", "corner_pixels", "image")
            add(
                "inspect.corners", "corners", {"max_corners": 32, "min_distance": 12.0}, (760, -350)
            )
            graph.connect("corner_pixels", "image", "corners", "image")
            points = add("cv.panorama_points", "points", {"source_index": 0}, (1520, -350))
            points.exposed_outputs.append("result_graphics")
            if options.get("masks"):
                points.exposed_inputs.append("source_mask")
                graph.connect("mask_" + order[0], "mask", "points", "source_mask")
            for wire in [
                ("panorama", "layout", "points", "layout"),
                ("corners", "points", "points", "points"),
                ("panorama", "coverage", "points", "mask"),
            ]:
                graph.connect(*wire)
            add("cv.overlay_result_graphics", "overlay", position=(1900, 0))
            for wire in [
                ("panorama", "image", "overlay", "image"),
                ("panorama", "result_graphics", "overlay", "graphics_1"),
                ("points", "result_graphics", "overlay", "graphics_2"),
            ]:
                graph.connect(*wire)
            add("output.filmstrip_image", "filmstrip", {"priority": 10}, (2280, 0))
            add("output.filmstrip_image", "diagnostic_filmstrip", {"priority": 0}, (1520, 450))
            graph.connect("overlay", "image", "filmstrip", "image")
            graph.connect("panorama", "diagnostic", "diagnostic_filmstrip", "image")
            note = add("layout.comment", "attribution", position=(0, -650))
            note.name = "Original rotating-camera controls - MIT"
            note.node_size = "large"
            note.description = (
                "Original MIT environment texture, pinhole captures, generator, projects and "
                "known landmarks by OpenCVDataflowGUI contributors. No customer or third-party "
                "images. Known camera focal 420 pixels on 480x360 views; yaw -36, 0, +36 degrees. "
                "Left/right have no direct overlap; center connects them even when input order "
                "puts the disconnected pair first. Native camera fitting is independently checked "
                "against feature homographies. Coverage excludes unobserved pixels. Point markers "
                "map source corners into the new panorama frame. Disconnected/distorted examples "
                "fail normally; partial policies are explicit. Diagnostics use a separate grid. "
                "These camera estimates are for panorama fitting, not physical metrology, lens "
                "calibration or robot extrinsics. See README/truth.json for numerical tolerances."
            )
            graph.save(HERE / f"{case}.json")
        finally:
            graph.close()
    truth = {
        "license": "MIT - original repository fixtures",
        "source_size": SIZE,
        "focal_pixels": FOCAL,
        "yaw_degrees": YAW,
        "landmarks": landmarks,
        "cases": CASES,
        "asset_sha256": hashes,
        "acceptance": {
            "focal_relative_error": 0.10,
            "relative_rotation_degrees": 3.0,
            "landmark_rms_pixels": 5.0,
        },
    }
    (HERE / "truth.json").write_bytes(
        (json.dumps(truth, indent=2, allow_nan=False) + "\n").encode()
    )


if __name__ == "__main__":
    build()
