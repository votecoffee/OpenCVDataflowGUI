"""Render a calibrated Mei stereo pair of two colored planes at known distances."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

SIZE = (384, 244)
K1 = np.array([[199.0, 0, 191.3], [0, 192.0, 121.4], [0, 0, 1]])
K2 = np.array([[194.0, 0, 193.0], [0, 198.0, 120.7], [0, 0, 1]])
XI = (0.71, 0.64)
RVEC = np.array([0.008, 0.031, 0.012])
ROTATION = cv2.Rodrigues(RVEC)[0]
CENTER = np.array([0.19, 0, 0])
TRANSLATION = -ROTATION @ CENTER
FOCAL = 180.0
FAR = 2.442857142857143
NEAR = 1.5545454545454547
RECTANGLE = (-0.32, -0.30, 0.34, 0.22)
RIGHT_HOLE = [202, 98, 16, 20]
COMMON = {"num_disparities": 48, "block_size": 5, "sample_step": 3, "maximum_depth_m": 4.0}


def _render(matrix, xi, world_to_camera, camera_center, texture):
    y, x = np.indices(SIZE[::-1], dtype=np.float64)
    u, v = (x - matrix[0, 2]) / matrix[0, 0], (y - matrix[1, 2]) / matrix[1, 1]
    radius2 = u * u + v * v
    factor = (xi + np.sqrt(1 + (1 - xi * xi) * radius2)) / (1 + radius2)
    rays = np.stack((u * factor, v * factor, factor - xi), axis=-1) @ world_to_camera
    near = camera_center + rays * ((NEAR - camera_center[2]) / rays[..., 2:3])
    left, top, right, bottom = RECTANGLE
    foreground = (
        (near[..., 0] >= left)
        & (near[..., 0] <= right)
        & (near[..., 1] >= top)
        & (near[..., 1] <= bottom)
    )
    far = camera_center + rays * ((FAR - camera_center[2]) / rays[..., 2:3])
    world = np.where(foreground[..., None], near, far)
    mx = (world[..., 0] * 126 + 800 + 300 * foreground).astype(np.float32)
    my = (world[..., 1] * 126 + 550 + 305 * foreground).astype(np.float32)
    return cv2.remap(texture, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def render_pair():
    rng = np.random.default_rng(124658)
    noise = rng.integers(12, 246, (1100, 1600), np.uint8)
    tex = cv2.GaussianBlur(noise, (5, 5), 0.9)
    texture = np.stack((tex, np.uint8(tex * 0.82 + 30), np.uint8(tex * 0.63 + 51)), axis=-1)
    return (
        _render(K1, XI[0], np.eye(3), np.zeros(3), texture),
        _render(K2, XI[1], ROTATION, CENTER, texture),
    )


def ideal_rays(size, matrix, mode):
    """Use the fixture's known rectified axes, equal to the raw left optical axes."""
    y, x = np.indices(size[::-1], dtype=np.float64)
    u, v = (x - matrix[0, 2]) / matrix[0, 0], (y - matrix[1, 2]) / matrix[1, 1]
    if mode == "perspective":
        return np.stack((u, v, np.ones_like(u)), axis=-1)
    return np.stack((-np.cos(u), -np.sin(u) * np.cos(v), np.sin(u) * np.sin(v)), axis=-1)


def write_assets(assets):
    assets.mkdir(parents=True, exist_ok=True)
    for side, image in zip(("left", "right"), render_pair(), strict=True):
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise ValueError(f"Could not encode {side}")
        (assets / f"{side}.png").write_bytes(encoded.tobytes())
    truth = {
        "image_size": list(SIZE),
        "units": "m",
        "far_plane_z": FAR,
        "near_plane_z": NEAR,
        "near_plane_xy_bounds": list(RECTANGLE),
        "left_intrinsics": K1.tolist(),
        "right_intrinsics": K2.tolist(),
        "xi": list(XI),
        "left_to_right_rvec": RVEC.tolist(),
        "left_to_right_translation": TRANSLATION.tolist(),
        "right_center_in_left_frame": CENTER.tolist(),
        "right_mask_hole": RIGHT_HOLE,
    }
    (assets / "truth.json").write_text(
        json.dumps(truth, indent=2) + "\n", encoding="utf8", newline="\n"
    )


def build(output):
    path = output / "omnidir_stereo_reconstruction.json"
    if path.exists():
        raise FileExistsError("Use an empty output directory")
    write_assets(output / "assets")
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, name, x, y, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.name, block.position = name, [x, y]
        return block

    try:
        note = add(
            "layout.comment", "note", "Measure two planes with an omnidirectional pair", 0, -260
        )
        note.node_size = "large"
        note.description = (
            "Original synthetic Mei stereo pair by OpenCVDataflowGUI contributors (MIT). "
            "Both branches measure a foreground plane and a background. The right-view mask "
            "excludes a selected patch. Compare the perspective and angular grids."
        )
        for i, (side, matrix, xi) in enumerate(zip(("left", "right"), (K1, K2), XI, strict=True)):
            source = add(
                "input.file",
                side,
                f"Raw {side} camera",
                0,
                215 * i,
                path=f"assets/{side}.png",
                path_base="project",
                color_format="bgr",
            )
            source.ticket_settings = {"mode": "fixed", "value": "omnidir-plane-pair"}
            add(
                "3d.camera",
                f"{side}_model",
                f"{side.title()} Mei camera",
                0,
                475 + 210 * i,
                width=SIZE[0],
                height=SIZE[1],
                lens="omnidir",
                xi=xi,
                frame=f"omni_{side}",
                distortion=[0, 0, 0, 0],
                fx=matrix[0, 0],
                fy=matrix[1, 1],
                cx=matrix[0, 2],
                cy=matrix[1, 2],
            )
        add(
            "3d.pose",
            "relative_pose",
            "Left-to-right camera transform",
            0,
            880,
            rvec=RVEC.tolist(),
            translation=TRANSLATION.tolist(),
            source_frame="omni_left",
            target_frame="omni_right",
        )
        add("3d.stereo_rig", "rig", "Metric Mei stereo rig", 390, 610)
        graph.connect("left_model", "camera", "rig", "left_camera")
        graph.connect("right_model", "camera", "rig", "right_camera")
        graph.connect("relative_pose", "transform", "rig", "transform")
        add(
            "region.shape",
            "exclude",
            "Exclude a patch from the right view",
            390,
            240,
            rectangle=RIGHT_HOLE,
            invert=True,
        )
        graph.connect("right", "image", "exclude", "image")
        for mode, label, height in (
            ("perspective", "Perspective reconstruction", 60),
            ("longitude_latitude", "Longitude/latitude reconstruction", 460),
        ):
            node = add(
                "3d.omnidir_stereo_reconstruct",
                mode,
                label,
                800,
                height,
                **COMMON,
                projection=mode,
                omnidir_focal_length=FOCAL,
                horizontal_span_degrees=108.0,
                vertical_span_degrees=76.0,
            )
            node.node_size = "large"
            for side in ("left", "right"):
                graph.connect(side, "image", mode, side)
            graph.connect("rig", "rig", mode, "rig")
            graph.connect("exclude", "mask", mode, "right_mask")
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise ValueError(validation["issues"])
        graph.save(path)
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
