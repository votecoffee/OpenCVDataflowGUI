"""Build a calibrated three-camera scene with independent projection and depth truth."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

SIZE = (256, 160)
NAMES = ("reference", "second", "third")
UNITS = "m"
DEPTH = 1.15
CENTERS = ((0, 0, 0), (0.054, 0, 0), (0.117, 0, 0))
YAWS = (-0.025, 0.025, -0.055)
DISTORTION = (-0.076, 0.018, 0.001, -0.0008, 0.0)
INTRINSICS = ((212, 208, 126, 78), (211, 213, 130, 81), (207, 212, 128, 79))


def yaw(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def brown(xy, distortion):
    x, y = np.moveaxis(xy, -1, 0)
    k1, k2, p1, p2, k3 = distortion
    r2 = x * x + y * y
    radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
    return np.stack(
        (
            x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
            y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y,
        ),
        axis=-1,
    )


def project(xyz, center, rotation, intrinsics, distortion=DISTORTION):
    camera = (xyz - center) @ rotation.T
    model = brown(camera[..., :2] / camera[..., 2:3], distortion)
    return model * intrinsics[:2] + intrinsics[2:]


def render(texture, center, rotation, intrinsics):
    y, x = np.indices(SIZE[::-1], dtype=np.float64)
    distorted = (np.stack((x, y), axis=-1) - intrinsics[2:]) / intrinsics[:2]
    normalized = distorted.copy()
    for _ in range(14):
        normalized += distorted - brown(normalized, DISTORTION)
    rays = np.dstack((normalized, np.ones_like(x))) @ rotation
    xy = center[:2] + rays[..., :2] * ((DEPTH - center[2]) / rays[..., 2:3])
    texture_pixels = xy * (204 / DEPTH) + np.array([212, 150])
    mx, my = np.moveaxis(texture_pixels.astype(np.float32), -1, 0)
    return cv2.remap(texture, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def check_observations(centers=CENTERS, rotations=None, intrinsics=INTRINSICS, units=UNITS):
    if rotations is None:
        rotations = [yaw(angle) for angle in YAWS]
    scale = 1000 if units == "mm" else 1
    board = np.array(
        [(x, y, 0) for y in np.linspace(-0.13, 0.13, 5) for x in np.linspace(-0.18, 0.18, 7)]
    )
    captures = (board + [-0.02, 0.03, 0.92], board @ yaw(-0.18).T + [0.09, -0.07, 1.3])
    return {
        "model": "multicamera_observations_v1",
        "units": units,
        "camera_frames": list(NAMES),
        "image_sizes": [list(SIZE)] * 3,
        "capture_ids": ["check-near", "check-far"],
        "visibility": [[True, True] for _ in NAMES],
        "object_points": (board * scale).tolist(),
        "image_points": [
            [project(view, center, r, k).tolist() for view in captures]
            for center, r, k in zip(centers, rotations, intrinsics, strict=True)
        ],
    }


def write_png(path, image):
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError(f"Could not write {path}")
    path.write_bytes(encoded.tobytes())


def build(directory):
    assets = directory / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(712540)
    texture = cv2.GaussianBlur(rng.integers(10, 245, (300, 424), np.uint8), (5, 5), 0.9)
    rotations = [yaw(angle) for angle in YAWS]
    for name, center, rotation, k in zip(NAMES, CENTERS, rotations, INTRINSICS, strict=True):
        write_png(assets / f"{name}.png", render(texture, np.asarray(center), rotation, k))
    depth_mm = np.full(SIZE[::-1], round(DEPTH * 1000), np.uint16)
    write_png(assets / "held_out_depth_mm.png", depth_mm)
    (assets / "check.json").write_text(
        json.dumps(check_observations(), separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    save_graph(directory / "collinear_stereo.json")


def save_graph(path):
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, name, pos, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.name, block.position = name, pos
        if block.ticket_source:
            block.ticket_settings = {"mode": "fixed", "value": "three-camera-plane"}
        return block

    try:
        note = add("layout.comment", "note", "Three-camera rectification and depth", [0, -310])
        note.node_size = "large"
        note.description = (
            "Synthetic images and independent geometry by OpenCVDataflowGUI contributors (MIT). "
            "Three cameras observe a textured plane at 1.15 m. Both baselines match on one "
            "reference grid. Separate sparse target observations check row alignment."
        )
        add("3d.rectify_collinear", "rectify", "Common-row rectification", [785, 250])
        for i, (name, center, angle, k) in enumerate(
            zip(NAMES, CENTERS, YAWS, INTRINSICS, strict=True)
        ):
            label = name.title()
            add(
                "input.file",
                name,
                f"{label} image",
                [0, i * 220],
                color_format="gray",
                path=f"assets/{name}.png",
                path_base="project",
            )
            add(
                "3d.camera",
                f"camera_{name}",
                f"{label} calibration",
                [310, i * 220],
                width=SIZE[0],
                height=SIZE[1],
                fx=k[0],
                fy=k[1],
                cx=k[2],
                cy=k[3],
                frame=name,
                distortion=list(DISTORTION),
            )
            graph.connect(name, "image", "rectify", name)
            graph.connect(f"camera_{name}", "camera", "rectify", f"{name}_camera")
            if i:
                add(
                    "3d.pose",
                    f"pose_{name}",
                    f"Reference to {name}",
                    [310, 440 + i * 230],
                    source_frame="reference",
                    target_frame=name,
                    rvec=[0, angle - YAWS[0], 0],
                    translation=(-yaw(angle) @ center).tolist(),
                )
                graph.connect(f"pose_{name}", "transform", "rectify", f"reference_to_{name}")
        add(
            "input.multicamera_observations",
            "check",
            "Independent target captures",
            [310, 1100],
            path="assets/check.json",
        )
        graph.connect("check", "observations", "rectify", "check")
        add(
            "input.depth_file",
            "truth",
            "Independent 1150 mm reference",
            [1260, 805],
            path="assets/held_out_depth_mm.png",
            path_base="project",
            units="mm",
            units_per_sample=1.0,
            display_min=1100,
            display_max=1200,
        )

        for label, suffix, other, row in (
            ("Short", "12", "second", 20),
            ("Long", "13", "third", 405),
        ):
            disparity, depth, check = (f"{name}_{suffix}" for name in ("match", "depth", "check"))
            add(
                "3d.disparity",
                disparity,
                f"{label} baseline disparity",
                [1230, row],
                search_mode="depth",
                minimum_depth=0.85,
                maximum_depth=1.4,
                block_size=7,
                uniqueness_ratio=8,
            )
            add(
                "3d.reproject",
                depth,
                f"{label} baseline metric depth",
                [1590, row],
                min_depth=0.85,
                max_depth=1.4,
                sample_step=4,
            )
            add(
                "3d.evaluate_depth",
                check,
                f"{label} baseline vs known plane",
                [1970, row],
                alignment="pixel_grid",
                error_tolerance_mm=35.0,
                minimum_coverage=0.72,
                maximum_bad_fraction=0.06,
            )
            for source, target in (
                (f"reference_{suffix}", "left"),
                (other, "right"),
                ("reference_valid", "left_mask"),
                (f"{other}_valid", "right_mask"),
                (f"rectification_{suffix}", "rectification"),
            ):
                graph.connect("rectify", source, disparity, target)
            graph.connect("rectify", f"rectification_{suffix}", depth, "rectification")
            graph.connect("rectify", f"reference_{suffix}", depth, "image")
            graph.connect(disparity, "disparity", depth, "disparity")
            graph.connect(depth, "depth", check, "estimate")
            graph.connect("truth", "depth", check, "reference")
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
