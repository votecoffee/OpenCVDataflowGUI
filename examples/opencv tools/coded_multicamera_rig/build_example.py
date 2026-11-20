"""Render a custom coded target for three cameras with disjoint marker subsets."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.coded_rig_observations import MANIFEST
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import camera_record

FRAMES = ("camera_a", "camera_b", "camera_c")
SIZE = (672, 504)
DICTIONARY = "5x5_100"
IDS = (37, 5, 22, 64, 8, 81, 17, 53, 28, 10)
MARKERS = (
    (-0.167, -0.108, 0.037, -11),
    (-0.081, -0.101, 0.041, 22),
    (0.015, -0.116, 0.032, 3),
    (0.123, -0.092, 0.047, -16),
    (-0.128, -0.020, 0.053, 12),
    (-0.011, -0.005, 0.042, 46),
    (0.142, 0.014, 0.040, -27),
    (-0.157, 0.088, 0.038, 17),
    (-0.048, 0.103, 0.049, -10),
    (0.084, 0.092, 0.048, 29),
)
TILE_SIZE = 140
TILE_EDGES = np.float32(
    [
        [-0.5, -0.5],
        [TILE_SIZE - 0.5, -0.5],
        [TILE_SIZE - 0.5, TILE_SIZE - 0.5],
        [-0.5, TILE_SIZE - 0.5],
    ]
)


def layout():
    squares = []
    for x, y, side, angle in MARKERS:
        theta = np.deg2rad(angle)
        a, b = np.cos(theta), np.sin(theta)
        xy = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]) * side
        xy = xy @ np.array([[a, b], [-b, a]]) + [x, y]
        squares.append(np.column_stack((xy, np.zeros(4))))
    return np.array(squares)


def rig():
    cameras, transforms = [], []
    rotations = [(0, 0, 0), (0.024, 0.052, -0.020), (-0.038, -0.044, 0.027)]
    offsets = [(0, 0, 0), (-0.127, 0.011, 0.023), (0.132, 0.038, -0.015)]
    width, height = SIZE
    for i, (name, r, t) in enumerate(zip(FRAMES, rotations, offsets, strict=True)):
        k = [[569 + 17 * i, 0, width / 2 + 3 * i], [0, 572 + 21 * i, height / 2 - 2 * i], [0, 0, 1]]
        cameras.append(camera_record(k, [], SIZE, "m", name))
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = cv2.Rodrigues(np.array(r, float))[0], t
        transforms.append(transform)
    return cameras, transforms


def paint(size, corners, tiles, oversample=2):
    w, h = size
    canvas = np.full((h * oversample, w * oversample), 255, np.uint8)
    for points, tile in zip(corners, tiles, strict=True):
        points = (points + 0.5) * oversample - 0.5
        left, top = np.maximum(np.floor(points.min(axis=0)).astype(int) - 3, 0)
        right, bottom = np.minimum(
            np.ceil(points.max(axis=0)).astype(int) + 4, [w * oversample, h * oversample]
        )
        if right <= left or bottom <= top:
            continue
        transform = cv2.getPerspectiveTransform(
            TILE_EDGES, (points - [left, top]).astype(np.float32)
        )
        image = cv2.warpPerspective(
            tile,
            transform,
            (int(right - left), int(bottom - top)),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
        patch = canvas[top:bottom, left:right]
        np.minimum(patch, image, out=patch)
    return cv2.resize(canvas, size, interpolation=cv2.INTER_AREA)


def write_png(path, image):
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Could not encode {path.name}")
    path.write_bytes(data.tobytes())


def write_json(path, record):
    path.write_bytes((json.dumps(record, indent=2) + "\n").encode("utf-8"))


def write_captures(root, name, target, corners, tiles, cameras, transforms, pairs, seed):
    rng = np.random.default_rng(seed)
    paths = [[None for _ in pairs] for _ in FRAMES]
    capture_ids = [f"{name}-{i:02}" for i in range(1, len(pairs) + 1)]
    for i, (capture, visible) in enumerate(zip(capture_ids, pairs, strict=True)):
        r = cv2.Rodrigues(rng.uniform([-0.34, -0.38, -0.25], [0.34, 0.38, 0.25]))[0]
        t = rng.uniform([-0.059, -0.055, 0.68], [0.057, 0.054, 0.96])
        scene = corners @ r.T + t
        for c in visible:
            camera, transform = cameras[c], transforms[c]
            xyz = scene @ transform[:3, :3].T + transform[:3, 3]
            k = np.array(camera["camera_matrix"])
            pixels = (xyz @ k.T)[..., :2] / xyz[..., 2, None]
            # A and C never observe the same training capture. B sees the
            # complementary set of five IDs, so shared poses connect the rig.
            indices = (np.arange(len(IDS)) + i) % len(IDS)
            used = indices[:5] if c in (0, 2) else indices[5:]
            if name == "holdout":
                used = np.roll(indices, 2 * c)[:7]
            image = paint(SIZE, pixels[used], [tiles[j] for j in used])
            image = cv2.GaussianBlur(image, (3, 3), 0.45 + 0.06 * c)
            path = f"{name}_{c}_{i + 1:02}.png"
            write_png(root / path, image)
            paths[c][i] = path
    write_json(
        root / f"{name}.json",
        {
            "model": MANIFEST,
            "camera_frames": list(FRAMES),
            "image_sizes": [list(SIZE)] * len(FRAMES),
            "capture_ids": capture_ids,
            "target": target,
            "images": paths,
            "camera_models": cameras,
        },
    )


def save_graph(path):
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, pos, **params):
        node = graph.add(registry.create(kind, block_id=key, parameters=params))
        node.position = pos
        return node

    try:
        note = add("layout.comment", "notes", [0, -230])
        note.name, note.node_size = "Calibrate a rig from partially visible coded markers", "large"
        note.description = (
            "Original synthetic captures and custom printed target by OpenCVDataflowGUI "
            "contributors (MIT). Camera B sees complementary IDs in each training pose. "
            "Different target poses check the fitted rig, including the A-to-C baseline."
        )
        for row, name in enumerate(("train", "holdout")):
            reader = add(
                "input.coded_rig_observations", name, [0, 250 * row], path=f"assets/{name}.json"
            )
            reader.name = "Detect fit captures" if row == 0 else "Detect held-out captures"
            reader.ticket_settings = {"mode": "fixed", "value": "coded-rig-example"}
        fit = add(
            "3d.calibrate_multicamera",
            "fit",
            [350, 0],
            intrinsics="fixed",
            minimum_shared_views=3,
            max_rms_px=0.9,
            max_view_rms_px=1.2,
            max_iterations=90,
        )
        hold = add(
            "3d.validate_multicamera",
            "check",
            [710, 250],
            max_rms_px=1.25,
            max_view_rms_px=2,
            max_anchor_rms_px=1.2,
        )
        graph.connect("train", "observations", fit.id, "observations")
        graph.connect("train", "camera_models", fit.id, "camera_models")
        graph.connect("holdout", "observations", hold.id, "observations")
        graph.connect(fit.id, "rig", hold.id, "rig")
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise ValueError(validation["issues"])
        graph.save(path)
    finally:
        graph.close()


def build(output):
    assets, project = output / "assets", output / "coded_multicamera_rig.json"
    if assets.exists() or project.exists():
        raise FileExistsError("Choose a new directory to preserve the example")
    assets.mkdir(parents=True)
    corners = layout()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    tiles = [cv2.aruco.generateImageMarker(dictionary, code, TILE_SIZE) for code in IDS]
    target = {
        "model": "fiducial_target_v1",
        "layout": "custom",
        "dictionary": DICTIONARY,
        "units": "m",
        "border_bits": 1,
        "marker_ids": list(IDS),
        "marker_corners_xyz": corners.tolist(),
        "print_size": [0.44, 0.32],
    }
    write_png(
        assets / "target.png",
        paint((1760, 1280), (corners[..., :2] + [0.22, 0.16]) * 4000 - 0.5, tiles, oversample=1),
    )
    cameras, transforms = rig()
    write_json(
        assets / "ground_truth.json",
        {
            "units": "m",
            "camera_frames": list(FRAMES),
            "anchor_to_camera": [transform.tolist() for transform in transforms],
        },
    )
    write_captures(
        assets,
        "train",
        target,
        corners,
        tiles,
        cameras,
        transforms,
        [(0, 1)] * 7 + [(1, 2)] * 7,
        84173,
    )
    write_captures(
        assets,
        "holdout",
        target,
        corners,
        tiles,
        cameras,
        transforms,
        [(0, 2), (0, 1), (1, 2)] * 2,
        74289,
    )
    save_graph(project)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
