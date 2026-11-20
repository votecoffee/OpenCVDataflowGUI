"""Render a depth sequence with a known camera path and one missing frame."""

import argparse
import json
from pathlib import Path

import cv2 as cv
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

SIZE = (160, 120)
K = np.array([[148.0, 0, 79.5], [0, 145.0, 59.5], [0, 0, 1]], np.float32)
X, Y = np.meshgrid(np.arange(SIZE[0]), np.arange(SIZE[1]))
RAYS = np.stack(((X - K[0, 2]) / K[0, 0], (Y - K[1, 2]) / K[1, 1], np.ones_like(X)), axis=-1)


def truth(i):
    matrix = np.eye(4)
    matrix[:3, :3] = cv.Rodrigues(np.array([-0.0012 * i, 0.0034 * i, -0.001 * i]))[0]
    matrix[:3, 3] = [0.007 * i, -0.0026 * i, 0.004 * i]
    return matrix


def render(pose):
    origin = pose[:3, 3]
    rays = RAYS @ pose[:3, :3].T
    normal = np.array([0.20, -0.13, 1.0])
    nearest = (1.52 - origin @ normal) / (rays @ normal)
    for center, radius in (
        ([-0.27, -0.08, 1.06], 0.23),
        ([0.24, 0.17, 1.11], 0.21),
        ([0.12, -0.28, 1.26], 0.17),
        ([-0.10, 0.34, 1.33], 0.205),
    ):
        v = origin - np.array(center)
        a = np.sum(rays * rays, axis=-1)
        b = rays @ v
        d = b * b - a * (v @ v - radius**2)
        front = np.where(d >= 0, (-b - np.sqrt(np.maximum(d, 0))) / a, np.inf)
        nearest = np.minimum(nearest, front)
    for a, b in (
        ([-0.40, 0.19, 0.84], [-0.20, 0.46, 1.46]),
        ([0.15, -0.22, 0.98], [0.40, -0.04, 1.54]),
    ):
        with np.errstate(divide="ignore", invalid="ignore"):
            first = (a - origin) / rays
            second = (b - origin) / rays
        start = np.maximum.reduce(np.minimum(first, second), axis=-1)
        end = np.minimum.reduce(np.maximum(first, second), axis=-1)
        nearest = np.minimum(nearest, np.where((start > 0) & (start <= end), start, np.inf))
    return (np.rint(nearest * 1000) / 1000).astype(np.float32)


CLOCK = 1_900_000_000_000_000_000
FRAME_COUNT = 10
DROPPED = 4
CAMERA = "depth_optical"


def write_png(path, image):
    ok, png = cv.imencode(".png", image)
    if not ok:
        raise ValueError(f"Could not encode {path.name}")
    path.write_bytes(png.tobytes())


def save_graph(path):
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, pos, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.position = pos
        return block

    try:
        note = add("layout.comment", "notes", [0, -230])
        note.name, note.node_size = "Track a depth recording through a dropped frame", "large"
        note.description = (
            "Original synthetic room and camera motion by OpenCVDataflowGUI contributors (MIT). "
            "Run All accumulates ten ordered depth frames. Frame 4 has no depth. "
            "Review trajectory data to see tracking resume on frame 5."
        )
        add(
            "input.rgbd_recording",
            "replay",
            [0, 0],
            path="assets/recording.json",
            maximum_skew_ms=0,
        )
        add("flow.batch", "segment", [335, 0], size=FRAME_COUNT)
        add(
            "3d.camera",
            "camera",
            [335, 250],
            frame=CAMERA,
            width=SIZE[0],
            height=SIZE[1],
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
        )
        tracker = add(
            "3d.track_depth_fusion",
            "track",
            [725, 90],
            resolution=[112, 96, 112],
            voxel_size_m=0.016,
            volume_origin_m=[-0.85, -0.7, 0.27],
            truncation_m=0.045,
            max_depth=2.0,
            max_depth_difference=0.1,
            max_weight=32,
            model_depth_tolerance_m=0.05,
            minimum_model_overlap=0.8,
            maximum_model_rms_m=0.024,
        )
        tracker.node_size = "large"
        tracker.preview_output = "surface_depth"
        graph.connect("replay", "depth", "segment", "item_1")
        graph.connect("segment", "group", "track", "frames")
        graph.connect("camera", "camera", "track", "camera")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise ValueError(checked["issues"])
        graph.save(path)
    finally:
        graph.close()


def build(output):
    assets, project = output / "assets", output / "depth_trajectory.json"
    if assets.exists() or project.exists():
        raise FileExistsError("Choose a new directory to preserve the recording")
    assets.mkdir(parents=True)
    color, depth, poses = [], [], []
    for i in range(FRAME_COUNT):
        pose = truth(i)
        poses.append(pose.tolist())
        samples = (
            np.zeros(SIZE[::-1], np.uint16)
            if i == DROPPED
            else np.rint(render(pose) * 1000).astype(np.uint16)
        )
        shade = np.uint8(np.clip((samples / 1000 - 0.7) * 255, 0, 255))
        preview = cv.applyColorMap(shade, cv.COLORMAP_TURBO)
        preview[samples == 0] = 0
        depth_path, color_path = f"depth_{i:02}.png", f"preview_{i:02}.png"
        write_png(assets / depth_path, samples)
        write_png(assets / color_path, preview)
        time = CLOCK + i * 33_000_000
        depth.append({"path": depth_path, "timestamp_ns": time})
        color.append({"path": color_path, "timestamp_ns": time})

    write_json(
        assets / "recording.json",
        {
            "format": "opencv_dataflow_rgbd_v1",
            "clock": "depth_trajectory_fixture",
            "color_frame": "synthetic_preview",
            "depth_frame": CAMERA,
            "depth_m_per_sample": 0.001,
            "color": color,
            "depth": depth,
        },
    )
    write_json(
        assets / "ground_truth.json",
        {
            "units": "m",
            "camera_to_first_camera": poses,
            "dropped_frames": [DROPPED],
        },
    )
    save_graph(project)


def write_json(path, record):
    path.write_bytes((json.dumps(record, indent=2) + "\n").encode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
