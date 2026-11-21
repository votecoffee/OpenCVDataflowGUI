"""Build a recorded RGB-D view of a textured, shearing and bending sheet."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

SIZE = (144, 104)
K = np.array([[171.0, 0.0, 71.5], [0.0, 169.0, 51.5], [0.0, 0.0, 1.0]])
Y, X = np.indices(SIZE[::-1], dtype=np.float32)
CAMERA = "sheet_depth_optical"
CLOCK = 1_900_000_000_000_000_000
TIME_STEP = 40_000_000
FRAME_COUNT, DROPPED = 5, 3


def shift(frame, y):
    return 1.35 * frame * np.sin(np.pi * (y - 12) / 120.0)


def depth_at(frame, x, y):
    u, v = (x - 79.0) / 50.0, (y - 46.0) / 40.0
    return 0.98 + (x - 72) * 0.00014 - 0.014 * frame * np.exp(-(u * u + v * v))


def truth_xyz(frame, source_indices):
    y, x = np.divmod(source_indices, SIZE[0])
    z = depth_at(frame, x, y)
    u = x + shift(frame, y)
    return np.column_stack(((u - K[0, 2]) * z / K[0, 0], (y - K[1, 2]) * z / K[1, 1], z))


def images(index):
    rng = np.random.default_rng(571232)
    texture = cv2.GaussianBlur(rng.integers(18, 235, SIZE[::-1], dtype=np.uint8), (3, 3), 0.58)
    source_x = X - shift(index, Y)
    image = cv2.remap(texture, source_x.astype(np.float32), Y, cv2.INTER_LINEAR)
    depth = np.uint16(np.rint(depth_at(index, source_x, Y) * 1000))
    in_sheet = (source_x >= 7) & (source_x < SIZE[0] - 7) & (Y >= 7) & (Y < SIZE[1] - 7)
    depth[~in_sheet] = 0
    if index == 2:
        depth[40:57, 62:80] = 0
    if index == DROPPED:
        depth.fill(0)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), depth


def write_json(path, record):
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_assets(directory):
    directory.mkdir(parents=True, exist_ok=True)
    color, depth = [], []
    for index in range(FRAME_COUNT):
        stamp = CLOCK + TIME_STEP * index
        for stream, name, pixels in zip(
            (color, depth), (f"guide_{index}.png", f"depth_{index}.png"), images(index), strict=True
        ):
            ok, encoded = cv2.imencode(".png", pixels)
            if not ok:
                raise ValueError(f"Could not encode {name}")
            (directory / name).write_bytes(encoded.tobytes())
            stream.append({"path": name, "timestamp_ns": stamp})
    write_json(
        directory / "recording.json",
        {
            "format": "opencv_dataflow_rgbd_v1",
            "clock": "deforming_sheet_fixture",
            "color_frame": CAMERA,
            "depth_frame": CAMERA,
            "depth_m_per_sample": 0.001,
            "color": color,
            "depth": depth,
        },
    )


def build(output):
    path = output / "deforming_depth_surface.json"
    if path.exists():
        raise FileExistsError("Choose an empty output directory")
    write_assets(output / "assets")
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add(
            "layout.comment", "note", "Track a bending sheet through a missing exposure", 0, -245
        )
        note.node_size = "large"
        note.description = (
            "Original synthetic sheet recording by OpenCVDataflowGUI contributors (MIT). "
            "Run All starts a reference on pair 0. Pair 2 has a depth hole. "
            "Pair 3 has no depth, and pair 4 recovers from the original reference."
        )
        add(
            "input.rgbd_recording",
            "replay",
            "Registered guide and depth",
            0,
            0,
            path="assets/recording.json",
            maximum_skew_ms=0,
        )
        add(
            "3d.camera",
            "camera",
            "Fixed pinhole camera",
            0,
            260,
            width=SIZE[0],
            height=SIZE[1],
            frame=CAMERA,
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
        )
        tracker = add(
            "3d.track_deforming_surface",
            "track",
            "Reconstruct the visible sheet",
            420,
            105,
            sample_step=4,
            tracking_window=13,
            max_pyramid_level=2,
            minimum_fraction=0.7,
            max_displacement_m=0.09,
        )
        tracker.node_size = "large"
        tracker.preview_output = "Reference-grid displacement"
        graph.connect("replay", "depth", "track", "image")
        graph.connect("replay", "image", "track", "guide")
        graph.connect("camera", "camera", "track", "camera")
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
