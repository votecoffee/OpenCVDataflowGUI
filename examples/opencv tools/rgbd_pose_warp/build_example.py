"""Build a replay graph with an independently predictable moving RGB-D view."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

WIDTH, HEIGHT = 56, 40
SOURCE = "registered_camera"
TARGET = "moved_camera"
TIMESTAMP = 1_900_000_000_000_000_000


def write_png(path, array):
    success, encoded = cv2.imencode(".png", array)
    if not success:
        raise ValueError(f"Could not encode {path.name}")
    path.write_bytes(encoded.tobytes())


def build(output):
    assets = output / "assets"
    project = output / "rgbd_pose_warp.json"
    if assets.exists() or project.exists():
        raise FileExistsError("Use an empty output directory to preserve this example")
    assets.mkdir(parents=True)
    depth = np.full((HEIGHT, WIDTH), 2000, np.uint16)
    depth[11:29, 15:35] = 1000
    depth[17:21, 20:23] = 0
    y, x = np.indices(depth.shape, dtype=np.uint8)
    color = np.dstack((x * 4, y * 6, np.where(depth == 1000, 235, 95).astype(np.uint8)))
    color[depth == 0] = 0
    write_png(assets / "source_depth.png", depth)
    write_png(assets / "source_color.png", color)
    recording = {
        "format": "opencv_dataflow_rgbd_v1",
        "clock": "synthetic_warp_clock",
        "color_frame": SOURCE,
        "depth_frame": SOURCE,
        "depth_m_per_sample": 0.001,
        "color": [{"path": "source_color.png", "timestamp_ns": TIMESTAMP}],
        "depth": [{"path": "source_depth.png", "timestamp_ns": TIMESTAMP + 1_000_000}],
    }
    (assets / "recording.json").write_bytes((json.dumps(recording, indent=2) + "\n").encode())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = position
        return block

    try:
        note = add("layout.comment", "note", [0, -300])
        note.name, note.node_size = "Move a calibrated RGB-D view", "large"
        note.description = (
            "Original synthetic RGB-D frame by OpenCVDataflowGUI contributors (MIT). "
            "Translate points 0.75 m right and 1.0 m forward into the target frame. "
            "Inspect the warped color, camera Z and validity where depth samples compete."
        )
        add("input.rgbd_recording", "source", [0, 0], path="assets/recording.json")
        add(
            "3d.camera",
            "camera",
            [0, 215],
            width=WIDTH,
            height=HEIGHT,
            fx=16.0,
            fy=16.0,
            cx=28.0,
            cy=20.0,
            frame=SOURCE,
        )
        add(
            "3d.pose",
            "pose",
            [0, 425],
            translation=[750, 0, 1000],
            units="mm",
            source_frame=SOURCE,
            target_frame=TARGET,
        )
        warp = add(
            "3d.warp_rgbd_pose",
            "warp",
            [400, 170],
            color_alignment="assume_registered",
            minimum_coverage=0.25,
        )
        warp.preview_output = "color"
        graph.connect("source", "depth", "warp", "depth")
        graph.connect("source", "image", "warp", "color")
        graph.connect("camera", "camera", "warp", "camera")
        graph.connect("pose", "transform", "warp", "transform")
        check = validate_graph(graph)
        if not check["valid"]:
            raise ValueError(check["issues"])
        graph.save(project)
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
