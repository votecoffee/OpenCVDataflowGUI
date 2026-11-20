"""Build a staggered recording with independent color and depth pixel grids."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

CLOCK = 1_900_000_000_000_000_000


def write_png(path, image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Could not encode {path}")
    path.write_bytes(encoded.tobytes())


def write_recording(folder):
    color_stream, depth_stream = [], []
    for i, time_ms in enumerate((0, 33, 66, 100)):
        color = np.full((48, 80, 3), [40, 48, 55], np.uint8)
        cv2.rectangle(color, (18 + i, 11), (49 + i, 32), (75, 185, 236), -1)
        for x in (13, 56):
            cv2.line(color, (x, 0), (x, 47), (95, 120, 138), 1)
        name = f"color_{i}.png"
        write_png(folder / name, color)
        color_stream.append({"path": name, "timestamp_ns": CLOCK + time_ms * 1_000_000})
    for i, time_ms in enumerate((2, 26, 69, 101)):
        depth = np.full((32, 48), 2000, np.uint16)
        depth[8:22, 10 + i : 29 + i] = 1000 + 50 * i
        depth[2:6, 6:11] = 0
        confidence = np.full(depth.shape, 220, np.uint8)
        confidence[11:20, 14:22] = 65
        depth_name, quality_name = f"depth_{i}.png", f"confidence_{i}.png"
        write_png(folder / depth_name, depth)
        write_png(folder / quality_name, confidence)
        depth_stream.append(
            {
                "path": depth_name,
                "timestamp_ns": CLOCK + time_ms * 1_000_000,
                "confidence": quality_name,
            }
        )
    manifest = {
        "format": "opencv_dataflow_rgbd_v1",
        "clock": "synthetic_capture_clock",
        "color_frame": "color_optical",
        "depth_frame": "depth_optical",
        "depth_m_per_sample": 0.001,
        "color": color_stream,
        "depth": depth_stream,
    }
    (folder / "recording.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode())


def save_graph(path):
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, pos, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = pos
        return block

    try:
        note = add("layout.comment", "notes", [0, -220])
        note.name, note.node_size = "Replay and register RGB-D", "large"
        note.description = (
            "Original synthetic frames by the OpenCVDataflowGUI contributors (MIT). "
            "Run All pairs three frames within 5 ms. Compare the time stamps and unmatched "
            "counts, then review depth and quality projected into the color camera."
        )
        add("input.rgbd_recording", "replay", [0, 0], path="assets/recording.json")
        add(
            "3d.camera",
            "depth_camera",
            [0, 240],
            fx=32.0,
            fy=32.0,
            cx=23.5,
            cy=15.5,
            width=48,
            height=32,
            frame="depth_optical",
        )
        add(
            "3d.camera",
            "color_camera",
            [0, 435],
            fx=52.0,
            fy=50.0,
            cx=39.5,
            cy=23.5,
            width=80,
            height=48,
            frame="color_optical",
        )
        add(
            "3d.pose",
            "mount",
            [0, 630],
            source_frame="depth_optical",
            target_frame="color_optical",
            translation=[0.04, 0, 0],
        )
        register = add("3d.register_rgbd", "register", [420, 245], minimum_confidence=0.5)
        register.preview_output = "depth"
        add("3d.depth_cloud", "cloud", [800, 350], sample_step=1)
        graph.connect("replay", "image", "register", "color")
        for output in ("depth", "confidence"):
            graph.connect("replay", output, "register", output)
        for camera in ("depth_camera", "color_camera"):
            graph.connect(camera, "camera", "register", camera)
        graph.connect("mount", "transform", "register", "transform")
        graph.connect("color_camera", "camera", "cloud", "camera")
        graph.connect("register", "depth", "cloud", "depth")
        graph.connect("register", "color", "cloud", "image")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise ValueError(checked["issues"])
        graph.save(path)
    finally:
        graph.close()


def build(output):
    folder = output / "assets"
    project = output / "recorded_rgbd.json"
    if folder.exists() or project.exists():
        raise FileExistsError("Choose a new output directory to preserve the recording")
    folder.mkdir(parents=True)
    write_recording(folder)
    save_graph(project)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
