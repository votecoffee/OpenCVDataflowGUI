"""Build a three-frame RGB-D replay with labeled time-of-flight depth artifacts."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

WIDTH, HEIGHT = 96, 56
CLOCK = 1_900_000_000_000_000_000


def write_png(path, image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Could not encode {path}")
    path.write_bytes(encoded.tobytes())


def save_recording(folder):
    manifest = {
        "format": "opencv_dataflow_rgbd_v1",
        "clock": "synthetic_tof_clock",
        "color_frame": "tof_color_optical",
        "depth_frame": "tof_depth_optical",
        "depth_m_per_sample": 0.001,
        "color": [],
        "depth": [],
    }
    for i in range(3):
        edge = WIDTH // 2 + i
        truth = np.full((HEIGHT, WIDTH), 1000, np.uint16)
        truth[:, edge:] = 1600
        truth[7:15, 30 + i] = 750  # Thin but supported foreground feature.
        cleaned_rgb = np.full((HEIGHT, WIDTH, 3), (100, 170, 205), np.uint8)
        cleaned_rgb[:, edge:] = (122, 101, 82)
        cleaned_rgb[7:15, 30 + i] = (50, 210, 180)
        measured = truth.copy()
        quality = np.full(truth.shape, 230, np.uint8)
        expected = np.zeros(truth.shape, np.uint8)
        measured[[10, 18, 28, 37, 46], edge - 1] = [1210, 1240, 1300, 1340, 1260]
        expected[[10, 18, 28, 37, 46], edge - 1] = 2
        measured[12, 68 + i], measured[32, 20 + i], measured[49, 78 + i] = 2050, 640, 2200
        expected[12, 68 + i] = expected[32, 20 + i] = expected[49, 78 + i] = 3
        quality[19:23, 10 + i : 15 + i] = 32 + i
        expected[19:23, 10 + i : 15 + i] = 1
        measured[39:41, 62 + i : 65 + i] = 0
        expected[39:41, 62 + i : 65 + i] = 4

        for name, image in (
            ("color", cleaned_rgb),
            ("depth", measured),
            ("confidence", quality),
            ("truth", truth),
            ("reasons", expected),
        ):
            write_png(folder / f"{name}_{i}.png", image)
        stamp = CLOCK + i * 33_000_000
        manifest["color"].append({"path": f"color_{i}.png", "timestamp_ns": stamp})
        manifest["depth"].append(
            {
                "path": f"depth_{i}.png",
                "timestamp_ns": stamp + 1_000_000,
                "confidence": f"confidence_{i}.png",
            }
        )
    (folder / "recording.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode())


def save_graph(path):
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = position
        return block

    try:
        note = add("layout.comment", "notes", [0, -255])
        note.name, note.node_size = "Replay a time-of-flight depth step", "large"
        note.description = (
            "Original synthetic RGB-D fixture by OpenCVDataflowGUI contributors (MIT). "
            "Run All replays three frames with a moving depth step, known confidence dropouts, "
            "mixed edge samples and isolated outliers. Compare raw depth, cleaned depth and "
            "the rejection diagnostic before reconstructing the point cloud."
        )
        add("input.rgbd_recording", "replay", [0, 0], path="assets/recording.json")
        clean = add("3d.clean_tof_depth", "clean", [390, 0])
        clean.preview_output = "depth"
        add(
            "3d.camera",
            "camera",
            [390, 300],
            fx=95.0,
            fy=95.0,
            cx=(WIDTH - 1) / 2,
            cy=(HEIGHT - 1) / 2,
            width=WIDTH,
            height=HEIGHT,
            frame="tof_depth_optical",
        )
        add("3d.depth_cloud", "cloud", [810, 0], sample_step=1)
        graph.connect("replay", "depth", "clean", "depth")
        graph.connect("replay", "confidence", "clean", "confidence")
        graph.connect("clean", "depth", "cloud", "depth")
        graph.connect("camera", "camera", "cloud", "camera")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise ValueError(checked["issues"])
        graph.save(path)
    finally:
        graph.close()


def build(output):
    assets, project = output / "assets", output / "tof_depth_cleanup.json"
    if assets.exists() or project.exists():
        raise FileExistsError("Choose a new output directory to preserve the recording")
    assets.mkdir(parents=True)
    save_recording(assets)
    save_graph(project)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
