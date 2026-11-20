"""Build a six-pose sweep for comparing seeded and bounded HashTSDF growth."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import pose_record

FRAME_COUNT = 6
SIZE = (64, 48)


def write_png(path, values):
    ok, encoded = cv2.imencode(".png", values)
    if not ok:
        raise RuntimeError(f"Could not encode {path.name}")
    path.write_bytes(encoded.tobytes())


def build(output):
    assets = output / "assets"
    project = output / "hash_tsdf_growth.json"
    if assets.exists() or project.exists():
        raise FileExistsError("Choose an empty output directory")
    assets.mkdir(parents=True)
    width, height = SIZE
    write_png(assets / "color.png", np.zeros((height, width, 3), np.uint8))
    manifest = {
        "format": "opencv_dataflow_rgbd_v1",
        "clock": "synthetic_plane_sweep",
        "color_frame": "camera",
        "depth_frame": "camera",
        "depth_m_per_sample": 0.001,
        "color": [],
        "depth": [],
    }
    poses = []
    for index in range(FRAME_COUNT):
        depth = f"depth_{index}.png"
        write_png(assets / depth, np.full((height, width), 1009 - 25 * index, np.uint16))
        stamp = (index + 1) * 20_000_000
        manifest["color"].append({"timestamp_ns": stamp, "path": "color.png"})
        manifest["depth"].append({"timestamp_ns": stamp, "path": depth})
        transform = np.eye(4)
        transform[:3, 3] = [380 * index, 0, 25 * index]
        poses.append(pose_record(transform, "camera", "plane", "mm"))
    (assets / "sweep.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = position
        return block

    try:
        note = add("layout.comment", "notes", [0, -240])
        note.name, note.node_size = "Compare sparse TSDF expansion", "large"
        note.description = (
            "Original analytic plane sweep by OpenCVDataflowGUI contributors (MIT). "
            "Run All accumulates six poses of one plane. Compare the growing volume "
            "with a volume that keeps its first frame's allocation."
        )
        add("input.rgbd_recording", "source", [0, 10], path="assets/sweep.json")
        add("flow.batch", "batch", [360, 10], size=FRAME_COUNT)
        add(
            "3d.camera",
            "camera",
            [360, 265],
            width=width,
            height=height,
            fx=90,
            fy=90,
            cx=31.5,
            cy=23.5,
        )
        graph.connect("source", "depth", "batch", "item_1")
        for index, (key, mode, name) in enumerate(
            (
                ("growing", "bounded", "Grow across all six views"),
                ("seeded", "seed_only", "Keep first view's allocation"),
            )
        ):
            block = add(
                "3d.tsdf_hash",
                key,
                [735, 255 * index],
                growth_mode=mode,
                camera_poses=poses,
                block_limit=224,
                surface_tolerance_m=0.025,
                max_depth_m=2,
                volume_origin_m=[-0.52, -0.40, 0.55],
            )
            block.name = name
            block.exposed_inputs = ["frames", "camera"]
            graph.connect("batch", "group", key, "frames")
            graph.connect("camera", "camera", key, "camera")
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise ValueError(validation["issues"])
        graph.save(project)
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
