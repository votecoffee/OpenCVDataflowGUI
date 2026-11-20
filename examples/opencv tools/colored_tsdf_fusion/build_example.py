"""Render three known posed RGB-D views of a painted plane and save the fusion project."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

WIDTH, HEIGHT, FOCAL = 96, 72, 120
POSES = ((-70, 0, 0.0), (60, 60, 0.05), (10, 10, -0.04))
VOXEL_M = 0.0125


def planes(x_mm, z_mm, yaw):
    u, v = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
    direction = np.stack(((u - 47.5) / FOCAL, (v - 35.5) / FOCAL, np.ones_like(u)), axis=2)
    rotation = cv2.Rodrigues(np.array([0, yaw, 0]))[0]
    direction = direction @ rotation.T
    origin = np.array([x_mm, 0, z_mm]) * 0.001
    normal = np.array([-0.06, 0.03, 1.0])
    depth = (1.023 - origin @ normal) / (direction @ normal)
    world = origin + depth[..., None] * direction
    x, y = world[..., 0], world[..., 1]
    bgr = np.empty((HEIGHT, WIDTH, 3), np.uint8)
    bgr[x < 0] = (18, 35, 225)
    bgr[x >= 0] = (221, 50, 22)
    bgr[(x > 0.12) & (y > 0.10)] = (0, 0, 0)
    low = (np.abs(x) < 0.055) & (y < -0.13)
    bgr[low] = (255, 0, 255)
    depth[low] -= 0.15
    confidence = np.where(low, 32, 255).astype(np.uint8)
    return bgr, np.rint(depth * 10_000).astype(np.uint16), confidence


def save_view(assets, key, index, pose):
    for field, raster in zip(("color", "depth", "confidence"), planes(*pose), strict=True):
        ok, data = cv2.imencode(".png", raster)
        if not ok:
            raise RuntimeError(f"Could not encode {field} for {key}")
        (assets / f"{key}_{field}.png").write_bytes(data.tobytes())
    stamp = (index + 1) * 10_000_000
    manifest = {
        "format": "opencv_dataflow_rgbd_v1",
        "clock": "synthetic_color_panel",
        "color_frame": "camera",
        "depth_frame": "camera",
        "depth_m_per_sample": 0.0001,
        "color": [{"timestamp_ns": stamp, "path": f"{key}_color.png"}],
        "depth": [
            {
                "timestamp_ns": stamp,
                "path": f"{key}_depth.png",
                "confidence": f"{key}_confidence.png",
            }
        ],
    }
    (assets / f"{key}.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def build(output):
    project = output / "colored_tsdf_fusion.json"
    assets = output / "assets"
    if assets.exists() or project.exists():
        raise FileExistsError("Choose an empty output directory to preserve the example")
    assets.mkdir(parents=True)
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = position
        return block

    try:
        note = add("layout.comment", "notes", [0, -260])
        note.name, note.node_size = "Fuse three views of a colored panel", "large"
        note.description = (
            "Original synthetic fixture by OpenCVDataflowGUI contributors (MIT). "
            "Known camera poses observe red, blue and black on one sloped plane. "
            "Confidence excludes magenta samples with deliberately wrong depth."
        )
        add(
            "3d.camera",
            "camera",
            [780, -165],
            width=WIDTH,
            height=HEIGHT,
            fx=FOCAL,
            fy=FOCAL,
            cx=47.5,
            cy=35.5,
        )
        add("flow.batch", "batch", [760, 550], size=len(POSES), distinct_sources=True)
        for index, pose in enumerate(POSES, start=1):
            source, transform, packed = f"record_{index}", f"pose_{index}", f"view_{index}"
            save_view(assets, source, index, pose)
            y = 425 * (index - 1)
            add("input.rgbd_recording", source, [0, y], path=f"assets/{source}.json", repeat="loop")
            x_mm, z_mm, yaw = pose
            add(
                "3d.pose",
                transform,
                [0, y + 215],
                units="mm",
                rvec=[0, yaw, 0],
                translation=[x_mm, 0, z_mm],
                source_frame="camera",
                target_frame="panel",
            )
            add("3d.pack_posed_rgbd", packed, [390, y + 50])
            for field in ("color", "depth", "confidence"):
                graph.connect(source, "image" if field == "color" else field, packed, field)
            graph.connect(transform, "transform", packed, "pose")
            graph.connect(packed, "view", "batch", f"item_{index}")
        add(
            "3d.tsdf_color",
            "surface",
            [1130, 440],
            grid_policy="calibrated_grid",
            resolution=[56, 48, 32],
            volume_origin_m=[-0.35, -0.30, 0.78],
            voxel_size_m=VOXEL_M,
            truncation_m=0.04,
            minimum_confidence=0.5,
            maximum_workspace_mib=64,
            max_depth_m=2,
        )
        graph.connect("batch", "group", "surface", "views")
        graph.connect("camera", "camera", "surface", "camera")
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
