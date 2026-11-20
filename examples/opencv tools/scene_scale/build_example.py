"""Build a unitless four-view scene and independently known metric control coordinates."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import camera_record, pose_record

SCALE = 0.18
SHIFT = np.array([0.19, -0.14, 0.45])
ROTATION, _ = cv2.Rodrigues(np.array([0.14, -0.21, 0.16]))
GAUGE, METRIC = "reconstruction_gauge", "survey_frame"
SIZE = (960, 704)
CAMERA_K = np.array([[805.0, 0, 478], [0, 818, 351], [0, 0, 1]])
DISTORTION = np.array([0.024, -0.014, 0.001, 0.0005, 0.0])


def build_data():
    rng = np.random.default_rng(2291)
    landmark_ids = [f"P{i:02d}" for i in range(26)]
    nominal = rng.uniform([-1.15, -0.85, 4.0], [1.15, 0.85, 6.2], (len(landmark_ids), 3))
    surveyed = SCALE * nominal @ ROTATION.T + SHIFT
    gauge = (surveyed - SHIFT) @ ROTATION / SCALE
    views = []
    geometry = (
        ([0.015, -0.075, 0.021], [-0.95, -0.21, 0.10]),
        ([0.022, 0.11, -0.013], [0.88, -0.29, 0.04]),
        ([-0.10, -0.022, 0.027], [0.15, 0.47, -0.04]),
        ([0.04, 0.012, -0.063], [0.36, -0.49, 0.16]),
    )
    for i, (rvec, center) in enumerate(geometry):
        rv = np.array(rvec)
        rotation, _ = cv2.Rodrigues(rv)
        translation = -rotation @ np.asarray(center)
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation, translation
        # Every landmark is observed in at least three views.
        selected = np.array([(j + 2 * i) % 11 != 0 for j in range(len(gauge))])
        xy, _ = cv2.projectPoints(gauge[selected], rv, translation, CAMERA_K, DISTORTION)
        xy = xy.reshape(-1, 2) + rng.normal(0, 0.075, (int(selected.sum()), 2))
        name = f"camera_{i + 1}"
        views.append(
            {
                "model": "landmark_view_v1",
                "id": name,
                "camera": camera_record(CAMERA_K, DISTORTION, SIZE, "unitless", name),
                "world_to_camera": pose_record(pose, GAUGE, name, "unitless"),
                "image_points": xy.tolist(),
                "landmark_ids": [key for key, visible in zip(landmark_ids, selected) if visible],
            }
        )
    refs = [
        {"id": key, "xyz": (xyz * 1000).tolist()}
        for key, xyz in zip(landmark_ids, surveyed, strict=True)
    ]
    anchors = [dict(refs[i]) for i in (8, 2, 0, 7, 4, 9, 11, 3, 1, 6)]
    # Wrong surveyed coordinate for one training ID. It should be rejected by consensus.
    damaged = np.asarray(anchors[1]["xyz"]) + [32, -18, 22]
    anchors[1]["xyz"] = damaged.tolist()
    controls = {
        "model": "metric_landmark_controls_v1",
        "frame": METRIC,
        "units": "mm",
        "anchors": anchors,
        "holdouts": refs[12:20],
    }
    truth = {
        "scale_m_per_unit": SCALE,
        "rotation": ROTATION.tolist(),
        "translation_m": SHIFT.tolist(),
        "points": refs,
        "outlier_id": "P02",
    }
    return views, controls, truth


def save_graph(path, controls):
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, pos, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.position = pos
        return block

    try:
        note = add("layout.comment", "notes", [0, -240])
        note.name, note.node_size = "Recover metric scale from four views", "large"
        note.description = (
            "Original synthetic observations by the OpenCVDataflowGUI contributors (MIT). "
            "The recorded views begin in a unitless gauge. Fit ten measured controls, including "
            "one deliberate outlier. Eight independent landmarks check the world alignment."
        )
        add(
            "3d.reconstruct_sparse_views",
            "unitless",
            [0, 35],
            path="assets/views.json",
            maximum_error_px=0.8,
        )
        add(
            "3d.recover_scene_scale",
            "metric",
            [385, 35],
            controls=controls,
            anchor_tolerance_m=0.003,
            maximum_holdout_error_m=0.003,
        )
        measured = add("3d.cloud_measure", "extent", [760, 35])
        measured.name = "Measure the reconstructed cloud in meters"
        graph.connect("unitless", "cloud", "metric", "cloud")
        graph.connect("unitless", "views", "metric", "views")
        graph.connect("metric", "cloud", "extent", "cloud")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise ValueError(checked["issues"])
        graph.save(path)
    finally:
        graph.close()


def write_json(path, datum):
    path.write_bytes((json.dumps(datum, indent=2, allow_nan=False) + "\n").encode())


def build(output):
    assets = output / "assets"
    project = output / "scene_scale.json"
    if assets.exists() or project.exists():
        raise FileExistsError("Choose a new directory to preserve the example")
    assets.mkdir(parents=True)
    views, controls, truth = build_data()
    write_json(assets / "views.json", views)
    write_json(assets / "ground_truth.json", truth)
    save_graph(project, controls)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
