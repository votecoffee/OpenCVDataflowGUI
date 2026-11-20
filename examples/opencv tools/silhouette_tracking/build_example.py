"""Rebuild the metric silhouette replay and independent pose truth."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import pose_matrix, pose_record

SIZE = (440, 320)
K = np.array([[490.0, 0, 220], [0, 487.0, 160], [0, 0, 1]])
VERTICES = np.array(
    [
        [-60, -40, -20],
        [60, -40, -20],
        [60, 40, -20],
        [-60, 40, -20],
        [-60, -40, 20],
        [60, -40, 20],
        [60, 40, 20],
        [-60, 40, 20],
    ]
)
FACES = [
    [2, 1, 0],
    [3, 2, 0],
    [4, 5, 6],
    [4, 6, 7],
    [5, 4, 0],
    [1, 5, 0],
    [6, 5, 1],
    [2, 6, 1],
    [7, 6, 2],
    [3, 7, 2],
    [4, 7, 3],
    [0, 4, 3],
]
OBJECT, CAMERA = "reference_part", "fixture_optical"
FRAME_COUNT = 12
VIDEO = "assets/part_motion.avi"


def known_pose(index):
    rvec = np.array([0.23, -0.47, 0.14]) + index * np.array([0.003, 0.006, -0.008])
    translation = [0, 0, 0.52] + index * np.array([0.0018, -0.0011, 0.0012])
    return pose_matrix(cv2.Rodrigues(rvec)[0], translation)


def frame(pose):
    width, height = SIZE
    if pose is None:
        return np.full((height, width), 26, np.uint8)
    # Independent solid-object renderer. Area sampling eliminates the inclusive
    # one-pixel halo of drawing a filled polygon directly on the target grid.
    pixels = cv2.projectPoints(
        VERTICES.astype(np.float64) / 1000,
        cv2.Rodrigues(pose[:3, :3])[0],
        pose[:3, 3],
        K,
        None,
    )[0].reshape(-1, 2)
    factor = 8
    large = np.full((height * factor, width * factor), 26, np.uint8)
    hull = cv2.convexHull(np.rint((pixels + 0.5) * factor - 0.5).astype(np.int32))
    cv2.fillConvexPoly(large, hull, 222)
    return cv2.resize(large, SIZE, interpolation=cv2.INTER_AREA)


def save_graph(path):
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = position
        return block

    try:
        note = add("layout.comment", "notes", [0, -240])
        note.name, note.node_size = "Track a known part through video", "large"
        note.description = (
            "Synthetic part motion and metric geometry by OpenCVDataflowGUI contributors (MIT). "
            "Run All replays 12 frames. Inspect the mesh, matched edges, and tracking report."
        )
        add(
            "input.video_file",
            "video",
            [0, 0],
            path=VIDEO,
            path_base="project",
            backend="opencv_mjpeg",
            color_format="gray",
            end_frame=FRAME_COUNT,
        )
        add(
            "3d.camera",
            "camera",
            [0, 220],
            width=SIZE[0],
            height=SIZE[1],
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
            frame=CAMERA,
        )
        seed = known_pose(0)
        seed[:3, 3] = (seed[:3, 3] + [0.002, -0.002, 0.003]) * 1000
        tracker = add(
            "3d.track_silhouette",
            "tracker",
            [395, 95],
            model={"vertices": VERTICES.tolist(), "faces": FACES, "units": "mm", "frame": OBJECT},
            initial_pose=pose_record(seed, OBJECT, CAMERA, "mm"),
        )
        tracker.node_size = "large"
        tracker.preview_output = "Tracked mesh"
        graph.connect("video", "image", "tracker", "image")
        graph.connect("camera", "camera", "tracker", "camera")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise ValueError(checked["issues"])
        graph.save(path)
    finally:
        graph.close()


def build(output):
    video = output / VIDEO
    project = output / "silhouette_tracking.json"
    if video.parent.exists() or project.exists():
        raise FileExistsError("Choose a new directory to preserve the example")
    video.parent.mkdir(parents=True)
    poses = [known_pose(index) for index in range(FRAME_COUNT)] + [None, known_pose(0)]
    writer = cv2.VideoWriter(
        str(video), cv2.CAP_OPENCV_MJPEG, cv2.VideoWriter_fourcc(*"MJPG"), 12, SIZE
    )
    if not writer.isOpened():
        raise RuntimeError("The installed OpenCV needs the built-in Motion JPEG backend")
    try:
        for pose in poses:
            writer.write(cv2.cvtColor(frame(pose), cv2.COLOR_GRAY2BGR))
    finally:
        writer.release()
    truth = {
        "object_frame": OBJECT,
        "camera_frame": CAMERA,
        "units": "m",
        "metric_poses": [pose.tolist() if pose is not None else None for pose in poses],
    }
    (video.parent / "ground_truth.json").write_text(
        json.dumps(truth, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    save_graph(project)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
