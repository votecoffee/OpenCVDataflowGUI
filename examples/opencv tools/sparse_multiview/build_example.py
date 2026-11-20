"""Generate a four-camera ChArUco replay and independent reconstruction truth."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import pose_matrix

SIZE = (640, 480)
K = np.array([[720.0, 0, 320.4], [0, 719.0, 239.5], [0, 0, 1]])
BOARD = {"columns": 8, "rows": 6, "square_length": 30.0, "marker_length": 22.0}
UNITS = "mm"
WORLD = "board"
CENTERS = ([-65, 67, -690], [290, 112, -745], [65, 188, -685], [198, -46, -710], [245, 150, -615])


def camera_pose(center):
    z = np.asarray([120, 90, 0]) - center
    z = z / np.linalg.norm(z)
    x = np.cross([0, 1, 0], z)
    x /= np.linalg.norm(x)
    rotation = np.stack((x, np.cross(z, x), z))
    return pose_matrix(rotation, -rotation @ center)


def render_board(board, transform):
    pixels_per_mm = 4.0
    u, v = transform[:3, 0] / pixels_per_mm, transform[:3, 1] / pixels_per_mm
    camera = K.copy()
    camera[:2] *= 2
    camera[:2, 2] += 0.5
    homography = camera @ np.column_stack((u, v, transform[:3, 3] + (u + v) / 2))
    image = cv2.warpPerspective(
        board, homography, (2 * SIZE[0], 2 * SIZE[1]), flags=cv2.INTER_LINEAR, borderValue=229
    )
    return cv2.resize(image, SIZE, interpolation=cv2.INTER_AREA)


def make_project(target, transforms):
    graph, registry = Graph(), builtin_registry()

    def add(type_id, block_id, x, y, **parameters):
        block = graph.add(registry.create(type_id, block_id=block_id, parameters=parameters))
        block.position = [x, y]
        return block

    try:
        comment = add("layout.comment", "credit", 0, -450)
        comment.name = "Sparse reconstruction from four views"
        comment.node_size = "large"
        comment.description = (
            "Original synthetic ChArUco views by OpenCVDataflowGUI contributors (MIT). "
            "Camera poses C and D contain small injected errors. The joint fit keeps "
            "A and B fixed. Corner IDs match tracks despite the missing corners."
        )
        camera = add(
            "3d.camera",
            "intrinsics",
            270,
            -425,
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
            width=SIZE[0],
            height=SIZE[1],
            units=UNITS,
        )
        group = add("flow.batch", "batch_views", 1000, 190, size=4, distinct_sources=True)
        for index, view_id in enumerate("ABCD"):
            rotation, translation = transforms[index][:3, :3], transforms[index][:3, 3].copy()
            axis_angle = cv2.Rodrigues(rotation)[0].ravel()
            if index >= 2:
                sign = (-1) ** index
                axis_angle += sign * np.array([0.0035, -0.008, 0.0018])
                translation += sign * np.array([1.5, -0.75, 1.0])
            y = index * 225
            source = add(
                "input.file",
                f"load_{view_id}",
                0,
                y,
                path=f"assets/{view_id}.png",
                path_base="project",
                color_format="gray",
            )
            source.name = f"Load view {view_id}"
            detector = add(
                "calibration.charuco",
                f"detect_{view_id}",
                270,
                y,
                **BOARD,
                units=UNITS,
                minimum_corners=12,
            )
            pose = add(
                "3d.pose",
                f"pose_{view_id}",
                270,
                y + 105,
                rvec=axis_angle.tolist(),
                translation=translation.tolist(),
                source_frame=WORLD,
                units=UNITS,
            )
            pack = add("3d.landmark_view", f"pack_{view_id}", 635, y, view_id=view_id)
            graph.connect(source.id, "image", detector.id, "image")
            graph.connect(camera.id, "camera", detector.id, "camera")
            graph.connect(camera.id, "camera", pack.id, "camera")
            graph.connect(pose.id, "transform", pack.id, "world_to_camera")
            graph.connect(detector.id, "corner_ids", pack.id, "landmark_ids")
            graph.connect(detector.id, "image_points", pack.id, "image_points")
            graph.connect(pack.id, "view", group.id, f"item_{index + 1}")

        fixed = add(
            "3d.reconstruct_sparse_views",
            "fixed",
            1310,
            95,
            refinement="points",
            maximum_error_px=15.0,
        )
        fixed.name = "Fit points with supplied camera poses"
        joint = add(
            "3d.reconstruct_sparse_views",
            "joint",
            1310,
            335,
            refinement="poses_and_points",
            anchor_view_ids=["A", "B"],
            maximum_error_px=1.0,
        )
        joint.name = "Refine landmarks and camera poses"
        for block in (fixed, joint):
            graph.connect(group.id, "group", block.id, "views")

        validated = validate_graph(graph)
        if not validated["valid"]:
            raise ValueError(validated["issues"])
        graph.save(target / "sparse_multiview.json")
    finally:
        graph.close()


def build(target):
    assets = target / "assets"
    if assets.exists() or (target / "sparse_multiview.json").exists():
        raise FileExistsError("Choose a new output directory to preserve the saved examples")
    assets.mkdir(parents=True)
    board = cv2.aruco.CharucoBoard(
        (BOARD["columns"], BOARD["rows"]),
        BOARD["square_length"],
        BOARD["marker_length"],
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    )
    texture = board.generateImage((960, 720), marginSize=0)
    transforms = [camera_pose(np.asarray(center)) for center in CENTERS]
    for view_id, transform in zip("ABCDE", transforms, strict=True):
        image = render_board(texture, transform)
        if view_id == "C":
            image[130:320, :242] = 229
        if view_id == "D":
            image[130:320, 404:] = 229
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise ValueError(f"Could not encode view {view_id}")
        (assets / f"{view_id}.png").write_bytes(encoded.tobytes())
    truth = {
        "intrinsics": K.tolist(),
        "board": BOARD,
        "landmarks_mm": board.getChessboardCorners().tolist(),
        "world_to_camera": {
            key: pose.tolist() for key, pose in zip("ABCDE", transforms, strict=True)
        },
    }
    (assets / "truth.json").write_bytes((json.dumps(truth, indent=2) + "\n").encode("ascii"))
    make_project(target, transforms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
