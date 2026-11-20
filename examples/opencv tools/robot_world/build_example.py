"""Render independent board images and save offline robot-world calibration examples."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import camera_record, pose_record

SIZE = (800, 600)
K = np.array([[861.0, 0, 397.4], [0, 854.0, 299.3], [0, 0, 1]])
BOARD = {"columns": 7, "rows": 5, "square_length": 30.0, "marker_length": 22.0}
TRAIN_COUNT, HOLDOUT_COUNT = 12, 4
SOLUTIONS = {
    "fixed_camera": {
        "mount": ([0.12, -0.17, 0.08], [30, -35, 65]),
        "station": ([-0.28, 0.11, 0.3], [420, -110, 470]),
    },
    "robot_mounted_camera": {
        "mount": ([0.18, -0.10, 0.22], [42, -25, 95]),
        "station": ([0.15, 0.05, -0.20], [170, 90, 820]),
    },
}


def matrix(rvec, translation):
    result = np.eye(4)
    result[:3, :3] = cv2.Rodrigues(np.asarray(rvec, float))[0]
    result[:3, 3] = translation
    return result


def render_board(board, pose):
    sx = sy = 4.0  # Projector-independent board texture pixels per millimeter.
    transform = pose[:3, :]
    injected = np.column_stack(
        (
            transform[:, 0] / sx,
            transform[:, 1] / sy,
            transform[:, 3] + transform[:, 0] * (0.5 / sx) + transform[:, 1] * (0.5 / sy),
        )
    )
    camera = K.copy()
    camera[:2] *= 2
    camera[:2, 2] += 0.5
    image = cv2.warpPerspective(
        board,
        camera @ injected,
        tuple(2 * size for size in SIZE),
        flags=cv2.INTER_LINEAR,
        borderValue=237,
    )
    return cv2.resize(image, SIZE, interpolation=cv2.INTER_AREA)


def samples(assets):
    registry = builtin_registry()
    board = cv2.aruco.CharucoBoard(
        (BOARD["columns"], BOARD["rows"]),
        BOARD["square_length"],
        BOARD["marker_length"],
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    ).generateImage((840, 600), marginSize=0, borderBits=1)
    camera = camera_record(K, [], SIZE, units="mm", frame="camera")
    detector = registry.create(
        "calibration.charuco", parameters={**BOARD, "units": "mm", "minimum_corners": 16}
    )
    pnp = registry.create(
        "3d.pnp", parameters={"source_frame": "target", "ransac": False, "max_rms": 0.5}
    )
    rng = np.random.default_rng(32271)
    observations = []
    for index in range(TRAIN_COUNT + HOLDOUT_COUNT):
        angles = rng.uniform(-0.40, 0.40, 3)
        center = rng.uniform([-62, -54, 560], [62, 54, 840])
        pose = matrix(angles, center)
        pose[:3, 3] -= pose[:3, :3] @ [105, 75, 0]
        image = render_board(board, pose)
        corners = detector.execute({"image": image, "camera": camera})
        if not corners or not corners.passed:
            raise ValueError(
                detector.error or "The rendered ChArUco board did not have sixteen corners"
            )
        pixels = np.asarray(corners.outputs["image_points"], np.float32).reshape(-1, 1, 2)
        refined = cv2.cornerSubPix(
            image,
            pixels,
            (4, 4),
            (-1, -1),
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 32, 0.001),
        )
        found = pnp.execute(
            {
                "camera": camera,
                "object_points": corners.outputs["object_points"],
                "image_points": refined.reshape(-1, 2).tolist(),
            }
        )
        if not found or not found.passed:
            raise ValueError(pnp.error or "The rendered-board pose exceeded the pixel RMS limit")
        if index >= TRAIN_COUNT:
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                raise ValueError("Could not encode a held-out camera image")
            (assets / f"heldout-{index - TRAIN_COUNT:02}.png").write_bytes(encoded.tobytes())
        observations.append((pose, found.outputs["pose"]))
    return camera, observations


def save_graph(output, arrangement):
    graph, registry = Graph(), builtin_registry()
    try:
        fit = graph.add(
            registry.create(
                "calibration.robot_world_hand_eye",
                block_id="fit",
                parameters={
                    "path": f"assets/{arrangement}.json",
                    "arrangement": arrangement,
                    "max_translation_rms_mm": 1.5,
                    "max_rotation_rms_degrees": 0.4,
                },
            )
        )
        fit.position = [355, 0]
        comment = graph.add(registry.create("layout.comment", block_id="credit"))
        comment.name, comment.node_size, comment.position = (
            "Offline calibration control",
            "large",
            [0, 0],
        )
        comment.description = (
            "Original synthetic ChArUco images and robot poses by OpenCVDataflowGUI "
            "contributors (MIT). Twelve image-derived poses train the fit. Four separate "
            "image-derived poses check the fitted transforms."
        )
        report = validate_graph(graph)
        if not report["valid"]:
            raise ValueError(report["issues"])
        graph.save(output / f"{arrangement}.json")
    finally:
        graph.close()


def build(output):
    assets = output / "assets"
    if assets.exists() or any((output / f"{key}.json").exists() for key in SOLUTIONS):
        raise FileExistsError("Choose an empty destination. Existing fixtures are preserved.")
    assets.mkdir(parents=True)
    camera, observations = samples(assets)
    truth = {"camera": camera, "board": BOARD, "expected": {}}
    for arrangement, known in SOLUTIONS.items():
        mount, station = (matrix(*known[key]) for key in ("mount", "station"))
        moving = arrangement == "robot_mounted_camera"
        gripper = [
            pose_record(
                station @ (np.linalg.inv(pose) if moving else pose) @ np.linalg.inv(mount),
                "gripper",
                "base",
                "mm",
            )
            for pose, _ in observations
        ]
        target = [detected for _, detected in observations]
        training = {
            "gripper_poses": gripper[:TRAIN_COUNT],
            "target_poses": target[:TRAIN_COUNT],
            "heldout_gripper_poses": gripper[TRAIN_COUNT:],
            "heldout_target_poses": target[TRAIN_COUNT:],
        }
        path = assets / f"{arrangement}.json"
        path.write_bytes((json.dumps(training, separators=(",", ":")) + "\n").encode("ascii"))
        truth["expected"][arrangement] = {"mount": mount.tolist(), "station": station.tolist()}
        save_graph(output, arrangement)
    (assets / "truth.json").write_bytes((json.dumps(truth, indent=2) + "\n").encode("ascii"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
