"""Generate the metric ChArUco geometry and a frozen 20+5 acquisition protocol."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np


def xyzrpy(t):
    r = t[:3, :3]
    return [
        *t[:3, 3].tolist(),
        math.atan2(r[2, 1], r[2, 2]),
        math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2])),
        math.atan2(r[1, 0], r[0, 0]),
    ]


def rectangles(image):
    active, result = {}, []
    for y, row in enumerate(image):
        changes = np.diff(np.r_[False, row < 128, False].astype(int))
        spans = list(
            zip(np.flatnonzero(changes == 1).tolist(), np.flatnonzero(changes == -1).tolist())
        )
        for key in list(active):
            if key not in spans:
                start = active.pop(key)
                result.append(
                    [
                        key[0] * 0.0001 - 0.02,
                        start * 0.0001 - 0.02,
                        (key[1] - key[0]) * 0.0001,
                        (y - start) * 0.0001,
                    ]
                )
        for key in spans:
            active.setdefault(key, y)
    for (left, right), start in active.items():
        result.append(
            [
                left * 0.0001 - 0.02,
                start * 0.0001 - 0.02,
                (right - left) * 0.0001,
                (len(image) - start) * 0.0001,
            ]
        )
    return result


def build(robot_description, output, diverse=False):
    output = Path(output)
    if output.exists():
        raise ValueError("Choose a new output directory")
    # Reuse the existing robot-bound review chain; it is not an IK or collision solver.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "opencv_dataflow_pick_backend"))
    from xml.etree import ElementTree as ET

    from opencv_dataflow_pick_backend.kinematics import SerialChain, origin

    raw = Path(robot_description).read_text(encoding="utf-8")
    chain = SerialChain(raw, "base_link", "tool0")
    names = list(chain.required)
    reference = np.array(
        [0, -0.5, -1.4, -1.24, 0.7, 0.05] if diverse else [0, -1.25, -0.65, -1.25, 0.30, 0.05]
    )
    mount = origin(ET.fromstring('<origin xyz="0.11 -0.025 0.08" rpy="0.12 -0.09 0.08"/>'))
    optical_axes = origin(
        ET.fromstring(f'<origin xyz="0 0 0" rpy="{-math.pi / 2} 0 {-math.pi / 2}"/>')
    )
    base_from_camera = chain.forward(dict(zip(names, reference)))[0] @ mount
    camera_from_board = np.eye(4)
    camera_from_board[:3, 3] = [-0.105, -0.075, 0.65]
    board = base_from_camera @ camera_from_board
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    target = cv2.aruco.CharucoBoard((7, 5), 0.03, 0.022, dictionary)
    image = np.full((1900, 2500), 255, np.uint8)
    image[200:1700, 200:2300] = target.generateImage((2100, 1500), marginSize=0, borderBits=1)
    ok, png = cv2.imencode(".png", image)
    assert ok
    rng = np.random.default_rng(4107)
    views = []
    for i in range(25):
        q = reference + rng.uniform(-1, 1, 6) * [0.05, 0.07, 0.08, 0.14, 0.18, 0.18]
        if diverse:
            from scipy.optimize import least_squares

            desired = base_from_camera.copy()
            desired[:3, :3] = (
                base_from_camera[:3, :3]
                @ cv2.Rodrigues(rng.uniform(-1, 1, 3) * [0.25, 0.32, 0.7])[0]
            )
            desired[:3, 3] += base_from_camera[:3, :3] @ (
                rng.uniform(-1, 1, 3) * [0.025, 0.025, 0.04]
            )
            target_tool = desired @ np.linalg.inv(mount)

            def residual(values):
                actual = chain.forward(dict(zip(names, values)))[0]
                error = target_tool[:3, :3].T @ actual[:3, :3]
                return np.r_[
                    (actual[:3, 3] - target_tool[:3, 3]) * 4,
                    0.5
                    * np.array(
                        [
                            error[2, 1] - error[1, 2],
                            error[0, 2] - error[2, 0],
                            error[1, 0] - error[0, 1],
                        ]
                    ),
                ]

            solution = least_squares(
                residual,
                reference,
                bounds=(-6.2, 6.2),
                max_nfev=200,
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
            )
            if np.linalg.norm(residual(solution.x)) > 1e-6:
                raise ValueError(
                    f"Diverse view {i} is unreachable "
                    f"(residual {np.linalg.norm(residual(solution.x))}); redesign acquisition"
                )
            q = solution.x
        # Acquisition design preflight only; image observations remain independent.
        pose = chain.forward(dict(zip(names, q)))[0] @ mount
        c = np.linalg.inv(pose) @ board
        corners = np.array([[0, 0, 0], [0.21, 0, 0], [0.21, 0.15, 0], [0, 0.15, 0]])
        projected = corners @ c[:3, :3].T + c[:3, 3]
        xy = projected[:, :2] / projected[:, 2, None] * 1000 + [639.5, 479.5]
        if not (np.all(projected[:, 2] > 0.2) and np.all(xy > 20) and np.all(xy < [1260, 940])):
            raise ValueError(
                f"Frozen capture {i} leaves the camera field; redesign before acquisition"
            )
        views.append(
            dict(
                id=f"capture_{i + 1:02d}", role="train" if i < 20 else "held_out", joints=q.tolist()
            )
        )
    scene = dict(
        schema="opencv.hand_eye_scene/v1",
        units="m",
        board_pose_xyzrpy=xyzrpy(board),
        tool_from_camera_body_xyzrpy=xyzrpy(mount @ np.linalg.inv(optical_axes)),
        black_rectangles_m=rectangles(image),
        joint_names=names,
        captures=views,
        target_sha256=hashlib.sha256(png.tobytes()).hexdigest(),
        acquisition_design="camera_pose_diversity" if diverse else "limited_joint_perturbations",
        board_definition=dict(
            columns=7,
            rows=5,
            square_length=0.03,
            marker_length=0.022,
            units="m",
            dictionary="4x4_50",
            legacy_pattern=False,
        ),
        gates=dict(
            max_reprojection_rms_px=1.0,
            max_mount_error_mm=2.0,
            max_mount_error_degrees=1.0,
            max_held_out_translation_rms_mm=2.0,
            max_held_out_rotation_rms_degrees=1.0,
        ),
        evaluation_only=dict(
            tool0_from_camera_optical=mount.tolist(), base_from_board=board.tolist()
        ),
    )
    output.mkdir(parents=True)
    (output / "target.png").write_bytes(png.tobytes())
    (output / "scene.json").write_text(
        json.dumps(scene, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return dict(
        rectangles=len(scene["black_rectangles_m"]), captures=len(views), board_pose=xyzrpy(board)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robot-description", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--diverse", action="store_true")
    print(json.dumps(build(**vars(p.parse_args()))))


if __name__ == "__main__":
    main()
