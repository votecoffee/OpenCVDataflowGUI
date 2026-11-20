"""Generate pinhole stereo fit points and an independent rendered validation set."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import camera_record, pose_matrix

SIZE = (800, 600)
BOARD = {"columns": 7, "rows": 5, "square_length": 30.0, "marker_length": 22.0}
K = (
    np.array([[861.0, 0, 397], [0, 856, 298], [0, 0, 1]]),
    np.array([[837.0, 0, 401], [0, 846, 305], [0, 0, 1]]),
)
SIDES = ("left", "right")
FIELDS = ("object_views", "left_views", "right_views")
FIT_COUNT = 10
HOLDOUT_COUNT = 3


def render(texture, pose, intrinsics):
    u, v = pose[:3, 0] / 4, pose[:3, 1] / 4
    pixel_mapping = np.column_stack((u, v, pose[:3, 3] + (u + v) / 2))
    oversample = intrinsics.copy()
    oversample[:2] *= 2
    oversample[:2, 2] += 0.5
    large = cv2.warpPerspective(
        texture,
        oversample @ pixel_mapping,
        tuple(2 * n for n in SIZE),
        flags=cv2.INTER_LINEAR,
        borderValue=233,
    )
    return cv2.resize(large, SIZE, interpolation=cv2.INTER_AREA)


def observe(detector, image, camera):
    result = detector.execute({"image": image, "camera": camera})
    if result is None or not result.passed:
        raise ValueError(detector.error or "Rendered stereo board was not detected")
    out = result.outputs
    return dict(zip(out["corner_ids"], out["image_points"], strict=True))


def captures(assets, registry):
    rng = np.random.default_rng(3195)
    transform = pose_matrix(cv2.Rodrigues(np.array([0.015, -0.029, 0.013]))[0], [-118, 2.5, 4.0])
    board = cv2.aruco.CharucoBoard(
        (BOARD["columns"], BOARD["rows"]),
        BOARD["square_length"],
        BOARD["marker_length"],
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    )
    texture = board.generateImage((840, 600), marginSize=0)
    objects = board.getChessboardCorners().astype(np.float64)
    fit, heldout = ({key: [] for key in FIELDS} for _ in range(2))
    cams = [camera_record(k, [], SIZE, "mm", side) for k, side in zip(K, SIDES, strict=True)]
    detector = registry.create(
        "calibration.charuco", parameters={**BOARD, "units": "mm", "minimum_corners": 12}
    )
    holdout_ids, holdout_poses = [], []
    for i in range(FIT_COUNT + HOLDOUT_COUNT):
        angles = rng.uniform(-0.42, 0.42, 3)
        rotation = cv2.Rodrigues(angles)[0]
        center = rng.uniform([40, -42, 560], [115, 42, 810])
        l_pose = pose_matrix(rotation, center - rotation @ [105, 75, 0])
        poses = (l_pose, transform @ l_pose)
        if i < FIT_COUNT:
            ids = [j for j in range(len(objects)) if (i + 3 * j) % (5 + i % 3) != 0]
            xyz = objects[ids]
            fit["object_views"].append(xyz.tolist())
            for side, pose, k in zip(SIDES, poses, K, strict=True):
                uv = cv2.projectPoints(xyz, cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3], k, None)[
                    0
                ].reshape(-1, 2)
                uv += rng.normal(0, 0.065, uv.shape)
                fit[f"{side}_views"].append(uv.tolist())
            continue
        index = i - FIT_COUNT
        detected = []
        for side, pose, k, camera in zip(SIDES, poses, K, cams, strict=True):
            image = render(texture, pose, k)
            detected.append(observe(detector, image, camera))
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                raise ValueError("Could not encode rendered stereo image")
            (assets / f"{side}-{index}.png").write_bytes(encoded.tobytes())
        shared_ids = sorted(detected[0].keys() & detected[1].keys())
        if len(shared_ids) < 12:
            raise ValueError("Too few shared ChArUco correspondences")
        holdout_ids.append(shared_ids)
        holdout_poses.append(l_pose.tolist())
        heldout["object_views"].append(objects[shared_ids].tolist())
        for side, points in zip(SIDES, detected, strict=True):
            heldout[f"{side}_views"].append([points[key] for key in shared_ids])
    expected = {
        "target": BOARD,
        "left_to_right": transform.tolist(),
        "target_to_left": holdout_poses,
        "heldout_ids": holdout_ids,
    }
    return fit, heldout, expected


def save_program(output, registry, training, heldout):
    graph = Graph()

    def add(type_id, block_id, x, y, **parameters):
        block = graph.add(registry.create(type_id, block_id=block_id, parameters=parameters))
        block.position = [x, y]
        return block

    try:
        comment = add("layout.comment", "description", 0, -195)
        comment.name = "Fit and validate stereo calibration"
        comment.node_size = "large"
        comment.description = (
            "Original synthetic controls by OpenCVDataflowGUI contributors (MIT). "
            "Ten captures fit the stereo rig. Three independent rendered ChArUco pairs "
            "are checked by predicting each camera from the other camera."
        )
        recipe = graph.recipes.put("StereoQuality", {"fit": training, "validation": heldout})
        graph.recipes.activate(recipe, project_default=True)
        fit = add("3d.stereo_calibrate", "fit", 395, 230, max_rms=0.5, max_view_rms=0.4)
        check = add(
            "3d.validate_stereo_calibration",
            "validate",
            815,
            230,
            max_rms=0.55,
            max_view_rms=0.65,
            max_anchor_rms=0.35,
        )
        graph.connect(fit.id, "rig", check.id, "rig")
        for idx, side in enumerate(SIDES):
            k = K[idx]
            camera = add(
                "3d.camera",
                f"{side}_camera",
                0,
                idx * 150,
                width=SIZE[0],
                height=SIZE[1],
                fx=k[0, 0],
                fy=k[1, 1],
                cx=k[0, 2],
                cy=k[1, 2],
                frame=side,
                units="mm",
            )
            graph.connect(camera.id, "camera", fit.id, f"{side}_camera")
        for split, solver, x, first_y in (("fit", fit, 0, 310), ("validation", check, 410, 480)):
            for i, field in enumerate(FIELDS):
                source = add(
                    "recipe.field",
                    f"{split}_{field}",
                    x,
                    first_y + i * 145,
                    path=f"settings.{split}.{field}",
                )
                source.name = f"{split.title()} {field.replace('_', ' ')}"
                graph.connect(source.id, "value", solver.id, field)
        validated = validate_graph(graph)
        if not validated["valid"]:
            raise ValueError(validated["issues"])
        graph.save(output / "stereo_calibration_quality.json")
    finally:
        graph.close()


def build(output):
    assets = output / "assets"
    if assets.exists() or (output / "stereo_calibration_quality.json").exists():
        raise FileExistsError("Choose a new output directory to preserve the fixture")
    assets.mkdir(parents=True)
    registry = builtin_registry()
    training, heldout, expected = captures(assets, registry)
    (assets / "truth.json").write_bytes((json.dumps(expected, indent=2) + "\n").encode("ascii"))
    save_program(output, registry, training, heldout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
