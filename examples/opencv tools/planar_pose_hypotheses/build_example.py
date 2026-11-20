"""Render two planar target views and save a native IPPE candidate review graph."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import pose_matrix

SIZE = (800, 600)
K = np.array([[861.0, 0, 397.4], [0, 854.0, 299.3], [0, 0, 1]])
BOARD = {"columns": 7, "rows": 5, "square_length": 30.0, "marker_length": 22.0}
VIEWS = {
    "near_frontal": ([0.036, -0.021, 0.012], [0, 0, 929]),
    "oblique": ([0.31, -0.38, -0.10], [27, -9, 633]),
}


def render_board(texture, pose):
    u, v = pose[:3, 0] / 4, pose[:3, 1] / 4
    mapping = np.column_stack((u, v, pose[:3, 3] + (u + v) / 2))
    camera = K.copy()
    camera[:2] *= 2
    camera[:2, 2] += 0.5
    image = cv2.warpPerspective(
        texture,
        camera @ mapping,
        tuple(2 * n for n in SIZE),
        flags=cv2.INTER_LINEAR,
        borderValue=232,
    )
    return cv2.resize(image, SIZE, interpolation=cv2.INTER_AREA)


def save_graph(output):
    graph, registry = Graph(), builtin_registry()

    def add(type_id, block_id, pos, **parameters):
        block = graph.add(registry.create(type_id, block_id=block_id, parameters=parameters))
        block.position = list(pos)
        return block

    try:
        credit = add("layout.comment", "credit", (0, -260))
        credit.name = "Planar pose ambiguity"
        credit.node_size = "large"
        credit.description = (
            "Original synthetic ChArUco images by OpenCVDataflowGUI contributors (MIT). "
            "The two views use the same board and calibrated camera. Compare the number "
            "and reprojection error of positive-depth pose candidates."
        )
        camera = add(
            "3d.camera",
            "camera",
            (290, -210),
            width=SIZE[0],
            height=SIZE[1],
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
            units="mm",
        )
        for index, name in enumerate(VIEWS):
            y = 260 * index
            source = add(
                "input.file",
                f"load_{name}",
                (0, y),
                path_base="project",
                path=f"assets/{name}.png",
                color_format="gray",
            )
            detector = add(
                "calibration.charuco",
                f"detect_{name}",
                (290, y),
                **BOARD,
                units="mm",
                minimum_corners=16,
            )
            pose = add(
                "3d.planar_pose_hypotheses",
                name,
                (595, y),
                source_frame="board",
                ambiguity_gap_px=0.2,
                max_rms_px=8.0,
            )
            pose.name = ("Near-frontal" if index == 0 else "Oblique") + " pose hypotheses"
            graph.connect(source.id, "image", detector.id, "image")
            graph.connect(source.id, "image", pose.id, "image")
            graph.connect(camera.id, "camera", detector.id, "camera")
            graph.connect(camera.id, "camera", pose.id, "camera")
            for name in ("object_points", "image_points"):
                graph.connect(detector.id, name, pose.id, name)

        check = validate_graph(graph)
        if not check["valid"]:
            raise ValueError(check["issues"])
        graph.save(output / "planar_pose_hypotheses.json")
    finally:
        graph.close()


def build(output):
    assets = output / "assets"
    if assets.exists() or (output / "planar_pose_hypotheses.json").exists():
        raise FileExistsError("Choose a new output directory to preserve the fixture")
    assets.mkdir(parents=True)
    board = cv2.aruco.CharucoBoard(
        (BOARD["columns"], BOARD["rows"]),
        BOARD["square_length"],
        BOARD["marker_length"],
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    )
    texture = board.generateImage((840, 600), marginSize=0)
    truth = {"board": BOARD, "size": list(SIZE), "camera_matrix": K.tolist(), "poses": {}}
    for name, (angles, position) in VIEWS.items():
        rotation = cv2.Rodrigues(np.asarray(angles))[0]
        # Position specifies the center of the board in the camera frame.
        pose = pose_matrix(rotation, position - rotation @ [105, 75, 0])
        ok, encoded = cv2.imencode(".png", render_board(texture, pose))
        if not ok:
            raise ValueError(f"Could not encode {name} camera view")
        (assets / f"{name}.png").write_bytes(encoded.tobytes())
        truth["poses"][name] = pose.tolist()
    (assets / "truth.json").write_bytes((json.dumps(truth, indent=2) + "\n").encode("ascii"))
    save_graph(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
