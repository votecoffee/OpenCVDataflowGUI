"""Build a calibrated camera image of a damaged, partially visible ArUco grid."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

DAMAGE_ID, OCCLUDED_ID, FOREIGN_ID = 7, 12, 42
ORDER = [DAMAGE_ID, 3, 18, 1, 22, OCCLUDED_ID]
SETTINGS = {
    "layout": "aruco_grid",
    "dictionary": "6x6_50",
    "columns": 3,
    "rows": 2,
    "marker_ids": ORDER,
    "marker_length": 24.0,
    "marker_separation": 10.0,
    "margin": 8.0,
    "dpi": 127,
}
SIZE = (640, 480)
INTRINSICS = np.array([[920.0, 0, 320.0], [0, 905.0, 240.0], [0, 0, 1.0]])
ROTATION = cv2.Rodrigues(np.array([0.34, -0.28, 0.16]))[0]
TRANSLATION = np.array([-48.0, -30.0, 228.0])


def capture(*, foreign_id=FOREIGN_ID, damaged_foreign=False):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    scale, margin, pitch, side = 5, 40, 170, 120
    texture = np.full((370, 540), 255, dtype=np.uint8)
    xyz = {}
    for i, tag in enumerate(ORDER):
        row, col = divmod(i, 3)
        x, y = col * pitch, row * pitch
        quad = np.array([[x, y], [x + side, y], [x + side, y + side], [x, y + side]])
        xyz[tag] = np.column_stack((quad / scale, np.zeros(4)))
        printed_id = foreign_id if tag == OCCLUDED_ID else tag
        marker = dictionary.generateImageMarker(printed_id, side)
        if tag == DAMAGE_ID or (tag == OCCLUDED_ID and damaged_foreign):
            marker[30:45, 45:60] ^= 255
        texture[y + margin : y + margin + side, x + margin : x + margin + side] = marker

    world_from_raster = np.array([[0.2, 0, -8], [0, 0.2, -8], [0, 0, 1]])
    homography = INTRINSICS @ np.column_stack((ROTATION[:, :2], TRANSLATION)) @ world_from_raster
    image = cv2.warpPerspective(texture, homography, SIZE, flags=cv2.INTER_LINEAR, borderValue=255)
    camera_xyz = {tag: quad @ ROTATION.T + TRANSLATION for tag, quad in xyz.items()}
    pixels = {tag: quad @ INTRINSICS.T for tag, quad in camera_xyz.items()}
    canonical_pixels = {tag: quad[:, :2] / quad[:, 2:] for tag, quad in pixels.items()}
    return image, xyz, canonical_pixels


def write_assets(directory):
    directory.mkdir(parents=True, exist_ok=True)
    image, xyz, pixels = capture()
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode the board capture")
    (directory / "partial_board.png").write_bytes(encoded.tobytes())
    truth = {
        "units": "mm",
        "intrinsics": INTRINSICS.tolist(),
        "rotation": ROTATION.tolist(),
        "translation": TRANSLATION.tolist(),
        "board_ids": ORDER,
        "damaged_id": DAMAGE_ID,
        "occluded_id": OCCLUDED_ID,
        "foreign_id": FOREIGN_ID,
        "object_corners": {tag: quad.tolist() for tag, quad in xyz.items()},
        "ideal_pixels": {tag: quad.tolist() for tag, quad in pixels.items()},
    }
    (directory / "truth.json").write_text(
        json.dumps(truth, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def build(directory):
    path = directory / "aruco_board_recovery.json"
    if path.exists():
        raise FileExistsError("Use an empty output directory")
    write_assets(directory / "assets")
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, name, position, **settings):
        node = graph.add(registry.create(kind, block_id=key, parameters=settings))
        node.name, node.position = name, position
        return node

    try:
        note = add("layout.comment", "note", "Recover a damaged board marker", [0, -250])
        note.node_size = "large"
        note.description = (
            "Original synthetic marker capture by OpenCVDataflowGUI contributors (MIT). "
            "The board image contains four intact IDs, damaged marker 7 and foreign marker 42 "
            "covering position 12. Compare initial detection and board-guided recovery."
        )
        add(
            "calibration.generate_fiducial_target",
            "target",
            "Reference board layout",
            [0, 0],
            **SETTINGS,
        )
        add(
            "input.file",
            "capture",
            "Partial board capture",
            [0, 240],
            path="assets/partial_board.png",
            path_base="project",
            color_format="gray",
        ).ticket_settings = {"mode": "fixed", "value": "partial-board"}
        add(
            "3d.camera",
            "camera",
            "Calibrated camera",
            [0, 480],
            width=SIZE[0],
            height=SIZE[1],
            units="mm",
            frame="camera",
            fx=INTRINSICS[0, 0],
            fy=INTRINSICS[1, 1],
            cx=INTRINSICS[0, 2],
            cy=INTRINSICS[1, 2],
        )
        for index, (key, name, recover) in enumerate(
            (
                ("seed", "Decode the board without recovery", False),
                ("recover", "Recover marker 7", True),
            )
        ):
            add(
                "3d.recover_aruco_board",
                key,
                name,
                [400, 130 + 275 * index],
                error_correction_rate=0,
                recover_rejected=recover,
            )
            for source, port in (("target", "target"), ("capture", "image"), ("camera", "camera")):
                graph.connect(source, port, key, port)
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise ValueError(validation["issues"])
        graph.save(path)
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
