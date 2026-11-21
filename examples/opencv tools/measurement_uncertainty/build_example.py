"""Create a tilted, distorted planar distance fixture with correlated errors."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

SIZE = (640, 480)
CAMERA = np.array([[724.0, 0, 319.5], [0, 710.0, 239.5], [0, 0, 1]])
DISTORTION = [-0.10, 0.026, 0.001, -0.0007, 0.0]
ROTATION = [0.23, -0.32, 0.14]
TRANSLATION = [15.0, -12.0, 420.0]
ENDPOINTS = np.array([[-65.0, -22.0, 0.0], [72.0, 25.0, 0.0]])
TERMS = ["u0", "u1", "rx", "tz", "fx", "k1", "cx"]
FACTOR = np.array(
    [
        [0.34, 0, 0, 0, 0, 0, 0],
        [0.28, 0.19, 0, 0, 0, 0, 0],
        [0, 0, 0.004, 0, 0, 0, 0],
        [0, 0, 0, 0.85, 0, 0, 0],
        [0, 0, 0, 0.74, 0.69, 0, 0],
        [0, 0, 0, 0, 0, 0.0018, 0],
        [0.12, 0, 0, 0, 0, 0, 0.45],
    ]
)
PIXEL_STDDEV = 0.25


def draw_plane(path):
    points = np.array([[-92, -62, 0], [103, -62, 0], [103, 65, 0], [-92, 65, 0]])
    grids = [points, ENDPOINTS]
    for x in range(-80, 101, 15):
        grids.append([(x, -61, 0), (x, 64, 0)])
    for y in range(-55, 64, 15):
        grids.append([(-91, y, 0), (102, y, 0)])
    rvec, tvec = np.asarray(ROTATION), np.asarray(TRANSLATION)
    projected = [
        np.rint(
            cv2.projectPoints(
                np.asarray(group, dtype=float), rvec, tvec, CAMERA, np.array(DISTORTION)
            )[0].reshape(-1, 2)
        ).astype(np.int32)
        for group in grids
    ]
    image = np.full((*SIZE[::-1], 3), (67, 57, 50), dtype=np.uint8)
    cv2.fillConvexPoly(image, projected[0], (198, 209, 214), cv2.LINE_AA)
    for line in projected[2:]:
        cv2.polylines(image, [line], False, (162, 178, 185), 1, cv2.LINE_AA)
    for x, y in projected[1]:
        cv2.circle(image, (int(x), int(y)), 9, (24, 29, 34), 2, cv2.LINE_AA)
    ok, data = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise OSError("Could not write synthetic image")
    path.write_bytes(data.tobytes())


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True)
    draw_plane(assets / "tilted_plane.png")
    registry, graph = builtin_registry(), Graph()

    def add(kind, key, label, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = label, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Estimate calibrated distance uncertainty", 0, -250)
        note.node_size = "large"
        note.description = (
            "Original synthetic board by OpenCVDataflowGUI contributors (MIT). "
            "Project a known segment to the camera, then recover its length and estimate "
            "uncertainty using declared correlated pixel, pose and lens errors."
        )
        add(
            "3d.camera",
            "camera",
            "Distorted pinhole camera",
            0,
            0,
            fx=CAMERA[0, 0],
            fy=CAMERA[1, 1],
            cx=CAMERA[0, 2],
            cy=CAMERA[1, 2],
            width=SIZE[0],
            height=SIZE[1],
            units="mm",
            distortion=DISTORTION,
        )
        add(
            "3d.pose",
            "plane",
            "Tilted board pose",
            0,
            215,
            rvec=ROTATION,
            translation=TRANSLATION,
            units="mm",
            source_frame="board",
        )
        add(
            "3d.points",
            "segment",
            "Known endpoints",
            0,
            425,
            points=ENDPOINTS.tolist(),
            units="mm",
            frame="board",
        )
        add("3d.project", "project", "Generate image coordinates", 380, 290)
        add(
            "input.file",
            "reference",
            "Load annotated scene",
            370,
            0,
            path_base="project",
            path="assets/tilted_plane.png",
        )
        add(
            "calibration.measure_uncertainty",
            "measure",
            "Distance with uncertainty",
            800,
            95,
            pixel_stddev_px=PIXEL_STDDEV,
            input_covariance={"terms": TERMS, "matrix": (FACTOR @ FACTOR.T).tolist()},
        )
        for block, port, consumer, terminal in (
            ("camera", "camera", "project", "camera"),
            ("plane", "transform", "project", "transform"),
            ("segment", "points", "project", "points"),
            ("camera", "camera", "measure", "camera"),
            ("plane", "transform", "measure", "plane_to_camera"),
            ("project", "points", "measure", "image_points"),
            ("reference", "image", "measure", "image"),
        ):
            graph.connect(block, port, consumer, terminal)
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise ValueError(checked["issues"])
        graph.save(output / "measurement_uncertainty.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty output directory")
    build(args.output)
