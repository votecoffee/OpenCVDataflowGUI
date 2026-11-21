"""Build a known camera and independent 3D-to-2D projection fixture."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

SIZE = (800, 600)
SENSOR_MM = (12.8, 9.6)
CENTER = np.array([2.4, -1.6, 0.83])
INTRINSICS = np.array([[1120.0, 0, 371.2], [0, 1040.0, 270.1], [0, 0, 1]])
GAUGE = -3.7e-60
OPTICAL = "inspection_camera"
WORLD = "fixture"
# Known points across the device's field of view and at different depths.
CAMERA_XYZ = np.array(
    [
        [-0.53, -0.39, 2.7],
        [0.25, -0.46, 2.4],
        [0.56, -0.11, 3.2],
        [-0.41, 0.17, 2.9],
        [0.36, 0.31, 2.5],
        [-0.07, 0.39, 3.4],
        [0.07, -0.08, 2.1],
        [0.69, 0.45, 3.7],
    ]
)


def known_rotation():
    sx, sy, sz = np.sin([0.086, -0.28, 0.125])
    cx, cy, cz = np.cos([0.086, -0.28, 0.125])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def fixture_geometry(skew=0.0):
    k = INTRINSICS.copy()
    k[0, 1] = skew
    rotation = known_rotation()
    translation = -rotation @ CENTER
    world = CAMERA_XYZ @ rotation + CENTER
    norm = CAMERA_XYZ[:, :2] / CAMERA_XYZ[:, 2:3]
    uv = norm @ k[:2, :2].T + k[:2, 2]
    projection = k @ np.column_stack((rotation, translation))
    return k, rotation, translation, world, uv, projection


def write_assets(assets, uv, world):
    assets.mkdir(parents=True, exist_ok=True)
    image = np.full((SIZE[1], SIZE[0], 3), [43, 42, 38], np.uint8)
    for point in uv:
        center = tuple(np.rint(point).astype(int))
        cv2.circle(image, center, 10, (92, 86, 73), -1, cv2.LINE_AA)
        cv2.circle(image, center, 10, (170, 164, 151), 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode the fixture image")
    (assets / "target.png").write_bytes(encoded.tobytes())
    truth = {
        "center_m": CENTER.tolist(),
        "rotation": known_rotation().tolist(),
        "object_points_m": world.tolist(),
        "image_points_px": uv.tolist(),
        "sensor_size_mm": list(SENSOR_MM),
        "image_size": list(SIZE),
        "intrinsics": INTRINSICS.tolist(),
        "matrix_input_scale": GAUGE,
    }
    (assets / "truth.json").write_text(
        json.dumps(truth, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def build(directory):
    path = directory / "projection_matrix.json"
    if path.exists():
        raise FileExistsError("Use an empty output directory")
    k, rotation, translation, world, uv, matrix = fixture_geometry()
    write_assets(directory / "assets", uv, world)
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, name, pos, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.name, block.position = name, pos
        return block

    try:
        note = add("layout.comment", "note", "Recover a known projection", [0, -275])
        note.node_size = "large"
        note.description = (
            "Original synthetic target by OpenCVDataflowGUI contributors (MIT). "
            "Recover camera geometry from a projection stored at a negative scale. "
            "Known intrinsics and pose generate eight reference image points. "
            "Recovered intrinsics and pose drive the second projection."
        )
        add(
            "3d.points",
            "points",
            "Known world points",
            [0, 0],
            points=world.tolist(),
            units="m",
            frame=WORLD,
        )
        add(
            "3d.camera",
            "camera",
            "Known intrinsics",
            [0, 200],
            fx=k[0, 0],
            fy=k[1, 1],
            cx=k[0, 2],
            cy=k[1, 2],
            width=SIZE[0],
            height=SIZE[1],
            frame=OPTICAL,
        )
        add(
            "3d.pose",
            "pose",
            "Known world-to-camera pose",
            [0, 402],
            rvec=cv2.Rodrigues(rotation)[0].ravel().tolist(),
            translation=translation.tolist(),
            source_frame=WORLD,
            target_frame=OPTICAL,
        )
        add("3d.project", "reference", "Reference point projection", [365, 195])
        add(
            "input.file",
            "target",
            "Synthetic observed target",
            [365, 475],
            path="assets/target.png",
            path_base="project",
        )
        add(
            "3d.inspect_projection_matrix",
            "inspect",
            "Recover and verify",
            [755, 200],
            projection=(matrix * GAUGE).tolist(),
            width=SIZE[0],
            height=SIZE[1],
            world_frame=WORLD,
            camera_frame=OPTICAL,
            units="m",
            sensor_width_mm=SENSOR_MM[0],
            sensor_height_mm=SENSOR_MM[1],
            maximum_reprojection_rms_px=0.01,
        )
        add("3d.project", "reproject", "Project with recovered geometry", [1150, 200])
        for source, port, target, terminal in (
            ("points", "points", "reference", "points"),
            ("camera", "camera", "reference", "camera"),
            ("pose", "transform", "reference", "transform"),
            ("points", "points", "inspect", "object_points"),
            ("reference", "points", "inspect", "image_points"),
            ("target", "image", "inspect", "image"),
            ("points", "points", "reproject", "points"),
            ("inspect", "camera", "reproject", "camera"),
            ("inspect", "world_to_camera", "reproject", "transform"),
        ):
            graph.connect(source, port, target, terminal)
        report = validate_graph(graph)
        if not report["valid"]:
            raise ValueError(report["issues"])
        graph.save(path)
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
