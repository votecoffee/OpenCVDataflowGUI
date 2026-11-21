"""Build the projector prewarp example and its analytical target geometry."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

RAW_SIZE = (320, 240)
IDEAL_SIZE = (430, 316)
RAW_K = np.array([[306.0, 0, 158.2], [0, 318.0, 116.3], [0, 0, 1]])
IDEAL_K = np.array([[333.0, 0, 196.6], [0, 330.0, 145.5], [0, 0, 1]])
DISTORTION = (-0.137, 0.031, 0.0014, -0.0022, 0.0)
ROTATION = (-0.062, 0.08, 0.019)
MARKERS = ((92, 86), (177, 109), (304, 83), (102, 217), (245, 199), (332, 211))


def rotation_matrix(vector):
    # Independently evaluate the Rodrigues formula for the reference model.
    angle = np.linalg.norm(vector)
    x, y, z = np.asarray(vector) / angle
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


def project_brown(xyz):
    x, y = np.moveaxis(xyz[..., :2] / xyz[..., 2:3], -1, 0)
    k1, k2, p1, p2, k3 = DISTORTION
    rr = x * x + y * y
    radial = 1 + rr * (k1 + rr * (k2 + k3 * rr))
    u = x * radial + 2 * p1 * x * y + p2 * (rr + 2 * x * x)
    v = y * radial + p1 * (rr + 2 * y * y) + 2 * p2 * x * y
    return np.stack((RAW_K[0, 0] * u + RAW_K[0, 2], RAW_K[1, 1] * v + RAW_K[1, 2]), axis=-1)


def front_plane_points(ideal_pixels):
    pixels = np.asarray(ideal_pixels)
    x, y = np.moveaxis((pixels - IDEAL_K[:2, 2]) / IDEAL_K.diagonal()[:2], -1, 0)
    raw_rays = np.stack((x, y, np.ones_like(x)), axis=-1) @ rotation_matrix(ROTATION)
    # Intersect an oblique planar target at a measured distance in the projector frame.
    normal, plane_distance = np.array([0.18, -0.11, 1]), 1.25
    return raw_rays * (plane_distance / (raw_rays @ normal))[..., None]


def make_pattern():
    y, x = np.indices(IDEAL_SIZE[::-1])
    pattern = np.zeros(y.shape, np.float64)
    for index, (cx, cy) in enumerate(MARKERS):
        amplitude = 165 + index * 15
        radius = 3.7 + index * 0.15
        pattern += amplitude * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * radius**2))
    return np.rint(pattern).astype(np.uint8)


def save_assets(path):
    path.mkdir(parents=True, exist_ok=True)
    ok, png = cv2.imencode(".png", make_pattern())
    if not ok:
        raise ValueError("Could not encode the pattern")
    (path / "pattern.png").write_bytes(png.tobytes())
    calibration = {
        "model": "camera_model_v1",
        "lens": "pinhole",
        "frame": "projector",
        "camera_matrix": RAW_K.tolist(),
        "image_size": list(RAW_SIZE),
        "distortion": list(DISTORTION),
        "units": "m",
        "rms_pixels": 0.0,
    }
    truth = [
        {"ideal_pixel": marker, "target_xyz_m": xyz.tolist(), "projector_pixel": uv.tolist()}
        for marker, xyz, uv in zip(
            MARKERS,
            front_plane_points(MARKERS),
            project_brown(front_plane_points(MARKERS)),
            strict=True,
        )
    ]
    for name, value in (("projector_calibration", calibration), ("target_truth", truth)):
        (path / f"{name}.json").write_text(
            json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n"
        )


def build(directory):
    path = directory / "projector_inverse_rectification.json"
    if path.exists():
        raise FileExistsError("Use an empty output directory")
    save_assets(directory / "assets")
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, name, pos, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.name, block.position = name, pos
        if block.ticket_source:
            block.ticket_settings = {"mode": "fixed", "value": "six-fiducials"}
        return block

    try:
        note = add(
            "layout.comment", "note", "Prewarp a pattern for a calibrated projector", [0, -280]
        )
        note.node_size = "large"
        note.description = (
            "Original synthetic target and optics by OpenCVDataflowGUI contributors (MIT). "
            "Run once and compare the ideal pattern, prewarped projector raster and restored "
            "view. A synthetic plane supplies independent known target positions."
        )
        add(
            "input.file",
            "pattern",
            "Ideal six-fiducial pattern",
            [0, 0],
            path="assets/pattern.png",
            path_base="project",
            color_format="gray",
        )
        add(
            "input.calibration",
            "optics",
            "Raw projector intrinsics",
            [0, 230],
            path="assets/projector_calibration.json",
            path_base="project",
        )
        add(
            "3d.camera",
            "ideal",
            "Ideal pattern intrinsics",
            [0, 460],
            width=IDEAL_SIZE[0],
            height=IDEAL_SIZE[1],
            fx=IDEAL_K[0, 0],
            fy=IDEAL_K[1, 1],
            cx=IDEAL_K[0, 2],
            cy=IDEAL_K[1, 2],
            frame="ideal",
        )
        add(
            "3d.projector_inverse_rectify",
            "prewarp",
            "Prewarp for projection",
            [390, 140],
            rotation_vector=list(ROTATION),
            minimum_valid_fraction=0.8,
        )
        add(
            "calibration.lens_maps",
            "forward",
            "Build forward lens map",
            [390, 490],
            width=IDEAL_SIZE[0],
            height=IDEAL_SIZE[1],
            camera_matrix_mode="custom",
            new_camera_matrix=IDEAL_K.tolist(),
            rotation_vector=list(ROTATION),
        )
        add("cv.remap", "recover", "Restore ideal view", [800, 245])
        add(
            "inspect.image_correlation",
            "compare",
            "Check restored fiducials",
            [1200, 245],
            minimum_similarity=0.997,
            emit_residual=True,
        )
        for source, output, target, terminal in (
            ("pattern", "image", "prewarp", "pattern"),
            ("optics", "profile", "prewarp", "projector"),
            ("ideal", "camera", "prewarp", "pattern_camera"),
            ("optics", "profile", "forward", "calibration"),
            ("prewarp", "image", "recover", "image"),
            ("forward", "map_x", "recover", "map_x"),
            ("forward", "map_y", "recover", "map_y"),
            ("recover", "image", "compare", "image"),
            ("pattern", "image", "compare", "reference"),
        ):
            graph.connect(source, output, target, terminal)
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
