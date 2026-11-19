"""Render a calibration board through an independent fisheye camera model."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
SIZE = (640, 480)
K = np.float64([[311.0, 0.0, 317.0], [0.0, 307.0, 236.0], [0.0, 0.0, 1.0]])
D = np.float64([-0.048, 0.006, 0.0, 0.0])
COLS, ROWS, PITCH = 9, 6, 10.0
TRAINING = (
    ([0.15, 0.26, -0.06], [-42, -28, 108]),
    ([-0.26, 0.28, 0.14], [-88, -48, 100]),
    ([0.35, -0.23, -0.19], [4, -54, 104]),
    ([-0.18, -0.34, 0.23], [-94, 3, 111]),
    ([0.21, 0.16, 0.10], [3, 3, 117]),
    ([-0.42, 0.12, -0.28], [-24, -54, 125]),
    ([0.26, -0.40, 0.20], [-66, -16, 88]),
    ([-0.36, -0.15, 0.05], [-30, -25, 101]),
)
HELD_OUT = ([0.27, -0.17, 0.11], [-77, -55, 105])


def camera_rays(samples=3):
    width, height = SIZE
    # Sample subpixel centers and invert the camera's equidistant polynomial.
    y, x = np.indices((height * samples, width * samples), dtype=np.float64)
    x, y = (x + 0.5) / samples - 0.5, (y + 0.5) / samples - 0.5
    dx, dy = (x - K[0, 2]) / K[0, 0], (y - K[1, 2]) / K[1, 1]
    r = np.hypot(dx, dy)
    theta = r.copy()
    for _ in range(6):
        t2 = theta**2
        theta -= (theta * (1 + D[0] * t2 + D[1] * t2**2) - r) / (
            1 + 3 * D[0] * t2 + 5 * D[1] * t2**2
        )
    scale = np.divide(np.tan(theta), r, out=np.ones_like(r), where=r > 0)
    return np.stack((dx * scale, dy * scale, np.ones_like(r)), axis=-1)


def render(rays, rvec, translation):
    rotation = cv2.Rodrigues(np.asarray(rvec, np.float64))[0]
    offset = np.asarray(translation) @ rotation
    directions = rays @ rotation
    distance = np.divide(
        offset[2], directions[..., 2], out=np.zeros(rays.shape[:2]), where=directions[..., 2] > 1e-9
    )
    x, y = (directions[..., :2] * distance[..., None] - offset[:2]).transpose(2, 0, 1)
    u, v = np.floor(x / PITCH), np.floor(y / PITCH)
    board = (u >= -1) & (u < COLS) & (v >= -1) & (v < ROWS) & (distance > 0)
    pixels = np.where(board, np.where((u + v) % 2 == 0, 26, 235), 181).astype(np.uint8)
    return cv2.resize(pixels, SIZE, interpolation=cv2.INTER_AREA)


def fixture_names():
    return [f"training-{i:02}.png" for i in range(len(TRAINING))]


def graph_for(files):
    registry, graph = builtin_registry(), Graph()

    def add(kind, key, xy, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = xy
        return block

    add(
        "3d.fisheye_calibrate",
        "fit",
        [0, 0],
        image_paths=[f"assets/{file}" for file in files],
        columns=COLS,
        rows=ROWS,
        spacing=PITCH,
        fix_k3=True,
        fix_k4=True,
        max_rms_pixels=0.4,
        max_view_rms_pixels=0.6,
        minimum_coverage=0.35,
    )
    add("input.file", "heldout", [0, 330], path="assets/held-out.png", path_base="project")
    add(
        "calibration.validate_lens",
        "validate",
        [390, 0],
        columns=COLS,
        rows=ROWS,
        spacing=PITCH,
        max_rms_pixels=0.4,
        max_point_error_pixels=0.9,
    )
    add("3d.rectify_camera", "rectify", [390, 330])
    for target, port in (("validate", "calibration"), ("rectify", "camera")):
        graph.connect("fit", "camera", target, port)
        graph.connect("heldout", "image", target, "image")
    note = add("layout.comment", "attribution", [0, -280])
    note.name, note.node_size = "Synthetic fisheye targets", "large"
    note.description = (
        "Original simulated checkerboards by OpenCVDataflowGUI contributors, MIT. "
        "Eight tilted training views fit the profile. The separate held-out view checks "
        "reprojection and shows rectification. All coordinates are in millimeters and pixels."
    )
    check = validate_graph(graph)
    assert check["valid"], check["issues"]
    return graph


def build(output):
    output.mkdir(parents=True, exist_ok=True)
    directory = output / "assets"
    directory.mkdir(exist_ok=True)
    files = fixture_names()
    images = [*files, "held-out.png"]
    project = output / "fisheye_calibration.json"
    truth = output / "truth.json"
    existing = [p for p in [project, truth, *(directory / name for name in images)] if p.exists()]
    if existing:
        raise FileExistsError(f"Existing files preserved: {existing}")

    rays = camera_rays()
    for name, (rvec, translation) in zip(images, (*TRAINING, HELD_OUT), strict=True):
        image = render(rays, rvec, translation)
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise ValueError(f"Cannot encode {name}")
        (directory / name).write_bytes(encoded.tobytes())
    graph = graph_for(files)
    try:
        graph.save(project)
    finally:
        graph.close()
    record = {
        "camera_matrix": K.tolist(),
        "distortion": D.tolist(),
        "size": SIZE,
        "target_units": "mm",
        "training": len(files),
        "held_out_rvec": HELD_OUT[0],
        "held_out_translation": HELD_OUT[1],
    }
    truth.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    build(parser.parse_args().output)
