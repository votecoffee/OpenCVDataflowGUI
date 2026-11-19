"""Build checkerboard views rendered with independent Mei-model ray equations."""

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
K = np.array([[453.0, 0.0, 319.0], [0.0, 447.0, 238.0], [0.0, 0.0, 1.0]])
XI = 0.75
COLS, ROWS, PITCH = 9, 6, 10.0
TRAINING = (
    ([0.15, 0.26, -0.06], [-42, -28, 108]),
    ([-0.26, 0.28, 0.14], [-114, -62, 99]),
    ([0.35, -0.23, -0.19], [39, -74, 98]),
    ([-0.18, -0.34, 0.23], [-117, -3, 92]),
    ([0.21, 0.16, 0.10], [56, 14, 103]),
    ([-0.42, 0.12, -0.28], [-23, -80, 112]),
    ([0.26, -0.40, 0.20], [-66, 37, 81]),
    ([-0.36, -0.15, 0.05], [-32, -26, 88]),
    ([0.26, 0.13, -0.25], [-127, 38, 120]),
    ([-0.16, 0.27, 0.28], [7, -17, 142]),
)
HELD_OUT = ([0.17, -0.15, 0.08], [-119, -53, 79])


def pixel_rays(supersample=3):
    w, h = SIZE
    y, x = np.indices((h * supersample, w * supersample), dtype=np.float64)
    x = ((x + 0.5) / supersample - 0.5 - K[0, 2]) / K[0, 0]
    y = ((y + 0.5) / supersample - 0.5 - K[1, 2]) / K[1, 1]
    r2 = x * x + y * y
    # Intersect the lifted image rays with the unit sphere of the Mei model.
    lam = (XI + np.sqrt(1 + (1 - XI**2) * r2)) / (1 + r2)
    return np.stack((x * lam, y * lam, lam - XI), axis=-1)


def render(rays, rvec, translation):
    rotation = cv2.Rodrigues(np.asarray(rvec))[0]
    origin = np.asarray(translation) @ rotation
    directions = rays @ rotation
    distance = np.divide(
        origin[2],
        directions[..., 2],
        out=np.zeros(rays.shape[:2]),
        where=directions[..., 2] > 1e-9,
    )
    u, v = np.floor((directions[..., :2] * distance[..., None] - origin[:2]) / PITCH).transpose(
        2, 0, 1
    )
    target = (u >= -1) & (v >= -1) & (u < COLS) & (v < ROWS) & (distance > 0)
    checker = np.where((u + v) % 2 == 0, 25, 234)
    image = np.where(target, checker, 176).astype(np.uint8)
    return cv2.resize(image, SIZE, interpolation=cv2.INTER_AREA)


def build_graph(training):
    graph, registry = Graph(), builtin_registry()

    def add(type_id, block_id, x, y, **settings):
        block = graph.add(registry.create(type_id, block_id=block_id, parameters=settings))
        block.position = [x, y]
        return block

    grid = dict(columns=COLS, rows=ROWS, spacing=PITCH)
    add(
        "3d.omnidir_calibrate",
        "fit",
        0,
        0,
        **grid,
        image_paths=[f"assets/{file}" for file in training],
        fix_k1=True,
        fix_k2=True,
        fix_p1=True,
        fix_p2=True,
        view_policy="require_all",
        minimum_views=10,
        minimum_coverage=0.5,
        max_rms_pixels=0.45,
        max_view_rms_pixels=0.8,
    )
    add("input.file", "heldout", 0, 320, path="assets/held-out.png", path_base="project")
    add(
        "calibration.validate_lens",
        "validate",
        390,
        0,
        **grid,
        max_rms_pixels=0.5,
        max_point_error_pixels=1.25,
    )
    add("3d.rectify_camera", "rectify", 390, 320, focal_scale=0.65)
    for target, port in (("validate", "calibration"), ("rectify", "camera")):
        graph.connect("fit", "camera", target, port)
        graph.connect("heldout", "image", target, "image")
    note = add("layout.comment", "credits", 0, -300)
    note.name, note.node_size = "Synthetic omnidirectional grid", "large"
    note.description = (
        "Original rendered checkerboards under the repository MIT license. "
        "Ten views fit a Mei-model camera. The separate left-edge view checks "
        "reprojection and shows the perspective rectification. The solver reports "
        "any training views it excluded."
    )
    check = validate_graph(graph)
    assert check["valid"], check["issues"]
    return graph


def build(output):
    folder = output / "assets"
    folder.mkdir(parents=True, exist_ok=True)
    ids = [f"training-{i + 1:02}.png" for i in range(len(TRAINING))]
    images = [*ids, "held-out.png"]
    project, truth = output / "omnidir_calibration.json", output / "truth.json"
    existing = [p for p in [project, truth, *(folder / name for name in images)] if p.exists()]
    if existing:
        raise FileExistsError(f"Existing fixture files preserved: {existing}")
    rays = pixel_rays()
    for name, (rvec, tvec) in zip(images, (*TRAINING, HELD_OUT), strict=True):
        ok, encoded = cv2.imencode(".png", render(rays, rvec, tvec))
        if not ok:
            raise ValueError(f"Cannot encode fixture {name}")
        (folder / name).write_bytes(encoded.tobytes())
    graph = build_graph(ids)
    try:
        graph.save(project)
    finally:
        graph.close()
    record = dict(
        size=SIZE,
        camera_matrix=K.tolist(),
        xi=XI,
        distortion=[0.0] * 4,
        training=ids,
        heldout_rvec=HELD_OUT[0],
        heldout_translation=HELD_OUT[1],
    )
    truth.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    build(parser.parse_args().output)
