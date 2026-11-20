"""Build original calibration observations and a separate stepped-depth camera view."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

CAMERA_SIZE, PROJECTOR_SIZE = (152, 116), (256, 192)
CAMERA_K = np.array([[157, 0, 74.2], [0, 151, 57.4], [0, 0, 1]], dtype=float)
PROJECTOR_K = np.array([[224, 0, 127.6], [0, 231, 93.3], [0, 0, 1]], dtype=float)
CAMERA_D = (0.024, -0.031, 0.0012, -0.0008, 0.014)
PROJECTOR_D = (-0.053, 0.071, -0.0010, 0.0011, -0.023)
ROT_VEC = np.array([0.031, 0.069, -0.027])
TRANSLATION = np.array([-0.137, 0.019, 0.009])
DEPTHS = (1.1, 1.33)


def distort(xy, d):
    x, y = np.moveaxis(xy, -1, 0)
    k1, k2, p1, p2, k3 = d
    r2 = x * x + y * y
    radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
    return np.stack(
        (
            x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
            y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y,
        ),
        axis=-1,
    )


def project(xyz, k, d):
    uv = distort(xyz[..., :2] / xyz[..., 2, None], d)
    return uv * [k[0, 0], k[1, 1]] + k[:2, 2]


def visible(xy, size):
    return bool(np.all((xy >= 4) & (xy < np.asarray(size) - 4)))


def training_views(rotation):
    x, y = np.meshgrid(np.arange(7) - 3, np.arange(6) - 2.5)
    grid = np.c_[0.037 * x.ravel(), 0.037 * y.ravel(), np.zeros(x.size)]
    rng = np.random.default_rng(76425)
    groups = {k: [] for k in ("object_views", "camera_views", "projector_views")}
    for _ in range(160):
        rvec = rng.uniform(-0.53, 0.53, 3)
        shift = rng.uniform([-0.21, -0.15, 0.72], [0.27, 0.15, 1.25])
        a = grid @ cv2.Rodrigues(rvec)[0].T + shift
        b = a @ rotation.T + TRANSLATION
        u, v = project(a, CAMERA_K, CAMERA_D), project(b, PROJECTOR_K, PROJECTOR_D)
        if not (visible(u, CAMERA_SIZE) and visible(v, PROJECTOR_SIZE)):
            continue
        count = len(grid) - len(groups["object_views"]) % 4
        choose = rng.choice(len(grid), count, replace=False)
        u += rng.normal(0, 0.008, u.shape)
        v += rng.normal(0, 0.008, v.shape)
        for rows, values in zip(groups.values(), (grid, u, v), strict=True):
            rows.append(np.round(values[choose], 9).tolist())
        if len(groups["object_views"]) == 14:
            return groups
    raise ValueError("Could not generate fourteen visible training frames")


def stepped_scene(rotation):
    w, h = CAMERA_SIZE
    y, x = np.mgrid[:h, :w]
    distorted = (np.stack((x, y), axis=-1) - CAMERA_K[:2, 2]) / [157, 151]
    rays = distorted.copy()
    for _ in range(12):
        rays += distorted - distort(rays, CAMERA_D)
    depth = np.where(x < 72, DEPTHS[0], DEPTHS[1])
    xyz = np.concatenate((rays, np.ones_like(depth)[..., None]), axis=-1) * depth[..., None]
    projected = project(xyz @ rotation.T + TRANSLATION, PROJECTOR_K, PROJECTOR_D)
    supported = np.all((projected >= 1) & (projected < np.asarray(PROJECTOR_SIZE) - 1), axis=-1)
    supported[:, 71:75] = False
    supported[32:45, 47:58] = False
    # An epipolar inconsistency is kept in the inputs to exercise the quality gate.
    check = supported.copy()
    check[48:56, 95:104] = False
    projected[48:56, 95:104, 1] += 9
    projected[~supported] = -1
    truth = np.rint(np.where(check, depth, 0) * 1000).astype(np.uint16)
    return projected.astype(np.float32), truth


def save_graph(output):
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = position
        return block

    try:
        for key, k, d, size, y in (
            ("camera", CAMERA_K, CAMERA_D, CAMERA_SIZE, 0),
            ("projector", PROJECTOR_K, PROJECTOR_D, PROJECTOR_SIZE, 280),
        ):
            add(
                "3d.camera",
                key,
                [0, y],
                fx=k[0, 0],
                fy=k[1, 1],
                cx=k[0, 2],
                cy=k[1, 2],
                width=size[0],
                height=size[1],
                frame=key,
                distortion=list(d),
            )
        add(
            "3d.camera_projector_calibrate",
            "fit",
            [350, 0],
            path="assets/calibration_views.json",
            max_rms_px=0.05,
            max_view_rms_px=0.075,
        )
        for axis, y in (("x", 320), ("y", 610)):
            add(
                "input.file",
                f"map_{axis}",
                [350, y],
                path=f"assets/projector_{axis}.tiff",
                path_base="project",
                color_format="gray",
                pixel_depth="unchanged",
                filename_ticket=False,
            ).ticket_settings = {"mode": "fixed", "value": "stepped_surface"}
        add(
            "3d.camera_projector_reconstruct",
            "depth",
            [745, 30],
            max_reprojection_error_px=0.25,
            maximum_z_m=2.0,
        )
        comment = add("layout.comment", "credit", [0, -290])
        comment.name, comment.node_size = "Synthetic calibration and held-out step", "large"
        comment.description = (
            "Original metric observations and scene by OpenCVDataflowGUI contributors (MIT). "
            "Fourteen board poses calibrate the rig. The example then reconstructs a separate "
            "two-depth scene from simulated subpixel correspondences, with shadow and bad matches."
        )
        graph.connect("camera", "camera", "fit", "camera")
        graph.connect("projector", "camera", "fit", "projector")
        graph.connect("fit", "rig", "depth", "rig")
        for axis in ("x", "y"):
            graph.connect(f"map_{axis}", "image", "depth", f"projector_{axis}")
        report = validate_graph(graph)
        if not report["valid"]:
            raise ValueError(report["issues"])
        graph.save(output / "camera_projector.json")
    finally:
        graph.close()


def build(output):
    assets = output / "assets"
    if assets.exists() or (output / "camera_projector.json").exists():
        raise FileExistsError("Choose an empty destination. Existing fixtures are preserved.")
    assets.mkdir(parents=True)
    rotation = cv2.Rodrigues(ROT_VEC)[0]
    views = json.dumps(training_views(rotation), separators=(",", ":")) + "\n"
    (assets / "calibration_views.json").write_bytes(views.encode("ascii"))
    maps, truth = stepped_scene(rotation)
    for name, image in (
        ("projector_x.tiff", maps[..., 0]),
        ("projector_y.tiff", maps[..., 1]),
        ("truth_depth_mm.png", truth),
    ):
        ok, encoded = cv2.imencode(Path(name).suffix, image)
        if not ok:
            raise ValueError(f"Could not encode {name}")
        (assets / name).write_bytes(encoded.tobytes())
    save_graph(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
