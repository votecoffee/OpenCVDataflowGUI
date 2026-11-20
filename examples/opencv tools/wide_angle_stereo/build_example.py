"""Render two calibrated wide-angle rigs and an independent held-out plane."""

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
DEPTH = 3.0
LENSES = ("fisheye", "omnidir")
RVEC = np.array([0.012, -0.025, 0.008])
T = np.array([-0.12, 0.004, 0.002])
R = cv2.Rodrigues(RVEC)[0]


def models(lens):
    data = [
        (np.array([[305.0, 0.0, 319.0], [0.0, 302.0, 238.0], [0.0, 0.0, 1.0]]), None),
        (np.array([[310.0, 0.0, 321.0], [0.0, 306.0, 240.0], [0.0, 0.0, 1.0]]), None),
    ]
    if lens == "omnidir":
        data = [(k, xi) for (k, _), xi in zip(data, (0.78, 0.85), strict=True)]
        for (k, _), scale in zip(data, (1.47, 1.5), strict=True):
            k[0, 0] *= scale
            k[1, 1] *= scale
    return data


def project(xyz, model):
    k, xi = model
    if xi is None:
        ray = xyz[..., :2] / xyz[..., 2, None]
        radius = np.linalg.norm(ray, axis=-1)
        factor = np.divide(np.arctan(radius), radius, out=np.ones_like(radius), where=radius > 0)
        xy = ray * factor[..., None]
    else:
        xy = xyz[..., :2] / (xyz[..., 2] + xi * np.linalg.norm(xyz, axis=-1))[..., None]
    return xy * np.diag(k)[:2] + k[:2, 2]


def rays(model):
    k, xi = model
    y, x = np.indices(SIZE[::-1], dtype=np.float64)
    x, y = (x - k[0, 2]) / k[0, 0], (y - k[1, 2]) / k[1, 1]
    if xi is None:
        theta = np.hypot(x, y)
        scale = np.sinc(theta / np.pi)
        return np.stack((x * scale, y * scale, np.cos(theta)), axis=-1)
    r2 = x * x + y * y
    lam = (xi + np.sqrt(1 + (1 - xi**2) * r2)) / (1 + r2)
    return np.stack((x * lam, y * lam, lam - xi), axis=-1)


def target_views(cameras):
    world = np.c_[np.mgrid[:8, :6].reshape(2, -1).T * 0.025, np.zeros(48)]
    world[:, :2] -= [0.0875, 0.0625]
    groups = {key: [] for key in ("object_views", "left_views", "right_views")}
    rng = np.random.default_rng(9923)
    for _ in range(200):
        rotation = cv2.Rodrigues(rng.uniform(-0.57, 0.57, 3))[0]
        offset = np.r_[rng.uniform([-0.14, -0.20], [0.28, 0.16]), rng.uniform(0.30, 0.48)]
        a = world @ rotation.T + offset
        p, q = project(a, cameras[0]), project(a @ R.T + T, cameras[1])
        if np.all([((v >= 8) & (v < np.asarray(SIZE) - 8)).all() for v in (p, q)]):
            for group, values in zip(groups.values(), (world, p, q), strict=True):
                group.append(np.round(values, 9).tolist())
        if len(groups["object_views"]) == 14:
            return groups
    raise ValueError("Failed to construct fourteen visible training pairs")


def plane_images(cameras):
    rng = np.random.default_rng(51811)
    texture = cv2.GaussianBlur(rng.integers(15, 241, (768, 1024), np.uint8), (5, 5), 0.9)
    for side, cam in enumerate(cameras):
        directions = rays(cam)
        center = -R.T @ T if side else np.zeros(3)
        if side:
            directions = directions @ R
        z = directions[..., 2]
        step = np.divide(DEPTH - center[2], z, out=np.zeros_like(z), where=z > 1e-4)
        xy = (directions[..., :2] * step[..., None] + center[:2]) * 120
        xy += [texture.shape[1] / 2, texture.shape[0] / 2]
        image = cv2.remap(
            texture, xy[..., 0].astype(np.float32), xy[..., 1].astype(np.float32), cv2.INTER_LINEAR
        )
        image[step == 0] = 0
        yield image


def save_graph(output, lens, cameras):
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **settings):
        item = graph.add(registry.create(kind, block_id=key, parameters=settings))
        item.position = position
        return item

    for (k, xi), side, y in zip(cameras, ("left", "right"), (0, 330), strict=True):
        add(
            "3d.camera",
            f"{side}_model",
            [0, y],
            fx=k[0, 0],
            fy=k[1, 1],
            cx=k[0, 2],
            cy=k[1, 2],
            distortion=[0.0] * 4,
            width=SIZE[0],
            height=SIZE[1],
            frame=f"{lens}_{side}",
            lens=lens,
            **({"xi": xi} if xi is not None else {}),
        )
        add(
            "input.file",
            side,
            [330, 325 + y],
            path=f"assets/{lens}-{side}.png",
            path_base="project",
            color_format="gray",
        )
    add(
        "3d.wide_stereo_calibrate",
        "fit",
        [350, 0],
        path=f"assets/{lens}-views.json",
        minimum_views=8,
        view_policy="allow_subset",
        max_rms=0.05,
        max_view_rms=0.1,
        **{f"omni_fix_{term}": True for term in ("k1", "k2", "p1", "p2") if lens == "omnidir"},
    )
    add("3d.stereo_rectify", "rectify", [700, 0], omnidir_focal_length=235.0)
    add("3d.disparity", "disparity", [1050, 0], num_disparities=48)
    add("3d.reproject", "cloud", [1400, 0], min_depth=1.0, max_depth=6.0)
    for side in ("left", "right"):
        graph.connect(f"{side}_model", "camera", "fit", f"{side}_camera")
        graph.connect(side, "image", "rectify", side)
        graph.connect("rectify", side, "disparity", side)
    graph.connect("fit", "rig", "rectify", "rig")
    graph.connect("rectify", "rectification", "cloud", "rectification")
    graph.connect("rectify", "left", "cloud", "image")
    graph.connect("disparity", "disparity", "cloud", "disparity")
    comment = add("layout.comment", "credit", [0, -325])
    comment.name, comment.node_size = "Original synthetic stereo fixture", "large"
    comment.description = (
        "Original target observations and images by OpenCVDataflowGUI contributors (MIT). "
        "Fourteen projected board poses train the rig. The rendered plane at Z = 3 m is held out."
    )
    try:
        check = validate_graph(graph)
        assert check["valid"], check["issues"]
        graph.save(output / f"{lens}.json")
    finally:
        graph.close()


def build(output):
    assets = output / "assets"
    if assets.exists() or any(output.glob("*.json")):
        raise FileExistsError("Choose an empty destination. Existing fixtures are preserved.")
    assets.mkdir(parents=True)
    truth = {"size": SIZE, "plane_depth_m": DEPTH, "rvec": RVEC.tolist(), "translation": T.tolist()}
    for lens in LENSES:
        cameras = models(lens)
        truth[lens] = [{"matrix": k.tolist(), "xi": xi} for k, xi in cameras]
        views = json.dumps(target_views(cameras), separators=(",", ":")) + "\n"
        (assets / f"{lens}-views.json").write_bytes(views.encode())
        for side, image in zip(("left", "right"), plane_images(cameras), strict=True):
            ok, encoded = cv2.imencode(".png", image)
            assert ok
            (assets / f"{lens}-{side}.png").write_bytes(encoded.tobytes())
        save_graph(output, lens, cameras)
    (output / "truth.json").write_bytes((json.dumps(truth, indent=2) + "\n").encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    build(parser.parse_args().output)
