"""Build original two-plane stereo fixtures with both signed baselines."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

WIDTH, HEIGHT = 224, 144
FOCAL, BASELINE = 144.0, 0.125


def images():
    rng = np.random.default_rng(719)

    def texture(height, width):
        return cv2.GaussianBlur(rng.integers(18, 232, (height, width), dtype=np.uint8), (3, 3), 0)

    background = texture(HEIGHT, WIDTH + 6)
    left, right = background[:, :WIDTH].copy(), background[:, 6:].copy()
    target = texture(64, 75)
    left[40:104, 86:161] = target
    right[40:104, 74:149] = target
    right = np.rint(right.astype(float) * 0.81 + 20).astype(np.uint8)
    rasters = {"left.png": left, "right.png": right}

    y, x = np.mgrid[:HEIGHT, :WIDTH]
    for label, sign, origin, target_origin in (
        ("forward", 1, 86, 74),
        ("reverse", -1, 74, 86),
    ):
        vertical = (y >= 40) & (y < 104)
        foreground = vertical & (x >= origin) & (x < origin + 75)
        disparity = np.where(foreground, 12 * sign, 6 * sign).astype(np.float32)
        counterpart_x = x - disparity
        visible = (counterpart_x >= 0) & (counterpart_x < WIDTH)
        blocked = vertical & (counterpart_x >= target_origin) & (counterpart_x < target_origin + 75)
        disparity[~visible | (~foreground & blocked)] = 0
        rasters[f"truth_{label}.tiff"] = disparity
    return rasters


def save_graph(path, sign):
    graph, registry = Graph(), builtin_registry()
    title = "forward" if sign > 0 else "reverse"

    def add(kind, key, position, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = position
        return block

    try:
        note = add("layout.comment", "notes", [0, -230])
        note.name, note.node_size = f"Quasi-dense stereo, {title} baseline", "large"
        note.description = (
            "Original synthetic stereo images by OpenCVDataflowGUI contributors (MIT). "
            "Seed matches grow over two textured depths. Inspect disparity and "
            "validity checks before reprojecting the reconstructed planes."
        )
        for index, role in enumerate(("left", "right")):
            file = ("left.png", "right.png")[index if sign > 0 else 1 - index]
            reader = add(
                "input.file",
                role,
                [0, 240 * index],
                path=f"assets/{file}",
                path_base="project",
                color_format="gray",
            )
            reader.ticket_settings = {"mode": "fixed", "value": "quasi-dense-two-plane"}
            add(
                "3d.camera",
                f"camera_{role}",
                [0, 500 + 205 * index],
                width=WIDTH,
                height=HEIGHT,
                fx=FOCAL,
                fy=FOCAL,
                cx=(WIDTH - 1) / 2,
                cy=(HEIGHT - 1) / 2,
                frame=f"{role}_optical",
            )
        truth = add(
            "input.file",
            "truth",
            [1200, 530],
            path=f"assets/truth_{title}.tiff",
            path_base="project",
            color_format="gray",
            pixel_depth="unchanged",
        )
        truth.name = "Independent signed disparity reference"
        truth.ticket_settings = {"mode": "fixed", "value": "quasi-dense-two-plane"}
        add(
            "3d.pose",
            "baseline",
            [0, 920],
            source_frame="left_optical",
            target_frame="right_optical",
            translation=[-sign * BASELINE, 0.0, 0.0],
        )
        add("3d.stereo_rig", "rig", [310, 685])
        add("3d.stereo_rectify", "rectify", [570, 240], interpolation="nearest")
        mask = add(
            "region.shape",
            "exclusion",
            [925, 0],
            rectangle=[104, 62, 12, 19],
            invert=True,
        )
        mask.name = "Exclude the right-view patch"
        low, high = (5, 13) if sign > 0 else (-13, -5)
        matcher = add(
            "3d.quasi_dense_disparity",
            "matcher",
            [1200, 240],
            min_disparity=low,
            max_disparity=high,
            minimum_coverage=0.63,
            minimum_correlation=0.72,
            feature_separation=5,
            lk_template_size=15,
            lk_iterations=20,
        )
        matcher.preview_output = "Quasi-dense disparity"
        add(
            "3d.evaluate_disparity",
            "evaluate",
            [1590, 505],
            alignment="pixel_grid",
            reference_invalid_mode="also_use_code",
            reference_invalid_code=0,
            bad_threshold_px=0.5,
            maximum_bad_fraction=0.015,
            minimum_coverage=0.65,
        )
        add("3d.reproject", "cloud", [1590, 135], sample_step=2, max_depth=3.8)
        for role in ("left", "right"):
            graph.connect(f"camera_{role}", "camera", "rig", f"{role}_camera")
            graph.connect(role, "image", "rectify", role)
            graph.connect("rectify", role, "matcher", role)
        graph.connect("baseline", "transform", "rig", "transform")
        graph.connect("rig", "rig", "rectify", "rig")
        graph.connect("rectify", "right", "exclusion", "image")
        graph.connect("exclusion", "mask", "matcher", "right_mask")
        graph.connect("rectify", "rectification", "matcher", "rectification")
        graph.connect("rectify", "rectification", "cloud", "rectification")
        graph.connect("matcher", "disparity", "evaluate", "estimate")
        graph.connect("matcher", "disparity", "cloud", "disparity")
        graph.connect("truth", "image", "evaluate", "reference")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise ValueError(checked["issues"])
        graph.save(path)
    finally:
        graph.close()


def build(output):
    assets = output / "assets"
    projects = [output / f"quasi_dense_{tag}.json" for tag in ("forward", "reverse")]
    if assets.exists() or any(path.exists() for path in projects):
        raise FileExistsError("Choose a new directory to preserve the stereo fixtures")
    assets.mkdir(parents=True)
    for name, raster in images().items():
        ok, data = cv2.imencode(Path(name).suffix, raster)
        if not ok:
            raise ValueError(f"Could not encode {name}")
        (assets / name).write_bytes(data.tobytes())
    for path, sign in zip(projects, (1, -1), strict=True):
        save_graph(path, sign)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
