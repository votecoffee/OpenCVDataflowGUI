"""Build a calibrated stereo comparison with separate fit and evaluation regions."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HEIGHT, WIDTH = 128, 224
DISPARITY = 8
FOCAL_LENGTH = 160.0
BASELINE_M = 0.05
GAINS_BGR = (0.115, 0.14, 0.155)
OFFSETS_BGR = (125, 64, 98)
FIT = [23, 5, 184, 53]
EVALUATE = [25, 69, 180, 53]


def images():
    rng = np.random.default_rng(954)
    signal = rng.integers(15, 235, (HEIGHT, WIDTH), np.uint8)
    texture = cv2.GaussianBlur(signal, (5, 5), 1.35).astype(np.float64)
    left = (
        np.rint(
            texture[..., None] * [0.9, 1, 1.05]
            + rng.normal(0, 1.5, (HEIGHT, WIDTH, 3))
            + [12, 0, -8]
        )
        .clip(4, 250)
        .astype(np.uint8)
    )
    right = np.zeros_like(left)
    right[:, :-DISPARITY] = np.rint(left[:, DISPARITY:] * GAINS_BGR + OFFSETS_BGR).astype(np.uint8)
    truth = np.full((HEIGHT, WIDTH), DISPARITY * 16, np.uint16)
    return {"left": left, "right": right, "truth": truth}


def save_graph(path):
    graph, registry = Graph(), builtin_registry()

    def add(kind, name, position, **parameters):
        block = graph.add(registry.create(kind, block_id=name, parameters=parameters))
        block.position = position
        return block

    def wire(source, output, target, input=None):
        graph.connect(source, output, target, input or output)

    try:
        note = add("layout.comment", "note", [0, -260])
        note.name, note.node_size = "Stereo exposure correction with a held-out region", "large"
        note.description = (
            "Original synthetic images by OpenCVDataflowGUI contributors (MIT). "
            "An ideal horizontal rig views a plane at 1 m. Census supplies input matches. "
            "Fit is confined to the top of the image. The bottom is used to compare "
            "native StereoBM with identical settings before and after correction."
        )
        for i, name in enumerate(("left", "right", "truth")):
            source = add(
                "input.file",
                name,
                [0, 210 * i],
                path=f"assets/{name}.png",
                path_base="project",
                color_format="gray" if name == "truth" else "bgr",
                pixel_depth="unchanged",
            )
            source.name = {
                "left": "Reference left color",
                "right": "Reduced exposure right",
                "truth": "8 px ground truth",
            }[name]
            source.ticket_settings = {"mode": "fixed", "value": "stereo-exposure-control"}
        for i, side in enumerate(("left", "right")):
            key = f"{side}_camera"
            add(
                "3d.camera",
                key,
                [0, 640 + 170 * i],
                fx=FOCAL_LENGTH,
                fy=FOCAL_LENGTH,
                cx=(WIDTH - 1) / 2,
                cy=(HEIGHT - 1) / 2,
                width=WIDTH,
                height=HEIGHT,
                frame=key,
            )
        add(
            "3d.pose",
            "pose",
            [0, 985],
            source_frame="left_camera",
            target_frame="right_camera",
            translation=[-BASELINE_M, 0, 0],
        )
        add("3d.stereo_rig", "rig", [260, 720])
        add("3d.stereo_rectify", "rectify", [530, 145])
        for side in ("left", "right"):
            wire(side, "image", "rectify", side)
            wire(f"{side}_camera", "camera", "rig", f"{side}_camera")
        wire("pose", "transform", "rig")
        wire("rig", "rig", "rectify")

        for key, side, bounds, pos, title in (
            ("fit_region", "left", FIT, [810, 330], "Fit region (top)"),
            ("check_region", "left", EVALUATE, [1090, 770], "Evaluation region (bottom)"),
            (
                "right_support",
                "right",
                [0, 0, WIDTH - DISPARITY, HEIGHT],
                [810, 540],
                "Right image support",
            ),
        ):
            chosen = add("region.shape", key, pos, rectangle=bounds)
            chosen.name = title
            wire("rectify", side, key, "image")

        add(
            "3d.binary_disparity", "guide", [1090, 5], num_disparities=16, left_right_tolerance_px=0
        )
        add("3d.stereo_color_match", "correct", [1380, 270], minimum_signal_std=0.002)
        for block in ("guide", "correct"):
            for name in ("left", "right", "rectification"):
                wire("rectify", name, block)
            wire("right_support", "mask", block, "right_mask")
        wire("guide", "disparity", "correct")
        wire("fit_region", "region", "correct", "left_region")
        before_after = (
            ("raw_bm", 0, "Raw right image"),
            ("corrected_bm", 1, "Corrected right image"),
        )
        for key, row, label in before_after:
            native = add(
                "3d.disparity",
                key,
                [1740, 260 * row],
                algorithm="bm",
                num_disparities=16,
                block_size=13,
                uniqueness_ratio=15,
            )
            native.name = f"StereoBM · {label}"
            for name in ("left", "rectification"):
                wire("rectify", name, key)
            wire("rectify" if row == 0 else "correct", "right", key)
            wire("right_support", "mask", key, "right_mask")
            check = add(
                "3d.evaluate_disparity",
                f"{key}_check",
                [2070, 260 * row],
                alignment="pixel_grid",
            )
            check.name = f"Check held-out disparity · {label}"
            wire(key, "disparity", check.id, "estimate")
            wire("truth", "image", check.id, "reference")
            wire("check_region", "region", check.id)

        validated = validate_graph(graph)
        if not validated["valid"]:
            raise ValueError(validated["issues"])
        graph.save(path)
    finally:
        graph.close()


def build(output):
    assets = output / "assets"
    project = output / "stereo_color.json"
    if assets.exists() or project.exists():
        raise FileExistsError("Choose a new directory to preserve the example")
    assets.mkdir(parents=True)
    for name, image in images().items():
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise ValueError(f"Could not encode {name}")
        (assets / f"{name}.png").write_bytes(encoded.tobytes())
    save_graph(project)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
