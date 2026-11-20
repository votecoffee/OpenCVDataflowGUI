"""Build a three-baseline stereo fixture with known occlusion and bad matches."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

WIDTH, HEIGHT = 176, 96
FOCAL = 150.0
BASELINES = (0.04, 0.08, 0.12)
KEYS = ("a", "b", "c")
FOREGROUND = (75, 25, 37, 37)
OUTLIER = (127, 8, 9, 8)
DISAGREEMENT = (41, 72, 6, 6)


def zone(bounds):
    x, y, w, h = bounds
    return np.s_[y : y + h, x : x + w]


def camera_view(baseline):
    y, x = np.mgrid[:HEIGHT, :WIDTH]
    # Intersect the foreground plane at 1 m, then test the physical rectangular patch.
    left_x = x + round(FOCAL * baseline)
    x0, y0, w, h = FOREGROUND
    front = (left_x >= x0) & (left_x < x0 + w) & (y >= y0) & (y < y0 + h)
    z = np.where(front, 1.0, 1.5)
    projected_left_x = x + (FOCAL * baseline / z).astype(int)
    checker = ((projected_left_x * 29 + y * 17) % 41) * 2
    color = np.stack((56 + checker, 45 + checker + front * 70, 64 + checker + front * 84), axis=2)
    return color.astype(np.uint8), z


def write_png(out, raster):
    ok, encoded = cv2.imencode(".png", raster)
    if not ok:
        raise ValueError(f"Could not encode {out}")
    out.write_bytes(encoded.tobytes())


def write_assets(out):
    left, z = camera_view(0.0)
    write_png(out / "left.png", left)
    write_png(out / "held_out_depth_mm.png", (z * 1000).astype(np.uint16))
    y, x = np.mgrid[:HEIGHT, :WIDTH]
    for name, baseline in zip(KEYS, BASELINES, strict=True):
        right, right_z = camera_view(baseline)
        write_png(out / f"right_{name}.png", right)
        shift = (FOCAL * baseline / z).astype(int)
        xr = x - shift
        visible = (xr >= 0) & (right_z[y, xr.clip(0, WIDTH - 1)] == z)
        disparity = np.where(visible, 16 * shift, 0).astype(np.uint16)
        if name == "a":
            disparity[zone(DISAGREEMENT)] = 0
        if name == "c":
            disparity[zone(OUTLIER)] = 16 * 24
            disparity[zone(DISAGREEMENT)] = 16 * 24
        write_png(out / f"disparity_{name}.png", disparity)
    known = {
        "width": WIDTH,
        "height": HEIGHT,
        "baselines_m": BASELINES,
        "foreground_rectangle": FOREGROUND,
        "near_outlier_rectangle": OUTLIER,
        "tie_rectangle": DISAGREEMENT,
        "foreground_z_m": 1,
        "background_z_m": 1.5,
    }
    (out / "fixture.json").write_bytes((json.dumps(known, indent=2) + "\n").encode())


def save_graph(path):
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, pos, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = pos
        if block.ticket_source:
            block.ticket_settings = {"mode": "fixed", "value": "three-baseline-scene"}
        return block

    def wire(source, output, target, terminal=None):
        graph.connect(source, output, target, terminal or output)

    def load(key, file, pos, grayscale=False):
        return add(
            "input.file",
            key,
            pos,
            path=f"assets/{file}",
            path_base="project",
            color_format="gray" if grayscale else "bgr",
            pixel_depth="unchanged",
        )

    def camera(key, pos):
        return add(
            "3d.camera",
            key,
            pos,
            width=WIDTH,
            height=HEIGHT,
            fx=FOCAL,
            fy=FOCAL,
            cx=(WIDTH - 1) / 2,
            cy=(HEIGHT - 1) / 2,
            frame=key,
        )

    try:
        note = add("layout.comment", "notes", [0, -285])
        note.name, note.node_size = "Compare three calibrated stereo baselines", "large"
        note.description = (
            "Original synthetic optical scene by OpenCVDataflowGUI contributors (MIT). "
            "The three disparity files encode known correspondences, true geometric occlusion, "
            "and controlled errors. Reference depth is connected only to the evaluation blocks."
        )
        load("left", "left.png", [0, 100])
        camera("left_camera", [0, 340])
        add(
            "input.depth_file",
            "truth",
            [1840, 1180],
            path="assets/held_out_depth_mm.png",
            path_base="project",
            units="mm",
            units_per_sample=1.0,
            display_min=950,
            display_max=1550,
        )
        for i, (name, baseline) in enumerate(zip(KEYS, BASELINES, strict=True)):
            y = 440 * i
            load(f"right_{name}", f"right_{name}.png", [320, y])
            camera(f"camera_{name}", [320, y + 140])
            load(f"disp_{name}", f"disparity_{name}.png", [320, y + 280], grayscale=True)
            add(
                "3d.pose",
                f"pose_{name}",
                [620, y + 235],
                source_frame="left_camera",
                target_frame=f"camera_{name}",
                translation=[-baseline, 0, 0],
            )
            add("3d.stereo_rig", f"rig_{name}", [620, y + 75])
            add("3d.stereo_rectify", f"rectify_{name}", [920, y + 70])
            add("3d.stereo_rectified_camera", f"geometry_{name}", [1200, y + 245])
            add(
                "3d.reproject",
                f"q_{name}",
                [1200, y + 65],
                input_scale=0.0625,
                min_depth=0.5,
                max_depth=2.0,
            )
            add("3d.register_rgbd", f"registered_{name}", [1520, y + 75], display_max_m=2.0)
            wire("left_camera", "camera", f"rig_{name}", "left_camera")
            wire(f"camera_{name}", "camera", f"rig_{name}", "right_camera")
            wire(f"pose_{name}", "transform", f"rig_{name}")
            wire(f"rig_{name}", "rig", f"rectify_{name}")
            wire("left", "image", f"rectify_{name}", "left")
            wire(f"right_{name}", "image", f"rectify_{name}", "right")
            wire(f"rectify_{name}", "rectification", f"q_{name}")
            wire(f"disp_{name}", "image", f"q_{name}", "disparity")
            wire(f"rectify_{name}", "rectification", f"geometry_{name}")
            wire("left_camera", "camera", f"geometry_{name}", "left_camera")
            wire(f"q_{name}", "depth", f"registered_{name}")
            wire(f"geometry_{name}", "camera", f"registered_{name}", "depth_camera")
            wire(f"geometry_{name}", "transform", f"registered_{name}")
            wire("left_camera", "camera", f"registered_{name}", "color_camera")
            wire("left", "image", f"registered_{name}", "color")
            graph.groups.append(
                {
                    "id": f"baseline_{name}",
                    "name": f"{baseline * 1000:.0f} mm baseline",
                    "members": [
                        f"{key}_{name}"
                        for key in (
                            "right",
                            "camera",
                            "disp",
                            "pose",
                            "rig",
                            "rectify",
                            "geometry",
                            "q",
                            "registered",
                        )
                    ],
                }
            )
        for i, mode in enumerate(("consensus", "nearest")):
            fused = add(
                "3d.fuse_registered_depth",
                mode,
                [1870, 300 + 440 * i],
                surface_policy=mode,
                minimum_support=2,
                surface_tolerance_m=0.005,
            )
            fused.name = f"{mode.title()} surface, minimum two baselines"
            fused.preview_output = "Fusion support"
            for name in KEYS:
                wire(f"registered_{name}", "depth", mode, f"depth_{name}")
            check = add(
                "3d.evaluate_depth",
                f"evaluate_{mode}",
                [2220, 300 + 440 * i],
                alignment="pixel_grid",
                error_tolerance_mm=1,
                minimum_coverage=0.93,
                maximum_bad_fraction=0,
            )
            check.name = f"Check {mode} with held-out depth"
            wire(mode, "depth", check.id, "estimate")
            wire("truth", "depth", check.id, "reference")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise ValueError(checked["issues"])
        graph.save(path)
    finally:
        graph.close()


def build(output):
    assets = output / "assets"
    project = output / "multi_baseline_depth.json"
    if assets.exists() or project.exists():
        raise FileExistsError("Choose a new directory to preserve the example")
    assets.mkdir(parents=True)
    write_assets(assets)
    save_graph(project)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
