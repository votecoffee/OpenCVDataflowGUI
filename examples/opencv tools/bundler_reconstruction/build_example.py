"""Write a Bundler scene and independent metric projection of a held-out camera."""

import argparse
import json
from pathlib import Path

import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

BASELINE = 0.4
RAW_UNITS_PER_METER = 2.5
IMAGE_SIZES = [(640, 480), (656, 488), (632, 476), None, (641, 481)]
FOCALS = (562.0, 588.0, 545.0, 0.0, 600.0)
RADIAL = [(-0.09, 0.018), (-0.075, 0.012), (-0.105, 0.024), (0.0, 0.0), (-0.092, 0.015)]
CENTERS_M = np.array(
    [
        [-0.20, 0.04, 0.00],
        [0.20, 0.04, 0.00],
        [0.05, -0.18, 0.045],
        [0.0, 0.0, 0.0],
        [0.11, 0.08, -0.06],
    ]
)
ANGLES = (
    (0.012, -0.045, 0.020),
    (-0.025, 0.060, -0.015),
    (0.045, -0.025, 0.018),
    (0.0, 0.0, 0.0),
    (-0.035, 0.054, 0.026),
)
HELD_OUT = 4


def rotation(angles):
    ax, ay, az = angles
    a, b, c = np.sin([ax, ay, az])
    d, e, f = np.cos([ax, ay, az])
    rx = np.array([[1, 0, 0], [0, d, -a], [0, a, d]])
    ry = np.array([[e, 0, b], [0, 1, 0], [-b, 0, e]])
    rz = np.array([[f, -c, 0], [c, f, 0], [0, 0, 1]])
    return rz @ ry @ rx


def project(index, xyz):
    camera = (xyz - CENTERS_M[index]) @ rotation(ANGLES[index]).T
    uv = camera[:, :2] / camera[:, 2:3]
    squared_radius = np.sum(uv * uv, axis=1)
    k1, k2 = RADIAL[index]
    radial = 1 + squared_radius * (k1 + k2 * squared_radius)
    return uv * (FOCALS[index] * radial[:, None]) + np.asarray(IMAGE_SIZES[index]) / 2, camera[:, 2]


def _text(values):
    return " ".join(f"{number:.17g}" for number in values)


def scene():
    x, y = np.meshgrid(np.linspace(-0.42, 0.44, 7), np.linspace(-0.32, 0.34, 6))
    z = 2.15 + 0.31 * np.sin(3 * x) + 0.18 * np.cos(4 * y)
    xyz = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    colors = np.array(
        [
            [45 + 26 * (index % 7), 44 + 30 * (index // 7), 203 - 3 * index]
            for index in range(len(xyz))
        ],
        dtype=np.uint8,
    )
    return xyz, colors


def write_assets(assets):
    assets.mkdir(parents=True, exist_ok=True)
    xyz, colors = scene()
    rows = ["# Bundle file v0.3", f"{len(IMAGE_SIZES)} {len(xyz)}"]
    source_axes = np.diag([1.0, -1.0, -1.0])
    for i, focal in enumerate(FOCALS):
        rows.append(_text([focal, *RADIAL[i]]))
        if IMAGE_SIZES[i] is None:
            rows.extend(["0 0 0"] * 4)
            continue
        transform = source_axes @ rotation(ANGLES[i])
        rows.extend(_text(row) for row in transform)
        rows.append(_text(-transform @ CENTERS_M[i] * RAW_UNITS_PER_METER))

    for i, (point, color) in enumerate(zip(xyz, colors, strict=True)):
        rows.extend((_text(point * RAW_UNITS_PER_METER), _text(color)))
        views = [0, 1, 2] if i % 5 else [0, 2]
        fields = [str(len(views))]
        for camera in views:
            pixels, _ = project(camera, point[None])
            u, v = (pixels[0] - np.asarray(IMAGE_SIZES[camera]) / 2) * [1, -1]
            fields.extend((str(camera), str(10 + i * 3), _text([u, v])))
        rows.append(" ".join(fields))
    (assets / "bundle.out").write_text("\n".join(rows) + "\n", encoding="ascii", newline="\n")
    pixels, depths = project(HELD_OUT, xyz)
    truth = {
        "units": "m",
        "baseline_m": BASELINE,
        "camera_centers_m": CENTERS_M.tolist(),
        "point_xyz_m": xyz.tolist(),
        "color_rgb": colors.tolist(),
        "image_sizes": IMAGE_SIZES,
        "held_out_camera": HELD_OUT,
        "held_out_pixels": pixels.tolist(),
        "held_out_z_m": depths.tolist(),
    }
    (assets / "truth.json").write_text(
        json.dumps(truth, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def build(output):
    path = output / "bundler_reconstruction.json"
    if path.exists():
        raise FileExistsError("Use an empty output directory")
    write_assets(output / "assets")
    registry, graph = builtin_registry(), Graph()

    def add(kind, key, label, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = label, [x, y]
        return block

    try:
        note = add(
            "layout.comment", "note", "Import a sparse scene and project its held-out view", 0, -250
        )
        note.node_size = "large"
        note.description = (
            "Original synthetic reconstruction by OpenCVDataflowGUI contributors (MIT). "
            "The import uses a measured 0.4 m baseline. Camera 4 is held out of the observation "
            "tracks. The projection block shows where the colored points land in that view."
        )
        add(
            "input.bundler_reconstruction",
            "bundle",
            "Import cameras and colored points",
            0,
            0,
            path="assets/bundle.out",
            world_frame="fixture_world",
            image_sizes=[None if size is None else list(size) for size in IMAGE_SIZES],
            camera_index=HELD_OUT,
            scale_mode="known_baseline",
            baseline_length_m=BASELINE,
            maximum_reprojection_error_px=0.01,
        )
        add("3d.cloud_project", "held_out", "Project into held-out camera 4", 450, 0)
        graph.connect("bundle", "cloud", "held_out", "cloud")
        graph.connect("bundle", "camera", "held_out", "camera")
        graph.connect("bundle", "world_to_camera", "held_out", "transform")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise ValueError(checked["issues"])
        graph.save(path)
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
