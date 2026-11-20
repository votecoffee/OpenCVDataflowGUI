"""Create a tilted datum with two raised rectangular parts and missing edge depth."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

WIDTH, HEIGHT = 1280, 944
K = np.array([[2600, 0, 639.5], [0, 2480, 471.5], [0, 0, 1.0]])
NORMAL = np.array([0.20, -0.24, -0.95])
NORMAL /= np.linalg.norm(NORMAL)
OFFSET = 0.8
SAMPLE_M = 0.0001
PARTS = (
    ("part_a", 0.070, 0.035, 0.012, -0.077, -0.023, 25, 36),
    ("part_b", 0.046, 0.024, 0.020, 0.059, 0.036, -36, 21),
)


def synthesize():
    y, x = np.mgrid[:HEIGHT, :WIDTH]
    rays = np.stack(((x - K[0, 2]) / K[0, 0], (y - K[1, 2]) / K[1, 1], np.ones(x.shape)), axis=2)
    denominator = rays @ NORMAL
    depth = -OFFSET / denominator
    center = np.array([0, 0, -OFFSET / NORMAL[2]])
    u = np.array([1.0, 0, 0]) - NORMAL[0] * NORMAL
    u /= np.linalg.norm(u)
    v = np.cross(u, NORMAL)
    expected = {}
    for key, length, width, thickness, px, py, angle, missing in PARTS:
        c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
        a, b = c * u + s * v, -s * u + c * v
        top = center + px * u + py * v + thickness * NORMAL
        signs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
        corners = top + signs[:, :1] * length * a / 2 + signs[:, 1:] * width * b / 2
        image = corners @ K.T
        vertices = np.rint(image[:, :2] / image[:, 2, None]).astype(np.int32)
        mask = np.zeros((HEIGHT, WIDTH), np.uint8)
        cv2.fillPoly(mask, [vertices], 255)
        depth[mask != 0] = (thickness - OFFSET) / denominator[mask != 0]
        inner = cv2.erode(mask, np.ones((3, 3), np.uint8))
        edge = np.argwhere((mask != 0) & (inner == 0))
        away = np.min(np.linalg.norm(edge[:, None, ::-1] - vertices[None], axis=2), axis=1) > 6
        removed = edge[away][np.linspace(0, np.count_nonzero(away) - 1, missing, dtype=int)]
        depth[removed[:, 0], removed[:, 1]] = 0
        expected[key] = {
            "length_m": length,
            "width_m": width,
            "thickness_m": thickness,
            "vertices": vertices.tolist(),
            "lost_yx": removed.tolist(),
        }
    return np.rint(depth / SAMPLE_M).astype(np.uint16), expected


def save_graph(path, expected):
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, pos, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = pos
        return block

    try:
        note = add("layout.comment", "notes", [0, -245])
        note.name, note.node_size = "Measure two raised parts on a tilted datum", "large"
        note.description = (
            "Original analytic depth raster by OpenCVDataflowGUI contributors (MIT). "
            "Known part A is 70 × 35 × 12 mm and part B is 46 × 24 × 20 mm. "
            "The masks include the edge pixels with missing depth."
        )
        add(
            "input.depth_file",
            "depth",
            [0, 30],
            path="assets/raised_parts.png",
            path_base="project",
            units="mm",
            units_per_sample=SAMPLE_M * 1000,
            display_min=720,
            display_max=950,
        )
        add(
            "3d.camera",
            "camera",
            [0, 275],
            width=WIDTH,
            height=HEIGHT,
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            frame="depth_optical",
        )
        add(
            "3d.separate_depth_plane",
            "datum",
            [330, 345],
            inlier_tolerance_m=0.00025,
            foreground_min_m=0.003,
        )
        graph.connect("depth", "depth", "datum", "depth")
        graph.connect("camera", "camera", "datum", "camera")
        for row, key in enumerate(expected):
            selection = add(
                "region.shape",
                f"{key}_silhouette",
                [330, 590 * row],
                shape="polygon",
                vertices=expected[key]["vertices"],
            )
            selection.name = f"Full silhouette {row + 1}"
            graph.connect("depth", "preview", selection.id, "image")
            tool = add("3d.depth_silhouette", key, [730, 330 * row])
            tool.name = f"Measure part {row + 1}"
            for source, output, target in (
                ("depth", "depth", "depth"),
                ("camera", "camera", "camera"),
                ("datum", "plane", "plane"),
                (selection.id, "mask", "mask"),
            ):
                graph.connect(source, output, tool.id, target)
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise ValueError(checked["issues"])
        graph.save(path)
    finally:
        graph.close()


def build(output):
    assets = output / "assets"
    project = output / "depth_silhouette.json"
    if assets.exists() or project.exists():
        raise FileExistsError("Choose a new output directory to preserve the example")
    assets.mkdir(parents=True)
    array, expected = synthesize()
    ok, encoded = cv2.imencode(".png", array)
    if not ok:
        raise ValueError("Could not encode the synthetic depth")
    (assets / "raised_parts.png").write_bytes(encoded.tobytes())
    (assets / "ground_truth.json").write_bytes((json.dumps(expected, indent=2) + "\n").encode())
    save_graph(project, expected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
