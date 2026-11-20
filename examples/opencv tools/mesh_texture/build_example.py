"""Build the original two-panel camera texture example."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.mesh_io import write_mesh
from opencv_dataflow.meshes import TriangleMesh

WIDTH, HEIGHT = 80, 60
FX, FY, CX, CY = 71.0, 64.0, 39.5, 29.5
FRAME = "texture_camera"
BACK = (-6.2, 3.2, 79.2, 56.2)
FRONT = (27.2, 19.3, 53.8, 38.7)


def _corners(bounds, slope):
    x0, y0, x1, y1 = bounds
    xy = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
    ray = (xy - [CX, CY]) / [FX, FY]
    z = 1.0 / (1 - ray @ [0.3, -0.1]) if slope else np.full(4, 2.0)
    return np.column_stack((ray * z[:, None], z))


def build(output):
    project = output / "mesh_texture.json"
    image = output / "assets/camera.png"
    model = output / "assets/panels.ply"
    if any(path.exists() for path in (project, image, model)):
        raise FileExistsError("Use an empty output directory")
    model.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.vstack((_corners(BACK, False), _corners(FRONT, True)))
    uv = [
        [0, 1],
        [0.625, 1],
        [0, 0],
        [0.625, 0],
        [0.6875, 0.8125],
        [1, 0.8125],
        [0.6875, 0.1875],
        [1, 0.1875],
    ]
    faces = [[0, 1, 3], [0, 3, 2], [4, 5, 7], [4, 7, 6]]
    write_mesh(model, TriangleMesh(xyz, faces, frame=FRAME, texcoords=uv))
    y, x = np.mgrid[:HEIGHT, :WIDTH]
    x0, y0, x1, y1 = FRONT
    foreground = (x > x0) & (x < x1) & (y > y0) & (y < y1)
    bgr = np.stack((2 * x + 10, 3 * y + 15, x + y + np.where(foreground, 100, 10)), axis=-1)
    if not cv2.imwrite(str(image), bgr.astype(np.uint8)):
        raise OSError(f"Could not write {image}")

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.position = position
        return block

    try:
        note = add("layout.comment", "note", [0, -265])
        note.name, note.node_size = "Registered camera color on a mesh", "large"
        note.description = (
            "Original synthetic panels and image by OpenCVDataflowGUI contributors (MIT). "
            "The left UV island is the back panel. Its dark rectangular gap is occluded "
            "by the slanted front panel shown on the right island. The white mask "
            "marks valid texels, including valid black pixels in other images."
        )
        add("input.file", "image", [0, 0], path="assets/camera.png", path_base="project")
        add("input.mesh", "panels", [0, 245], path="assets/panels.ply")
        add(
            "3d.camera",
            "camera",
            [0, 490],
            fx=FX,
            fy=FY,
            cx=CX,
            cy=CY,
            width=WIDTH,
            height=HEIGHT,
            frame=FRAME,
        )
        for index, sampling in enumerate(("bilinear", "nearest")):
            bake = add(
                "3d.mesh_texture",
                sampling,
                [450, 95 + 300 * index],
                texture_width=128,
                texture_height=64,
                sampling=sampling,
                visibility_tolerance_m=0.00001,
            )
            bake.name = f"Bake {sampling} texture"
            bake.preview_output = "texture"
            bake.exposed_outputs = ["texture", "texture_mask", "mesh", "report"]
            for block_id, port in (("image", "image"), ("panels", "mesh"), ("camera", "camera")):
                graph.connect(block_id, port, bake.id, port)
        report = validate_graph(graph)
        if not report["valid"]:
            raise ValueError(report["issues"])
        graph.save(project)
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
