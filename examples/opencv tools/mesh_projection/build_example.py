"""Build a calibrated raster example with overlap, clipping and opposite winding."""

import argparse
from pathlib import Path

import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.mesh_io import write_mesh
from opencv_dataflow.meshes import TriangleMesh

WIDTH, HEIGHT = 49, 41
FX, FY, CX, CY = 39.0, 32.0, 19.2, 23.7
FRAME = "inspection_camera"
NEAR, FAR = 0.55, 2.7


def xyz(uv, z):
    uv, z = np.asarray(uv), np.asarray(z)
    return np.column_stack(((uv[:, 0] - CX) * z / FX, (uv[:, 1] - CY) * z / FY, z))


def fixture():
    background = xyz([(-3.2, 3.61), (46.6, 4.2), (11.3, 37.7)], [2, 2, 2])
    slanted = xyz([(5.37, 7.29), (37.8, 11.2), (13.2, 36.35)], [0.22, 0.88, 3.9])
    reverse = xyz([(33.8, 24.3), (42, 37.2), (44, 23.6)], [1.1, 1.1, 1.1])
    coords = np.vstack((background, slanted, reverse))
    colors = np.repeat([[5, 150, 220], [235, 70, 15], [220, 240, 11]], 3, axis=0)
    return TriangleMesh(coords, np.arange(9).reshape(3, 3), "m", FRAME, colors=colors)


def build(output):
    project = output / "mesh_projection.json"
    model = output / "assets/calibrated_triangles.ply"
    if project.exists() or model.exists():
        raise FileExistsError("Use an empty output directory to preserve the example")
    model.parent.mkdir(parents=True, exist_ok=True)
    write_mesh(model, fixture())
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = position
        return block

    try:
        note = add("layout.comment", "note", [0, -300])
        note.name, note.node_size = "Project calibrated triangles", "large"
        note.description = (
            "Original synthetic fixture by OpenCVDataflowGUI contributors (MIT). "
            "A blue face extends beyond the left edge. An orange face crosses the "
            "near and far planes and occludes part of the blue face. A yellow face "
            "has the opposite winding."
        )
        add("input.mesh", "mesh", [0, 35], path="assets/calibrated_triangles.ply")
        add(
            "3d.camera",
            "camera",
            [0, 320],
            fx=FX,
            fy=FY,
            cx=CX,
            cy=CY,
            width=WIDTH,
            height=HEIGHT,
            frame=FRAME,
        )
        for i, (key, culling) in enumerate((("all", "none"), ("cull", "counterclockwise"))):
            block = add(
                "3d.mesh_project",
                key,
                [460, 35 + 285 * i],
                minimum_z_m=NEAR,
                maximum_z_m=FAR,
                culling=culling,
            )
            block.name = "Render all windings" if i == 0 else "Cull counterclockwise faces"
            block.preview_output = "color"
            graph.connect("mesh", "mesh", key, "mesh")
            graph.connect("camera", "camera", key, "camera")
        result = validate_graph(graph)
        if not result["valid"]:
            raise ValueError(result["issues"])
        graph.save(project)
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
