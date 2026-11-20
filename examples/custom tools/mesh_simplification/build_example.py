"""Build a triangulated panel with a rectangular opening and vertex attributes."""

import argparse
from pathlib import Path

import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.meshes import TriangleMesh, write_mesh_ply


def panel():
    width, height = 21, 15
    u, v = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    z = 6 * np.sin(np.pi * u) * np.sin(np.pi * v)
    vertices = np.column_stack((200 * u.ravel(), 140 * v.ravel(), z.ravel()))
    dy, dx = np.gradient(z, 10, 10)
    normals = np.column_stack((-dx.ravel(), -dy.ravel(), np.ones(u.size)))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    uv = np.column_stack((u.ravel(), v.ravel()))
    colors = np.column_stack((255 * uv, np.full(u.size, 128))).round().astype(np.uint8)
    faces = []
    for row in range(height - 1):
        for column in range(width - 1):
            if 6 <= row < 8 and 8 <= column < 12:
                continue
            a = row * width + column
            b, c, d = a + 1, a + width, a + width + 1
            faces.extend(((a, b, d), (a, d, c)))
    return TriangleMesh(vertices, faces, "mm", "panel", normals, colors, uv)


def build(output):
    target = output / "mesh_simplification.json"
    asset = output / "assets/crowned_panel.ply"
    if target.exists() or asset.exists():
        raise FileExistsError("Use an empty output directory to preserve the example")
    asset.parent.mkdir(parents=True, exist_ok=True)
    write_mesh_ply(asset, panel())
    graph = Graph()
    registry = builtin_registry()
    try:
        note = graph.add(registry.create("layout.comment", block_id="note"))
        note.name = "Simplify while retaining the panel perimeter and cutout"
        note.position, note.node_size = [0, -240], "large"
        note.description = (
            "Synthetic panel by OpenCVDataflowGUI contributors (MIT). "
            "The upper branch reduces this 544-face panel to 220 triangles. "
            "The lower branch requests 220 triangles with a zero displacement limit "
            "and should report that it cannot meet the budget."
        )
        source = graph.add(
            registry.create(
                "input.mesh", block_id="panel", parameters={"path": "assets/crowned_panel.ply"}
            )
        )
        source.name, source.position = "Load the crowned panel", [0, 80]
        for index, (key, name, limit) in enumerate(
            (
                ("review", "Simplify within 35 mm", 35.0),
                ("strict", "Zero displacement control", 0.0),
            )
        ):
            reduce = graph.add(
                registry.create(
                    "3d.mesh_simplify",
                    block_id=key,
                    parameters={
                        "target_faces": 220,
                        "deviation_mode": "Mesh units",
                        "max_deviation": limit,
                    },
                )
            )
            reduce.name, reduce.position = name, [420, index * 235]
            graph.connect(source.id, "mesh", reduce.id, "mesh")
        report = validate_graph(graph)
        if not report["valid"]:
            raise ValueError(report["issues"])
        graph.save(target)
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
