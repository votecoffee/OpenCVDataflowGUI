"""Build synthetic triangle meshes with known topology and dimensions."""

import argparse
from pathlib import Path

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.meshes import TriangleMesh, write_mesh_ply


def fixtures():
    vertices = [
        [0, 0, 0],
        [80, 0, 0],
        [80, 40, 0],
        [0, 40, 0],
        [0, 0, 20],
        [80, 0, 20],
        [80, 40, 20],
        [0, 40, 20],
    ]
    faces = [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ]
    return {
        "closed_box": TriangleMesh(vertices, faces, "mm", "fixture"),
        "open_box": TriangleMesh(vertices, faces[:2] + faces[4:], "mm", "fixture"),
        "flap_box": TriangleMesh(vertices + [[40, -10, 5]], faces + [[0, 1, 8]], "mm", "fixture"),
    }


def build(output):
    target = output / "mesh_validity.json"
    meshes = fixtures()
    paths = {key: output / f"assets/{key}.ply" for key in meshes}
    if target.exists() or any(path.exists() for path in paths.values()):
        raise FileExistsError("Use an empty output directory to preserve the example")
    (output / "assets").mkdir(parents=True, exist_ok=True)
    graph = Graph()
    registry = builtin_registry()
    try:
        note = graph.add(registry.create("layout.comment", block_id="note"))
        note.name = "Check geometry before reporting volume"
        note.position, note.node_size = [0, -240], "large"
        note.description = (
            "Synthetic fixtures by OpenCVDataflowGUI contributors (MIT). "
            "The closed box has 11,200 mm² area and 64,000 mm³ volume. "
            "Compare its report with the box missing a lid and the box with a third face "
            "attached to one edge. Both damaged controls should fail."
        )
        for row, (key, mesh) in enumerate(meshes.items()):
            path = paths[key]
            write_mesh_ply(path, mesh)
            label = key.replace("_", " ").capitalize()
            source = graph.add(
                registry.create(
                    "input.mesh",
                    block_id=f"load_{key}",
                    parameters={"path": path.relative_to(output).as_posix()},
                )
            )
            source.name, source.position = f"Load {label.lower()}", [0, row * 240]
            inspect = graph.add(
                registry.create(
                    "3d.mesh_inspect",
                    block_id=key,
                    parameters={"require_closed": True},
                )
            )
            inspect.name, inspect.position = f"Inspect {label.lower()}", [440, row * 240]
            graph.connect(source.id, "mesh", inspect.id, "mesh")
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
