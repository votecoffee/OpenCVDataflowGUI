"""Build equivalent OBJ and PLY triangle meshes with known vertex attributes."""

import argparse
from pathlib import Path

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.mesh_io import write_mesh
from opencv_dataflow.meshes import TriangleMesh


def fixture():
    return TriangleMesh(
        [
            [0, 0, 350],
            [47.123456, 0, 350],
            [0, 35, 350],
            [47.123456, 35, 350],
            [0, 61, 367],
            [47.123456, 61, 367],
        ],
        [[0, 1, 2], [2, 1, 3], [2, 3, 4], [4, 3, 5]],
        "mm",
        "folded_panel",
        normals=[[0, 0, 1]] * 4 + [[0, -0.547, 0.837]] * 2,
        colors=[
            [40, 120, 220],
            [30, 90, 200],
            [60, 150, 240],
            [60, 150, 240],
            [240, 160, 0],
            [230, 96, 4],
        ],
        texcoords=[[0, 0], [1, 0], [0, 0.5], [1, 0.5], [0, 1], [1, 1]],
    )


def build(output):
    target = output / "mesh_interchange.json"
    paths = {extension: output / f"assets/folded_panel.{extension}" for extension in ("obj", "ply")}
    if target.exists() or any(p.exists() for p in paths.values()):
        raise FileExistsError("Use an empty output directory to preserve the example")
    (output / "assets").mkdir(parents=True, exist_ok=True)
    graph = Graph()
    registry = builtin_registry()
    mesh = fixture()
    try:
        note = graph.add(registry.create("layout.comment", block_id="note"))
        note.name = "Compare triangle meshes"
        note.position, note.node_size = [0, -280], "large"
        note.description = (
            "Original synthetic meshes by OpenCVDataflowGUI contributors (MIT). "
            "Both formats contain the same six vertices and four triangles. Compare "
            "the wireframes, coordinate frame and vertex attribute reports."
        )
        for index, (extension, path) in enumerate(paths.items()):
            write_mesh(path, mesh)
            load = graph.add(
                registry.create(
                    "input.mesh",
                    block_id=f"load_{extension}",
                    parameters={"path": path.relative_to(output).as_posix()},
                )
            )
            load.name = f"Load {extension.upper()} panel"
            load.position = [index * 420, 30]
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
