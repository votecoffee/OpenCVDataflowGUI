"""Build equivalent attributed OBJ and PLY clouds and their inspection graph."""

import argparse
from pathlib import Path

import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud


def fixture():
    x, y = np.meshgrid([-12, -4, 4, 12], [-10, 0, 10])
    xyz = np.column_stack((x.ravel() + 0.123456789, y.ravel(), 400 + np.abs(x.ravel()) / 8))
    slope = np.sign(x.ravel()) / 8
    normals = np.column_stack((-slope, np.zeros_like(slope), np.ones_like(slope)))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    colors = np.array(
        [[0, 1, 255], [21, 128, 239], [100, 208, 30], [245, 173, 22]] * len(y),
        dtype=np.uint8,
    )
    return PointCloud(xyz, "mm", "folded_coupon", colors=colors, normals=normals)


def build(output):
    project = output / "cloud_obj_interchange.json"
    assets = output / "assets"
    paths = {suffix: assets / f"folded_coupon.{suffix}" for suffix in ("obj", "ply")}
    if project.exists() or any(path.exists() for path in paths.values()):
        raise FileExistsError("Use an empty output directory to preserve the example")
    assets.mkdir(parents=True, exist_ok=True)
    cloud = fixture()
    for path in paths.values():
        write_cloud(path, cloud)
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = position
        return block

    try:
        note = add("layout.comment", "note", [0, -320])
        note.name, note.node_size = "Compare OBJ and PLY point clouds", "large"
        note.description = (
            "Original synthetic cloud by OpenCVDataflowGUI contributors (MIT). "
            "Both files contain the same twelve folded-surface points with aligned normals "
            "and RGB. The dataflow comment supplies millimeters and the cloud frame."
        )
        for index, (suffix, path) in enumerate(paths.items()):
            source = add(
                "input.cloud",
                f"source_{suffix}",
                [0, index * 275],
                path=path.relative_to(output).as_posix(),
            )
            source.name = f"Load {suffix.upper()} scan"
            measure = add("3d.cloud_measure", f"measure_{suffix}", [410, index * 275])
            graph.connect(source.id, "cloud", measure.id, "cloud")
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
