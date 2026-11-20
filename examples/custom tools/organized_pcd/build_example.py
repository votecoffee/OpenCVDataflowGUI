"""Build a portable demonstration of two organized PCD encodings."""

import argparse
from pathlib import Path

import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud

SIZE = (64, 44)


def fixture():
    width, height = SIZE
    y, x = np.mgrid[:height, :width]
    hole = ((x - 23) / 9) ** 2 + ((y - 22) / 7) ** 2 <= 1
    occluded = (x >= 41) & (x < 48) & (y >= 8) & (y < 35)
    keep = ~(hole | occluded)
    u, v = x[keep], y[keep]
    step = u >= 37
    points = np.column_stack((u - width / 2, v - height / 2, 350 - 11 * step))
    colors = np.where(step[:, None], [250, 171, 64], [69, 186, 220])
    normals = np.broadcast_to([0, 0, -1], points.shape)
    return PointCloud(
        points,
        "mm",
        "reference_scan",
        colors=colors,
        normals=normals,
        pixels=np.column_stack((u, v)),
        source_size=SIZE,
    )


def build(output):
    project = output / "organized_pcd.json"
    assets = output / "assets"
    if assets.exists() or project.exists():
        raise FileExistsError("Choose an empty output directory")
    assets.mkdir(parents=True)
    cloud = fixture()
    for mode in ("binary", "ascii"):
        path = assets / f"coupon_{mode}.pcd"
        write_cloud(path, cloud, mode == "binary", pcd_organization="source_pixels")

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = position
        return block

    try:
        note = add("layout.comment", "note", [0, -230])
        note.name, note.node_size = "Compare organized PCD encodings", "large"
        note.description = (
            "Original synthetic scan by OpenCVDataflowGUI contributors (MIT). "
            "Both files hold the same 64 × 44 scanning grid with holes. "
            "The cloud keeps valid XYZ, RGB, normals and the original pixel of each point."
        )
        for row, mode in enumerate(("binary", "ascii")):
            source = add("input.cloud", mode, [0, 255 * row], path=f"assets/coupon_{mode}.pcd")
            source.name = f"Load organized {mode} PCD"
            render = add(
                "3d.cloud_render",
                f"view_{mode}",
                [395, 255 * row],
                width=512,
                height=384,
                point_size=3,
            )
            graph.connect(source.id, "cloud", render.id, "cloud")
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise ValueError(validation["issues"])
        graph.save(project)
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
