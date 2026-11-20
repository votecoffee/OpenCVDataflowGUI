"""Build a triangulated reference coupon and a scan with known dimensional defects."""

import argparse
from pathlib import Path

import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.mesh_io import write_mesh
from opencv_dataflow.meshes import TriangleMesh
from opencv_dataflow.pointclouds import PointCloud

FRAME = "fixture_panel"


def fixtures():
    # The two faces duplicate their shared vertex IDs as an imported UV seam would.
    mesh = TriangleMesh(
        [[0, 0, 500], [100, 0, 500], [0, 100, 500], [0, 100, 500], [100, 0, 500], [100, 100, 500]],
        [[0, 1, 2], [3, 4, 5]],
        "mm",
        FRAME,
        texcoords=[[0, 0], [1, 0], [0, 1], [0, 0.5], [1, 0.5], [1, 1]],
    )
    mm = np.array(
        [
            [25, 25, 500.4],
            [75, 25, 499.5],
            [60, 65, 501.6],
            [68, 20, 498.5],
            [-1.2, 40, 500.4],
            [0, 0, 500.3],
            [145, 30, 500],
            [15, 65, 500],
        ]
    )
    return mesh, PointCloud(mm / 1000, "m", FRAME, colors=[[75, 185, 220]] * len(mm))


def build(output):
    project = output / "scan_mesh_comparison.json"
    part = output / "assets/nominal_panel.obj"
    xyz = output / "assets/measured_panel.ply"
    if any(path.exists() for path in (project, part, xyz)):
        raise FileExistsError("Use an empty output directory to preserve the example")
    part.parent.mkdir(parents=True, exist_ok=True)
    mesh, scan = fixtures()
    write_mesh(part, mesh)
    write_cloud(xyz, scan, binary=True)
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = position
        return block

    try:
        comment = add("layout.comment", "note", [0, -300])
        comment.name, comment.node_size = "Inspect a scan against a reference surface", "large"
        comment.description = (
            "Original synthetic example by OpenCVDataflowGUI contributors (MIT). "
            "Known offsets exercise signed tolerances. A split vertex seam, two "
            "open-edge hits and an unsupported point exercise coverage."
        )
        model = add("input.mesh", "reference", [0, 0], path="assets/nominal_panel.obj")
        source = add("input.cloud", "scan", [0, 270], path="assets/measured_panel.ply")
        for capture in (model, source):
            capture.ticket_settings = {"mode": "fixed", "value": "panel_scan"}
        cases = (
            ("pass", "Allow 2 mm deviation", 0.002, 0.55),
            ("size", "Inspect at 1 mm", 0.001, 0.55),
            ("coverage", "Require 80% usable scan support", 0.002, 0.8),
        )
        for i, (key, name, tolerance, coverage) in enumerate(cases):
            block = add(
                "3d.scan_mesh_compare",
                key,
                [430, (i - 1) * 285],
                comparison="normal",
                maximum_distance_m=0.004,
                negative_tolerance_m=tolerance,
                positive_tolerance_m=tolerance,
                minimum_support_fraction=coverage,
            )
            block.name = name
            graph.connect("reference", "mesh", key, "reference")
            graph.connect("scan", "cloud", key, "scan")
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
