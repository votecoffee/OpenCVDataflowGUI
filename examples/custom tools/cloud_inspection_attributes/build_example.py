"""Build two small registered scans with different inspection columns."""

import argparse
from pathlib import Path

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud

SCAN_A = """# .PCD v0.7
# dataflow_metadata {"units":"mm","frame":"fixture"}
VERSION 0.7
FIELDS x y z intensity confidence
SIZE 4 4 4 2 8
TYPE F F F U F
COUNT 1 1 1 1 1
WIDTH 6
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 6
DATA ascii
0 0 400 8 0.2
nan 0 400 999 0.1
11 0 400 41 0.95
20 0 400 38 nan
30 0 400 5 0.11
40 0 400 55 0.75
"""


def build(output):
    project = output / "cloud_inspection_attributes.json"
    assets = output / "assets"
    if project.exists() or assets.exists():
        raise FileExistsError("Choose an empty output directory")
    assets.mkdir(parents=True)
    (assets / "upper_scan.pcd").write_text(SCAN_A, encoding="ascii", newline="\n")
    side = PointCloud(
        [[11.1, 0, 400], [15, 8, 400], [45, 0, 400], [55, 6, 400]],
        "mm",
        "fixture",
        attributes={"confidence": [0.5, 0.96, 0.87, 0.77], "source_index": [2, 3, 5, 8]},
    )
    write_cloud(assets / "side_scan.ply", side, binary=True)
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = position
        return block

    try:
        note = add("layout.comment", "note", [0, -245])
        note.name, note.node_size = "Trace inspection fields through two scans", "large"
        note.description = (
            "Original analytic scans by OpenCVDataflowGUI contributors (MIT). "
            "Merge keeps measured intensity, confidence, and original row IDs. "
            "Confidence filtering and voxel sampling retain aligned fields."
        )
        upper = add(
            "input.cloud", "upper", [0, 0], path="assets/upper_scan.pcd", track_source_rows=True
        )
        side = add("input.cloud", "side", [0, 260], path="assets/side_scan.ply")
        for source, title in (
            (upper, "Upper scan with intensity"),
            (side, "Side scan with confidence"),
        ):
            source.name = title
            source.ticket_settings = {"mode": "fixed", "value": "inspection-attributes"}
        join = add(
            "3d.cloud_merge",
            "merge",
            [335, 90],
            output_units="mm",
            merge_mode="prefer_earlier",
            overlap_radius_m=0.0005,
        )
        join.exposed_inputs = ["cloud_1", "cloud_2"]
        select = add(
            "3d.cloud_filter",
            "confidence",
            [670, 90],
            attribute="confidence",
            attribute_minimum=0.7,
            attribute_maximum=1.0,
        )
        select.name = "Keep confidence 0.7–1.0"
        select.exposed_inputs = ["cloud"]
        add("3d.cloud_voxel", "sample", [990, 90], voxel_size=18.0, mode="first")
        add("3d.cloud_measure", "inspect", [1310, 90])
        graph.connect("upper", "cloud", "merge", "cloud_1")
        graph.connect("side", "cloud", "merge", "cloud_2")
        for before, after in (
            ("merge", "confidence"),
            ("confidence", "sample"),
            ("sample", "inspect"),
        ):
            graph.connect(before, "cloud", after, "cloud")
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
