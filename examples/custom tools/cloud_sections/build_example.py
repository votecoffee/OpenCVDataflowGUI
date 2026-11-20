"""Build an original stepped channel and two pose-relative section measurements."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud

RVEC = [0.32, -0.27, 0.44]
ORIGIN_MM = [260, -90, 350]


def scan():
    local = []
    for height, widths in ((25, (16, 18, None, 20, 24)), (55, (28, 28, 32, 32, 36))):
        for index, width in enumerate(widths):
            if width is None:
                continue
            u = 12.5 + 25 * index
            local.extend(((u, -width / 2, height - 0.5), (u, width / 2, height + 0.5)))
    local.extend(((12.5, 100.0, 25.0), (130.0, 10.0, 25.0), (38.0, 20.0, 40.0)))
    local = np.array(local)[::-1]  # Input order differs from profile bin order.
    rotation = Rotation.from_rotvec(RVEC).as_matrix()
    xyz = local @ rotation.T + ORIGIN_MM
    color = np.where((local[:, 2] < 40)[:, None], [60, 180, 224], [238, 170, 66])
    return PointCloud(
        xyz,
        "mm",
        "bench_scan",
        colors=color,
        attributes={"source_index": 1000 + np.arange(len(local)), "intensity": abs(local[:, 1])},
    )


def build(output):
    project = output / "cloud_sections.json"
    asset = output / "assets/stepped_channel.ply"
    if project.exists() or asset.exists():
        raise FileExistsError("Build to a new directory")
    asset.parent.mkdir(parents=True, exist_ok=True)
    write_cloud(asset, scan(), binary=True)
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.position = position
        return block

    try:
        note = add("layout.comment", "note", [0, -260])
        note.name, note.node_size = "Measure a stepped channel at two heights", "large"
        note.description = (
            "Original synthetic scan by OpenCVDataflowGUI contributors (MIT). "
            "Two sections share a fixture pose but use separate heights. The lower "
            "section has an intentional gap at the center bin. Compare observed "
            "widths and coverage, then inspect the original selected point rows."
        )
        add("input.cloud", "scan", [0, 20], path="assets/stepped_channel.ply")
        add(
            "3d.pose",
            "fixture",
            [0, 300],
            source_frame="fixture",
            target_frame="bench_scan",
            units="m",
            rvec=RVEC,
            translation=(np.asarray(ORIGIN_MM) / 1000).tolist(),
        )
        for i, (key, height) in enumerate((("lower", 0.025), ("upper", 0.055))):
            section = add(
                "3d.cloud_section",
                key,
                [460, 20 + 310 * i],
                offset_m=height,
                thickness_m=0.0025,
                u_min_m=0.0,
                u_max_m=0.125,
                bins=5,
                restrict_v=True,
                v_min_m=-0.02,
                v_max_m=0.02,
                minimum_coverage=0.75,
            )
            section.name = f"{key.capitalize()} channel section"
            graph.connect("scan", "cloud", key, "cloud")
            graph.connect("fixture", "transform", key, "pose")
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
