"""Build calibrated nominal and measured scans with controlled dimensional defects."""

import argparse
from pathlib import Path

import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud

FRAME = "inspection_fixture"


def fixture():
    x, y = np.meshgrid(np.arange(8) * 10, np.arange(4) * 10)
    nominal_mm = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 400)))
    offsets = np.where(np.arange(len(nominal_mm)) % 2 == 0, 0.25, -0.35)
    offsets[[5, 13, 28]] = [-1.5, 1.2, -2.0]
    measured_mm = nominal_mm.astype(np.float64)
    measured_mm[:, 2] += offsets
    measured_mm = np.vstack((np.delete(measured_mm, [3, 26], axis=0), [107, 71, 400]))
    measured_mm = measured_mm[np.random.default_rng(88).permutation(len(measured_mm))]
    reference = PointCloud(
        nominal_mm,
        "mm",
        FRAME,
        normals=np.tile([0, 0, 1], (len(nominal_mm), 1)),
        colors=np.tile([75, 185, 225], (len(nominal_mm), 1)),
    )
    measured = PointCloud(
        measured_mm / 1000,
        "m",
        FRAME,
        normals=np.tile([0, 0, 1], (len(measured_mm), 1)),
        colors=np.tile([215, 165, 65], (len(measured_mm), 1)),
    )
    return {"measured": measured, "reference": reference}


def build(output):
    project = output / "cloud_dimensional_comparison.json"
    assets = {kind: output / f"assets/{kind}.ply" for kind in ("measured", "reference")}
    if project.exists() or any(path.exists() for path in assets.values()):
        raise FileExistsError("Use an empty output directory to preserve the example")
    (output / "assets").mkdir(parents=True, exist_ok=True)
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, pos, **params):
        tool = graph.add(registry.create(kind, block_id=key, parameters=params))
        tool.position = pos
        return tool

    try:
        note = add("layout.comment", "note", [0, -340])
        note.name, note.node_size = "Inspect distance and coverage", "large"
        note.description = (
            "Original synthetic comparison by OpenCVDataflowGUI contributors (MIT). "
            "Three measured rows violate the strict tolerance. Two nominal rows are "
            "missing and one measured row is outside the overlap region."
        )
        for index, (kind, cloud) in enumerate(fixture().items()):
            write_cloud(assets[kind], cloud, binary=True)
            source = add(
                "input.cloud",
                kind,
                [0, index * 300],
                path=assets[kind].relative_to(output).as_posix(),
            )
            source.name = f"Load {kind} cloud"
            source.ticket_settings = {"mode": "fixed", "value": "dimensional-fixture"}
        cases = (
            ("pass", "Allow up to 2.1 mm", 0.0021, 0.0021, 0.9),
            ("size", "Inspect inward and outward tolerance", 0.0008, 0.0009, 0.9),
            ("overlap", "Require 98% coverage", 0.0021, 0.0021, 0.98),
        )
        for index, (key, name, inward, outward, coverage) in enumerate(cases):
            comparison = add(
                "3d.cloud_compare",
                key,
                [435, (index - 1) * 280],
                comparison="normal",
                correspondence_radius_m=0.004,
                negative_tolerance_m=inward,
                positive_tolerance_m=outward,
                minimum_measured_overlap=coverage,
                minimum_reference_overlap=coverage,
            )
            comparison.name = name
            for kind in ("measured", "reference"):
                graph.connect(kind, "cloud", key, kind)
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
