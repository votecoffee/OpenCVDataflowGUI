"""Build nominal, raised and warped scans in one known datum frame."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud

RVEC = [0.25, -0.46, 0.31]
ORIGIN_MM = [420, -125, 680]


def scan(case):
    y, x = np.indices((4, 4))
    u, v = (10 + 20 * x).ravel(), (10 + 20 * y).ravel()
    h = np.full(16, 2.0)
    rows = np.arange(16)
    if case == "raised":
        h += 4
    elif case == "warped":
        h[0], h[-1] = -1, 5
        rows = rows[rows != 6]
    # Add one return outside the measurement footprint.
    local = np.vstack((np.column_stack((u[rows], v[rows], h[rows])), [120, 10, 70]))
    rotation = Rotation.from_rotvec(RVEC).as_matrix()
    points = local @ rotation.T + ORIGIN_MM
    return PointCloud(
        points,
        "mm",
        "station",
        attributes={"source_index": np.r_[100 + rows, 500], "intensity": np.r_[100 + u[rows], 0]},
    )


def build(output):
    project = output / "cloud_datum.json"
    asset_folder = output / "assets"
    cases = ("nominal", "raised", "warped")
    if project.exists() or any((asset_folder / f"{case}.ply").exists() for case in cases):
        raise FileExistsError("Build in a new directory")
    asset_folder.mkdir(parents=True, exist_ok=True)
    for case in cases:
        write_cloud(asset_folder / f"{case}.ply", scan(case), binary=True)
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = position
        return block

    try:
        note = add("layout.comment", "note", [0, -260])
        note.name, note.node_size = "Check height and coverage against a fixture datum", "large"
        note.description = (
            "Original synthetic scans by OpenCVDataflowGUI contributors (MIT). "
            "Nominal passes. Raised remains nearly flat and fails height. "
            "Warped fails height, coverage, and peak-to-valley spread."
        )
        add(
            "3d.pose",
            "datum",
            [10, 0],
            units="m",
            source_frame="fixture",
            target_frame="station",
            rvec=RVEC,
            translation=(np.asarray(ORIGIN_MM) / 1000).tolist(),
        )
        for i, case in enumerate(cases):
            y = 210 + 310 * i
            source = add("input.cloud", f"{case}_scan", [10, y], path=f"assets/{case}.ply")
            source.name = f"{case.capitalize()} surface"
            measure = add(
                "3d.cloud_datum",
                case,
                [470, y],
                u_min_m=0.0,
                u_max_m=0.08,
                v_min_m=0.0,
                v_max_m=0.08,
                grid_width=4,
                grid_height=4,
                nominal_height_m=0.002,
                negative_tolerance_m=0.001,
                positive_tolerance_m=0.001,
                minimum_coverage=0.97,
                check_peak_to_valley=True,
                maximum_peak_to_valley_m=0.003,
            )
            measure.name = f"Check {case} surface"
            measure.preview_output = "deviation"
            graph.connect(source.id, "cloud", measure.id, "cloud")
            graph.connect("datum", "transform", measure.id, "datum")
        review = validate_graph(graph)
        if not review["valid"]:
            raise ValueError(review["issues"])
        graph.save(project)
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
