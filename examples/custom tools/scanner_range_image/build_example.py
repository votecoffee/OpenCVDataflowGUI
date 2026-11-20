"""Build a calibrated scanner projection with independently defined returns."""

import argparse
from pathlib import Path

import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud

GRID = {
    "columns": 6,
    "azimuth_start_deg": 165.0,
    "azimuth_span_deg": 60.0,
    "elevation_edges_deg": [-15.0, -5.0, 4.0, 25.0],
    "minimum_range_m": 0.5,
    "maximum_range_m": 15.0,
}


def scan():
    returns = [
        (170, 0, 2, 10, 101),
        (170, 0, 4, 90, 201),
        (170, 0, 3, 90, 20_000_003),
        (170, 0, 3, 90, 401),
        (190, 12, 6.5, 2, 60),
        (200, -10, 8, np.nan, -1),
        (180, 0, 1.5, 11, 702),
        (220, 8, 12, 0, 800),
        (149, 0, 2, 2, 901),
        (170, 29, 2, 2, 902),
        (170, 0, 0.3, 2, 903),
        (170, 0, 20, 2, 904),
    ]
    az, el, distance, intensity, indices = np.asarray(returns).T
    az, el = np.deg2rad(az), np.deg2rad(el)
    radial_xy = distance * np.cos(el)
    x, y, z = radial_xy * np.cos(az), radial_xy * np.sin(az), distance * np.sin(el)

    # The building frame is a quarter-turn clockwise with a 0.5 m Z offset.
    building = np.column_stack((y, -x, z + 0.5))
    building = np.vstack((building, [0, 0, 2.5]))
    return PointCloud(
        building * 1000,
        "mm",
        "building",
        attributes={
            "intensity": np.r_[intensity, 8.0],
            "source_index": np.r_[indices, 905].astype(np.int32),
        },
    )


def build(output):
    asset = output / "assets/angular_returns.ply"
    project = output / "scanner_range_image.json"
    if asset.exists() or project.exists():
        raise FileExistsError("Use an empty output directory to preserve the example")
    asset.parent.mkdir(parents=True, exist_ok=True)
    write_cloud(asset, scan(), binary=True)

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, name, position, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = name, position
        return block

    try:
        note = add(
            "layout.comment", "note", "Compare scanner returns on an angular grid", [0, -265]
        )
        note.node_size = "large"
        note.description = (
            "Synthetic scanner returns by OpenCVDataflowGUI contributors (MIT). "
            "The grid wraps across 180 degrees and has three unequal elevation bands. "
            "The same bin contains four returns. Compare its range, intensity and "
            "source ID for nearest, farthest and strongest policies."
        )
        add(
            "input.cloud",
            "source",
            "Load angular returns",
            [0, 0],
            path="assets/angular_returns.ply",
        )
        add(
            "3d.pose",
            "pose",
            "Building to scanner",
            [0, 230],
            source_frame="building",
            target_frame="scanner",
            units="m",
            rvec=[0, 0, np.pi / 2],
            translation=[0, 0, -0.5],
        )
        for row, mode in enumerate(("nearest", "farthest", "strongest", "empty")):
            settings = dict(GRID, return_policy=mode if mode != "empty" else "nearest")
            if mode == "empty":
                settings["azimuth_start_deg"] = 25.0
            add(
                "3d.cloud_range_image",
                mode,
                f"{mode.capitalize()} returns",
                [440, row * 265],
                **settings,
            )
            graph.connect("source", "cloud", mode, "cloud")
            graph.connect("pose", "transform", mode, "transform")
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
