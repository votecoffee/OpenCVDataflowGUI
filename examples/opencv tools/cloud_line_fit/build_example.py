"""Build a tilted measured line with independent truth and labeled clutter."""

import argparse
import json
from pathlib import Path

import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud

FRAME = "fixture"
CENTER = np.array([116.0, -73.0, 430.0])
AXIS = np.array([0.71, -0.39, 0.58])
AXIS /= np.linalg.norm(AXIS)
INLIER_LIMITS = [-82.0, 97.0]


def fixture():
    rng = np.random.default_rng(1202)
    u = np.cross(AXIS, [0, 0, 1])
    u /= np.linalg.norm(u)
    v = np.cross(AXIS, u)
    t = np.linspace(*INLIER_LIMITS, 120)
    support = (
        CENTER
        + t[:, None] * AXIS
        + rng.normal(0, 0.06, (len(t), 1)) * u
        + rng.normal(0, 0.08, (len(t), 1)) * v
    )
    clutter = CENTER + rng.uniform(30, 98, (22, 1)) * AXIS + rng.uniform(16, 44, (22, 1)) * v
    xyz = np.concatenate((support, clutter))
    order = rng.permutation(len(xyz))
    is_support = order < len(support)
    colors = np.array([[247, 155, 83], [125, 189, 238]], np.uint8)[is_support.astype(int)]
    source_index = 101 + 5 * order
    source_scan = np.where(is_support, 4, 6).astype(np.int32)
    cloud = PointCloud(
        xyz[order],
        "mm",
        FRAME,
        colors=colors,
        attributes={"source_index": source_index, "source_scan": source_scan},
    )
    return cloud, {
        "axis": AXIS.tolist(),
        "center_mm": CENTER.tolist(),
        "observed_limits_mm": INLIER_LIMITS,
        "reference_orientation": (-AXIS).tolist(),
        "inlier_rows": np.flatnonzero(is_support).tolist(),
        "outlier_rows": np.flatnonzero(~is_support).tolist(),
        "inlier_source_index": source_index[is_support].tolist(),
        "outlier_source_index": source_index[~is_support].tolist(),
    }


def save_project(output, robust):
    registry, graph = builtin_registry(), Graph()

    def add(kind, key, label, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = label, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Fit a measured line through clutter", 0, -250)
        note.node_size = "large"
        note.description = "Original synthetic scan by OpenCVDataflowGUI contributors (MIT). " + (
            "Welsch fits the observed axis and the fixture pose establishes its direction."
            if robust
            else "Least squares is pulled toward the clutter and fails the support gate."
        )
        add(
            "input.cloud",
            "source",
            "Load 142 identified XYZ rows",
            0,
            0,
            path="assets/line_with_clutter.ply",
            units="mm",
            frame=FRAME,
        )
        add(
            "3d.cloud_fit_line",
            "fit",
            "Fit robust line" if robust else "Fit least-squares line",
            480,
            0,
            distance="welsch" if robust else "l2",
            inlier_tolerance=0.6,
            minimum_inliers=60,
            minimum_inlier_fraction=0.75,
            minimum_span=160.0,
        )
        graph.connect("source", "cloud", "fit", "cloud")
        if robust:
            ref = -AXIS
            around = np.cross([0, 0, 1], ref)
            rvec = around * (np.arccos(ref[2]) / np.linalg.norm(around))
            add(
                "3d.pose",
                "datum",
                "Fixture direction (+Z)",
                0,
                255,
                rvec=rvec.tolist(),
                units="mm",
                source_frame="datum",
                target_frame=FRAME,
            )
            graph.connect("datum", "transform", "fit", "reference_pose")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise ValueError(checked["issues"])
        name = "cloud_line_fit" if robust else "least_squares_control"
        graph.save(output / f"{name}.json")
    finally:
        graph.close()


def build(output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty output directory")
    assets = output / "assets"
    assets.mkdir(parents=True)
    cloud, truth = fixture()
    write_cloud(assets / "line_with_clutter.ply", cloud, binary=False)
    (assets / "truth.json").write_text(
        json.dumps(truth, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    for robust in (True, False):
        save_project(output, robust)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
