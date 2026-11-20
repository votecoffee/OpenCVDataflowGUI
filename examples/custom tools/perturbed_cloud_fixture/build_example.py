"""Build a seeded benchmark for sampling, registration and scan tolerances."""

import argparse
from pathlib import Path

import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud


def reference_cloud():
    x, y = np.meshgrid(np.arange(12) * 9 - 40, np.arange(8) * 11 - 30)
    xyz = np.column_stack((x.ravel(), y.ravel(), 500 + 0.2 * x.ravel() - 0.12 * y.ravel()))
    levels = np.vstack((xyz, xyz * [0.8, 1.1, 1] + [16, 6, 55]))
    colors = np.repeat([[56, 182, 180], [228, 160, 94]], len(xyz), axis=0)
    directions = np.array([[-0.2, 0.12, 1], [-0.2 / 0.8, 0.12 / 1.1, 1]])
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    normals = np.repeat(directions, len(xyz), axis=0)
    return PointCloud(
        levels,
        "mm",
        "fixture_reference",
        colors=colors,
        normals=normals,
        attributes={"source_index": np.arange(100, 100 + len(levels))},
    )


def build(output):
    asset = output / "assets/reference.ply"
    saved = output / "perturbed_cloud_fixture.json"
    if asset.exists() or saved.exists():
        raise FileExistsError("Use an empty output directory to preserve the example")
    asset.parent.mkdir(parents=True, exist_ok=True)
    write_cloud(asset, reference_cloud(), binary=True)
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, name, position, **params):
        block = graph.add(registry.create(kind, block_id=key, parameters=params))
        block.name, block.position = name, position
        return block

    try:
        note = add("layout.comment", "note", "Test a known scan pose and noise", [0, -260])
        note.node_size = "large"
        note.description = (
            "Original synthetic geometry by OpenCVDataflowGUI contributors (MIT). "
            "The fixture generates a known rigid transform, drops 38 rows and adds "
            "seeded metric noise. Downstream blocks sample, recover the pose, "
            "and compare the noisy scan against the clean scan at two tolerances."
        )
        add("input.cloud", "source", "Two-surface reference", [0, 10], path="assets/reference.ply")
        add(
            "3d.cloud_perturb",
            "fixture",
            "Seeded pose and noisy scan",
            [410, 0],
            seed=7631,
            pose_mode="seeded",
            max_rotation_deg=15.0,
            translation=[25.0, 19.0, 31.0],
            target_frame="fixture_scan",
            noise_scale=[0.2, 0.15, 0.1],
            dropout_fraction=0.2,
        )
        add(
            "3d.estimate_matched_transform",
            "fit",
            "Recover the known rigid pose",
            [870, 0],
            output_units="mm",
            inlier_threshold_m=0.001,
        )
        add(
            "3d.cloud_sample",
            "sample",
            "Sample the perturbed scan",
            [870, 230],
            method="farthest",
            max_points=24,
            seed=7631,
        )
        graph.connect("source", "cloud", "fixture", "cloud")
        graph.connect("source", "cloud", "fit", "source")
        graph.connect("fixture", "cloud", "fit", "target")
        graph.connect("fixture", "pairs", "fit", "pairs")
        graph.connect("fixture", "cloud", "sample", "cloud")
        for row, key, cutoff in ((2, "wide", 0.8), (3, "tight", 0.08)):
            add(
                "3d.cloud_compare",
                key,
                f"{cutoff:g} mm tolerance",
                [870, 230 * row],
                distance_tolerance_m=cutoff / 1000,
                correspondence_radius_m=0.001,
            )
            graph.connect("fixture", "cloud", key, "measured")
            graph.connect("fixture", "clean", key, "reference")
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise ValueError(validation["issues"])
        graph.save(saved)
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
