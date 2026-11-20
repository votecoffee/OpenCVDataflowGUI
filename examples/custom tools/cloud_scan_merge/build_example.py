"""Build three registered scans with independently known target coordinates."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud

RVEC = [np.pi / 2, 0, 0]
TRANSLATION_MM = [10, -40, 400]


def fixtures():
    first = [[0, 0, 400], [10, 0, 400], [0, 10, 400], [10, 10, 400]]
    second = np.array([[0.2, 0.1, 400], [10.2, 0, 400], [20, 0, 400], [20, 10, 400]])
    third = np.array([[20.2, 0.1, 400], [30, 0, 405], [30.1, 0.1, 405]])
    rotation, _ = cv2.Rodrigues(np.array(RVEC))
    return (
        PointCloud(
            first,
            "mm",
            "fixture",
            colors=[[75, 170, 235]] * 4,
            normals=[[0, 0, 1]] * 4,
        ),
        PointCloud(
            ((second - TRANSLATION_MM) / 1000) @ rotation,
            "m",
            "side_camera",
            colors=[[80, 235, 130]] * 4,
            normals=np.tile([0, 0.6, 0.8], (4, 1)) @ rotation,
        ),
        PointCloud(
            third / 1000,
            "m",
            "fixture",
            colors=[[235, 180, 65]] * 3,
            normals=[[0, 0, 1]] * 3,
        ),
    )


def build(output):
    project = output / "cloud_scan_merge.json"
    assets = output / "assets"
    if assets.exists() or project.exists():
        raise FileExistsError("Choose an empty output directory to preserve the example")
    assets.mkdir(parents=True)
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = position
        return block

    try:
        note = add("layout.comment", "notes", [0, -285])
        note.name, note.node_size = "Merge three scans in fixture coordinates", "large"
        note.description = (
            "Original synthetic fixture by OpenCVDataflowGUI contributors (MIT). "
            "The side scan uses a millimeter pose and meter point coordinates. "
            "Its green points and the gold third scan extend the blue reference."
        )
        for index, cloud in enumerate(fixtures(), start=1):
            key = f"scan_{index}"
            write_cloud(assets / f"{key}.ply", cloud, binary=True)
            source = add("input.cloud", key, [0, (index - 1) * 255], path=f"assets/{key}.ply")
            source.name = ("Load reference", "Load side view", "Load extension")[index - 1]
            source.ticket_settings = {"mode": "fixed", "value": "registered-scan-fixture"}
        pose = add(
            "3d.pose",
            "side_pose",
            [0, 765],
            source_frame="side_camera",
            target_frame="fixture",
            rvec=RVEC,
            translation=TRANSLATION_MM,
            units="mm",
        )
        pose.name = "Side camera to fixture"
        for index, (key, name, mode) in enumerate(
            (
                ("append", "Append all scan rows", "append"),
                ("merge", "Suppress cross-scan overlap", "prefer_earlier"),
            )
        ):
            node = add(
                "3d.cloud_merge",
                key,
                [510, 70 + index * 330],
                target_frame="fixture",
                output_units="mm",
                merge_mode=mode,
                overlap_radius_m=0.0005,
                attribute_policy="require_matching",
            )
            node.name = name
            node.exposed_inputs = ["cloud_1", "cloud_2", "pose_2", "cloud_3"]
            for slot in range(1, 4):
                graph.connect(f"scan_{slot}", "cloud", key, f"cloud_{slot}")
            graph.connect(pose.id, "transform", key, "pose_2")
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
