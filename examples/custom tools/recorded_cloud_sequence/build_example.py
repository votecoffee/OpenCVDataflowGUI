"""Build the mixed-format recorded point-cloud example with original data."""

import argparse
import json
from pathlib import Path

import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.cloud_recording import MANIFEST_FORMAT
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud

EPOCH = 1_900_000_000_000_000_000


def build(output):
    project = output / "recorded_cloud_sequence.json"
    manifest = output / "assets/recording.json"
    names = ["z_entrance.ply", "middle.pcd", "a_review.obj"]
    if any(path.exists() for path in (project, manifest, *(manifest.parent / n for n in names))):
        raise FileExistsError("Choose a new directory for the generated example")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.array([[0, 0, 0], [0.05, 0, 0], [0.05, 0.02, 0], [0, 0.02, 0], [0.02, 0.01, 0.01]])
    rgb = np.array([[32, 172, 200], [32, 172, 200], [253, 182, 70], [253, 182, 70], [226, 91, 52]])
    scans = [
        PointCloud(xyz * 1000, "mm", "conveyor", colors=rgb),
        PointCloud(
            xyz + [0.03, 0.0, 0.0],
            "m",
            "conveyor",
            colors=rgb,
            attributes={"source_index": [21, 16, 8, 31, 13]},
        ),
        PointCloud(xyz + [0.01, 0.02, 0.25], "m", "tool_camera"),
    ]
    records = []
    for name, offset, cloud in zip(names, (3, 45_000_007, 105_000_017), scans, strict=True):
        write_cloud(manifest.parent / name, cloud, binary=not name.endswith(".obj"))
        records.append(
            {
                "path": name,
                "timestamp_ns": EPOCH + offset,
                "frame": cloud.frame,
                "units": cloud.units,
            }
        )
    manifest.write_text(
        json.dumps(
            {"format": MANIFEST_FORMAT, "clock": "scanner_clock", "scans": records}, indent=2
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = position
        return block

    try:
        note = add("layout.comment", "note", [0, -240])
        note.name, note.node_size = "Replay and group recorded scans", "large"
        note.description = (
            "Original synthetic point clouds by OpenCVDataflowGUI contributors (MIT). "
            "Run All replays the manifest order across PLY, PCD and OBJ. Batch collects "
            "three scan packets. Unbatch restores each item with its own capture "
            "timestamp, ticket, units and frame."
        )
        add("input.cloud_recording", "replay", [0, 10], path="assets/recording.json")
        batch = add("flow.batch", "batch", [400, 10], size=3, max_pending=4, max_bytes_mb=1)
        batch.name = "Collect all three scans"
        unbatch = add("flow.unbatch", "unbatch", [800, 10])
        unbatch.name = "Replay collected scans in order"
        graph.connect("replay", "cloud", "batch", "item_1")
        graph.connect("batch", "group", "unbatch", "group")
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
