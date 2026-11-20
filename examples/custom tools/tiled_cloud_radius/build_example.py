"""Build a seam-crossing PLY example or a large analytic grid."""

import argparse
from pathlib import Path

import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud


def write_grid(path, shape):
    nx, ny, nz = shape
    count = nx * ny * nz
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        'comment dataflow_metadata {"units": "m", "frame": "grid"}\n'
        f"element vertex {count + 1}\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    )
    with path.open("xb") as output:
        output.write(header.encode("ascii"))
        for first in range(0, count, 8192):
            index = np.arange(first, min(first + 8192, count))
            x = index % nx - nx // 2
            y = index // nx % ny - ny // 2
            z = index // (nx * ny)
            output.write((np.column_stack((x, y, z)) / 128).astype("<f4").tobytes())
        output.write(np.array([[3, 3, 3]], dtype="<f4").tobytes())


def build(output, grid=None):
    path = output / "assets/scan.ply"
    project = output / "tiled_cloud_radius.json"
    if path.exists() or project.exists():
        raise FileExistsError("Build to a new directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    if grid is None:
        # Distances of 125 mm straddle both negative and positive tile boundaries.
        vertices = [
            [125, 0, 0],
            [-125, 0, 0],
            [825, 100, 0],
            [250, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [-250, 0, 0],
            [-850, 300, 0],
        ]
        colors = np.tile([44, 173, 206], (len(vertices), 1))
        for i in (2, 7):
            colors[i] = [245, 96, 20]
        ids = np.array([31, 9, 44, 89, 12, 14, 22, 17], np.int32)
        write_cloud(
            path,
            PointCloud(
                vertices, "mm", "inspection", colors=colors, attributes={"source_index": ids}
            ),
            binary=True,
        )
        settings = {"tile_size_m": 0.25, "radius_m": 0.125, "minimum_neighbors": 1}
        note_text = (
            "Original synthetic scan by OpenCVDataflowGUI contributors (MIT). "
            "The radius crosses tile boundaries, includes points exactly 125 mm away, "
            "and counts the two coincident rows separately. Two isolated rows are rejected."
        )
    else:
        write_grid(path, grid)
        settings = {
            "tile_size_m": 1 / 16,
            "radius_m": 1 / 128,
            "minimum_neighbors": 4,
            "max_window_points": 14_000,
        }
        note_text = (
            "An original regular grid with cubic 1/128 m spacing and one isolated "
            "point. Radius counts are the axial grid neighbors. The eight corners "
            "and isolated point fail the four-neighbor requirement."
        )

    registry, graph = builtin_registry(), Graph()
    try:
        note = graph.add(registry.create("layout.comment", block_id="note"))
        note.name, note.description, note.node_size = (
            "Filter a scan with tile halos",
            note_text,
            "large",
        )
        note.position = [0, -250]
        block = graph.add(
            registry.create(
                "input.cloud_tiled_radius",
                block_id="filter",
                parameters={"path": "assets/scan.ply", **settings},
            )
        )
        block.position = [0, 0]
        report = validate_graph(graph)
        if not report["valid"]:
            raise ValueError(report["issues"])
        graph.save(project)
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--grid",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        help="Generate a regular grid with these three dimensions",
    )
    args = parser.parse_args()
    if args.grid is not None and (
        min(args.grid) < 3 or np.prod(args.grid, dtype=object) >= 4_000_000
    ):
        parser.error("Grid needs three dimensions of at least 3 and fewer than 4,000,000 rows")
    build(args.output, args.grid)
