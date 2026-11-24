"""Build an interpolation example with analytic sparse motion on two panels."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

ASSET = Path(__file__).resolve().parents[1] / "sparse_to_dense_flow/assets/00_reference.png"
SEAM = 88
SLOPE = np.float32(0.003)


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    source = cv2.imread(str(ASSET), cv2.IMREAD_COLOR)
    if source is None:
        raise FileNotFoundError(ASSET)
    height, width = source.shape[:2]
    xx, yy = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    shift = np.where(xx < SEAM, np.float32(4), np.float32(-3))
    donor_y = (yy - shift + (height / 2) * SLOPE) / (1 + SLOPE)
    target = cv2.remap(source, xx, donor_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    for name, image in (("source.png", source), ("target.png", target)):
        ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode {name}")
        (assets / name).write_bytes(encoded.tobytes())

    x, y = np.meshgrid(np.arange(8, width, 8), np.arange(8, height - 8, 8))
    source_points = np.stack((x, y), axis=-1).reshape(-1, 2)
    target_points = source_points.astype(np.float64)
    shifts = np.where(source_points[:, 0] < SEAM, 4, -3)
    target_points[:, 1] += shifts + 0.003 * (source_points[:, 1] - height / 2)
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Interpolate across a motion seam", 0, -245)
        note.node_size = "large"
        note.description = (
            "Run Once. The block starts with 252 supplied correspondences. "
            "The dark and light panels move in opposite directions, with a small vertical "
            "stretch. Check the flow preview around the center seam. Original MIT fixture."
        )
        for key, title, y in (("source", "Query image", 0), ("target", "Train image", 240)):
            add(
                "input.file",
                key,
                title,
                0,
                y,
                path=f"assets/{key}.png",
                path_base="project",
                color_format="bgr",
            )
        add(
            "cv.edge_aware_flow",
            "interpolate",
            "Interpolate supplied matches",
            315,
            90,
            source_points=source_points.tolist(),
            target_points=target_points.round(6).tolist(),
            neighbors=96,
        )
        graph.connect("source", "image", "interpolate", "image")
        graph.connect("target", "image", "interpolate", "target")
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise RuntimeError(validation["issues"])
        graph.save(output / "edge_aware_matches.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
