"""Create a RIC example with a known motion seam, outliers and a coverage gap."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

ASSET = Path(__file__).resolve().parents[1] / "sparse_to_dense_flow/assets/00_reference.png"


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    source = cv2.imread(str(ASSET), cv2.IMREAD_COLOR)
    if source is None:
        raise FileNotFoundError(ASSET)
    height, width = source.shape[:2]
    y, x = np.indices((height, width), dtype=np.float32)
    drift = np.where(x < 88, 7, -6).astype(np.float32)
    slope = np.float32(0.012)
    vertical = drift + (y - height / 2) * slope
    target = cv2.remap(source, x, (y - drift + height / 2 * slope) / (1 + slope), cv2.INTER_LINEAR)
    for name, image in (("query", source), ("train", target)):
        ok, data = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode {name}")
        (assets / f"{name}.png").write_bytes(data.tobytes())

    xx, yy = np.meshgrid(np.arange(8, width, 8), np.arange(12, height - 12, 8))
    points = np.stack((xx, yy), axis=-1).reshape(-1, 2)
    points = points[~((points[:, 0] > 130) & (points[:, 1] < 52))]
    matches = points.astype(np.float32)
    matches[:, 1] += vertical[points[:, 1], points[:, 0]]
    outliers = np.arange(5, len(matches), 17)
    matches[outliers, 0] = np.clip(matches[outliers, 0] + 11, 0, width - 1)

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Fit motion across two moving panels", 0, -225)
        note.node_size = "large"
        note.description = (
            "Run Once. RIC fits 206 matches including 12 outliers across a 13-pixel motion seam. "
            "The upper-right gap has no nearby matches. Inspect the support mask and how it "
            "limits the warped image. Original MIT fixture."
        )
        for role, y in (("query", 0), ("train", 220)):
            add(
                "input.file", role, f"{role.title()} image", 0, y,
                path=f"assets/{role}.png", path_base="project", color_format="bgr",
            )
        add(
            "cv.ric_flow", "ric", "RIC motion with support radius", 310, 85,
            source_points=points.tolist(), target_points=matches.tolist(),
            superpixel_neighbors=48, post_smoothing=False, support_radius=13.0,
        )
        warp = add("cv.warp_flow", "warp", "Warp only supported pixels", 630, 120)
        warp.exposed_inputs = ["image", "flow", "flow_support", "target"]
        graph.connect("query", "image", "ric", "image")
        graph.connect("train", "image", "ric", "target")
        graph.connect("train", "image", "warp", "image")
        graph.connect("query", "image", "warp", "target")
        graph.connect("ric", "flow", "warp", "flow")
        graph.connect("ric", "support", "warp", "flow_support")
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise RuntimeError(validation["issues"])
        graph.save(output / "ric_correspondences.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
