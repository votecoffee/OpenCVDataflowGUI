"""Build two synthetic edge-rich views with a known camera motion."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def images():
    query = np.full((258, 352), 26, np.uint8)
    outlines = (
        ([(42, 38), (127, 43), (118, 101), (52, 112)], 194),
        ([(182, 46), (245, 27), (277, 92), (220, 116), (176, 81)], 135),
        ([(52, 160), (153, 143), (162, 184), (116, 224), (42, 204)], 158),
        ([(209, 163), (287, 150), (310, 211), (234, 223)], 220),
    )
    for points, intensity in outlines:
        cv2.fillPoly(query, [np.array(points, np.int32)], intensity)
    accents = (
        ((66, 59), (100, 63), 90, 2),
        ((220, 55), (245, 76), 195, 2),
        ((56, 198), (117, 210), 45, 2),
        ((236, 174), (291, 191), 153, 3),
    )
    for start, end, intensity, width in accents:
        cv2.line(query, start, end, intensity, width)
    cv2.circle(query, (167, 125), 8, 90, 1)
    transform = cv2.getRotationMatrix2D((176, 129), 3, 1)
    transform[:, 2] += (16, -9)
    train = cv2.warpAffine(query, transform, (352, 258), borderValue=26)
    return query, train


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for key, image in zip(("query", "train"), images(), strict=True):
        ok, png = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode {key} line image")
        (assets / f"{key}.png").write_bytes(png.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Match part edges across two views", 0, -210)
        note.node_size = "large"
        note.description = (
            "Run Once. The second image is rotated 3 degrees and translated by (16, -9) pixels. "
            "Review matching segments and their midpoint links side by side. Original MIT assets."
        )
        for role, pos_y in (("query", 40), ("train", 280)):
            add(
                "input.file",
                role,
                f"{role.title()} view",
                0,
                pos_y,
                path=f"assets/{role}.png",
                color_format="gray",
                path_base="project",
            )
            add(
                "inspect.line_descriptors",
                f"{role}_lines",
                f"Describe {role} lines",
                265,
                pos_y,
                min_length=26,
                max_features=96,
            )
            graph.connect(role, "image", f"{role}_lines", "image")
        add(
            "inspect.match_line_descriptors",
            "match",
            "Match described lines",
            545,
            125,
            min_matches=12,
        )
        add("inspect.review_matches", "review", "Review line pairs", 820, 125, endpoints=False)
        for role in ("query", "train"):
            graph.connect(f"{role}_lines", "features", "match", role)
            graph.connect(f"{role}_lines", "image", "review", f"{role}_image")
        graph.connect("match", "matches", "review", "matches")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise RuntimeError(checked["issues"])
        graph.save(output / "line_descriptors.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
