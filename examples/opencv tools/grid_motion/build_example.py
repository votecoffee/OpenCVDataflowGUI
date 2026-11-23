"""Build a controlled registration pair with occlusion and a known transform."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def images():
    height, width = 440, 600
    rng = np.random.default_rng(842)
    query = cv2.GaussianBlur(rng.integers(0, 256, (height, width), np.uint8), (7, 7), 1.1)
    for _ in range(135):
        x, y = int(rng.integers(28, width - 36)), int(rng.integers(28, height - 36))
        shade = int(rng.integers(20, 240))
        extent = (x + int(rng.integers(4, 18)), y + int(rng.integers(4, 18)))
        cv2.rectangle(query, (x, y), extent, shade, -1)
    transform = cv2.getRotationMatrix2D((width / 2, height / 2), 7, 0.92)
    transform[:, 2] += (19, -13)
    train = cv2.warpAffine(query, transform, (width, height), borderValue=105)
    train[22:272, 28:253] = rng.integers(0, 256, (250, 225), np.uint8)
    train[170:406, 347:524] = rng.integers(0, 256, (236, 177), np.uint8)
    return query, train


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for key, image in zip(("query", "train"), images(), strict=True):
        ok, png = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode {key} registration image")
        (assets / f"{key}.png").write_bytes(png.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Filter outliers before fitting motion", 0, -235)
        note.node_size = "large"
        note.description = (
            "Run Once. The train view has a known rotation, scale, translation and two "
            "occluders. Compare the raw and filtered pairs and their affine fits. "
            "Both fits use the same 160-iteration budget. Original MIT assets."
        )
        for role, pos_y in (("query", 35), ("train", 290)):
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
                "inspect.orb",
                f"{role}_orb",
                f"Find {role} features",
                265,
                pos_y,
                max_features=4200,
                fast_threshold=0,
            )
            graph.connect(role, "image", f"{role}_orb", "image")
        add(
            "inspect.match_orb",
            "raw",
            "Nearest ORB candidates",
            535,
            125,
            mode="nearest",
            max_distance=256,
        )
        graph.connect("query_orb", "features", "raw", "query")
        graph.connect("train_orb", "features", "raw", "train")
        add(
            "inspect.filter_matches_gms",
            "filter",
            "Keep consistent grid motion",
            810,
            280,
            with_rotation=True,
            with_scale=True,
            min_matches=50,
        )
        graph.connect("raw", "matches", "filter", "matches")
        for branch, matches_id, pos_y in (("raw", "raw", 35), ("filtered", "filter", 300)):
            fit, review = f"{branch}_fit", f"{branch}_review"
            add(
                "inspect.affine",
                fit,
                f"Fit {branch} pairs",
                1085,
                pos_y,
                model="partial",
                max_iterations=160,
                min_inlier_ratio=0.0,
            )
            add(
                "inspect.review_matches",
                review,
                f"Review {branch} pairs",
                1360,
                pos_y,
                max_pairs=80,
                endpoints=False,
            )
            graph.connect(matches_id, "matches", fit, "matches")
            graph.connect(matches_id, "matches", review, "matches")
            graph.connect(fit, "transform", review, "fit")
            for role in ("query", "train"):
                graph.connect(role, "image", review, f"{role}_image")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise RuntimeError(checked["issues"])
        graph.save(output / "grid_motion.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
