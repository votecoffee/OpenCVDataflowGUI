"""Compare LATCH, BEBLID, and TEBLID on a known image transform."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

DISTANCE_LIMIT = 48
FAMILIES = ("latch", "beblid", "teblid")


def images():
    height, width = 256, 336
    rng = np.random.default_rng(372)
    query = cv2.GaussianBlur(rng.integers(14, 232, (height, width), np.uint8), (5, 5), 0.8)
    for _ in range(64):
        start = tuple(map(int, rng.integers([22, 22], [width - 24, height - 24])))
        end = (start[0] + int(rng.integers(3, 19)), start[1] + int(rng.integers(3, 19)))
        cv2.rectangle(query, start, end, int(rng.integers(20, 225)), -1)
    affine = cv2.getRotationMatrix2D((width / 2, height / 2), 7, 1.04)
    affine[:, 2] += (11, -9)
    train = cv2.add(cv2.warpAffine(query, affine, (width, height), borderValue=84), 9)
    return {"query": query, "train": train}, np.vstack((affine, [0, 0, 1])).tolist()


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    pair, transform = images()
    for name, image in pair.items():
        ok, png = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode {name} image")
        (assets / f"{name}.png").write_bytes(png.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Compare binary patch descriptors", 0, -238)
        note.node_size = "large"
        note.description = (
            "Run Once. All three extractors share the ORB detections and use 256-bit descriptors. "
            "Evaluate their Hamming matches against a 7-degree rotation, 4% enlargement and shift. "
            "Both images are original MIT assets."
        )
        for role, top in (("query", 30), ("train", 665)):
            source = add(
                "input.file",
                f"{role}_image",
                f"{role.title()} view",
                0,
                top,
                path=f"assets/{role}.png",
                color_format="gray",
                path_base="project",
            )
            detector = add(
                "inspect.orb",
                f"{role}_orb",
                f"{role.title()} keypoints",
                245,
                top,
                max_features=420,
                fast_threshold=8,
            )
            detector.exposed_outputs = ("features",)
            graph.connect(source.id, "image", detector.id, "image")
            for index, family in enumerate(FAMILIES):
                settings = (
                    {"descriptor_bytes": "32"} if family == "latch" else {"descriptor_bits": "256"}
                )
                block = add(
                    f"inspect.{family}_descriptor",
                    f"{family}_{role}",
                    f"{role.title()} {family.upper()}",
                    492,
                    top + index * 190,
                    **settings,
                )
                block.exposed_outputs = ("image", "features", "count")
                graph.connect(source.id, "image", block.id, "image")
                graph.connect(detector.id, "features", block.id, "keypoints")

        for index, family in enumerate(FAMILIES):
            row = 90 + index * 265
            match = add(
                "inspect.match_descriptor_radius",
                f"{family}_match",
                f"Match {family.upper()}",
                790,
                row,
                max_distance=DISTANCE_LIMIT,
                max_neighbors=1,
            )
            evaluate = add(
                "inspect.evaluate_features",
                f"{family}_evaluate",
                f"Evaluate {family.upper()}",
                1120,
                row,
                reference_transform=transform,
                reprojection_tolerance=2.5,
                match_threshold=DISTANCE_LIMIT,
                minimum_correspondences=70,
                minimum_precision=0.8,
                maximum_graphic_matches=100,
            )
            for role in ("query", "train"):
                graph.connect(f"{family}_{role}", "features", match.id, role)
                graph.connect(f"{family}_{role}", "features", evaluate.id, role)
                graph.connect(f"{family}_{role}", "image", evaluate.id, f"{role}_image")
            graph.connect(match.id, "matches", evaluate.id, "matches")
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise RuntimeError(validation["issues"])
        graph.save(output / "binary_patch_descriptors.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
