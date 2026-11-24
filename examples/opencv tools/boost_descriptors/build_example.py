"""Build an image pair with known viewpoint and illumination changes for BoostDesc."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

MODES = (("binboost", "BINBOOST_256", 48), ("bgm", "BGM_BILINEAR", 48), ("lbgm", "LBGM", 1.3))


def make_images():
    height, width = 252, 332
    rng = np.random.default_rng(741)
    query = cv2.GaussianBlur(rng.integers(10, 240, (height, width), np.uint8), (5, 5), 0.75)
    for index in range(52):
        x, y = map(int, rng.integers([20, 20], [width - 25, height - 25]))
        intensity = int(rng.integers(25, 232))
        if index % 2:
            cv2.circle(query, (x, y), int(rng.integers(3, 10)), intensity, -1)
        else:
            cv2.rectangle(query, (x, y), (x + 13, y + 7), intensity, -1)
    affine = cv2.getRotationMatrix2D((width / 2, height / 2), 9, 1.065)
    affine[:, 2] += (10, -8)
    warped = cv2.warpAffine(query, affine, (width, height), borderMode=cv2.BORDER_REFLECT101)
    train = np.rint(0.76 * warped.astype(np.float32) + 32).astype(np.uint8)
    return {"query": query, "train": train}, np.vstack((affine, [0, 0, 1])).tolist()


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    images, transform = make_images()
    for role, image in images.items():
        ok, data = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode the {role} image")
        (assets / f"{role}.png").write_bytes(data.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, position, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, list(position)
        return block

    try:
        note = add("layout.comment", "note", "Compare BoostDesc variants", (0, -225))
        note.node_size = "large"
        note.description = (
            "Run Once. Each variant uses the same SIFT detections on a rotated, scaled and "
            "relit image pair. The evaluation blocks compare their matches to the known mapping. "
            "Open each evaluation's overlay to review the accepted and rejected pairs. "
            "Original generated images under MIT."
        )
        for role, top in (("query", 30), ("train", 680)):
            source = add(
                "input.file", f"{role}_image", f"{role.title()} image", (0, top),
                path=f"assets/{role}.png", path_base="project", color_format="gray",
            )
            detector = add(
                "inspect.sift", f"{role}_sift", f"{role.title()} SIFT", (230, top),
                max_features=350,
            )
            detector.exposed_outputs = ("features",)
            graph.connect(source.id, "image", detector.id, "image")
            for index, (key, mode, _) in enumerate(MODES):
                block = add(
                    "inspect.boost_descriptor", f"{key}_{role}", f"{role.title()} {mode}",
                    (470, top + 185 * index), variant=mode, scale_factor=6.75,
                )
                block.exposed_outputs = ("image", "features", "count")
                graph.connect(source.id, "image", block.id, "image")
                graph.connect(detector.id, "features", block.id, "keypoints")
        for index, (key, mode, limit) in enumerate(MODES):
            top = 80 + 290 * index
            match = add(
                "inspect.match_descriptor_radius", f"{key}_match", f"Match {mode}", (780, top),
                max_distance=limit, max_neighbors=1,
            )
            evaluate = add(
                "inspect.evaluate_features", f"{key}_evaluate", f"Review {mode}", (1090, top),
                reference_transform=transform, reprojection_tolerance=2.5, match_threshold=limit,
                minimum_correspondences=100, minimum_precision=0.9, maximum_graphic_matches=120,
            )
            for role in ("query", "train"):
                graph.connect(f"{key}_{role}", "features", match.id, role)
                graph.connect(f"{key}_{role}", "features", evaluate.id, role)
                graph.connect(f"{key}_{role}", "image", evaluate.id, f"{role}_image")
            graph.connect(match.id, "matches", evaluate.id, "matches")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise RuntimeError(checked["issues"])
        graph.save(output / "boost_descriptors.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
