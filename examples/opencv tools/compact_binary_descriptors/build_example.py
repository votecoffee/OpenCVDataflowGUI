"""Generate a registered image pair and compare BRIEF and FREAK matches."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

QUERY_TO_TRAIN = [[1, 0, 9], [0, 1, 7], [0, 0, 1]]


def images():
    rng = np.random.default_rng(41751)
    query = np.full((208, 256), 42, np.uint8)
    for n in range(55):
        x, y = rng.integers([28, 26], [215, 164])
        w, h = rng.integers(6, 18, 2)
        shade = int(rng.integers(95, 245))
        cv2.rectangle(query, (x, y), (x + w, y + h), shade, -1)
        cv2.line(query, (x + 1, y + h), (x + w, y + 1), 56 + n, 1)
        if n % 3 == 0:
            cv2.circle(query, (x + w + 4, y + 2), 2, 235, -1)
    translated = cv2.warpAffine(
        query,
        np.asarray(QUERY_TO_TRAIN[:2], np.float64),
        query.shape[::-1],
        flags=cv2.INTER_NEAREST,
        borderValue=42,
    )
    train = np.rint(translated * 0.92 + 8).astype(np.uint8)
    return {"query": query, "train": train}


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name, image in images().items():
        ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode {name}")
        (assets / f"{name}.png").write_bytes(encoded.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(type_id, key, name, x, y, **parameters):
        block = graph.add(registry.create(type_id, block_id=key, parameters=parameters))
        block.name, block.position = name, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Compare binary descriptors", 0, -285)
        note.node_size = "large"
        note.description = (
            "ORB supplies keypoints to BRIEF and FREAK. Hamming matching pairs the retained "
            "features. The evaluator scores the matches against the known 9 px right and "
            "7 px down translation. Both synthetic images are original MIT-licensed assets."
        )
        for side, top in (("query", 0), ("train", 490)):
            add(
                "input.file",
                f"{side}_image",
                f"{side.title()} image",
                0,
                top,
                path=f"assets/{side}.png",
                path_base="project",
                color_format="gray",
            )
            detector = add(
                "inspect.orb",
                f"{side}_orb",
                f"{side.title()} ORB keypoints",
                255,
                top,
                max_features=240,
                levels=1,
                edge_threshold=18,
                fast_threshold=10,
            )
            detector.exposed_outputs = ("features",)
            graph.connect(f"{side}_image", "image", detector.id, "image")
            for method, offset, settings in (
                ("brief", 0, {}),
                ("freak", 200, {"pattern_scale": 9.0}),
            ):
                node = add(
                    f"inspect.{method}_descriptor",
                    f"{method}_{side}",
                    f"{side.title()} {method.upper()}",
                    520,
                    top + offset,
                    **settings,
                )
                node.exposed_outputs = ("image", "features", "count")
                graph.connect(detector.id, "features", node.id, "keypoints")
                graph.connect(f"{side}_image", "image", node.id, "image")

        for method, y in (("brief", 0), ("freak", 450)):
            matcher = add(
                "inspect.match_descriptor_radius",
                f"{method}_match",
                f"Match {method.upper()}",
                805,
                y,
                max_neighbors=1,
                max_distance=36.0,
            )
            evaluation = add(
                "inspect.evaluate_features",
                f"{method}_evaluate",
                f"Verify {method.upper()}",
                1100,
                y,
                reference_transform=QUERY_TO_TRAIN,
                reprojection_tolerance=1.0,
                match_threshold=36,
                minimum_correspondences=25,
                minimum_precision=0.9,
                minimum_recall=0.8,
            )
            for side in ("query", "train"):
                graph.connect(f"{method}_{side}", "features", matcher.id, side)
                graph.connect(f"{method}_{side}", "features", evaluation.id, side)
                graph.connect(f"{side}_image", "image", evaluation.id, f"{side}_image")
            graph.connect(matcher.id, "matches", evaluation.id, "matches")
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise RuntimeError(validation["issues"])
        graph.save(output / "compact_binary_descriptors.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
