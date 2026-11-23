"""Build a matched DAISY lattice on a translated synthetic texture."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

TRANSLATION = [[1, 0, 8], [0, 1, 6], [0, 0, 1]]
DISTANCE_LIMIT = 0.03


def images():
    rng = np.random.default_rng(42530)
    noise = rng.integers(15, 230, (152, 184), dtype=np.uint8)
    query = cv2.GaussianBlur(noise, (0, 0), 1.1)
    cv2.rectangle(query, (41, 40), (68, 66), 45, 2)
    cv2.circle(query, (112, 80), 16, 195, 2)
    cv2.line(query, (69, 121), (138, 108), 200, 2)
    train = cv2.warpAffine(
        query,
        np.asarray(TRANSLATION[:2], np.float64),
        query.shape[::-1],
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return {"query": query, "train": train + 8}


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
        note = add("layout.comment", "note", "Measure a translated texture", 0, -250)
        note.node_size = "large"
        note.description = (
            "DAISY measures local gradients on two 16-pixel grids. The train grid is shifted "
            "8 pixels right and 6 down. The matcher pairs rows by L2 distance, and the evaluator "
            "checks every match against the known translation. Both images are original MIT assets."
        )
        match = add(
            "inspect.match_descriptor_radius",
            "match",
            "Find nearest DAISY matches",
            560,
            0,
            max_neighbors=1,
            max_distance=DISTANCE_LIMIT,
        )
        evaluate = add(
            "inspect.evaluate_features",
            "evaluate",
            "Verify against translation",
            850,
            0,
            reference_transform=TRANSLATION,
            reprojection_tolerance=0.25,
            match_threshold=DISTANCE_LIMIT,
            minimum_correspondences=60,
            minimum_precision=0.95,
            minimum_recall=0.95,
        )
        for side, y, origin in (("query", 0, [32, 32]), ("train", 280, [40, 38])):
            source = add(
                "input.file",
                f"{side}_image",
                f"{side.title()} image",
                0,
                y,
                path=f"assets/{side}.png",
                path_base="project",
                color_format="gray",
            )
            descriptor = add(
                "inspect.daisy_descriptor",
                side,
                f"{side.title()} DAISY grid",
                280,
                y,
                radius=8.0,
                grid_step=16,
                grid_origin=origin,
                normalization="full",
            )
            descriptor.exposed_outputs = ("image", "features", "count")
            graph.connect(source.id, "image", descriptor.id, "image")
            graph.connect(descriptor.id, "features", match.id, side)
            graph.connect(descriptor.id, "features", evaluate.id, side)
            graph.connect(source.id, "image", evaluate.id, f"{side}_image")
        graph.connect(match.id, "matches", evaluate.id, "matches")
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise RuntimeError(validation["issues"])
        graph.save(output / "daisy_descriptors.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
