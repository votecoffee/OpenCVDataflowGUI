"""Build a Harris–Laplace repeatability example with a doubled image."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

QUERY_TO_TRAIN = [[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]]


def target():
    image = np.full((236, 320), 37, np.uint8)
    rng = np.random.default_rng(372)
    for row in range(3):
        for col in range(4):
            x, y = 28 + col * 72, 25 + row * 69
            width, height = map(int, rng.integers(18, 40, 2))
            shade = 98 + ((row * 4 + col) * 11) % 146
            if (row + col) % 2:
                corners = [
                    (x, y),
                    (x + width, y + 5),
                    (x + width - 4, y + height),
                    (x + 1, y + height - 5),
                ]
                cv2.fillPoly(image, [np.array(corners, np.int32)], shade)
            else:
                cv2.rectangle(image, (x, y), (x + width, y + height), shade, -1)
            cv2.circle(image, (x + 6, y + 5), int(rng.integers(2, 5)), 44 + 10 * col, -1)
            cv2.line(
                image, (x + 5, y + height - 9), (x + width - 6, y + 10), shade // 2, 1 + row % 2
            )
    return cv2.GaussianBlur(image, (0, 0), 0.85)


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    valid, png = cv2.imencode(".png", target(), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not valid:
        raise OSError("Could not encode the corner target")
    (assets / "corners.png").write_bytes(png.tobytes())
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Track interest points through a size change", 0, -220)
        note.node_size = "large"
        note.description = (
            "Run Once. Find Harris–Laplace points at the original scale and at 2×. Each circle "
            "shows a detected diameter. BEBLID describes those points before evaluation against "
            "the known resize transform. The image is an original MIT asset."
        )
        add(
            "input.file",
            "source",
            "Corner target",
            0,
            80,
            path="assets/corners.png",
            color_format="gray",
            path_base="project",
        )
        add(
            "cv.resize",
            "enlarge",
            "Double image size",
            235,
            310,
            size_mode="scale",
            scale_x=2,
            scale_y=2,
        )
        graph.connect("source", "image", "enlarge", "image")
        add(
            "inspect.evaluate_features",
            "evaluate",
            "Measure 2× repeatability",
            1010,
            125,
            reference_transform=QUERY_TO_TRAIN,
            reprojection_tolerance=2.5,
            match_threshold=20,
            minimum_correspondences=140,
            minimum_repeatability=0.85,
            maximum_graphic_matches=120,
        )
        for role, key, y in (("query", "source", 40), ("train", "enlarge", 310)):
            detect = add(
                "inspect.harris_laplace",
                f"{role}_corners",
                f"{role.title()} Harris–Laplace points",
                490,
                y,
            )
            describe = add(
                "inspect.beblid_descriptor",
                f"{role}_descriptor",
                f"Describe {role} points",
                750,
                y,
                descriptor_bits="256",
                scale_factor=5.0,
            )
            detect.exposed_outputs = ("image", "keypoints")
            describe.exposed_outputs = ("features",)
            graph.connect(key, "image", detect.id, "image")
            graph.connect(detect.id, "image", describe.id, "image")
            graph.connect(detect.id, "keypoints", describe.id, "keypoints")
            graph.connect(detect.id, "image", "evaluate", f"{role}_image")
            graph.connect(describe.id, "features", "evaluate", role)
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise RuntimeError(validation["issues"])
        graph.save(output / "harris_laplace.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
