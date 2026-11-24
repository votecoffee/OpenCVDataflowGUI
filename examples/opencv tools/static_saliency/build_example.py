"""Build a controlled saliency comparison with a known part silhouette."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def fixture():
    height, width = 232, 312
    yy, xx = np.indices((height, width))
    base = np.rint(78 + 5 * np.cos(xx / 19) + 5 * np.sin(yy / 27)).astype(np.uint8)
    image = np.dstack((base + 6, base + 2, base))
    cv2.rectangle(image, (34, 45), (89, 127), (95, 96, 103), -1)
    cv2.circle(image, (251, 172), 23, (110, 103, 100), -1)
    cv2.line(image, (24, 195), (145, 191), (96, 95, 91), 2)
    truth = np.zeros((height, width), np.uint8)
    vertices = np.array(
        [
            [141, 79],
            [181, 79],
            [181, 89],
            [194, 89],
            [194, 120],
            [184, 120],
            [184, 138],
            [141, 138],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(truth, [vertices], 255)
    image[truth > 0] = (189, 198, 220)
    cv2.circle(image, (165, 107), 6, (117, 138, 158), -1)
    return image, truth


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name, image in zip(("scene", "truth"), fixture(), strict=True):
        valid, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not valid:
            raise OSError(f"Could not encode {name} image")
        (assets / f"{name}.png").write_bytes(encoded.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Compare visually prominent regions", 0, -215)
        note.node_size = "large"
        note.description = (
            "Run Once. Compare the two saliency responses and their masks at the same "
            "94th-percentile cutoff. The results show the selected outlines and the peak. "
            "The scene and reference silhouette are original MIT assets."
        )
        add(
            "input.file",
            "scene",
            "Inspection scene",
            0,
            40,
            path="assets/scene.png",
            path_base="project",
        )
        add(
            "input.file",
            "truth",
            "Known part silhouette",
            0,
            355,
            path="assets/truth.png",
            path_base="project",
            color_format="gray",
        )
        for row, (name, method) in enumerate(
            (("spectral", "spectral residual"), ("fine", "fine-grained"))
        ):
            y = row * 245 + 45
            tool = add(
                "cv.static_saliency",
                name,
                method.capitalize(),
                300,
                y,
                method=method,
                threshold_mode="percentile",
                percentile=94,
            )
            tool.exposed_inputs = ["image"]
            graph.connect("scene", "image", name, "image")
            compare = add(
                "inspect.mask_boundary",
                f"{name}_overlap",
                f"{method.capitalize()} overlap",
                620,
                y,
                alignment="pixel_grid",
                check_boundary=False,
            )
            compare.exposed_inputs = ["reference", "sample", "image"]
            compare.exposed_outputs = ["image", "measurements"]
            graph.connect("truth", "image", compare.id, "reference")
            graph.connect(name, "mask", compare.id, "sample")
            graph.connect(name, "image", compare.id, "image")
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise RuntimeError(validation["issues"])
        graph.save(output / "static_saliency.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
