"""Build a shape-context comparison from three original silhouettes."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def silhouettes():
    reference = np.array(
        [
            [37, 41],
            [123, 41],
            [123, 60],
            [149, 60],
            [149, 111],
            [120, 111],
            [120, 143],
            [66, 143],
            [66, 120],
            [37, 120],
        ]
    )
    deformation = np.array(
        [[0, 1], [0, -2], [3, -1], [4, -2], [5, 2], [3, -1], [0, 3], [-3, 2], [-2, 1], [-3, 0]]
    )
    angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    circle = np.column_stack([93 + 52 * np.cos(angles), 92 + 52 * np.sin(angles)])
    for name, vertices in (
        ("reference", reference),
        ("deformed", reference + deformation),
        ("circle", circle),
    ):
        image = np.zeros((184, 184), np.uint8)
        cv2.fillPoly(image, [np.rint(vertices).astype(np.int32)], 255)
        yield name, image


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name, image in silhouettes():
        valid, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not valid:
            raise OSError(f"Could not encode {name} silhouette")
        (assets / f"{name}.png").write_bytes(encoded.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Rank silhouettes against a reference", 0, -220)
        note.node_size = "large"
        note.description = (
            "Run Once. The comparison blocks score an identical outline, a mildly deformed "
            "part, and a circle. Smaller distances indicate closer silhouettes. "
            "The supplied silhouettes are original MIT assets."
        )
        for row, (name, title) in enumerate(
            (("reference", "Reference part"), ("deformed", "Deformed part"), ("circle", "Circle"))
        ):
            y = row * 220
            source = add(
                "input.file",
                name,
                title,
                0,
                y,
                path=f"assets/{name}.png",
                path_base="project",
                color_format="gray",
            )
            contour = add(
                "inspect.contours", f"{name}_outline", f"{title} outline", 270, y, expected_count=1
            )
            contour.exposed_outputs = ("collection",)
            graph.connect(source.id, "image", contour.id, "image")
            compare = add(
                "inspect.shape_context_distance",
                f"{name}_distance",
                f"Score {title.lower()}",
                630,
                y,
            )
            graph.connect("reference_outline", "collection", compare.id, "reference")
            graph.connect(contour.id, "collection", compare.id, "sample")
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise RuntimeError(validation["issues"])
        graph.save(output / "shape_context.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
