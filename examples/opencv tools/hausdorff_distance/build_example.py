"""Build a Hausdorff comparison of a reference, a translated part and a defect."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def silhouettes():
    points = np.array([[36, 36], [123, 36], [123, 96], [103, 96], [103, 132], [36, 132]])
    for name, vertices in (("reference", points), ("shifted", points + [3, 4]), ("defect", points)):
        image = np.zeros((170, 170), np.uint8)
        cv2.fillPoly(image, [vertices.astype(np.int32)], 255)
        if name == "defect":
            cv2.rectangle(image, (123, 62), (133, 69), 255, -1)
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
        note = add("layout.comment", "note", "Measure shifts and local outline defects", 0, -245)
        note.node_size = "large"
        note.description = (
            "Run Once. The (3, 4)-pixel shift measures 5 px with L2 or 7 px with L1. "
            "The 10-pixel tab is detected at full rank and excluded at rank 0.90. "
            "The three silhouettes are original MIT assets."
        )
        for row, (name, title) in enumerate(
            (
                ("reference", "Reference part"),
                ("shifted", "Shifted part"),
                ("defect", "10-pixel tab"),
            )
        ):
            y = row * 245
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
                "inspect.contours",
                f"{name}_outline",
                f"{title} outline",
                270,
                y,
                expected_count=1,
                approximation="none",
            )
            contour.exposed_outputs = ("collection",)
            graph.connect(source.id, "image", contour.id, "image")
        for row, (name, title, source, metric, rank) in enumerate(
            (
                ("shift_l2", "Shift · Euclidean", "shifted", "L2", 1.0),
                ("shift_l1", "Shift · Manhattan", "shifted", "L1", 1.0),
                ("tab_full", "Tab · full rank", "defect", "L2", 1.0),
                ("tab_partial", "Tab · 90% rank", "defect", "L2", 0.90),
            )
        ):
            compare = add(
                "inspect.hausdorff_distance",
                name,
                title,
                660,
                95 + row * 165,
                metric=metric,
                rank_proportion=rank,
                maximum_distance=5,
            )
            graph.connect("reference_outline", "collection", compare.id, "reference")
            graph.connect(f"{source}_outline", "collection", compare.id, "sample")
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise RuntimeError(validation["issues"])
        graph.save(output / "hausdorff_distance.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
