"""Create public synthetic part images and two perceptual hash projects."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

METHODS = ("color_moment", "radial_variance")
BACKGROUND = (37, 46, 55)


def samples():
    reference = np.full((256, 256, 3), BACKGROUND, np.uint8)
    profile = np.array([[55, 80], [155, 53], [207, 95], [192, 175], [75, 202], [52, 153]])
    cv2.fillConvexPoly(reference, profile, (105, 168, 225), cv2.LINE_AA)
    for center, radius in (((87, 101), 14), ((155, 114), 27), ((101, 161), 11)):
        cv2.circle(reference, center, radius, BACKGROUND, -1, cv2.LINE_AA)
    cv2.rectangle(reference, (137, 157), (174, 172), (231, 102, 62), -1)
    cv2.line(reference, (57, 149), (73, 182), (210, 230, 193), 5, cv2.LINE_AA)

    angle = cv2.getRotationMatrix2D((127.5, 127.5), 15, 1)
    rotated = cv2.warpAffine(reference, angle, (256, 256), borderValue=BACKGROUND)
    resampled = cv2.resize(reference, (173, 173), interpolation=cv2.INTER_AREA)
    changed = np.full_like(reference, BACKGROUND)
    cv2.rectangle(changed, (60, 68), (199, 191), (148, 93, 61), -1, cv2.LINE_AA)
    for x in (76, 112, 148, 184):
        cv2.line(changed, (x, 69), (x, 190), (41, 203, 155), 9, cv2.LINE_AA)
    return {"reference": reference, "rotated": rotated, "resampled": resampled, "changed": changed}


def make_graph(output, method):
    registry = builtin_registry()
    graph = Graph()

    def add(type_id, key, name, x, y, **parameters):
        block = graph.add(registry.create(type_id, block_id=key, parameters=parameters))
        block.name, block.position = name, [x, y]
        return block

    try:
        title = method.replace("_", " ").title()
        comment = add("layout.comment", "comment", f"{title} descriptors", 0, -220)
        comment.node_size = "large"
        comment.description = (
            f"Measure {title} scores for identical, rotated, resampled and different parts. "
            "The original synthetic part images are covered by this repository's MIT license."
        )
        for index, (key, caption) in enumerate(
            (
                ("reference", "Reference part"),
                ("rotated", "Rotated 15 degrees"),
                ("resampled", "Resampled to 173 × 173"),
                ("changed", "Different part"),
            )
        ):
            y = 185 * index
            add("input.file", key, caption, 0, y, path=f"assets/{key}.png", path_base="project")
            add("cv.image_hash", f"{key}_hash", f"{caption} hash", 270, y, method=method)
            graph.connect(key, "image", f"{key}_hash", "image")
            name = "Identical pair" if key == "reference" else f"{caption} comparison"
            comparison = add(
                "inspect.compare_perceptual_descriptors", f"{key}_compare", name, 575, y
            )
            port = "distance" if method == "color_moment" else "similarity"
            comparison.exposed_outputs = (port, "measurements")
            graph.connect("reference_hash", "hash", comparison.id, "reference")
            graph.connect(f"{key}_hash", "hash", comparison.id, "sample")
        qualified = validate_graph(graph)
        if not qualified["valid"]:
            raise RuntimeError(qualified["issues"])
        graph.save(output / f"{method}.json")
    finally:
        graph.close()


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for key, image in samples().items():
        ok, png = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode {key}")
        (assets / f"{key}.png").write_bytes(png.tobytes())
    for method in METHODS:
        make_graph(output, method)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
