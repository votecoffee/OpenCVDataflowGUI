"""Rasterize a shape and an independently transformed copy."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

SIZE = (240, 220)
POLYGON = [[45, 49], [154, 49], [154, 95], [120, 95], [120, 158], [87, 158], [87, 95], [45, 95]]
AFFINE = [[1.06, 0.26, -27], [-0.18, 0.84, 54]]


def make_images():
    points = np.asarray(POLYGON, dtype=np.float64)
    affine = np.asarray(AFFINE, dtype=np.float64)
    distorted = points @ affine[:, :2].T + affine[:, 2]
    images = {}
    for name, polygon in (("reference", points), ("distorted", distorted)):
        mask = np.zeros(SIZE[::-1], np.uint8)
        cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 255)
        images[name] = mask
    return images


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name, image in make_images().items():
        ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode {name}")
        (assets / f"{name}.png").write_bytes(encoded.tobytes())
    truth = {"reference_to_distorted": AFFINE, "reference_vertices": POLYGON}
    (output / "truth.json").write_text(json.dumps(truth, indent=2) + "\n", encoding="utf-8")

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Normalize equivalent silhouettes", 0, -270)
        note.node_size = "large"
        note.description = (
            "Run Once and compare the original and normalized mask measurements. Both shapes "
            "come from the same polygon. The second is rasterized after translation, shear and "
            "anisotropic scale. Original MIT-licensed images."
        )
        for i, name in enumerate(("reference", "distorted")):
            add(
                "input.file",
                name,
                name.capitalize(),
                0,
                70 + i * 240,
                path=f"assets/{name}.png",
                path_base="project",
                color_format="unchanged",
            )
            add(
                "cv.peilin_normalization",
                f"normalize_{name}",
                f"Normalize {name}",
                350,
                70 + i * 240,
                width=200,
                height=200,
                interpolation="nearest",
            )
            graph.connect(name, "image", f"normalize_{name}", "image")
        add(
            "inspect.mask_boundary",
            "before",
            "Compare originals",
            350,
            -175,
            alignment="pixel_grid",
            check_boundary=False,
        )
        add(
            "inspect.mask_boundary",
            "after",
            "Compare normalized shapes",
            730,
            175,
            alignment="pixel_grid",
            check_boundary=False,
        )
        for key, target in (("reference", "reference"), ("distorted", "sample")):
            graph.connect(key, "image", "before", target)
            graph.connect(f"normalize_{key}", "image", "after", target)
        add("output.filmstrip_image", "filmstrip", "Review normalized shape", 1070, 175)
        graph.connect("after", "image", "filmstrip", "image")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise RuntimeError(checked["issues"])
        graph.save(output / "peilin_normalization.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
