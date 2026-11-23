"""Build the original edge and impulse-noise comparison."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

SIZE = (96, 144)
RECTANGLE = (19, 6, 114, 77)


def images():
    surface = np.zeros(SIZE, dtype=bool)
    surface[15:76, 58:117] = True
    surface[47:76, 44:58] = True
    surface[20:83, 31:33] = True
    clean = np.full(SIZE, 60, np.uint8)
    clean[surface] = 174
    guide = np.full((*SIZE, 3), (40, 78, 182), np.uint8)
    guide[surface] = (203, 179, 37)
    noisy = clean.copy()
    indices = np.random.default_rng(283).choice(clean.size, 900, replace=False)
    noisy.flat[indices[:450]] = 0
    noisy.flat[indices[450:]] = 255
    noisy[25:29, 82:86] = 255
    sample_mask = np.where(noisy == clean, 255, 0).astype(np.uint8)
    return {"clean": clean, "noisy": noisy, "guide": guide, "sample_mask": sample_mask}


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name, image in images().items():
        ok, png = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode {name}")
        (assets / f"{name}.png").write_bytes(png.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Remove impulse noise", 0, -230)
        note.node_size = "large"
        note.description = (
            "Run Once to compare an unweighted median, the same window with a color guide, "
            "and a sample mask that excludes the injected damage. The region branch limits "
            "where filtered pixels are written. Original MIT assets."
        )
        for index, (key, title) in enumerate(
            (
                ("noisy", "Signal with impulse noise"),
                ("guide", "Aligned edge guide"),
                ("sample_mask", "Trusted sample pixels"),
            )
        ):
            add(
                "input.file",
                key,
                title,
                0,
                index * 200 + 30,
                path=f"assets/{key}.png",
                path_base="project",
                color_format="unchanged",
            )
        region = add(
            "region.shape", "region", "Inspection rectangle", 275, 505, rectangle=list(RECTANGLE)
        )
        region.exposed_outputs = ("region",)
        graph.connect("noisy", "image", "region", "image")
        for index, (key, title, weight) in enumerate(
            (
                ("unweighted", "Unweighted median", "unweighted"),
                ("weighted", "Gaussian guide weights", "gaussian"),
                ("samples", "Exclude damaged samples", "gaussian"),
                ("selected", "Filter within a rectangle", "gaussian"),
            )
        ):
            add(
                "cv.weighted_median_filter",
                key,
                title,
                560,
                index * 205 - 65,
                radius=2,
                weight=weight,
                guide_alignment="pixel_grid",
            )
            graph.connect("noisy", "image", key, "image")
            graph.connect("guide", "image", key, "guide")
        graph.connect("sample_mask", "image", "samples", "sample_mask")
        graph.connect("region", "region", "selected", "region")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise ValueError(checked["issues"])
        graph.save(output / "weighted_median.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
