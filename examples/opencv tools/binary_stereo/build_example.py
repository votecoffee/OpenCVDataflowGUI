"""Build a known-shift pair and compare binary Census with OpenCV SGBM."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def images():
    rng = np.random.default_rng(83)
    left = cv2.GaussianBlur(rng.integers(0, 120, (100, 192), dtype=np.uint8), (3, 3), 0) * 2
    right = np.zeros_like(left)
    right[:, :-8] = left[:, 8:] // 2 + 70
    truth = np.full(left.shape, 8 * 16, dtype=np.uint16)
    return left, right, truth


def save_graph(path):
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, pos, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = pos
        return block

    try:
        comment = add("layout.comment", "notes", [0, -240])
        comment.name, comment.node_size = "Compare Census matching and SGBM", "large"
        comment.description = (
            "Original synthetic stereo pair by OpenCVDataflowGUI contributors (MIT). "
            "The right view has a different exposure. Both matchers search for the same "
            "8-pixel shift while excluding an unavailable patch in the right view."
        )
        for i, key in enumerate(("left", "right", "truth")):
            source = add(
                "input.file",
                key,
                [0, 205 * i],
                path=f"assets/{key}.png",
                path_base="project",
                color_format="gray",
                pixel_depth="unchanged",
            )
            source.name = key.title()
            source.ticket_settings = {"mode": "fixed", "value": "binary-stereo-shift"}
        mask = add(
            "region.shape", "right_support", [295, 210], rectangle=[96, 32, 9, 19], invert=True
        )
        mask.name = "Exclude the right-view patch"
        graph.connect("right", "image", mask.id, "image")
        for i, (key, kind, settings) in enumerate(
            (("census", "3d.binary_disparity", {}), ("sgbm", "3d.disparity", {"block_size": 9}))
        ):
            disparity = add(kind, key, [640, 350 * i], num_disparities=16, **settings)
            if key == "census":
                disparity.preview_output = "Binary disparity"
            compare = add(
                "3d.evaluate_disparity",
                f"{key}_errors",
                [1050, 350 * i],
                alignment="pixel_grid",
                minimum_coverage=0.7,
                maximum_bad_fraction=0,
            )
            compare.name = f"Check {key.upper()} against 8 px"
            for source in ("left", "right"):
                graph.connect(source, "image", disparity.id, source)
            graph.connect(mask.id, "mask", disparity.id, "right_mask")
            graph.connect(disparity.id, "disparity", compare.id, "estimate")
            graph.connect("truth", "image", compare.id, "reference")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise ValueError(checked["issues"])
        graph.save(path)
    finally:
        graph.close()


def build(output):
    assets = output / "assets"
    project = output / "binary_stereo.json"
    if assets.exists() or project.exists():
        raise FileExistsError("Choose a new output directory to preserve the example")
    assets.mkdir(parents=True)
    for name, array in zip(("left", "right", "truth"), images(), strict=True):
        ok, data = cv2.imencode(".png", array)
        if not ok:
            raise ValueError(f"Could not encode {name}")
        (assets / f"{name}.png").write_bytes(data.tobytes())
    save_graph(project)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
