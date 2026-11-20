"""Build a disparity reference example with exact error and coverage strata."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def save_assets(assets):
    shape = (96, 144)
    reference = np.broadcast_to(np.repeat([0, 2, 4, 6, 8, 10], 24), shape)
    estimate = np.broadcast_to(np.repeat([30, 2, 5, 4, -64, 13], 24), shape)
    confidence = np.broadcast_to(np.repeat([1, 1, 0.5, 0.25, 0, 0.75], 24), shape)
    rasters = {
        "reference.png": (reference * 256).astype(np.uint16),
        "estimate.tiff": estimate.astype(np.float32),
        "confidence.tiff": confidence.astype(np.float32),
    }
    for name, array in rasters.items():
        ok, encoded = cv2.imencode(Path(name).suffix, array)
        if not ok:
            raise ValueError(f"Could not encode {name}")
        (assets / name).write_bytes(encoded.tobytes())


def save_graph(path):
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, x, y, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = [x, y]
        return block

    try:
        notes = add("layout.comment", "notes", 0, -250)
        notes.name = "Evaluate stereo disparity"
        notes.node_size = "large"
        notes.description = (
            "Original synthetic maps by OpenCVDataflowGUI contributors (MIT). "
            "Compare the full field to a rectangular inspection region. The full field "
            "includes unknown reference, missing estimate and bad pixels. The selected "
            "region includes exact and one-pixel errors."
        )
        for i, (key, filename) in enumerate(
            (
                ("estimate", "estimate.tiff"),
                ("reference", "reference.png"),
                ("confidence", "confidence.tiff"),
            )
        ):
            block = add(
                "input.file",
                key,
                0,
                i * 205,
                path=f"assets/{filename}",
                path_base="project",
                color_format="gray",
                pixel_depth="unchanged",
            )
            block.name = key.title()
            block.ticket_settings = {"mode": "fixed", "value": "synthetic-stereo-reference"}
        roi = add("region.shape", "selection", 310, -55, rectangle=[24, 0, 48, 96])
        roi.name = "Center inspection region"
        graph.connect("estimate", "image", roi.id, "image")
        for i, (key, coverage, bad_limit) in enumerate(
            (("whole", 0.8, 0.5), ("selected", 0.95, 0.1))
        ):
            compare = add(
                "3d.evaluate_disparity",
                key,
                690,
                i * 360,
                alignment="pixel_grid",
                reference_samples_per_pixel=256,
                reference_invalid_mode="also_use_code",
                reference_invalid_code=0,
                estimate_invalid_mode="also_use_code",
                estimate_invalid_code=-64,
                minimum_coverage=coverage,
                maximum_bad_fraction=bad_limit,
            )
            compare.name = "Whole field evaluation" if not i else "Selected region evaluation"
            for source in ("estimate", "reference", "confidence"):
                graph.connect(source, "image", compare.id, source)
            if i:
                graph.connect(roi.id, "region", compare.id, "region")

        checked = validate_graph(graph)
        if not checked["valid"]:
            raise ValueError(checked["issues"])
        graph.save(path)
    finally:
        graph.close()


def build(output):
    assets = output / "assets"
    path = output / "disparity_reference.json"
    if assets.exists() or path.exists():
        raise FileExistsError("Choose a new output directory to preserve the fixture")
    assets.mkdir(parents=True)
    save_assets(assets)
    save_graph(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
