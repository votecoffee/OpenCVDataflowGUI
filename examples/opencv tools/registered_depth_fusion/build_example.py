"""Build two aligned range views and an independent reference for fusion."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def make_assets():
    shape = (96, 144)
    truth = np.full(shape, 2000, np.uint16)
    first = np.full(shape, 2010, np.uint16)
    second = np.full(shape, 1970, np.uint16)
    weight_a = np.full(shape, 255, np.uint8)
    weight_b = np.full(shape, 85, np.uint8)
    # Only view A sees the narrow foreground. Both views sample the background.
    truth[14:78, 67:69] = first[14:78, 67:69] = 1250
    first[24:34, 19:29] = 0
    second[42:54, 90:101] = 0
    first[81:85, 102:107] = second[81:85, 102:107] = 0
    # A false near surface in B has insufficient confidence to enter the fusion.
    second[15:19, 117:125] = 800
    weight_b[15:19, 117:125] = 10
    return {
        "depth_a.png": first,
        "depth_b.png": second,
        "confidence_a.png": weight_a,
        "confidence_b.png": weight_b,
        "truth.png": truth,
    }


def save_graph(path):
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, pos, **parameters):
        node = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        node.position = pos
        return node

    try:
        notes = add("layout.comment", "notes", [0, -250])
        notes.name, notes.node_size = "Fuse registered depth without blurring the edge", "large"
        notes.description = (
            "Original synthetic range images by OpenCVDataflowGUI contributors (MIT). "
            "Two aligned views have complementary holes and unequal confidence. "
            "One view sees a narrow foreground object. The other sees the background there."
        )
        for i, key in enumerate(("depth_a", "depth_b", "confidence_a", "confidence_b", "truth")):
            is_weight = key.startswith("confidence")
            kind = "input.file" if is_weight else "input.depth_file"
            units = "m" if key == "depth_b" else "mm"
            options = (
                {"color_format": "gray", "pixel_depth": "unchanged"}
                if is_weight
                else {
                    "units": units,
                    "units_per_sample": 0.001 if units == "m" else 1.0,
                    "display_min": 1 if units == "m" else 1000,
                    "display_max": 2.2 if units == "m" else 2200,
                }
            )
            source = add(
                kind, key, [0, 205 * i], path=f"assets/{key}.png", path_base="project", **options
            )
            source.name = key.replace("_", " ").title()
            source.ticket_settings = {"mode": "fixed", "value": "registered-depth-fusion"}
        fuse = add(
            "3d.fuse_registered_depth",
            "fuse",
            [415, 200],
            alignment="pixel_grid",
            surface_tolerance_m=0.06,
            minimum_coverage=0.99,
        )
        fuse.preview_output = "Fusion support"
        verify = add(
            "3d.evaluate_depth",
            "verify",
            [840, 430],
            alignment="pixel_grid",
            error_tolerance_mm=32,
            maximum_bad_fraction=0,
            minimum_coverage=0.99,
        )
        verify.name = "Compare against known depth"
        for key in ("depth_a", "depth_b", "confidence_a", "confidence_b"):
            graph.connect(key, "depth" if key.startswith("depth") else "image", fuse.id, key)
        graph.connect(fuse.id, "depth", verify.id, "estimate")
        graph.connect("truth", "depth", verify.id, "reference")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise ValueError(checked["issues"])
        graph.save(path)
    finally:
        graph.close()


def build(output):
    assets = output / "assets"
    project = output / "registered_depth_fusion.json"
    if assets.exists() or project.exists():
        raise FileExistsError("Choose a new output directory to preserve the example")
    assets.mkdir(parents=True)
    for name, image in make_assets().items():
        ok, data = cv2.imencode(".png", image)
        if not ok:
            raise ValueError(f"Could not encode {name}")
        (assets / name).write_bytes(data.tobytes())
    save_graph(project)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
