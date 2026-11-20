"""Build a synthetic depth image of touching parts with known footprints."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def image():
    depth = np.full((96, 160), 1700, np.uint16)
    depth[20:48, 28:60] = 1000
    depth[30:34, 39:44] = 0
    depth[24:50, 60:92] = 1250
    depth[64:78, 112:130] = 900
    depth[84, 20:22] = 800
    return depth


def save_graph(path):
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, pos, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = pos
        return block

    try:
        notes = add("layout.comment", "notes", [0, -260])
        notes.name, notes.node_size = "Measure separate parts on a depth grid", "large"
        notes.description = (
            "Original synthetic depth raster by OpenCVDataflowGUI contributors (MIT). "
            "Two parts touch in the image but have distinct Z. A third is isolated. "
            "The graph filters by visible width and reconstructs the selected source mask."
        )
        add(
            "input.depth_file",
            "depth",
            [0, 30],
            path="assets/depth_mm.png",
            path_base="project",
            units="mm",
            units_per_sample=1,
            display_min=750,
            display_max=1800,
        )
        add(
            "3d.camera",
            "camera",
            [0, 275],
            width=160,
            height=96,
            fx=200,
            fy=250,
            cx=79.5,
            cy=47.5,
            frame="depth_optical",
        )
        measure = add(
            "3d.depth_components",
            "measure",
            [350, 50],
            maximum_depth_m=1.4,
            maximum_depth_step_m=0.035,
            minimum_pixels=12,
        )
        measure.preview_output = "Depth components"
        filtered = add(
            "collection.filter",
            "wide",
            [710, 55],
            metric="custom",
            custom_metric="width_m",
            minimum=0.19,
        )
        filtered.name = "Retain width of at least 190 mm"
        mask = add("inspect.mask_from_labels", "selected_mask", [1040, 110])
        mask.name = "Rebuild the selected source mask"
        for src, output, dst, input_name in (
            ("depth", "depth", "measure", "depth"),
            ("camera", "camera", "measure", "camera"),
            ("measure", "collection", "wide", "collection"),
            ("measure", "labels", "selected_mask", "labels"),
            ("wide", "collection", "selected_mask", "collection"),
        ):
            graph.connect(src, output, dst, input_name)
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise ValueError(checked["issues"])
        graph.save(path)
    finally:
        graph.close()


def build(output):
    assets = output / "assets"
    project = output / "depth_components.json"
    if assets.exists() or project.exists():
        raise FileExistsError("Choose a new output directory to preserve the example")
    assets.mkdir(parents=True)
    ok, data = cv2.imencode(".png", image())
    if not ok:
        raise ValueError("Could not encode the depth image")
    (assets / "depth_mm.png").write_bytes(data.tobytes())
    save_graph(project)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
