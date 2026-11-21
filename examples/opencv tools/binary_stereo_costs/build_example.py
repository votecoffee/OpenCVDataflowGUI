"""Build known textured, flat and occluded probes on a synthetic stereo pair."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

SHAPE = (76, 208)
DISPARITY = 8
PROBES = [[52, 35], [112, 35], [174, 35]]
HOLE = [164, 30, 4, 11]


def fixture():
    rng = np.random.default_rng(122087)
    left = rng.integers(0, 190, SHAPE, dtype=np.uint8)
    left[:, 88:144] = 75
    right = np.full(SHAPE, 30, np.uint8)
    right[:, :-DISPARITY] = left[:, DISPARITY:] + 32
    x, y, w, h = HOLE
    right[y : y + h, x : x + w] = 5
    return left, right


def build(directory):
    path = directory / "binary_stereo_costs.json"
    if path.exists():
        raise FileExistsError("Use an empty output directory")
    assets = directory / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name, pixels in zip(("left", "right"), fixture(), strict=True):
        ok, encoded = cv2.imencode(".png", pixels)
        if not ok:
            raise ValueError(f"Could not encode {name}")
        (assets / f"{name}.png").write_bytes(encoded.tobytes())
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, name, position, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = name, position
        return block

    try:
        note = add(
            "layout.comment", "note", "Inspect the evidence behind a stereo match", [0, -250]
        )
        note.node_size = "large"
        note.description = (
            "Original synthetic rectified pair by OpenCVDataflowGUI contributors (MIT). "
            "Probe 1 has texture and a true 8 px match. Probe 2 is flat and ambiguous. "
            "Probe 3 lands in a masked right-view occlusion at 8 px."
        )
        for i, (key, label) in enumerate((("left", "Left view"), ("right", "Right view"))):
            source = add(
                "input.file",
                key,
                label,
                [0, 240 * i],
                path=f"assets/{key}.png",
                path_base="project",
                color_format="gray",
            )
            source.ticket_settings = {"mode": "fixed", "value": "stereo-probe-pair"}
        mask = add(
            "region.shape",
            "right_valid",
            "Exclude right occlusion",
            [180, 465],
            rectangle=HOLE,
            invert=True,
        )
        graph.connect("right", "image", mask.id, "image")
        for i, (key, label) in enumerate(
            (("center", "Center Census cost curves"), ("symmetric", "Opposite-pair cost curves"))
        ):
            add(
                "3d.binary_stereo_costs",
                key,
                label,
                [400, 90 + 280 * i],
                census_variant=key,
                num_disparities=16,
                probe_points=PROBES,
                plot_probe=1 if i == 0 else 3,
            )
            for source, port in (
                ("left", "left"),
                ("right", "right"),
            ):
                graph.connect(source, "image", key, port)
            graph.connect(mask.id, "mask", key, "right_mask")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise ValueError(checked["issues"])
        graph.save(path)
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
