"""Build fixed-point and float disparity fixtures with known speckle footprints."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

SPECKLES = ((15, 20, 2, 3), (50, 52, 4, 4), (68, 140, 3, 4))
HOLES = ((26, 120, 7, 9), (0, 0, 80, 4))
SHAPE = (80, 160)


def rectangle(y, x, h, w):
    return slice(y, y + h), slice(x, x + w)


def make_maps(assets):
    field = np.full(SHAPE, 8.25, np.float32)
    field[:, 83:] = 14
    field[20:26, 65:75] = 22
    for i, spec in enumerate(SPECKLES):
        field[rectangle(*spec)] = (10.5, 5, 19)[i]
    valid = np.ones(SHAPE, bool)
    for spec in HOLES:
        valid[rectangle(*spec)] = False
    fixed = np.uint16(field * 16)
    fixed[~valid] = 65535

    # Subtract a global offset and add a non-grid fraction to test untouched float output.
    pixels = field - np.float32(11.0125)
    pixels[~valid] = -100
    for name, values in (("fixed.png", fixed), ("signed.tiff", pixels)):
        ok, encoded = cv2.imencode(Path(name).suffix, values)
        if not ok:
            raise ValueError(f"Could not encode {name}")
        (assets / name).write_bytes(encoded.tobytes())


def save_project(output):
    graph, registry = Graph(), builtin_registry()

    def add(type_id, name, x, y, **parameters):
        block = graph.add(registry.create(type_id, block_id=name, parameters=parameters))
        block.position = [x, y]
        return block

    try:
        comment = add("layout.comment", "notes", 0, -230)
        comment.name = "Disparity speckle removal"
        comment.node_size = "large"
        comment.description = (
            "Original synthetic disparity maps by OpenCVDataflowGUI contributors (MIT). "
            "Both contain three small islands and one larger feature. The fixed-point "
            "map uses a 65535 invalid code. The signed float map uses -100 px."
        )
        for i, (kind, file) in enumerate((("fixed_point", "fixed.png"), ("float", "signed.tiff"))):
            y = i * 255
            src = add(
                "input.file",
                f"source_{kind}",
                0,
                y,
                path=f"assets/{file}",
                path_base="project",
                color_format="gray",
                pixel_depth="unchanged",
            )
            src.name = "Read 16× fixed-point disparity" if i == 0 else "Read signed float disparity"
            filt = add(
                "3d.filter_disparity_speckles",
                kind,
                405,
                y,
                max_speckle_size=16,
                max_difference_px=0.5,
                invalid_value_enabled=True,
                invalid_value=65535 if i == 0 else -100,
            )
            graph.connect(src.id, "image", filt.id, "disparity")
        validated = validate_graph(graph)
        if not validated["valid"]:
            raise ValueError(validated["issues"])
        graph.save(output / "disparity_speckles.json")
    finally:
        graph.close()


def build(output):
    assets = output / "assets"
    if assets.exists() or (output / "disparity_speckles.json").exists():
        raise FileExistsError("Choose a new output directory to preserve the fixture")
    assets.mkdir(parents=True)
    make_maps(assets)
    save_project(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
