"""Build a synthetic metric-depth image and uint16 round-trip project."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

STEP = 1 / 1024
SAMPLES = (0, STEP / 4, 1024.25 * STEP, 1536.5 * STEP, 1537.5 * STEP, 70)


def build(output):
    asset = output / "assets/depth_m.tiff"
    project = output / "metric_depth_encoding.json"
    if asset.exists() or project.exists():
        raise FileExistsError("Choose a new output directory to preserve the fixture")
    asset.parent.mkdir(parents=True, exist_ok=True)
    # Six equal-width vertical bands, including a missing-depth band.
    row = np.repeat(np.array(SAMPLES, np.float32), 16)
    depth = np.tile(row, (48, 1))
    ok, encoded = cv2.imencode(".tiff", depth)
    if not ok:
        raise RuntimeError("Could not encode the synthetic depth raster")
    asset.write_bytes(encoded.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = position
        return block

    try:
        note = add("layout.comment", "note", [0, -230])
        note.name, note.node_size = "Round-trip sensor codes", "large"
        note.description = (
            "Original synthetic depth raster by OpenCVDataflowGUI contributors (MIT). "
            "Encode at 1/1024 meter per code, then decode using the scale carried with the image. "
            "The report separates rounding error from clamping at the two range limits."
        )
        add(
            "input.depth_file",
            "source",
            [0, 30],
            path="assets/depth_m.tiff",
            path_base="project",
            units_per_sample=1.0,
            display_max=2.0,
        )
        add(
            "3d.quantize_depth",
            "encode",
            [360, 30],
            meters_per_code=STEP,
            invalid_code=65535,
            out_of_range="clip",
        )
        add(
            "3d.normalize_depth",
            "decode",
            [745, 30],
            output_units="mm",
            output_format="float64",
            display_max_m=2.0,
        )
        graph.connect("source", "depth", "encode", "depth")
        graph.connect("encode", "codes", "decode", "samples")
        check = validate_graph(graph)
        if not check["valid"]:
            raise ValueError(check["issues"])
        graph.save(project)
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
