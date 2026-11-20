"""Build a fronto-parallel two-plane depth comparison with known residuals."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def save_assets(assets):
    reference = np.full((80, 128), 1000, dtype=np.uint16)
    reference[:, 64:] = 1500
    estimate = reference.astype(np.float32) / 1000
    estimate[:, 64] = 1.0
    estimate[20:30, 5:15] += 0.015625
    estimate[42:52, 88:98] -= 0.03125
    estimate[30:38, 100:110] = 0
    reference[12:20, 10:20] = 0
    for filename, depth in (("reference_mm.png", reference), ("estimate_m.tiff", estimate)):
        ok, encoded = cv2.imencode(Path(filename).suffix, depth)
        if not ok:
            raise ValueError(f"Could not encode {filename}")
        (assets / filename).write_bytes(encoded.tobytes())


def save_graph(path):
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = position
        return block

    try:
        notes = add("layout.comment", "notes", [0, -240])
        notes.name, notes.node_size = "Evaluate a known depth step", "large"
        notes.description = (
            "Original synthetic depth planes by OpenCVDataflowGUI contributors (MIT). "
            "The estimate moves the step one pixel right and includes two biased patches. "
            "Different holes in each map affect overlap. Compare the whole field and the step."
        )
        for i, (key, units, filename) in enumerate(
            (
                ("estimate", "m", "estimate_m.tiff"),
                ("reference", "mm", "reference_mm.png"),
            )
        ):
            block = add(
                "input.depth_file",
                key,
                [0, 245 * i],
                path=f"assets/{filename}",
                path_base="project",
                units=units,
                units_per_sample=1.0,
                display_min=0.9 if units == "m" else 900,
                display_max=1.6 if units == "m" else 1600,
            )
            block.name = f"{key.title()} ({units})"
            block.ticket_settings = {"mode": "fixed", "value": "known-depth-step"}
        roi = add("region.shape", "step_band", [330, 40], rectangle=[62, 0, 4, 80])
        roi.name = "Narrow band around the depth step"
        graph.connect("estimate", "depth", roi.id, "image")
        for i, (key, limit) in enumerate((("whole", 0.02), ("step", 0.25))):
            compare = add(
                "3d.evaluate_depth",
                key,
                [680, 330 * i],
                alignment="pixel_grid",
                error_tolerance_mm=20,
                minimum_coverage=0.99,
                maximum_bad_fraction=limit,
                preview_max_mm=40,
            )
            compare.name = "Whole-field comparison" if not i else "Step-band comparison"
            for source in ("estimate", "reference"):
                graph.connect(source, "depth", compare.id, source)
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
    project = output / "depth_reference.json"
    if assets.exists() or project.exists():
        raise FileExistsError("Choose a new output directory to preserve the fixture")
    assets.mkdir(parents=True)
    save_assets(assets)
    save_graph(project)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
