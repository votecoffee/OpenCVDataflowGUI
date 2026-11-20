"""Build a measured-fringe phase fixture with two independent reference regions."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

WIDTH, HEIGHT = 144, 64


def phase_data():
    y, x = np.mgrid[:HEIGHT, :WIDTH]
    left = -17 + 0.29 * x + 0.076 * y + 0.3 * np.sin(x / 12)
    right = 27 + 0.24 * (x - 80) - 0.13 * y + 0.15 * np.cos(y / 6)
    truth = np.where(x < 72, left, right)
    lit = (x < 64) | (x >= 80)
    lit[17:25, 40:48] = False
    lit[5, 6] = False
    amplitude = np.where(lit, 0.34, 0.0)
    exposures = [
        np.rint(65535 * (0.51 + amplitude * np.sin(truth + angle))).astype(np.uint16)
        for angle in np.arange(3) * 2 * np.pi / 3
    ]
    a, b, c = (array.astype(np.float64) for array in exposures)
    wrapped = np.arctan2((2 * a - b - c) / 3, (b - c) / np.sqrt(3)).astype(np.float32)
    references = [
        {"pixel": [x, y], "phase_rad": float(truth[y, x])}
        for x, y in ((10, 15), (92, 14), (121, 41))
    ]
    return (
        {
            "wrapped_phase.tiff": wrapped,
            "reference_phase.tiff": truth.astype(np.float32),
            "shadow_mask.png": lit.astype(np.uint8) * 255,
            **{f"fringe_{i}.png": material for i, material in enumerate(exposures)},
        },
        references,
    )


def save_graph(path, references):
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, pos, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = pos
        if block.ticket_source:
            block.ticket_settings = {"mode": "fixed", "value": "two-phase-regions"}
        return block

    try:
        note = add("layout.comment", "notes", [0, -245])
        note.name, note.node_size = "Phase across a shadow gap", "large"
        note.description = (
            "Synthetic phase and fringe exposures by OpenCVDataflowGUI contributors (MIT). "
            "The shadow divides the scene into two regions with independent unknown cycles. "
            "Compare relative output with the branch referenced to three independent samples."
        )
        for row, (name, file, title) in enumerate(
            (
                ("wrapped", "wrapped_phase.tiff", "Wrapped phase from three fringe exposures"),
                ("shadow", "shadow_mask.png", "Lit pixels"),
            )
        ):
            source = add(
                "input.file",
                name,
                [0, 35 + 205 * row],
                path=f"assets/{file}",
                path_base="project",
                color_format="gray",
                pixel_depth="unchanged",
            )
            source.name = title

        for row, branch in enumerate(("relative", "referenced")):
            block = add(
                "3d.unwrap_phase",
                branch,
                [400, 35 + 260 * row],
                alignment="pixel_grid",
                reference_mode="required" if branch == "referenced" else "relative",
                references=references if branch == "referenced" else [],
            )
            block.name = (
                "Absolute references per region"
                if branch == "referenced"
                else "Relative phase per region"
            )
            block.preview_output = "Unwrapped phase"
            graph.connect("wrapped", "image", branch, "wrapped_phase")
            graph.connect("shadow", "image", branch, "shadow_mask")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise ValueError(checked["issues"])
        graph.save(path)
    finally:
        graph.close()


def build(output):
    assets = output / "assets"
    project = output / "phase_unwrap.json"
    if assets.exists() or project.exists():
        raise FileExistsError("Choose a new directory to preserve the example")
    assets.mkdir(parents=True)
    rasters, references = phase_data()
    for name, image in rasters.items():
        ok, data = cv2.imencode(Path(name).suffix, image)
        if not ok:
            raise ValueError(f"Could not encode {name}")
        (assets / name).write_bytes(data.tobytes())
    save_graph(project, references)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
