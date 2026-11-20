"""Build independent synthetic captures of absolute X and Y projector coordinates."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

WIDTH, HEIGHT = 161, 121
CAMERA_WIDTH, CAMERA_HEIGHT = 108, 72
FINE_PERIODS = 8


def scene(axis):
    y, x = np.mgrid[:CAMERA_HEIGHT, :CAMERA_WIDTH]
    truth = 12.35 + 1.15 * x + 0.035 * y if axis == "x" else 8.25 + 0.035 * x + 1.36 * y
    truth[:, 57:] += 3.6  # Independent surface on the other side of the shadow.
    coarse = WIDTH if axis == "x" else HEIGHT
    fine = coarse // FINE_PERIODS
    labels = np.zeros(truth.shape, np.uint8)
    labels[:, 52:57] = 1
    labels[8:18, 12:27] = 1
    labels[28:39, 76:85] = 2

    mean = 0.39 + 0.20 * y / CAMERA_HEIGHT
    amplitude = np.full_like(truth, 0.21) + 0.07 * x / CAMERA_WIDTH
    amplitude[labels == 1] = 0.006
    mean[:, 52:57] = 0.015
    distorted = truth.copy()
    distorted[labels == 2] += 0.4 * fine
    captures = []
    rng = np.random.default_rng(6124)
    for period, coordinate in ((coarse, truth), (fine, distorted)):
        for phase in (0, 2 * np.pi / 3, 4 * np.pi / 3):
            response = mean + amplitude * np.sin(2 * np.pi * coordinate / period + phase)
            response += rng.normal(0, 0.00002, truth.shape)
            captures.append(np.rint(np.clip(response, 0, 1) * 65535).astype(np.uint16))
    return captures, labels


def save_graph(output, axis):
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = position
        return block

    try:
        add(
            "input.tiff_file",
            "captures",
            [0, 0],
            path=f"assets/capture_{axis}.tiff",
            path_base="project",
        )
        add("flow.batch", "batch", [320, 0], size=6).name = "Collect six exposures"
        add(
            "3d.phase_shift_patterns",
            "patterns",
            [320, 280],
            width=WIDTH,
            height=HEIGHT,
            axis=axis,
            fine_periods=FINE_PERIODS,
        )
        add("3d.phase_shift_decode", "decode", [685, 0])
        comment = add("layout.comment", "credit", [0, -260])
        comment.name, comment.node_size = f"Synthetic absolute projector {axis.upper()}", "large"
        comment.description = (
            "Original fixtures by OpenCVDataflowGUI contributors (MIT). Run All reads six "
            "simulated exposures of a known subpixel correspondence map, including a "
            "shadow, a low-modulation patch and an intentionally inconsistent fine phase."
        )
        graph.connect("captures", "image", "batch", "item_1")
        graph.connect("batch", "group", "decode", "patterns")
        graph.connect("patterns", "specification", "decode", "specification")
        check = validate_graph(graph)
        if not check["valid"]:
            raise ValueError(check["issues"])
        graph.save(output / f"projector_{axis}.json")
    finally:
        graph.close()


def build(output):
    assets = output / "assets"
    if assets.exists() or any(output.glob("projector_*.json")):
        raise FileExistsError("Choose an empty destination. Existing fixtures are preserved.")
    assets.mkdir(parents=True)
    for axis in ("x", "y"):
        captures, labels = scene(axis)
        for (extension, image), name in zip(
            [(".tiff", captures), (".png", labels)],
            [f"capture_{axis}.tiff", f"labels_{axis}.png"],
            strict=True,
        ):
            if extension == ".tiff":
                ok, data = cv2.imencodemulti(extension, image)
            else:
                ok, data = cv2.imencode(extension, image)
            if not ok:
                raise ValueError(f"Could not encode {name}")
            (assets / name).write_bytes(data.tobytes())
        save_graph(output, axis)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
