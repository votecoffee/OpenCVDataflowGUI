"""Original 16-bit orientation/ambiguity fixture and selected-region graph."""

import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def fixture():
    width, height = 576, 352
    image = np.full((height, width), 0.12, np.float64)
    probes = {}
    specs = [
        ("vertical", "Vertical texture: 90 deg", 0, 90),
        ("diagonal", "Diagonal texture: 135 deg", 45, 135),
        ("horizontal", "Horizontal texture: 0 deg", 90, 0),
        ("corner", "Corner: ambiguous center", None, None),
        ("flat", "Flat: no direction", None, None),
        ("weak", "Weak: below energy limit", 0, None),
    ]
    y, x = np.mgrid[:120, :160]
    for index, (name, title, normal, tangent) in enumerate(specs):
        column, row = index % 3, index // 3
        left, top = column * 192 + 16, row * 176 + 40
        if normal is not None:
            radians = np.deg2rad(normal)
            amplitude = 0.00005 if name == "weak" else 0.35
            patch = 0.5 + amplitude * np.sin(
                2 * np.pi * (x * np.cos(radians) + y * np.sin(radians)) / 16
            )
        elif name == "corner":
            patch = np.where((x >= 80) & (y >= 60), 0.85, 0.15)
        else:
            patch = np.full(x.shape, 0.5)
        image[top : top + 120, left : left + 160] = patch
        rectangle = [left + 12, top + 12, 136, 96]
        if name == "corner":
            rectangle = [left + 80, top + 60, 1, 1]
        probes[name] = {"rectangle": rectangle, "tangent_degrees": tangent}
    encoded = np.rint(image * 65535).astype(np.uint16)
    labels = np.zeros(encoded.shape, np.uint8)
    for index, (_, title, _, _) in enumerate(specs):
        cv2.putText(
            labels,
            title,
            (index % 3 * 192 + 6, index // 3 * 176 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            255,
            1,
            cv2.LINE_AA,
        )
    alpha = labels.astype(float) / 255
    encoded = np.rint(encoded * (1 - alpha) + 58000 * alpha).astype(np.uint16)
    return encoded, {"probes": probes, "units": "unsigned current-grid tangent degrees"}


def main():
    folder = Path(__file__).resolve().parent
    assets = folder / "assets"
    assets.mkdir(exist_ok=True)
    image, truth = fixture()
    assert cv2.imwrite(str(assets / "orientation-fixture.png"), image)
    (assets / "truth.json").write_text(
        json.dumps(truth, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    registry, graph = builtin_registry(), Graph()
    try:
        source = graph.add(
            registry.create(
                "input.file",
                block_id="source",
                parameters={
                    "path": "assets/orientation-fixture.png",
                    "path_base": "project",
                    "color_format": "gray",
                    "pixel_depth": "unchanged",
                },
            )
        )
        source.name, source.position = "Original 16-bit orientation fixture", [0.0, 250.0]
        overview = graph.add(registry.create("inspect.structure_tensor", block_id="overview"))
        overview.name, overview.position = "All local orientations", [330.0, -310.0]
        graph.connect(source.id, "image", overview.id, "image")
        for index, (name, expected) in enumerate(truth["probes"].items()):
            region = graph.add(
                registry.create(
                    "region.shape",
                    block_id=f"{name}_region",
                    parameters={
                        "shape": "rectangle",
                        "rectangle": expected["rectangle"],
                    },
                )
            )
            region.name = f"{name.capitalize()} probe region"
            region.position, region.exposed_outputs = [330.0, index * 240.0], ["region"]
            graph.connect(source.id, "image", region.id, "image")
            tensor = graph.add(
                registry.create(
                    "inspect.structure_tensor", block_id=name, parameters={"glyph_step": 10}
                )
            )
            angle = expected["tangent_degrees"]
            tensor.name = f"{name.capitalize()}: " + (
                f"{angle} deg" if angle is not None else "no reliable direction"
            )
            tensor.position = [660.0, index * 240.0]
            tensor.description = "Expected tangent from the analytic texture generator: " + (
                f"{angle} degrees." if angle is not None else "undefined."
            )
            graph.connect(source.id, "image", tensor.id, "image")
            graph.connect(region.id, "region", tensor.id, "region")
        note = graph.add(registry.create("layout.comment", block_id="about"))
        note.name = "Original orientation and ambiguity fixture — MIT"
        note.description = (
            "Original generated 16-bit image, graph and code under the repository MIT license. "
            "Top row: vertical, diagonal and horizontal textures with 90/135/0 degree tangents. "
            "Bottom row: a corner-center probe, uniform patch and weak texture, each without a "
            "reliable direction at default limits. Select a tensor and inspect orientation, "
            "coherence, validity and eigenvalues. Zero orientation is meaningful only where "
            "valid is nonzero. Overview includes panel boundaries and text; selected probes "
            "exclude those. assets/truth.json stores generator-defined expectations."
        )
        note.position, note.node_size = [0.0, -310.0], "large"
        graph.save(folder / "project.json")
    finally:
        graph.close()


if __name__ == "__main__":
    main()
