"""Original sampled Gaussian edges with known analytic sharpness references."""

import math
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def main():
    folder = Path(__file__).resolve().parent
    assets = folder / "assets"
    assets.mkdir(exist_ok=True)
    y, x = np.mgrid[:192, :256]
    angle = math.radians(5)
    distance = (x - 128 - (y - 96) * math.tan(angle)) * math.cos(angle)
    for name, sigma in (("sharp", 0.8), ("blurred", 2.0)):
        profile = 0.1 + 0.4 * (1 + np.vectorize(math.erf)(distance / (math.sqrt(2) * sigma)))
        image = np.rint(profile * 65535).astype(np.uint16)
        success, encoded = cv2.imencode(".png", image)
        if not success:
            raise RuntimeError("Unable to encode the Gaussian edge fixture")
        (assets / f"{name}.png").write_bytes(encoded.tobytes())
    graph, registry = Graph(), builtin_registry()
    try:
        source = graph.add(
            registry.create(
                "input.file",
                block_id="source",
                parameters={
                    "source_mode": "files",
                    "paths": ["assets/sharp.png", "assets/blurred.png"],
                    "path_base": "project",
                    "repeat": "stop",
                    "color_format": "gray",
                    "pixel_depth": "uint16",
                },
            )
        )
        source.name = "Original sharp and blurred edges"
        measure = graph.add(
            registry.create(
                "inspect.slanted_edge",
                block_id="measure",
                parameters={
                    "check_threshold": True,
                    "minimum_mtf50": 0.15,
                },
            )
        )
        measure.position = [330.0, 0.0]
        graph.connect(source.id, "image", measure.id, "image")
        note = graph.add(registry.create("layout.comment", block_id="fixture_note"))
        note.name = "Original fixtures: Gaussian edge sharpness"
        note.description = (
            "Original synthetic images, project and generator under this repository's MIT license. "
            "Run all: sigma 0.8 passes; sigma 2.0 fails a 0.15 cycles/pixel MTF50 limit. "
            "Open the measurement's result tab for the ESF and SFR chart. "
            "This engineering estimate is not an ISO 12233 conformance instrument."
        )
        note.node_size, note.position = "large", [0.0, -270.0]
        graph.save(folder / "project.json")
    finally:
        graph.close()


if __name__ == "__main__":
    main()
