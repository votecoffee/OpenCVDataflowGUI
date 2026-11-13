"""Original synthetic part images for an aligned reference inspection."""

from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def main():
    folder = Path(__file__).resolve().parent
    assets = folder / "assets"
    assets.mkdir(exist_ok=True)
    y, x = np.mgrid[:180, :240]
    reference = np.dstack(
        (35 + x // 8 + y // 12, 40 + x // 10 + y // 10, 45 + x // 12 + y // 8)
    ).astype(np.uint8)
    cv2.rectangle(reference, (40, 30), (200, 150), (110, 135, 150), -1)
    cv2.circle(reference, (75, 65), 16, (25, 35, 40), -1)
    cv2.circle(reference, (165, 65), 16, (25, 35, 40), -1)
    cv2.rectangle(reference, (90, 100), (155, 125), (180, 190, 175), -1)
    good = np.rint(reference * 0.9 + 8).astype(np.uint8)
    defective = good.copy()
    cv2.rectangle(defective, (145, 45), (185, 85), (110, 130, 145), -1)  # Missing bore.
    for name, image in (("reference", reference), ("good", good), ("missing-bore", defective)):
        success, encoded = cv2.imencode(".png", image)
        if not success:
            raise RuntimeError("Unable to encode the reference inspection fixture")
        (assets / f"{name}.png").write_bytes(encoded.tobytes())
    graph, registry = Graph(), builtin_registry()
    try:
        source = graph.add(
            registry.create(
                "input.file",
                block_id="source",
                parameters={
                    "source_mode": "files",
                    "paths": ["assets/good.png", "assets/missing-bore.png"],
                    "path_base": "project",
                    "repeat": "stop",
                    "color_format": "bgr",
                },
            )
        )
        source.name = "Candidate: good / missing bore"
        reference_node = graph.add(
            registry.create(
                "input.file",
                block_id="reference",
                parameters={
                    "path": "assets/reference.png",
                    "path_base": "project",
                    "repeat": "loop",
                    "color_format": "bgr",
                },
            )
        )
        reference_node.position = [0.0, 250.0]
        compare = graph.add(
            registry.create(
                "inspect.image_correlation",
                block_id="compare",
                parameters={
                    "minimum_similarity": 0.98,
                    "emit_residual": True,
                },
            )
        )
        compare.position = [350.0, 100.0]
        graph.connect(source.id, "image", compare.id, "image")
        graph.connect(reference_node.id, "image", compare.id, "reference")
        note = graph.add(registry.create("layout.comment", block_id="fixture_note"))
        note.name = "Original reference inspection fixtures"
        note.description = (
            "Original synthetic images, generator and project under this repository's MIT license. "
            "Run all: the intact part passes; the missing-bore image fails. "
            "Open the comparison's normalized residual preview to locate differences. "
            "Images must already be aligned. Correlation ignores positive brightness gain/offset."
        )
        note.node_size, note.position = "large", [0.0, -280.0]
        graph.save(folder / "project.json")
    finally:
        graph.close()


if __name__ == "__main__":
    main()
