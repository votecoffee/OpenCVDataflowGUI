"""Original 16-bit inspection backdrop with portable multilingual labels."""

from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def main():
    folder = Path(__file__).resolve().parent
    assets = folder / "assets"
    assets.mkdir(exist_ok=True)
    yy, xx = np.indices((240, 640))
    image = np.empty((240, 640, 3), np.uint16)
    image[:] = np.stack((9000 + 14 * xx, 12000 + 16 * yy, 15000 + 10 * xx), axis=-1)
    cv2.rectangle(image, (40, 150), (590, 210), (31000, 30000, 29000), -1)
    for x in (110, 280, 450):
        cv2.circle(image, (x, 180), 18, (14000, 16000, 18000), -1)
    assert cv2.imwrite(str(assets / "inspection.png"), image)
    registry = builtin_registry()
    graph = Graph()
    try:
        # Labels are an intentional output even in Lean; Off remains available.
        graph.set_graphics_mode("on")
        source = graph.add(
            registry.create(
                "input.file",
                block_id="source",
                parameters={
                    "path": "assets/inspection.png",
                    "path_base": "project",
                    "pixel_depth": "unchanged",
                },
            )
        )
        source.name = "Original 16-bit inspection fixture"
        source.position = [0, 100]
        source.node_size = "large"
        labels = [
            ("latin", "Qualité : conforme ±5 µm", [36, 55], [120, 245, 160]),
            ("greek", "Έλεγχος : επιτυχία Δ", [36, 100], [255, 210, 90]),
            ("cyrillic", "Контроль : годен №2", [36, 145], [135, 210, 255]),
        ]
        for index, (name, text, origin, color) in enumerate(labels):
            label = graph.add(
                registry.create(
                    "cv.annotation_layer",
                    block_id=name,
                    parameters={
                        "shape": "label",
                        "label": text,
                        "origin": origin,
                        "layer_name": name,
                        "font_size": 24,
                        "color_rgb": color,
                        "text_outline": 1,
                    },
                )
            )
            label.name = name.title() + " inspection text"
            label.position = [440, index * 270]
            label.exposed_outputs = ["result_graphics"]
            graph.connect("source", "image", name, "image")
        overlay = graph.add(registry.create("cv.overlay_result_graphics", block_id="overlay"))
        overlay.position = [880, 220]
        overlay.node_size = "large"
        overlay.exposed_inputs = ["image", "graphics_1", "graphics_2", "graphics_3"]
        graph.connect("source", "image", "overlay", "image")
        for index, (name, *_rest) in enumerate(labels, 1):
            graph.connect(name, "result_graphics", "overlay", f"graphics_{index}")
        filmstrip = graph.add(registry.create("output.filmstrip_image", block_id="filmstrip"))
        filmstrip.position = [1310, 220]
        graph.connect("overlay", "image", "filmstrip", "image")
        note = graph.add(registry.create("layout.comment", block_id="about"))
        note.name = "Original fixture - MIT; Noto Sans - SIL OFL 1.1"
        note.position = [0, -280]
        note.node_size = "large"
        note.description = (
            "Original synthetic fixture, generator and graph: repository MIT license. "
            "Unmodified bundled Noto Sans: Copyright 2018 The Noto Project Authors, SIL OFL 1.1. "
            "https://github.com/notofonts/noto-fonts/blob/"
            "ffebf8c1ee449e544955a7e813c54f9b73848eac/LICENSE "
            "Latin, Greek and Cyrillic are actual FreeType glyphs, not Unicode escape strings. "
            "Annotations remain drawable objects until Overlay result graphics. "
            "Source is uint16 BGR; only covered pixels change. The ticket follows inspection.png. "
            "Font size is in rendered pixels; coordinate transforms move the label anchor. "
            "Graphics On is saved so Lean also produces the composed image. Off omits labels. "
            "Other scripts need a suitable font; complex shaping needs Pillow Raqm. "
            "https://pillow.readthedocs.io/en/stable/reference/ImageFont.html"
        )
        graph.save(folder / "project.json")
    finally:
        graph.close()


if __name__ == "__main__":
    main()
