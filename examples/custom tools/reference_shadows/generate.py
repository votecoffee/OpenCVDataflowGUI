"""Original calibrated synthetic part under clear, neutral and inconsistent lighting."""

import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def main():
    folder = Path(__file__).resolve().parent
    assets = folder / "assets"
    assets.mkdir(exist_ok=True)
    yy, xx = np.indices((96, 160))
    nominal = np.tile(np.array([24000, 20000, 16000], np.uint16), (96, 160, 1))
    nominal[20:76, 24:136] = [32000, 28000, 24000]
    nominal = (nominal.astype(np.int32) + np.where((xx + yy) % 2, 800, -800)[..., None]).astype(
        np.uint16
    )
    clear = nominal.copy()
    clear[43:49, 76:84] = nominal[43:49, 76:84].astype(np.uint32) * 3 // 4
    shadow = clear.copy()
    shadow[:, :104] //= 2
    inconsistent = clear.copy()
    inconsistent[:, :60] = np.rint(clear[:, :60] * 0.4)
    inconsistent[:, 60:104] = np.rint(clear[:, 60:104] * 0.7)
    shadow_mask = np.zeros((96, 160), np.uint8)
    shadow_mask[:, :104] = 255
    for name, image in (
        ("reference", nominal),
        ("clear", clear),
        ("shadow", shadow),
        ("inconsistent", inconsistent),
        ("shadow_mask", shadow_mask),
    ):
        assert cv2.imwrite(
            str(assets / (name + ".png")), image[..., ::-1] if image.ndim == 3 else image
        )
    truth = {
        "origin": "Original synthetic 16-bit RGB fixture, generator and graph; MIT",
        "shadow_rectangle": [0, 0, 104, 96],
        "shadow_pixels": 9984,
        "defect_rectangle": [76, 43, 8, 6],
        "defect_pixels": 48,
        "reference_minus_clear_defect_mean_rgb": [8000, 7000, 6000],
        "clear_gain": 1,
        "shadow_gain": 2,
        "expected_corrected_max_absolute_error": 0,
        "minimum_shadow_area": 256,
        "inconsistent_ratios": [0.4, 0.7],
        "note": "An image-derived dark/chromatic similarity is a shadow hypothesis, not proof. "
        "This fixture qualifies region-constant multiplicative correction only.",
    }
    (assets / "truth.json").write_text(
        json.dumps(truth, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    registry = builtin_registry()
    for filename, invalid in (("project.json", False), ("invalid_fit.json", True)):
        graph = Graph()
        try:
            for index, (name, path) in enumerate(
                (
                    ("source", "inconsistent" if invalid else "clear"),
                    ("reference", "reference"),
                    ("clear", "clear"),
                )
            ):
                source = graph.add(
                    registry.create(
                        "input.file",
                        block_id=name,
                        parameters={
                            "path": f"assets/{path}.png",
                            "path_base": "project",
                            "color_format": "rgb",
                            "pixel_depth": "unchanged",
                            "source_mode": "files"
                            if name == "source" and not invalid
                            else "single",
                            "paths": ["assets/clear.png", "assets/shadow.png"],
                            "repeat": "stop",
                        },
                    )
                )
                source.position = [0, index * 280]
                source.node_size = "large"
                source.name = {
                    "source": "Clear / shadow replay" if not invalid else "Bad model",
                    "reference": "Unshadowed nominal reference",
                    "clear": "Independent expected pixels",
                }[name]
            shadows = graph.add(
                registry.create(
                    "cv.reference_shadows", block_id="shadows", parameters={"minimum_area": 256}
                )
            )
            shadows.name = "Constant gain per qualified shadow"
            shadows.position = [450, 150]
            shadows.node_size = "large"
            shadows.exposed_inputs = ["image", "reference", "mask", "fit_mask"]
            shadows.exposed_outputs = ["image", "shadow_mask", "gain_map", "valid", "measurements"]
            graph.connect("source", "image", "shadows", "image")
            graph.connect("reference", "image", "shadows", "reference")
            filmstrip = graph.add(registry.create("output.filmstrip_image", block_id="filmstrip"))
            filmstrip.position = [900, 150]
            graph.connect("shadows", "image", "filmstrip", "image")
            note = graph.add(registry.create("layout.comment", block_id="about"))
            note.name = "Original shadow and defect fixtures - MIT"
            note.position = [0, -300]
            note.node_size = "large"
            note.description = (
                "Original synthetic images, truth, generator and projects: repository MIT. "
                "Run all: clear.png has no shadow candidates and retains the 48-pixel dark defect. "
                "shadow.png has exactly 9,984 shadow pixels (x < 104), attenuation 0.5 and gain 2. "
                "Both corrected outputs equal the independent clear.png pixel for pixel, including "
                "the sharp part boundary, cast-shadow boundary, texture and real dark defect. "
                "Defect contrast against the nominal reference remains R/G/B = 8000/7000/6000. "
                "Minimum shadow area 256 rejects the smaller isolated defect in the clear frame. "
                "The reference supplies a robust regional gain, never replacement image pixels. "
                "Invalid_fit.json has joined attenuation 0.4/0.7 and must withhold Image/Gain. "
                "This is a neutral constant-attenuation hypothesis; similar dark material changes "
                "can be confused with shadows. Use trusted Fit mask support and validate your job. "
                "No registration, penumbra model or final inspection judgment is implied. "
                "API: https://docs.opencv.org/5.0/main_modules/imgproc_shape.html"
            )
            graph.save(folder / filename)
        finally:
            graph.close()


if __name__ == "__main__":
    main()
