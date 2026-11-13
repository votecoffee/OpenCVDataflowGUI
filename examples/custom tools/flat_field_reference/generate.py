"""Original 16-bit sensor shading, repeated references and held-out reflectance truth."""

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
    y, x = np.mgrid[:64, :96]
    offset = 4 * x + 8 * y
    dark = np.stack([2000 + offset, 1800 + offset, 1600 + offset], axis=2)
    base = 20000 + 128 * x + 64 * y
    response = np.stack([base, base * 3 // 4, base // 2], axis=2)
    flat = dark + response
    reflectance = np.full((64, 96), 0.5)
    reflectance[22:43, 34:63] = 0.25
    observed = dark + response * reflectance[..., None]
    truth = np.full((64, 96, 3), 16384, np.uint16)
    truth[22:43, 34:63] = 8192
    images = {"held_out": observed, "truth": truth}
    for i, noise in enumerate((-16, 0, 16)):
        images[f"bright_{i}"] = flat + noise
        images[f"dark_{i}"] = dark + noise
    outlier = images["bright_2"].copy()
    outlier[12:28, 15:37] += 4000
    images["bright_outlier"] = outlier
    for name, image in images.items():
        assert cv2.imwrite(str(assets / f"{name}.png"), image.astype(np.uint16)[..., ::-1])
    (assets / "truth.json").write_text(
        json.dumps(
            {
                "reference": "Direct reflectance assignment, independent of correction blocks",
                "image_size": [96, 64],
                "channel_order": "RGB",
                "input_depth": "uint16",
                "training_noise": [-16, 0, 16],
                "median_outlier": {"rectangle": [15, 12, 22, 16], "amplitude": 4000},
                "held_out": "Offset + channel response times reflectance; not a training capture",
                "expected_background": 16384,
                "expected_rectangle": 8192,
                "reflectance_rectangle": [34, 22, 29, 21],
                "maximum_error": 0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    registry = builtin_registry()

    def source(graph, identifier, asset, position):
        block = graph.add(
            registry.create(
                "input.file",
                block_id=identifier,
                parameters={
                    "path": f"assets/{asset}.png",
                    "path_base": "project",
                    "color_format": "rgb",
                    "pixel_depth": "unchanged",
                },
            )
        )
        block.position = position
        return block

    for method, filename in (("mean", "project.json"), ("median", "median.json")):
        graph = Graph()
        try:
            for row, kind in enumerate(("bright", "dark")):
                batch = graph.add(
                    registry.create("flow.batch", block_id=f"{kind}_batch", parameters={"size": 3})
                )
                batch.position, batch.name = [-200.0, row * 800.0], f"Three {kind} captures"
                for i in range(3):
                    asset = f"{kind}_{i}"
                    if method == "median" and kind == "bright" and i == 2:
                        asset = "bright_outlier"
                    block = source(graph, f"{kind}_{i}", asset, [-600.0, row * 800.0 + i * 240.0])
                    graph.connect(block.id, "image", batch.id, f"item_{i + 1}")
            build = graph.add(
                registry.create(
                    "calibration.flat_field_reference",
                    block_id="reference",
                    parameters={
                        "method": method,
                        "alignment": "pixel_grid",
                        "retain_noise": True,
                    },
                )
            )
            build.name, build.position = f"Build {method} references", [200.0, 400.0]
            graph.connect("bright_batch", "group", "reference", "bright")
            graph.connect("dark_batch", "group", "reference", "dark")
            source(graph, "held_out", "held_out", [200.0, 0.0])
            source(graph, "truth", "truth", [1000.0, 0.0])
            correction = graph.add(
                registry.create(
                    "cv.flat_field",
                    block_id="correct",
                    parameters={
                        "target_mode": "fixed",
                        "target_level": 0.5,
                        "reference_depth": "raw_sample_units",
                    },
                )
            )
            correction.position = [600.0, 220.0]
            graph.connect("held_out", "image", "correct", "image")
            graph.connect("reference", "flat", "correct", "flat")
            graph.connect("reference", "dark", "correct", "dark")
            for name, source_id in (("before", "held_out"), ("verify", "correct")):
                metric = graph.add(
                    registry.create(
                        "inspect.image_quality",
                        block_id=name,
                        parameters={
                            "metric": "max_absolute_error",
                            "check_threshold": name == "verify",
                            "threshold": 0,
                        },
                    )
                )
                metric.name, metric.position = (
                    ("Verify independent truth" if name == "verify" else "Before correction error"),
                    [1400.0, 240.0 if name == "verify" else 0.0],
                )
                graph.connect(source_id, "image", name, "image_a")
                graph.connect("truth", "image", name, "image_b")
            note = graph.add(registry.create("layout.comment", block_id="about"))
            note.name, note.node_size, note.position = (
                "Original calibration fixtures — MIT",
                "large",
                [-600.0, -340.0],
            )
            note.description = (
                "Original 16-bit RGB simulation, projects and generator under the repository's MIT "
                "license. Three bright/dark captures build mean or median references. The median "
                "project has an outlier in the third bright capture. A separate held-out image "
                "has 1/2 reflectance outside and 1/4 inside its rectangle. Correction to a 1/2 "
                "full-scale target produces 16384 and 8192 counts, matching independent pixels "
                "exactly. This verifies arithmetic and metadata contracts; physical camera "
                "settings require separate qualification."
            )
            graph.save(folder / filename)
        finally:
            graph.close()


if __name__ == "__main__":
    main()
