"""Original translated 16-bit fixture with independent sampled intermediate truth."""

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
    rng = np.random.default_rng(130)
    world = rng.integers(55, 185, (224, 288, 3), dtype=np.uint8)
    world = cv2.GaussianBlur(world, (3, 3), 0.6)
    for row in range(3):
        for column in range(4):
            center = (48 + 56 * column, 48 + 56 * row)
            cv2.circle(world, center, 18, (220, 180, 90), 3)
            cv2.circle(world, center, 9, (35, 40, 45), -1)
            cv2.putText(
                world,
                f"{row}{column}",
                (center[0] - 10, center[1] + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (220, 225, 230),
                1,
                cv2.LINE_AA,
            )
    world = world.astype(np.uint16) * 257
    frames = {
        "first": world[16:208, 16:272],
        "middle": world[14:206, 12:268],
        "last": world[12:204, 8:264],
    }
    for name, image in frames.items():
        assert cv2.imwrite(str(assets / f"{name}.png"), image)
    (assets / "truth.json").write_text(
        json.dumps(
            {
                "source_displacement_pixels": [8, 4],
                "fraction": 0.5,
                "source_times_ms": [0, 40],
                "expected_timestamp_ms": 20,
                "truth_method": "Direct integer crop from the original larger texture",
                "quality_rectangle": [32, 24, 192, 144],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    graph, registry = Graph(), builtin_registry()
    try:
        for index, name in enumerate(("first", "last", "middle")):
            source = graph.add(
                registry.create(
                    "input.file",
                    block_id=name,
                    parameters={
                        "path": f"assets/{name}.png",
                        "path_base": "project",
                        "pixel_depth": "unchanged",
                    },
                )
            )
            source.name, source.position = f"Original {name} frame — uint16", [0.0, 250.0 * index]
        interpolate = graph.add(
            registry.create(
                "cv.interpolate_frames",
                block_id="interpolate",
                parameters={
                    "alignment": "pixel_grid",
                    "levels": 4,
                    "window": 25,
                    "iterations": 5,
                    "poly_n": "7",
                    "poly_sigma": 1.5,
                    "minimum_coverage": 0.8,
                    "timing_source": "explicit",
                    "timing_policy": "required",
                    "time_a_ms": 0.0,
                    "time_b_ms": 40.0,
                },
            )
        )
        interpolate.position = [340.0, 0.0]
        graph.connect("first", "image", interpolate.id, "image_a")
        graph.connect("last", "image", interpolate.id, "image_b")
        region = graph.add(
            registry.create(
                "region.shape",
                block_id="quality_region",
                parameters={
                    "shape": "rectangle",
                    "rectangle": [32, 24, 192, 144],
                },
            )
        )
        region.position, region.exposed_outputs = [680.0, 0.0], ["region"]
        graph.connect(interpolate.id, "image", region.id, "image")
        quality = graph.add(
            registry.create(
                "inspect.image_quality",
                block_id="verify",
                parameters={
                    "metric": "mae",
                    "check_threshold": True,
                    "threshold": 100.0,
                },
            )
        )
        quality.name, quality.position = "Compare to independent middle frame", [1020.0, 200.0]
        graph.connect(interpolate.id, "image", quality.id, "image_a")
        graph.connect("middle", "image", quality.id, "image_b")
        graph.connect(interpolate.id, "coverage", quality.id, "mask")
        graph.connect(region.id, "region", quality.id, "region")
        note = graph.add(registry.create("layout.comment", block_id="about"))
        note.name = "Original frame interpolation fixture — MIT"
        note.description = (
            "Original synthetic images, code and project under the repository MIT license. "
            "Endpoint displacement is 8 right/4 down pixels over 40 ms. The middle reference "
            "is independently cropped at 20 ms. Estimate both flow directions, invert the "
            "motion maps, and compare supported interior pixels to that reference. Missing "
            "motion support stays black by default and remains excluded from the quality "
            "check. Coverage colors distinguish both/one/neither source. This generated "
            "translation does not qualify arbitrary real video or physical synchronization."
        )
        note.position, note.node_size = [0.0, -300.0], "large"
        graph.save(folder / "project.json")
    finally:
        graph.close()


if __name__ == "__main__":
    main()
