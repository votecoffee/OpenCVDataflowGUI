"""Original asymmetric brightness fixture with analytic rectangle-mixture truth."""

import json
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
    image = np.full((192, 256), 1000, np.uint16)
    image[48:144, 48:208] = 7000
    image[112:132, 160:192] = 55000
    assert cv2.imwrite(str(assets / "asymmetric-brightness.png"), image)
    # Treat the bright patch as extra weight on top of the uniform base rectangle.
    base_mass, extra_mass = 160 * 96 * 6000, 32 * 20 * 48000
    fraction = extra_mass / (base_mass + extra_mass)
    base_center, patch_center = np.array([127.5, 95.5]), np.array([175.5, 121.5])
    delta = patch_center - base_center
    base_covariance = np.diag([(160**2 - 1) / 12, (96**2 - 1) / 12])
    patch_covariance = np.diag([(32**2 - 1) / 12, (20**2 - 1) / 12])
    covariance = (
        (1 - fraction) * base_covariance
        + fraction * patch_covariance
        + fraction * (1 - fraction) * np.outer(delta, delta)
    )
    center = base_center + fraction * delta
    angle = (
        -math.degrees(math.atan2(2 * covariance[0, 1], covariance[0, 0] - covariance[1, 1])) / 2
    ) % 180
    truth = {
        "method": "Analytic mixture of two discrete uniform rectangles",
        "background": 1000,
        "mass": base_mass + extra_mass,
        "geometric_center": base_center.tolist(),
        "geometric_angle_degrees": 0,
        "weighted_center": center.tolist(),
        "weighted_covariance": covariance.tolist(),
        "weighted_angle_degrees": angle,
    }
    (assets / "truth.json").write_text(
        json.dumps(truth, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    graph, registry = Graph(), builtin_registry()
    try:
        source = graph.add(
            registry.create(
                "input.file",
                block_id="source",
                parameters={
                    "path": "assets/asymmetric-brightness.png",
                    "path_base": "project",
                    "color_format": "gray",
                    "pixel_depth": "unchanged",
                },
            )
        )
        source.name = "Original 16-bit asymmetric brightness"
        region = graph.add(
            registry.create(
                "region.shape",
                block_id="selection",
                parameters={
                    "rectangle": [40, 40, 176, 112],
                },
            )
        )
        region.position, region.exposed_outputs = [0.0, 310.0], ["region"]
        graph.connect(source.id, "image", region.id, "image")
        for index, (name, weighting) in enumerate(
            (("weighted", "intensity"), ("geometric", "uniform_foreground"))
        ):
            block = graph.add(
                registry.create(
                    "inspect.intensity_moments",
                    block_id=name,
                    parameters={
                        "weighting": weighting,
                        "background": 1000.0,
                        "require_orientation": True,
                        "emit_weights": True,
                    },
                )
            )
            block.name = "Intensity-weighted datum" if name == "weighted" else "Geometric datum"
            block.position = [400.0, index * 340.0]
            graph.connect(source.id, "image", block.id, "image")
            graph.connect(region.id, "region", block.id, "region")
        follow = graph.add(
            registry.create(
                "region.pose",
                block_id="follow",
                parameters={
                    "shape": "rectangle",
                    "rectangle": [-8, -5, 17, 11],
                },
            )
        )
        follow.name, follow.position = "Inspection region follows weighted pose", [800.0, 0.0]
        graph.connect("weighted", "image", follow.id, "image")
        graph.connect("weighted", "pose", follow.id, "pose")
        note = graph.add(registry.create("layout.comment", block_id="about"))
        note.name = "Original weighted moments fixture — MIT"
        note.description = (
            "Original synthetic 16-bit image, generator and graph under the repository MIT "
            "license. The bright patch moves the weighted center from the geometric center "
            "and rotates its principal axis. Analytic rectangle-mixture truth is stored in "
            "assets/truth.json. Green marks/ellipses show the chosen measurement; blue marks "
            "show uniform-support reference geometry. Ellipses are covariance displays. "
            "The final region follows the reusable weighted pose in the same pixel grid."
        )
        note.position, note.node_size = [0.0, -310.0], "large"
        graph.save(folder / "project.json")
    finally:
        graph.close()


if __name__ == "__main__":
    main()
