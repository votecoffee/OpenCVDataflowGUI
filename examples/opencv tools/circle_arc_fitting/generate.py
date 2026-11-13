"""Original noisy inspection points and independent generating circle/arc truth."""

import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def points(sweep, *, noise=0.0, outliers=False, closed=False):
    angles = np.radians(np.linspace(350, 350 + sweep, 181))
    radius = 60 + np.random.default_rng(134).normal(0, noise, len(angles))
    xy = [128, 96] + radius[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
    if outliers:
        xy[::9] += np.random.default_rng(5).normal(0, 15, (len(xy[::9]), 2))
    if closed:
        xy[-1] = xy[0]
    # Vector data stores float32 rows. Pin that input quantization in the fixture.
    return xy.astype(np.float32).tolist()


def main():
    folder = Path(__file__).resolve().parent
    assets = folder / "assets"
    assets.mkdir(exist_ok=True)
    y, x = np.mgrid[:192, :256]
    scene = np.repeat((11000 + x * 15 + y * 10)[..., None], 3, axis=2).astype(np.uint16)
    cv2.circle(scene, (128, 96), 63, (41000, 43000, 45000), -1)
    cv2.circle(scene, (128, 96), 57, (18000, 20000, 22000), -1)
    for angle in range(0, 360, 30):
        direction = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
        a, b = np.rint([128, 96] + np.array([68, 72])[:, None] * direction).astype(int)
        cv2.line(scene, tuple(a), tuple(b), (35000, 35000, 35000), 1)
    assert cv2.imwrite(str(assets / "scene.png"), scene)
    data = {
        "arc": points(210, noise=0.15, outliers=True),
        "circle": points(360, noise=0.1, closed=True),
        "short": points(5),
    }
    (assets / "truth.json").write_text(
        json.dumps(
            {
                "center": [128, 96],
                "radius": 60,
                "units": "pixels",
                "arc_sweep_degrees": 210,
                "short_sweep_degrees": 5,
                "radial_noise_std": {"arc": 0.15, "circle": 0.1},
                "outlier_every": 9,
                "maximum_center_error": 0.1,
                "maximum_radius_error": 0.1,
                "reference": "Generating center/radius and polar sampling; float32 input rows",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    registry = builtin_registry()

    def base_graph():
        graph = Graph()
        image = graph.add(
            registry.create(
                "input.file",
                block_id="image",
                parameters={
                    "path": "assets/scene.png",
                    "path_base": "project",
                    "color_format": "rgb",
                        "pixel_depth": "unchanged",
                },
            )
        )
        image.position = [-400.0, 0.0]
        note = graph.add(registry.create("layout.comment", block_id="about"))
        note.name = "Original circle/arc fixtures — MIT"
        note.description = (
            "Original 16-bit illustrative part image, polar-generated points, noise/outliers, "
            "generator and graphs under the repository MIT license. Vector data holds known "
            "image-coordinate observations, not an ML model. Fits are checked against the "
            "independent center (128,96), radius 60 pixels. Geometric circles are distinct "
            "from minimum enclosing circles. Conditional radial-noise uncertainty does not "
            "include calibration/systematic errors. Reference: "
            "https://arxiv.org/abs/cs/0301001"
        )
        note.node_size, note.position = "large", [-400.0, -350.0]
        return graph

    graph = base_graph()
    try:
        for row, name in enumerate(("arc", "circle")):
            source = graph.add(
                registry.create(
                    "ml.vector_data", block_id=f"points_{name}", parameters={"values": data[name]}
                )
            )
            source.name, source.position = f"Original {name} observations", [-60.0, row * 400.0]
        for row, (name, source, parameters) in enumerate(
            (
                ("robust", "arc", {}),
                (
                    "ordinary",
                    "arc",
                    {
                        "method": "least_squares",
                        "maximum_rmse": 10.0,
                        "maximum_backtracking_fraction": 0.5,
                        "maximum_gap_degrees": 179.0,
                    },
                ),
                ("circle", "circle", {}),
            )
        ):
            block = graph.add(
                registry.create("inspect.fit_circle_arc", block_id=name, parameters=parameters)
            )
            block.name = {
                "robust": "Robust partial arc with outliers",
                "ordinary": "Ordinary fit keeps the outliers",
                "circle": "Full circle",
            }[name]
            block.position = [300.0, row * 360.0]
            graph.connect(f"points_{source}", "vectors", block.id, "points")
            graph.connect("image", "image", block.id, "image")
        region = graph.add(
            registry.create("region.fitted_shape", block_id="region", parameters={"margin": -5.0})
        )
        region.name, region.position = "Reusable circle-relative region", [670.0, 720.0]
        graph.connect("circle", "fit", region.id, "fit")
        graph.connect("image", "image", region.id, "image")
        graph.save(folder / "project.json")
    finally:
        graph.close()
    graph = base_graph()
    try:
        source = graph.add(
            registry.create(
                "ml.vector_data", block_id="points_short", parameters={"values": data["short"]}
            )
        )
        source.name, source.position = "Five-degree arc observations", [-60.0, 0.0]
        block = graph.add(registry.create("inspect.fit_circle_arc", block_id="short"))
        block.name, block.position = "Reject insufficient angular support", [300.0, 0.0]
        graph.connect(source.id, "vectors", block.id, "points")
        graph.connect("image", "image", block.id, "image")
        graph.save(folder / "short_arc.json")
    finally:
        graph.close()


if __name__ == "__main__":
    main()
