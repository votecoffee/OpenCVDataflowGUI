"""Create an original noisy color step and compare two smoothing methods."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7803)
    height, width, seam = 112, 160, 80
    clean = np.empty((height, width, 3), np.uint8)
    clean[:, :seam] = (58, 70, 88)
    clean[:, seam:] = (164, 184, 209)
    noisy = np.rint(clean + rng.normal(0.0, 9, clean.shape)).clip(0, 255).astype(np.uint8)
    impulses = rng.choice(height * width, 36, replace=False)
    y, x = np.unravel_index(impulses, (height, width))
    noisy[y, x] = rng.choice([0, 255], (len(impulses), 1))
    for name, image in (("clean", clean), ("noisy", noisy)):
        ok, data = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode {name}")
        (assets / f"{name}.png").write_bytes(data.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Measure noise and edge strength", 0, -230)
        note.node_size = "large"
        note.description = (
            "Run Once. Compare flat-area RMSE against the clean target and inspect mean column "
            "profiles across x=79.5. Both filters use a 7-by-7 neighborhood. The variance-adaptive "
            "branch checks RMSE <= 2 on the shaded patch. Original MIT fixtures."
        )
        for name, title, y in (("noisy", "Noisy color step", 30), ("clean", "Clean color step", 490)):
            add(
                "input.file", name, title, 0, y,
                path=f"assets/{name}.png", path_base="project", color_format="bgr",
            )
        add("cv.variance_adaptive_smoothing", "adaptive", "Variance-adaptive smoother", 300, 45)
        add("cv.gaussian_blur", "gaussian", "7 × 7 Gaussian", 300, 260, kernel=7, sigma=0.0)
        add(
            "region.shape", "flat", "Noise measurement patch", 300, 485,
            rectangle=[12, 10, 56, 92],
        )
        add(
            "region.shape", "strip", "Central profile strip", 300, 700,
            rectangle=[0, 10, 160, 92],
        )
        graph.connect("noisy", "image", "flat", "image")
        graph.connect("noisy", "image", "strip", "image")
        for key in ("adaptive", "gaussian"):
            graph.connect("noisy", "image", key, "image")
        for index, (key, title) in enumerate(
            (("adaptive", "Adaptive"), ("gaussian", "Gaussian"), ("noisy", "Unfiltered"))
        ):
            top = 30 + 230 * index
            comparison = add(
                "inspect.image_quality", f"{key}_rmse", f"{title} flat-area RMSE", 625, top,
                metric="rmse", check_threshold=(key == "adaptive"), threshold=2.0,
            )
            profile = add(
                "cv.intensity_profile", f"{key}_profile", f"{title} edge profile", 905, top,
            )
            graph.connect(key, "image", comparison.id, "image_a")
            graph.connect("clean", "image", comparison.id, "image_b")
            graph.connect("flat", "mask", comparison.id, "mask")
            graph.connect(key, "image", profile.id, "image")
            graph.connect("strip", "mask", profile.id, "mask")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise RuntimeError(checked["issues"])
        graph.save(output / "variance_smoothing.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
