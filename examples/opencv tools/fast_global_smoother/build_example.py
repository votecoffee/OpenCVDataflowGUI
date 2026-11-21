"""Generate a noisy inspection target and its registered synthetic guide."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HEIGHT, WIDTH = 128, 192
REGION = (16, 8, 160, 64)


def images():
    clean = np.full((HEIGHT, WIDTH), 62, np.uint8)
    clean[:, WIDTH // 2 :] = 192
    clean[85:112, 27:58] = 135
    guide = np.empty((HEIGHT, WIDTH, 3), np.uint8)
    guide[:] = (167, 63, 38)
    guide[:, WIDTH // 2 :] = (36, 195, 234)
    guide[85:112, 27:58] = (18, 220, 99)
    rng = np.random.default_rng(131)
    noisy = np.clip(np.rint(clean + rng.normal(0, 11, clean.shape)), 0, 255).astype(np.uint8)
    return {"ground_truth": clean, "guide": guide, "noisy": noisy}


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for key, pixels in images().items():
        ok, png = cv2.imencode(".png", pixels, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode {key}")
        (assets / f"{key}.png").write_bytes(png.tobytes())

    registry = builtin_registry()
    graph = Graph()

    def add(type_id, key, name, x, y, **parameters):
        block = graph.add(registry.create(type_id, block_id=key, parameters=parameters))
        block.name, block.position = name, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Image smoothing with a registered guide", 0, -230)
        note.node_size = "large"
        note.description = (
            "Compare self-guided smoothing, a clean registered color guide, and a region that "
            "limits where the smoothed result is written. The original synthetic images are "
            "covered by this repository's MIT license."
        )
        for key, title, y in (("noisy", "Noisy sample", 120), ("guide", "Registered guide", 330)):
            add(
                "input.file",
                key,
                title,
                0,
                y,
                path=f"assets/{key}.png",
                path_base="project",
                color_format="unchanged",
            )
        region = add(
            "region.shape", "region", "Upper search region", 260, 330, rectangle=list(REGION)
        )
        region.exposed_outputs = ("region",)
        graph.connect("noisy", "image", "region", "image")
        for key, title, y in (
            ("self_guided", "Smooth without a separate guide", -50),
            ("guided", "Preserve edges from the color guide", 135),
            ("selected", "Write the result only inside the region", 330),
        ):
            add(
                "cv.fast_global_smoother",
                key,
                title,
                565,
                y,
                sigma_color=14.0,
                guide_alignment="pixel_grid" if key != "self_guided" else "same_frame",
            )
            graph.connect("noisy", "image", key, "image")
            if key != "self_guided":
                graph.connect("guide", "image", key, "guide")
        graph.connect("region", "region", "selected", "region")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise RuntimeError(checked["issues"])
        graph.save(output / "fast_global_smoother.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
