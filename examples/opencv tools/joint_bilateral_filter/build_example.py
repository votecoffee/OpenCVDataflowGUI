"""Generate synthetic mixed-depth inputs and a joint bilateral example."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

SHAPE = (112, 168)
REGION = (13, 8, 144, 62)


def images():
    clean = np.full(SHAPE, 23000, np.uint16)
    clean[:, 84:] = 31000
    clean[18:94, 108:113] += 2900
    clean[67:88, 22:52] += 4000

    guide = np.full((*SHAPE, 3), (46, 79, 183), np.uint8)
    guide[:, 84:] = (182, 187, 51)
    guide[18:94, 108:113] = (27, 130, 247)
    guide[67:88, 22:52] = (193, 65, 80)

    rng = np.random.default_rng(4514)
    noisy = np.rint(clean + rng.normal(0, 2200, clean.shape)).astype(np.uint16)
    return {"clean": clean, "guide": guide, "noisy": noisy}


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name, image in images().items():
        ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode {name}")
        (assets / f"{name}.png").write_bytes(encoded.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(type_id, key, title, x, y, **parameters):
        block = graph.add(registry.create(type_id, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Joint bilateral filtering", 0, -245)
        note.node_size = "large"
        note.description = (
            "Compare bilateral filtering with and without a registered color guide. "
            "The guide shows the boundaries of the step and faint stripe in a 16-bit image. "
            "The original synthetic images are covered by this repository's MIT license."
        )
        for key, label, y in (
            ("noisy", "Noisy 16-bit source", 90),
            ("guide", "Registered 8-bit color guide", 330),
        ):
            add(
                "input.file",
                key,
                label,
                0,
                y,
                path=f"assets/{key}.png",
                path_base="project",
                pixel_depth="unchanged",
                color_format="unchanged",
            )
        area = add(
            "region.shape", "region", "Upper inspection area", 260, 330, rectangle=list(REGION)
        )
        area.exposed_outputs = ("region",)
        graph.connect("noisy", "image", "region", "image")
        for key, label, y in (
            ("self_guided", "Bilateral on source", -40),
            ("guided", "Edges from registered guide", 138),
            ("selected", "Update selected output pixels", 335),
        ):
            add(
                "cv.joint_bilateral_filter",
                key,
                label,
                545,
                y,
                diameter=17,
                sigma_color=0.12,
                sigma_space=4.0,
                guide_alignment="same_frame" if key == "self_guided" else "pixel_grid",
            )
            graph.connect("noisy", "image", key, "image")
            if key != "self_guided":
                graph.connect("guide", "image", key, "guide")
        graph.connect("region", "region", "selected", "region")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise RuntimeError(checked["issues"])
        graph.save(output / "joint_bilateral_filter.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
