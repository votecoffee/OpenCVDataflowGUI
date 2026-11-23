"""Build the original registered-guide filter comparison."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

SHAPE = (96, 160)
EDGE = 80
RECTANGLE = (13, 8, 134, 65)


def images():
    clean = np.full(SHAPE, 19300, np.uint16)
    clean[:, EDGE:] = 27500
    clean[20:78, 112:117] += 3600
    guide = np.full((*SHAPE, 3), (61, 48, 157), np.uint8)
    guide[:, EDGE:] = (180, 195, 52)
    guide[20:78, 112:117] = (34, 182, 217)
    rng = np.random.default_rng(571)
    noisy = np.rint(clean + rng.normal(0, 2150, SHAPE)).astype(np.uint16)
    return {"clean": clean, "guide": guide, "noisy": noisy}


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name, image in images().items():
        ok, png = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode {name}")
        (assets / f"{name}.png").write_bytes(png.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Denoise a registered signal", 0, -230)
        note.node_size = "large"
        note.description = (
            "Run Once and compare joint bilateral with adaptive manifold. Both use the same "
            "guide and noisy 16-bit source. The lower branch replaces pixels only inside "
            "the inspection rectangle. Original MIT assets."
        )
        for key, title, y in (
            ("noisy", "Noisy 16-bit signal", 45),
            ("guide", "Registered color guide", 275),
        ):
            add(
                "input.file",
                key,
                title,
                0,
                y,
                path=f"assets/{key}.png",
                path_base="project",
                pixel_depth="unchanged",
                color_format="unchanged",
            )
        area = add(
            "region.shape", "region", "Inspection rectangle", 285, 500, rectangle=list(RECTANGLE)
        )
        area.exposed_outputs = ("region",)
        graph.connect("noisy", "image", "region", "image")
        add(
            "cv.joint_bilateral_filter",
            "bilateral",
            "Joint bilateral baseline",
            570,
            -55,
            diameter=21,
            sigma_color=0.08,
            sigma_space=5.0,
            guide_alignment="pixel_grid",
        )
        for key, label, y in (
            ("manifold", "Adaptive manifold", 155),
            ("selected", "Adaptive manifold in region", 365),
        ):
            add(
                "cv.adaptive_manifold_filter",
                key,
                label,
                570,
                y,
                sigma_spatial=8.0,
                sigma_color=0.08,
                guide_alignment="pixel_grid",
            )
        for key in ("bilateral", "manifold", "selected"):
            graph.connect("noisy", "image", key, "image")
            graph.connect("guide", "image", key, "guide")
        graph.connect("region", "region", "selected", "region")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise ValueError(checked["issues"])
        graph.save(output / "adaptive_manifold.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
