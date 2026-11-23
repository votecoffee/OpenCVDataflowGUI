"""Generate a textured synthetic part and a rolling guidance example."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

SHAPE = (144, 216)
REGION = (20, 28, 176, 84)


def images():
    clean = np.full(SHAPE, 56, np.uint8)
    clean[20:122, 53:170] = 174
    clean[20:47, 135:170] = 56
    cv2.circle(clean, (111, 77), 17, 81, -1)
    y, x = np.indices(SHAPE)
    texture = np.where((x // 2 + y // 2) % 2 == 0, 18, -18)
    rng = np.random.default_rng(4515)
    sample = np.rint(clean + texture + rng.normal(0, 2.5, SHAPE)).astype(np.uint8)
    return {"clean": clean, "textured": sample}


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for key, image in images().items():
        ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode {key}")
        (assets / f"{key}.png").write_bytes(encoded.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(type_id, key, name, x, y, **parameters):
        block = graph.add(registry.create(type_id, block_id=key, parameters=parameters))
        block.name, block.position = name, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Rolling guidance on a textured part", 0, -240)
        note.node_size = "large"
        note.description = (
            "Compare one and four passes on a textured plate with a notch and a round cutout. "
            "A third branch writes the four-pass result inside a rectangle. "
            "The original synthetic assets are covered by this repository's MIT license."
        )
        add(
            "input.file",
            "textured",
            "Textured part",
            0,
            140,
            path="assets/textured.png",
            path_base="project",
            color_format="unchanged",
        )
        region = add("region.shape", "region", "Output rectangle", 270, 330, rectangle=list(REGION))
        region.exposed_outputs = ("region",)
        graph.connect("textured", "image", "region", "image")
        for key, title, y, iterations in (
            ("single", "Smooth once", -35, 1),
            ("rolling", "Roll the guide four times", 145, 4),
            ("selected", "Four passes inside the rectangle", 330, 4),
        ):
            add(
                "cv.rolling_guidance_filter",
                key,
                title,
                530,
                y,
                diameter=17,
                sigma_color=0.08,
                sigma_space=4,
                iterations=iterations,
            )
            graph.connect("textured", "image", key, "image")
        graph.connect("region", "region", "selected", "region")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise RuntimeError(checked["issues"])
        graph.save(output / "rolling_guidance_filter.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
