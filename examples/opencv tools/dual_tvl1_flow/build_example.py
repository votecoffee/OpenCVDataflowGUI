"""Build a moving-patch replay that compares Dual TV-L1 and Farneback."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

SHIFT = (5, -3)
MEASUREMENT = (43, 34, 33, 30)
TVL1_SETTINGS = {
    "scale_step": 0.65,
    "warps": 4,
    "inner_iterations": 15,
    "outer_iterations": 4,
}


def images():
    rng = np.random.default_rng(1650)
    background = cv2.GaussianBlur(rng.integers(22, 75, (96, 120), dtype=np.uint8), (0, 0), 1.4)
    patch = cv2.GaussianBlur(rng.integers(80, 250, (54, 57), dtype=np.uint8), (0, 0), 1.8)
    cv2.circle(patch, (18, 24), 8, 78, -1)
    cv2.rectangle(patch, (31, 19), (43, 32), 211, -1)
    previous, current = background.copy(), background.copy()
    x, y = 31, 22
    dx, dy = SHIFT
    previous[y : y + 54, x : x + 57] = patch
    current[y + dy : y + dy + 54, x + dx : x + dx + 57] = patch
    return {"previous": previous, "current": current}


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name, image in images().items():
        ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode {name}")
        (assets / f"{name}.png").write_bytes(encoded.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(type_id, key, name, x, y, **parameters):
        block = graph.add(registry.create(type_id, block_id=key, parameters=parameters))
        block.name, block.position = name, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Compare motion estimates", 0, -230)
        note.node_size = "large"
        note.description = (
            "Run All processes two frames. The textured patch shifts 5 px right and 3 px up. "
            "Compare visual direction, round-trip error, and status for the interior region. "
            "Both synthetic images are original MIT-licensed assets."
        )
        add(
            "input.file",
            "frames",
            "Two-frame replay",
            0,
            90,
            source_mode="files",
            paths=["assets/previous.png", "assets/current.png"],
            path_base="project",
            color_format="gray",
            repeat="stop",
        )
        region = add(
            "region.shape",
            "interior",
            "Patch measurement area",
            265,
            275,
            rectangle=list(MEASUREMENT),
        )
        region.exposed_outputs = ("region",)
        graph.connect("frames", "image", "interior", "image")
        for kind, title, y, settings in (
            ("dual_tvl1", "Dual TV-L1 on the patch", -20, TVL1_SETTINGS),
            ("farneback", "Farneback baseline", 230, {}),
        ):
            block = add(
                f"sequence.{kind}",
                kind,
                title,
                555,
                y,
                max_gap_ms=0,
                consistency_tolerance=0.5,
                **settings,
            )
            block.exposed_outputs = (
                "flow",
                "visualization",
                "consistency_mask",
                "consistency_error",
                "status",
            )
            block.preview_output = "visualization"
            graph.connect("frames", "image", kind, "image")
            graph.connect("interior", "region", kind, "region")
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise RuntimeError(validation["issues"])
        graph.save(output / "dual_tvl1_flow.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
