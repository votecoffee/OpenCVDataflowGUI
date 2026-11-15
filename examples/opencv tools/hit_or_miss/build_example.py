"""Original MIT one-pixel patterns with explicit expected anchor coordinates."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
KERNELS = {
    "isolated": [[-1, -1, -1], [-1, 1, -1], [-1, -1, -1]],
    "junctions": [[-1, 1, -1], [1, 1, 1], [-1, 1, -1]],
    "right_endpoints": [[-1, -1, -1], [1, 1, -1], [-1, -1, -1]],
    "corners": [[-1, 1, -1], [-1, 1, 1], [-1, -1, -1]],
}
POINTS = {
    "isolated": [[12, 12], [36, 12], [60, 12]],
    "junctions": [[20, 36], [52, 36]],
    "right_endpoints": [[28, 64], [73, 64]],
    "corners": [[84, 36]],
}
CASES = (
    "binary_patterns",
    "dark_patterns",
    "selected_centers",
    "border_support",
    "count_failure",
    "invalid_binary",
)


def fixture():
    image = np.zeros((80, 112), np.uint8)
    image[0, 0] = 255
    for x, y in POINTS["isolated"]:
        image[y, x] = 255
    for x, y in POINTS["junctions"]:
        image[y, x - 1 : x + 2] = 255
        image[y - 1 : y + 2, x] = 255
    image[35, 84] = image[36, 84] = image[36, 85] = 255
    image[64, 12:29] = image[64, 52:74] = 255
    return image


def build():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    image = fixture()
    gray = image.copy()
    gray[12, 12] = 127
    for name, value in (
        ("patterns", image),
        ("dark", 255 - image),
        ("invalid", gray),
    ):
        ok, data = cv2.imencode(".png", value)
        assert ok
        (assets / f"{name}.png").write_bytes(data.tobytes())
    registry = builtin_registry()
    for case in CASES:
        graph = Graph()

        def add(kind, name, position, **parameters):
            block = graph.add(registry.create(kind, block_id=name, parameters=parameters))
            block.position = list(position)
            return block

        file = (
            "dark"
            if case == "dark_patterns"
            else "invalid"
            if case == "invalid_binary"
            else "patterns"
        )
        add(
            "input.file",
            "source",
            (0, 200),
            path=f"assets/{file}.png",
            path_base="project",
            color_format="gray",
            pixel_depth="unchanged",
        )
        patterns = (
            tuple(KERNELS)
            if case == "binary_patterns"
            else ("isolated", "junctions")
            if case == "dark_patterns"
            else ("isolated",)
        )
        for row, name in enumerate(patterns):
            expected = len(POINTS[name])
            if case == "selected_centers":
                expected = 1
            elif case == "count_failure":
                expected = 4
            tool = add(
                "cv.hit_or_miss",
                name,
                (750, row * 330),
                custom_kernel=KERNELS[name],
                polarity="dark" if case == "dark_patterns" else "light",
                check_count=True,
                minimum_count=expected,
                maximum_count=expected,
            )
            tool.name = name.replace("_", " ").capitalize()
            graph.connect("source", "image", name, "image")
        if case == "selected_centers":
            add("region.shape", "mask", (380, 550), shape="rectangle", rectangle=[25, 5, 41, 15])
            add("region.shape", "region", (380, 200), shape="circle", center=[36, 12], radius=6)
            graph.connect("source", "image", "region", "image")
            graph.connect("source", "image", "mask", "image")
            graph.connect("mask", "mask", "isolated", "mask")
            graph.connect("region", "region", "isolated", "region")
        elif case == "border_support":
            add(
                "cv.hit_or_miss",
                "native_border",
                (750, 330),
                custom_kernel=KERNELS["isolated"],
                support="native_border",
                check_count=True,
                minimum_count=4,
                maximum_count=4,
            )
            graph.connect("source", "image", "native_border", "image")
        note = add("layout.comment", "scope", (0, -350))
        note.name, note.node_size = "Original fixture and expected matches", "large"
        note.description = (
            "Original synthetic images/projects by OpenCVDataflowGUI contributors, MIT. "
            "No customer or third-party images. A controlled 112x80 one-pixel fixture, not a "
            "physical inspection. Full support: isolated (12,12),(36,12),(60,12); cross junctions "
            "(20,36),(52,36); right endpoints (28,64),(73,64); north/east corner (84,36). "
            "The isolated pixel (0,0) is outside full-kernel support. Native neutral borders "
            "add it as a fourth match. The dark project inverts pixels and polarity. "
            "Selected centers intersects a mask with a circle and returns only (36,12). "
            "Count failure deliberately requires 4 full-support isolated pixels but finds 3: "
            "it emits a valid failed inspection and reason. Invalid binary has one gray 127 pixel "
            "and deliberately emits no processing outputs. Click Edit kernel for signed cells. "
            "Counts refer to matched anchor pixels, not connected parts. See truth.json."
        )
        graph.save(HERE / f"{case}.json")
        graph.close()
    truth = {
        "creator": "OpenCVDataflowGUI contributors",
        "license": "MIT",
        "image_size": [112, 80],
        "kernels": KERNELS,
        "points": POINTS,
        "native_border_extra": [[0, 0]],
        "cases": list(CASES),
        "asset_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(assets.glob("*.png"))
        },
    }
    (HERE / "truth.json").write_bytes((json.dumps(truth, indent=2) + "\n").encode())


if __name__ == "__main__":
    build()
