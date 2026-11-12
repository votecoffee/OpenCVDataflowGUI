"""Generate original edge fixtures and two editable line-pair inspection graphs."""

from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def save_image(path, image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Unable to encode the inspection fixture")
    path.write_bytes(encoded.tobytes())


def make_project(folder, mode):
    registry = builtin_registry()
    graph = Graph()
    names = [f"assets/{mode}-pass.png", f"assets/{mode}-fail.png"]
    source = graph.add(
        registry.create(
            "input.file",
            block_id="source",
            parameters={
                "source_mode": "files",
                "paths": names,
                "path_base": "project",
                "repeat": "stop",
                "color_format": "gray",
            },
        )
    )
    source.name = "Pass and fail fixtures"
    crops = (
        (
            {"x": 90, "y": 45, "width": 170, "height": 30},
            {"x": 25, "y": 90, "width": 95, "height": 120},
        )
        if mode == "corner"
        else (
            {"x": 40, "y": 68, "width": 220, "height": 20},
            {"x": 40, "y": 91, "width": 220, "height": 40},
        )
    )
    for i, (suffix, bounds) in enumerate(zip(("a", "b"), crops, strict=True)):
        previous, output = source.id, "image"
        tools = (
            ("crop", "cv.crop", bounds, "image"),
            (
                "detect",
                "inspect.hough_lines",
                {"votes": 30, "min_length": 40.0, "max_gap": 3.0},
                "collection",
            ),
            (
                "sort",
                "collection.sort",
                {"metric": "length", "direction": "descending"},
                "collection",
            ),
            ("fit", "inspect.fit_shape", {"shape": "line"}, "fit"),
        )
        for column, (name, kind, settings, outgoing) in enumerate(tools):
            block = graph.add(
                registry.create(kind, block_id=f"{name}_{suffix}", parameters=settings)
            )
            block.position = [280.0 + column * 265, i * 270.0]
            graph.connect(
                previous, output, block.id, "image" if name in ("crop", "detect") else "collection"
            )
            previous, output = block.id, outgoing
    parameters = (
        {
            "check_angle": True,
            "target_angle": 90.0,
            "angle_tolerance": 2.0,
            "maximum_extension": 80.0,
        }
        if mode == "corner"
        else {
            "mode": "parallel_gap",
            "check_gap": True,
            "minimum_gap": 19.0,
            "maximum_gap": 21.0,
            "minimum_overlap": 50.0,
            "maximum_parallel_angle": 2.0,
        }
    )
    measure = graph.add(
        registry.create("inspect.line_pair", block_id="measure", parameters=parameters)
    )
    measure.position = [1400.0, 120.0]
    graph.connect(source.id, "image", measure.id, "image")
    graph.connect("fit_a", "fit", measure.id, "line_a")
    graph.connect("fit_b", "fit", measure.id, "line_b")
    note = graph.add(registry.create("layout.comment", block_id="fixture_note"))
    note.name = "Original synthetic inspection fixtures"
    note.description = (
        "Original synthetic images, generator and projects under this repository's MIT license. "
        "Run all to compare a passing and a failing fixture. Each edge is detected, sorted "
        "and fitted in a separate crop. Measure line pair maps those crop coordinates back "
        "to the original image. Open the measurement result image to inspect its overlays."
    )
    note.position = [0.0, -260.0]
    note.node_size = "large"
    graph.save(folder / f"{mode}.json")
    graph.close()


def main():
    folder = Path(__file__).resolve().parent
    assets = folder / "assets"
    assets.mkdir(exist_ok=True)
    for label, tilt in (("pass", 0.0), ("fail", 10.0)):
        image = np.zeros((240, 320), np.uint8)
        cv2.line(image, (60, 60), (270, 60), 255, 1)
        end = (round(60 + 150 * np.tan(np.radians(tilt))), 210)
        cv2.line(image, (60, 60), end, 255, 1)
        save_image(assets / f"corner-{label}.png", image)
    for label, gap in (("pass", 20), ("fail", 30)):
        image = np.zeros((240, 320), np.uint8)
        cv2.line(image, (25, 80), (285, 80), 255, 1)
        cv2.line(image, (25, 80 + gap), (285, 80 + gap), 255, 1)
        save_image(assets / f"gap-{label}.png", image)
    for mode in ("corner", "gap"):
        make_project(folder, mode)


if __name__ == "__main__":
    main()
