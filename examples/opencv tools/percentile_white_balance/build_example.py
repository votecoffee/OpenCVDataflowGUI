"""Build four cast gray steps and bright and dark outliers."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def chart():
    colors = np.array([[24, 58, 40], [64, 92, 81], [104, 126, 122], [144, 160, 163]], np.uint8)
    image = np.tile(np.repeat(colors, 32, axis=0)[None, ...], (100, 1, 1))
    image[0, :] = (2, 3, 4)
    image[-1, :] = (240, 244, 252)
    return image


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", chart(), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise OSError("Could not encode the cast gray chart")
    (assets / "cast_chart.png").write_bytes(encoded.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Balance four gray steps with outliers", 0, -250)
        note.node_size = "large"
        note.description = (
            "Run Once. The first and last rows are outliers. Compare the branch that uses "
            "every sample with the two branches that trim 2% from each tail. The filmstrip "
            "shows the exact percentile result. Original MIT-licensed fixture."
        )
        add(
            "input.file", "source", "Gray steps with a color cast", 0, 180,
            path="assets/cast_chart.png", path_base="project", color_format="unchanged",
            pixel_depth="unchanged",
        )
        for key, title, y, method, tail in (
            ("untrimmed", "Exact without tail trim", -45, "exact_percentiles", 0),
            ("trimmed", "Exact with 2% tail trim", 180, "exact_percentiles", 2),
            ("native", "OpenCV histogram with 2% trim", 405, "opencv_histogram", 2),
        ):
            add(
                "cv.percentile_white_balance", key, title, 365, y, method=method,
                tail_percent=tail, range_mode="explicit", output_min=16, output_max=235,
            )
            graph.connect("source", "image", key, "image")
        add("output.filmstrip_image", "filmstrip", "Review the trimmed result", 745, 180)
        graph.connect("trimmed", "image", "filmstrip", "image")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise RuntimeError(checked["issues"])
        graph.save(output / "percentile_white_balance.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
