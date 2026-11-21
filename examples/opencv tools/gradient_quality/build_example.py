"""Build the synthetic reference, blur and noise GMSD example."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def chart():
    """Make reproducible edges, flat patches, ramps and narrow lines."""
    y, x = np.indices((192, 256))
    canvas = np.full((192, 256, 3), (68, 85, 94), np.uint8)
    checker = ((x[12:86, 12:124] // 10 + y[12:86, 12:124] // 10) % 2)[..., None]
    canvas[12:86, 12:124] = np.where(checker, (30, 200, 236), (190, 69, 34))
    canvas[98:179, 12:124] = (
        28 + (x[98:179, 12:124] - 12)[..., None] * np.array([1.8, 1.1, 0.6])
    ).astype(np.uint8)
    cv2.circle(canvas, (186, 49), 33, (225, 184, 224), -1, cv2.LINE_AA)
    cv2.circle(canvas, (186, 49), 15, (40, 49, 65), -1, cv2.LINE_AA)
    for i, width in enumerate((1, 2, 3, 4, 6, 8, 12)):
        start = 137 + 14 * i
        canvas[101:178, start : start + width] = (195, 245, 225)
    return canvas


def build(output):
    output.mkdir(parents=True, exist_ok=True)
    asset = output / "reference.png"
    ok, png = cv2.imencode(".png", chart(), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise OSError(f"Could not encode {asset}")
    asset.write_bytes(png.tobytes())
    registry = builtin_registry()
    graph = Graph()

    def add(type_id, key, name, x, y, **parameters):
        block = graph.add(registry.create(type_id, block_id=key, parameters=parameters))
        block.name, block.position = name, [x, y]
        return block

    try:
        comment = add("layout.comment", "comment", "Gradient quality", 0, -220)
        comment.node_size = "large"
        comment.description = (
            "Compare a reference to itself, a mild Gaussian blur, and seeded Gaussian noise. "
            "Inspect GMSD and the fixed-scale gradient similarity loss. "
            "The original synthetic chart is covered by this repository's MIT license."
        )
        add(
            "input.file",
            "reference",
            "Synthetic color chart",
            0,
            150,
            path="reference.png",
            path_base="project",
        )
        add("cv.gaussian_blur", "blur", "Blur the reference", 260, 150, kernel=5, sigma=1.1)
        add(
            "cv.add_image_noise",
            "noise",
            "Add seeded noise",
            260,
            360,
            seed=6771,
            standard_deviation=46,
        )
        for key in ("blur", "noise"):
            graph.connect("reference", "image", key, "image")
        for key, source, label, y in (
            ("identity_quality", "reference", "Identical pair", -40),
            ("blur_quality", "blur", "Blurred pair", 150),
            ("noise_quality", "noise", "Noisy pair", 360),
        ):
            add(
                "inspect.gmsd_quality",
                key,
                label,
                550,
                y,
                channel_mode="per_channel",
                image_alignment="same_frame",
            )
            graph.connect("reference", "image", key, "reference")
            graph.connect(source, "image", key, "image")
        validated = validate_graph(graph)
        if not validated["valid"]:
            raise RuntimeError(validated["issues"])
        graph.save(output / "gradient_quality.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
