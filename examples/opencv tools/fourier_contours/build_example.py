"""Build a transformed asymmetric silhouette and an altered-outline inspection."""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

VERTICES = [
    [35, 40], [110, 40], [124, 56], [123, 83], [96, 83], [96, 108],
    [117, 108], [110, 132], [44, 132], [32, 108], [47, 85], [35, 72],
]


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    angle = math.radians(23)
    linear = 1.24 * np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    reference = np.array(VERTICES, dtype=float)
    center = np.array([90.0, 85.0])
    moved = (reference - center) @ linear.T + center + [15, 9]
    defect = moved.copy()
    defect[4:6] += [17, -20]
    for name, vertices in (("reference", reference), ("moved", moved), ("defect", defect)):
        pixels = np.zeros((204, 224), np.uint8)
        cv2.fillPoly(pixels, [np.rint(vertices).astype(np.int32)], 255)
        ok, encoded = cv2.imencode(".png", pixels, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode {name}")
        (assets / f"{name}.png").write_bytes(encoded.tobytes())

    registry = builtin_registry()
    for sample, filename in (("moved", "align_fourier_contours"), ("defect", "reject_altered_contour")):
        graph = Graph()

        def add(kind, key, title, x, y, **parameters):
            block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
            block.name, block.position = title, [x, y]
            return block

        try:
            note = add("layout.comment", "note", "Align an asymmetric silhouette", 0, -220)
            note.node_size = "large"
            note.description = (
                "Run Once. The plate moved, rotated by 23° and scaled by 1.24×. Fourier alignment "
                "registers the ordered outline and gives a sample-to-reference matrix. "
                + ("This sample has an altered notch and should fail shape acceptance. "
                   if sample == "defect" else "The fitted matrix also warps the sample image. ")
                + "Original MIT silhouette."
            )
            fit = add(
                "inspect.fit_fourier_contours", "fit", "Align sample to reference", 850, 80,
                maximum_normalized_rms=0.045,
            )
            fit.exposed_inputs = ["reference", "sample", "reference_image"]
            for role, asset, y in (("reference", "reference", 10), ("sample", sample, 260)):
                add(
                    "input.file", role, f"{role.capitalize()} silhouette", 0, y,
                    path_base="project", path=f"assets/{asset}.png", color_format="gray",
                )
                contour = add(
                    "inspect.contours", f"{role}_outline", f"{role.capitalize()} outline",
                    260, y, expected_count=1, approximation="none",
                )
                contour.exposed_outputs = ["collection"]
                add(
                    "inspect.fourier_contour", f"{role}_spectrum", f"{role.capitalize()} spectrum", 560, y,
                )
                graph.connect(role, "image", contour.id, "image")
                graph.connect(contour.id, "collection", f"{role}_spectrum", "contour")
                graph.connect(f"{role}_spectrum", "descriptor", "fit", role)
            graph.connect("reference", "image", "fit", "reference_image")
            if sample == "moved":
                add("cv.warp_affine", "warp", "Warp sample to reference", 1130, 145)
                graph.connect("sample", "image", "warp", "image")
                graph.connect("fit", "transform", "warp", "transform")
            checked = validate_graph(graph)
            if not checked["valid"]:
                raise RuntimeError(checked["issues"])
            graph.save(output / f"{filename}.json")
        finally:
            graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
