"""Build a textured image pair with a known similarity transform and coarse seed."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def make_pair():
    height, width = 161, 193
    y, x = np.indices((height, width), dtype=float)
    fade = np.clip(np.minimum.reduce([x, width - 1 - x, y, height - 1 - y]) / 32, 0, 1)
    noise = cv2.GaussianBlur(np.random.default_rng(72).normal(0, 1, x.shape), (0, 0), 2)
    pattern = 0.35 + 0.085 * np.sin(x * 0.28 + y * 0.13) + 0.08 * np.cos(y * 0.34 - x * 0.12)
    reference = np.rint(np.clip(fade * (pattern + 0.8 * noise), 0, 1) * 255).astype(np.uint8)
    center = width / 2, height / 2
    truth = cv2.getRotationMatrix2D(center, 10, 1.06)
    truth[:, 2] += [14, -10]
    approximate = cv2.getRotationMatrix2D(center, 8.6, 1.055)
    approximate[:, 2] += [10, -8]
    moving = cv2.warpAffine(reference, truth, (width, height), flags=cv2.INTER_CUBIC)
    return reference, moving, np.vstack((truth, [0, 0, 1])), approximate


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    reference, moving, truth, approximate = make_pair()
    for name, image in (("reference", reference), ("moving", moving)):
        ok, data = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode {name}")
        (assets / f"{name}.png").write_bytes(data.tobytes())
    (assets / "truth.json").write_text(
        json.dumps({"reference_to_moving": truth.tolist()}, indent=2) + "\n",
        encoding="utf-8",
    )

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, position, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, list(position)
        return block

    try:
        note = add("layout.comment", "note", "Refine coarse alignment", (0, -230))
        note.node_size = "large"
        note.description = (
            "Run Once. A coarse map from three approximate points initializes the similarity "
            "fit. The registration block reports overlap and the residual before and after "
            "refinement. Both texture images were generated for this MIT-licensed example."
        )
        for name, height in (("moving", 30), ("reference", 235)):
            add(
                "input.file", name, f"{name.capitalize()} texture", (0, height),
                path=f"assets/{name}.png", path_base="project", color_format="unchanged",
            )
        reference_points = np.array([[26, 27], [143, 36], [57, 125]], np.float64)
        moving_points = cv2.transform(reference_points[:, None], approximate).reshape(-1, 2)
        add(
            "cv.control_point_transform", "coarse", "Approximate moving-to-reference map",
            (0, 440), source_points=moving_points.tolist(), target_points=reference_points.tolist(),
        )
        add(
            "cv.align_mapper", "register", "Refine rotation, scale and shift", (340, 128),
            motion="similarity", iterations_per_level=8,
        )
        add("output.filmstrip_image", "filmstrip", "Review the aligned texture", (685, 128))
        graph.connect("moving", "image", "register", "image")
        graph.connect("reference", "image", "register", "reference")
        graph.connect("coarse", "transform", "register", "initial_transform")
        graph.connect("register", "image", "filmstrip", "image")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise RuntimeError(checked["issues"])
        graph.save(output / "mapper_registration.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
