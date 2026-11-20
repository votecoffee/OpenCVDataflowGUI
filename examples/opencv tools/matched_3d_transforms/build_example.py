"""Build a comparison of four metric transform models with known outliers."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

TRANSLATION = [0.17, -0.11, 0.31]
OUTLIERS = [2, 9, 14, 22, 25]


def build(output):
    target = output / "matched_3d_transforms.json"
    if target.exists():
        raise FileExistsError("Choose a new example directory")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(741)
    source = rng.uniform(-0.16, 0.16, (26, 3))
    rotation = cv2.Rodrigues(np.array([0.25, -0.33, 0.48]))[0]
    linear_maps = {
        "translation": np.eye(3),
        "rigid": rotation,
        "similarity": 1.24 * rotation,
        "affine": np.array([[1.1, 0.25, -0.12], [0.07, 0.88, 0.16], [-0.08, 0.11, 1.2]]),
    }
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **settings):
        node = graph.add(registry.create(kind, block_id=key, parameters=settings))
        node.position = position
        return node

    try:
        note = add("layout.comment", "note", [0, -270])
        note.name, note.node_size = "Estimate metric transforms from matched landmarks", "large"
        note.description = (
            "Original synthetic landmarks by OpenCVDataflowGUI contributors (MIT). "
            "Each branch has 21 correct matches and five outliers. Compare rigid "
            "poses, the 1.24× similarity and a general affine map in their own outputs."
        )
        a = add(
            "3d.points",
            "source",
            [0, 0],
            points=(source * 1000).tolist(),
            units="mm",
            frame="reference",
        )
        a.name = "Source landmarks in mm"
        for index, (mode, linear) in enumerate(linear_maps.items()):
            measured = source @ linear.T + TRANSLATION
            measured[OUTLIERS] += rng.uniform(0.08, 0.19, (len(OUTLIERS), 3))
            b = add(
                "3d.points",
                f"{mode}_landmarks",
                [0, 235 + 235 * index],
                points=measured.tolist(),
                frame=f"{mode}_station",
            )
            b.name = f"{mode.capitalize()} target in meters"
            fit = add(
                "3d.estimate_matched_transform",
                mode,
                [415, 115 + 245 * index],
                model=mode,
                output_units="mm",
                inlier_threshold_m=0.0001,
            )
            fit.name = f"Fit {mode}"
            fit.exposed_inputs = ["source", "target"]
            output_port = "pose" if mode in ("translation", "rigid") else mode
            fit.exposed_outputs = [output_port, "inlier_pairs", "report"]
            graph.connect(a.id, "cloud", fit.id, "source")
            graph.connect(b.id, "cloud", fit.id, "target")
        report = validate_graph(graph)
        if not report["valid"]:
            raise ValueError(report["issues"])
        graph.save(target)
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
