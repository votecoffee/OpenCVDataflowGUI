"""Generate known stereo captures and rectify them using only image correspondences."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

SIZES = ((760, 550), (700, 510))
INTRINSICS = (
    np.array([[790.0, 0, 378], [0, 779, 271], [0, 0, 1]]),
    np.array([[735.0, 0, 349], [0, 747, 254], [0, 0, 1]]),
)
ANGLE = np.array([-0.045, 0.091, 0.076])
BASELINE = np.array([-0.20, 0.014, 0.021])
OUTLIERS = [2, 17, 31, 49]
TRAINING = 56


def fixture():
    rng = np.random.default_rng(5177)
    xyz = rng.uniform([-0.76, -0.56, 2.0], [0.72, 0.55, 4.4], (TRAINING + 20, 3))
    projected = [
        cv2.projectPoints(xyz, r, t, k, None)[0].reshape(-1, 2)
        for r, t, k in zip((np.zeros(3), ANGLE), (np.zeros(3), BASELINE), INTRINSICS, strict=True)
    ]
    fit_xy = [p[:TRAINING] + rng.normal(0, 0.10, (TRAINING, 2)) for p in projected]
    replacements = [26, 45, 7, 12]
    fit_xy[1][OUTLIERS] = fit_xy[1][replacements]
    return projected, fit_xy


def render(size, xy):
    width, height = size
    image = np.full((height, width, 3), (16, 17, 21), np.uint8)
    for i, point in reversed(list(enumerate(xy))):
        accent = (166, 171, 50) if i < TRAINING else (76, 168, 250)
        center = tuple(np.rint(point * 256).astype(int))
        cv2.circle(image, center, 3 * 256, accent, -1, cv2.LINE_AA, shift=8)
        x, y = np.rint(point).astype(int)
        cv2.putText(
            image,
            str(i),
            (x + 5, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            accent,
            1,
            cv2.LINE_AA,
        )
    return image


def save_program(output, fit_xy):
    graph, registry = Graph(), builtin_registry()

    def add(type_id, block_id, x, y, **parameters):
        block = graph.add(registry.create(type_id, block_id=block_id, parameters=parameters))
        block.position = [x, y]
        return block

    try:
        comment = add("layout.comment", "description", 0, -245)
        comment.name, comment.node_size = "Stereo rectification from image matches", "large"
        comment.description = (
            "Original synthetic images by OpenCVDataflowGUI contributors (MIT). "
            "The graph estimates the fundamental matrix using 56 match pairs. Four pairs "
            "are deliberately wrong. Orange points are separate validation observations."
        )
        recipe = graph.recipes.put(
            "StereoMatches",
            {"left": fit_xy[0].tolist(), "right": fit_xy[1].tolist()},
        )
        graph.recipes.activate(recipe, project_default=True)
        fit = add("3d.fundamental", "fundamental", 350, 105, threshold=0.7, method="ransac")
        rect = add(
            "3d.rectify_uncalibrated_stereo",
            "rectify",
            690,
            205,
            width=640,
            height=480,
            maximum_input_error_px=0.8,
            maximum_vertical_rms_px=0.6,
            minimum_common_support=0.65,
        )
        for i, side in enumerate(("left", "right")):
            y = 180 * i
            points = add("recipe.field", f"{side}_matches", 0, y, path=f"settings.{side}")
            points.name = f"{side.title()} matched points"
            image = add(
                "input.file",
                f"{side}_image",
                0,
                385 + y,
                path_base="project",
                path=f"assets/{side}.png",
            )
            image.name = f"{side.title()} original image"
            graph.connect(points.id, "value", fit.id, f"points_{side}")
            graph.connect(points.id, "value", rect.id, f"points_{side}")
            graph.connect(image.id, "image", rect.id, f"{side}_image")
        for port in ("matrix", "inliers"):
            graph.connect(fit.id, port, rect.id, port)
        result = validate_graph(graph)
        if not result["valid"]:
            raise ValueError(result["issues"])
        graph.save(output / "uncalibrated_stereo.json")
    finally:
        graph.close()


def build(output):
    assets = output / "assets"
    if assets.exists() or (output / "uncalibrated_stereo.json").exists():
        raise FileExistsError("Choose a new output directory to preserve the fixture")
    assets.mkdir(parents=True)
    projected, fit_xy = fixture()
    for side, xy, size in zip(("left", "right"), projected, SIZES, strict=True):
        ok, encoded = cv2.imencode(".png", render(size, xy))
        if not ok:
            raise ValueError(f"Could not encode the {side} view")
        (assets / f"{side}.png").write_bytes(encoded.tobytes())
    truth = {
        "heldout_pixels": [xy[TRAINING:].tolist() for xy in projected],
        "outlier_indices": OUTLIERS,
    }
    (assets / "truth.json").write_bytes((json.dumps(truth, indent=2) + "\n").encode("ascii"))
    save_program(output, fit_xy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
