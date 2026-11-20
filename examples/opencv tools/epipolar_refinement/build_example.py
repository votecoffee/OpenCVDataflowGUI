"""Create independently projected sparse stereo matches and a correction example."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

SIZES = ((800, 550), (720, 510))
INTRINSICS = (
    np.array([[790.0, 0, 402], [0, 785, 273], [0, 0, 1]]),
    np.array([[755.0, 0, 354], [0, 751, 242], [0, 0, 1]]),
)
ANGLE = np.array([-0.025, 0.09, 0.08])
BASELINE = np.array([-0.19, 0.023, 0.018])
OUTLIERS = [5, 19, 28]


def fixture():
    rng = np.random.default_rng(734)
    xyz = rng.uniform([-0.70, -0.46, 2.0], [0.72, 0.46, 4.0], (32, 3))
    rotation = cv2.Rodrigues(ANGLE)[0]
    true = [
        cv2.projectPoints(xyz, r, t, k, None)[0].reshape(-1, 2)
        for r, t, k in zip((np.zeros(3), ANGLE), (np.zeros(3), BASELINE), INTRINSICS, strict=True)
    ]
    noisy = [xy + rng.normal(0, 0.55, xy.shape) for xy in true]
    noisy[1][OUTLIERS, 1] += [12, -14, 13]
    tx, ty, tz = BASELINE
    cross = np.array([[0, -tz, ty], [tz, 0, -tx], [-ty, tx, 0]])
    f = np.linalg.inv(INTRINSICS[1]).T @ cross @ rotation @ np.linalg.inv(INTRINSICS[0])
    return noisy, true, f


def render(size, xy):
    width, height = size
    image = np.full((height, width, 3), (40, 38, 34), np.uint8)
    for i, point in enumerate(xy):
        fixed = tuple(np.rint(point * 256).astype(int))
        cv2.circle(image, fixed, 6 * 256, (190, 203, 218), 1, cv2.LINE_AA, shift=8)
        x, y = np.rint(point).astype(int)
        cv2.putText(
            image,
            str(i),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 195, 180),
            1,
            cv2.LINE_AA,
        )
    return image


def save_program(output, noisy, f):
    graph, registry = Graph(), builtin_registry()

    def add(type_id, block_id, x, y, **parameters):
        block = graph.add(registry.create(type_id, block_id=block_id, parameters=parameters))
        block.position = [x, y]
        return block

    try:
        note = add("layout.comment", "description", 0, -220)
        note.name, note.node_size = "Stereo match correction", "large"
        note.description = (
            "Original synthetic correspondences by OpenCVDataflowGUI contributors (MIT). "
            "There are 32 noisy pairs and three intentional bad matches. "
            "The known fundamental matrix is stored in the recipe. "
            "Select the correction block to review per-view movements and rejected pairs."
        )
        recipe = graph.recipes.put(
            "SyntheticStereo",
            {"left": noisy[0].tolist(), "right": noisy[1].tolist(), "fundamental": f.tolist()},
        )
        graph.recipes.activate(recipe, project_default=True)
        refine = add(
            "3d.refine_epipolar",
            "refine",
            390,
            110,
            maximum_sampson_px=2.0,
            maximum_shift_px=2.0,
            minimum_inlier_ratio=0.9,
        )
        check = add("3d.check_epipolar", "check", 765, 110, max_distance=0.0002, max_drawn_pairs=0)
        for index, side in enumerate(("left", "right")):
            matched = add("recipe.field", f"points_{side}", 0, 145 * index, path=f"settings.{side}")
            matched.name = f"{side.title()} observed points"
            image = add(
                "input.file",
                f"{side}_input",
                0,
                415 + 145 * index,
                path=f"assets/{side}.png",
                path_base="project",
            )
            image.name = f"{side.title()} synthetic view"
            graph.connect(matched.id, "value", refine.id, f"points_{side}")
            graph.connect(image.id, "image", refine.id, f"{side}_image")
            graph.connect(refine.id, f"points_{side}", check.id, f"points_{side}")
        matrix = add("recipe.field", "fundamental", 0, 290, path="settings.fundamental")
        matrix.name = "Known fundamental matrix"
        graph.connect(matrix.id, "value", refine.id, "matrix")
        graph.connect(matrix.id, "value", check.id, "matrix")
        report = validate_graph(graph)
        if not report["valid"]:
            raise ValueError(report["issues"])
        graph.save(output / "epipolar_refinement.json")
    finally:
        graph.close()


def build(output):
    assets = output / "assets"
    if assets.exists() or (output / "epipolar_refinement.json").exists():
        raise FileExistsError("Choose a new output directory to preserve the fixture")
    assets.mkdir(parents=True)
    noisy, truth, f = fixture()
    for side, xy, size in zip(("left", "right"), noisy, SIZES, strict=True):
        ok, encoded = cv2.imencode(".png", render(size, xy))
        if not ok:
            raise ValueError(f"Cannot encode the {side} view")
        (assets / f"{side}.png").write_bytes(encoded.tobytes())
    data = {"clean": [xy.tolist() for xy in truth], "outlier_indices": OUTLIERS}
    (assets / "truth.json").write_bytes((json.dumps(data, indent=2) + "\n").encode("ascii"))
    save_program(output, noisy, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
