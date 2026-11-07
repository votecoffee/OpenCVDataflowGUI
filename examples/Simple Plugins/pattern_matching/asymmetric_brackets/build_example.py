"""Generate the portable asymmetric bracket inspection from synthetic geometry."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from opencv_geometric_pattern_match import InspectionRecipe, Pose, Search, save_recipe, train
from opencv_geometric_pattern_match.dataflow import recipe_reference

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

BACKGROUND = 242
ORIGIN = (46.5, 43.5)
SHAPE = (290, 440)


def make_reference():
    reference = np.full((88, 94), BACKGROUND, np.uint8)
    outline = np.array(
        [(15, 12), (65, 12), (80, 27), (80, 73), (50, 73), (50, 65), (15, 65)], np.int32
    )
    cv2.fillPoly(reference, [outline], 70, lineType=cv2.LINE_AA)
    cv2.circle(reference, (35, 33), 9, BACKGROUND, -1, cv2.LINE_AA)
    cv2.rectangle(reference, (53, 41), (65, 52), BACKGROUND, -1)
    return reference


def make_scene(reference, poses, *, occluded=False):
    scene = np.full(SHAPE, BACKGROUND, np.uint8)
    for index, pose in enumerate(poses):
        matrix = pose.matrix()
        matrix[:, 2] -= matrix[:, :2] @ ORIGIN
        rendered = cv2.warpAffine(reference, matrix, SHAPE[::-1], borderValue=BACKGROUND)
        if occluded and index == 1:
            patch_center = tuple(np.rint(matrix @ [79, 43, 1]).astype(int))
            cv2.circle(rendered, patch_center, 11, BACKGROUND, -1, cv2.LINE_AA)
        np.minimum(scene, rendered, out=scene)
    if occluded:
        decoy = np.array([(333, 47), (389, 49), (371, 111), (342, 101)], np.int32)
        cv2.fillPoly(scene, [decoy], 82, lineType=cv2.LINE_AA)
        cv2.circle(scene, (359, 72), 9, BACKGROUND, -1, cv2.LINE_AA)
    noise = np.random.default_rng(38 if occluded else 12).normal(0, 1.2, SHAPE)
    gradient = np.linspace(-9, 8, SHAPE[1])
    return np.rint(scene + noise + gradient).clip(0, 255).astype(np.uint8)


def write_project(root, paths, *, wired):
    registry = builtin_registry()
    graph = Graph()
    try:
        source = graph.add(registry.create("input.file", block_id="scene", parameters={
            "source_mode": "files", "paths": paths, "path_base": "project", "color_format": "gray",
        }))
        source.name = "Generated tray scenes"
        reference = graph.add(registry.create("input.file", block_id="reference", parameters={
            "path": "assets/reference.png", "path_base": "project", "color_format": "gray",
        }))
        reference.name = "Training reference"
        reference.position = [0., 245.]
        pin = recipe_reference(root, "assets/bracket.npz")
        if wired:
            loader = graph.add(registry.create(
                "pattern.ogpm_load_recipe", block_id="recipe", parameters=pin
            ))
            loader.position = [375., 245.]
            matcher = graph.add(registry.create("pattern.ogpm_match_model", block_id="matcher"))
            graph.connect(loader.id, "recipe", matcher.id, "recipe")
        else:
            matcher = graph.add(registry.create(
                "pattern.ogpm_recipe", block_id="matcher", parameters=pin
            ))
        matcher.position = [375., 0.]
        matcher.exposed_outputs.append("result_graphics")
        graph.connect(source.id, "image", matcher.id, "image")
        credits = graph.add(registry.create("layout.comment", block_id="source_credits"))
        credits.name = "Bracket example credits"
        credits.description = (
            "The synthetic bracket geometry and both recipe routes were adapted from the "
            "MIT-licensed OpenCVGeometricPatternMatch Dataflow example.\n\nSource at "
            "https://github.com/votecoffee/OpenCVGeometricPatternMatch/blob/632e81bbd12c6640d1c6f65618bd85e45aa26b9e/examples/dataflow/build_example.py."
        )
        credits.node_size = "large"
        credits.position = [0.0, -254.0]
        graph.save(root / ("wired_recipe.json" if wired else "project.json"))
    finally:
        graph.close()


def build_example(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    assets = root / "assets"
    assets.mkdir()
    reference = make_reference()
    grid = [
        Pose(78.2 + 141 * (i % 3), 77.4 + 140 * (i // 3), angle, scale)
        for i, (angle, scale) in enumerate(
            [(-4, .98), (3, 1.02), (0, 1.), (4, 1.01), (-2, .99), (1, 1.)]
        )
    ]
    changed = [
        Pose(79.4, 74.8, 5., 1.03), Pose(221.6, 78.2, -5., .96),
        Pose(75.4, 214.9, -2., .99), Pose(362.8, 213.3, 4., 1.04),
    ]
    scenes = [
        ("tray.png", grid, make_scene(reference, grid)),
        ("partial_tray.png", changed, make_scene(reference, changed, occluded=True)),
    ]
    for filename, pixels in [
        ("reference.png", reference), ("bracket.reference.png", reference),
        *((name, pixels) for name, _, pixels in scenes)
    ]:
        ok, encoded = cv2.imencode(".png", pixels)
        if not ok:
            raise RuntimeError(f"Cannot encode {filename}")
        (assets / filename).write_bytes(encoded.tobytes())
    pattern = train(reference, origin=ORIGIN)
    search = Search(angle=(-8, 8, 4), scale=(.94, 1.06, .04), accept=.55,
                    coarse_accept=.25, max_results=8, xy_overlap=.35)
    save_recipe(assets / "bracket.npz", InspectionRecipe(pattern, search))
    expected = [
        {"path": f"assets/{name}", "poses": [asdict(pose) for pose in poses]}
        for name, poses, _ in scenes
    ]
    (root / "expected.json").write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    paths = [frame["path"] for frame in expected]
    write_project(root, paths, wired=False)
    write_project(root, paths, wired=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="New output directory for the projects and assets")
    build_example(parser.parse_args().directory)
