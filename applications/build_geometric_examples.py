"""Generate publishable geometric matcher applications with synthetic fixtures."""

import argparse
import json
import runpy
import tempfile
from pathlib import Path

import cv2
import numpy as np
from opencv_geometric_pattern_match import InspectionRecipe, Pose, Search, save_recipe, train
from opencv_geometric_pattern_match.feature_weights import weight_pattern

BACKGROUND = 236


def add_part(scene, part, datum, x, y, angle=0.0, scale=1.0, contrast=1.0):
    transform = Pose(x, y, angle, scale).matrix()
    transform[:, 2] -= transform[:, :2] @ datum
    placed = cv2.warpAffine(part, transform, scene.shape[::-1], borderValue=BACKGROUND)
    placed = np.rint(BACKGROUND + (placed.astype(float) - BACKGROUND) * contrast)
    np.minimum(scene, placed.clip(0, 255).astype(np.uint8), out=scene)


def to_uint16(image):
    return image.astype(np.uint16) * 257


def tray_grid():
    reference = np.full((58, 60), BACKGROUND, np.uint8)
    outline = np.array(
        [(10, 8), (42, 8), (51, 17), (51, 47), (33, 47), (33, 43), (10, 43)], np.int32
    )
    cv2.fillPoly(reference, [outline], 65, lineType=cv2.LINE_AA)
    cv2.circle(reference, (24, 23), 5, BACKGROUND, -1, cv2.LINE_AA)
    cv2.rectangle(reference, (35, 27), (43, 34), BACKGROUND, -1)
    datum = (29.5, 28.5)
    pattern = train(reference, origin=datum, coarse_factor=2)
    search = Search(
        angle=(-6, 6, 3),
        scale=(0.96, 1.04, 0.04),
        accept=0.52,
        coarse_accept=0.20,
        max_results=48,
        xy_overlap=0.35,
    )
    rng = np.random.default_rng(831)
    scenes, centers = [], []
    for _ in range(4):
        scene = np.full((348, 414), BACKGROUND, np.uint8)
        datums = []
        for row in range(6):
            for col in range(7):
                x, y = np.array([33 + 58 * col, 32 + 56 * row]) + rng.uniform(-1.7, 1.7, 2)
                add_part(
                    scene,
                    reference,
                    datum,
                    x,
                    y,
                    rng.uniform(-3.5, 3.5),
                    rng.uniform(0.98, 1.02),
                    rng.uniform(0.85, 1.05),
                )
                datums.append([float(x), float(y)])
        noise = rng.normal(0, 1.1, scene.shape)
        scenes.append(to_uint16(np.rint(scene + noise).clip(0, 255).astype(np.uint8)))
        centers.append(datums)
    return to_uint16(reference), scenes, InspectionRecipe(pattern, search), centers


def needle_row():
    reference = np.full((116, 32), BACKGROUND, np.uint8)
    shape = np.array(
        [(12, 10), (20, 10), (22, 18), (19, 24), (18, 94), (15, 106), (13, 96), (13, 25), (10, 20)],
        np.int32,
    )
    cv2.fillPoly(reference, [shape], 61, lineType=cv2.LINE_AA)
    cv2.line(reference, (13, 17), (19, 17), 161, 1, cv2.LINE_AA)
    datum = (15.5, 22.0)
    pattern = train(reference, (6, 5, 20, 107), origin=datum, coarse_factor=2)
    search = Search(
        angle=(-2, 2, 2),
        scale=(1, 1, 1),
        accept=0.52,
        coarse_accept=0.22,
        max_results=24,
        xy_overlap=0.35,
    )
    rng = np.random.default_rng(916)
    scenes, centers = [], []
    for omitted in (set(), {3, 9, 15}, set(range(19))):
        scene = np.full((140, 444), BACKGROUND, np.uint8)
        datums = []
        for column in range(19):
            if column in omitted:
                continue
            x, y = np.array([22 + 22 * column, 28]) + rng.uniform(-0.8, 0.8, 2)
            add_part(
                scene,
                reference,
                datum,
                x,
                y,
                rng.uniform(-0.9, 0.9),
                contrast=rng.uniform(0.91, 1.04),
            )
            datums.append([float(x), float(y)])
        noise = rng.normal(0, 0.8, scene.shape)
        scenes.append(to_uint16(np.rint(scene + noise).clip(0, 255).astype(np.uint8)))
        centers.append(datums)
    return to_uint16(reference), scenes, InspectionRecipe(pattern, search), centers


def weighted_blocks():
    blank = np.full((130, 163), BACKGROUND, np.uint8)
    cv2.rectangle(blank, (24, 20), (124, 103), 27, -1)
    reference = blank.copy()
    cv2.circle(reference, (94, 58), 11, BACKGROUND, -1, cv2.LINE_AA)
    datum = (76.5, 66.5)
    pattern = train(reference, (12, 11, 127, 112), origin=datum, coarse_factor=3)
    pattern = weight_pattern(pattern, [(68, 33, 30, 31, 12)])
    search = Search(angle=(0, 0, 1), scale=(1, 1, 1), accept=0.55, max_results=4, xy_overlap=0.35)
    scenes, centers = [], []
    for true_indices in ({0}, {0, 2}, set()):
        scene = np.full((156, 494), BACKGROUND, np.uint8)
        datums = []
        for index in range(3):
            true_part = index in true_indices
            x, y = 81 + 160 * index, 77
            add_part(scene, reference if true_part else blank, datum, x, y)
            if true_part:
                datums.append([x, y])
        scenes.append(to_uint16(scene))
        centers.append(datums)
    return to_uint16(reference), scenes, InspectionRecipe(pattern, search), centers


APPLICATIONS = {
    "tray_grid": ("Locate parts in a crowded tray", tray_grid),
    "needle_row": ("Inspect a needle row", needle_row),
    "weighted_blocks": ("Distinguish weighted block geometry", weighted_blocks),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matcher_checkout", type=Path)
    parser.add_argument("destination", type=Path, help="A new output directory")
    args = parser.parse_args()
    builder = args.matcher_checkout / "examples/dataflow/build_recipe_project.py"
    build_project = runpy.run_path(str(builder))["build_project"]
    args.destination.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="ogpm-synthetic-") as scratch:
        scratch = Path(scratch)
        for name, (title, fixture) in APPLICATIONS.items():
            reference, scenes, recipe, centers = fixture()
            reference_path = scratch / "reference.png"
            archive = scratch / "inspection.npz"
            save_recipe(archive, recipe)
            scene_paths = [scratch / f"scene-{i + 1:02d}.png" for i in range(len(scenes))]
            for path, image in zip(
                [reference_path, *scene_paths], [reference, *scenes], strict=True
            ):
                if not cv2.imwrite(str(path), image):
                    raise OSError(f"Could not write {path}")
            project = build_project(
                args.destination / name,
                archive,
                scene_paths,
                reference=reference_path,
                title=title,
                workers=1,
            )
            expected = {"match_counts": [len(frame) for frame in centers], "datums": centers}
            (project.parent / "expected.json").write_text(
                json.dumps(expected, indent=2) + "\n", encoding="utf-8", newline="\n"
            )
            print(project, flush=True)


if __name__ == "__main__":
    main()
