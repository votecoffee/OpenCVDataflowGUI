"""Original feature scenes with independently declared rendering transforms."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
CASES = (
    "identity",
    "orb_rotation",
    "sift_ratio",
    "external_matches",
    "selected_support",
    "wrong_transform",
    "empty_features",
    "pair_budget",
)


def images():
    rng = np.random.default_rng(163)
    base = np.full((320, 480), 35, np.uint8)
    for i in range(65):
        x, y = rng.integers([30, 25], [440, 280])
        width, height = rng.integers(6, 23, 2)
        shade = int(rng.integers(90, 245))
        cv2.rectangle(base, (x, y), (x + width, y + height), shade, int(rng.choice([-1, 2])))
        cv2.line(base, (x - 4, y + height + 3), (x + width + 5, y - 3), 250, 1)
        if i % 4 == 0:
            cv2.putText(base, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 255, 1)
    affine = np.vstack((cv2.getRotationMatrix2D((239.5, 159.5), 8, 1), [0, 0, 1]))
    affine[:2, 2] += [9, 6]
    perspective = affine.copy()
    perspective[2, :2] = [0.00012, -0.00008]
    warped = cv2.warpPerspective(base, affine, base.shape[::-1], borderValue=35)
    projective = cv2.warpPerspective(base, perspective, base.shape[::-1], borderValue=35)
    obscured = warped.copy()
    obscured[90:230, 270:420] = 35
    return (
        {
            "reference.png": base,
            "rotated.png": warped,
            "projective.png": projective,
            "occluded.png": obscured,
            "blank.png": np.full_like(base, 35),
        },
        affine,
        perspective,
    )


def build():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    data, affine, perspective = images()
    for name, image in data.items():
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        (assets / name).write_bytes(encoded.tobytes())
    registry = builtin_registry()
    truth = {
        "creator": "OpenCVDataflowGUI contributors",
        "license": "MIT",
        "cases": {},
        "asset_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(assets.glob("*.png"))
        },
    }
    for case in CASES:
        graph = Graph()
        query = "blank.png" if case == "empty_features" else "reference.png"
        train = (
            "reference.png"
            if case in ("identity", "pair_budget")
            else "projective.png"
            if case == "sift_ratio"
            else "occluded.png"
            if case == "selected_support"
            else "blank.png"
            if case == "empty_features"
            else "rotated.png"
        )
        matrix = (
            np.eye(3)
            if case in ("identity", "pair_budget", "empty_features")
            else perspective
            if case == "sift_ratio"
            else affine
        )
        supplied = matrix.copy()
        if case == "wrong_transform":
            supplied[0, 2] += 60
        settings = {
            "reference_transform": supplied.tolist(),
            "match_threshold": 0.75
            if case == "sift_ratio"
            else 32
            if case == "external_matches"
            else 64,
            "ranking": "ratio" if case == "sift_ratio" else "distance",
            "minimum_repeatability": 0.3,
            "minimum_precision": 0.45,
            "minimum_recall": 0.2,
            "maximum_curve_points": 24,
            "maximum_pairs": 20 if case == "pair_budget" else 4000000,
        }
        evaluation = graph.add(
            registry.create("inspect.evaluate_features", block_id="evaluate", parameters=settings)
        )
        evaluation.position = [900, 50]
        detector_type = "inspect.sift" if case == "sift_ratio" else "inspect.orb"
        for role, filename, y in (("query", query, 0), ("train", train, 380)):
            source = graph.add(
                registry.create(
                    "input.file",
                    block_id=role + "_source",
                    parameters={
                        "path": "assets/" + filename,
                        "path_base": "project",
                        "color_format": "gray",
                    },
                )
            )
            source.position = [0, y]
            detect = graph.add(
                registry.create(detector_type, block_id=role, parameters={"max_features": 250})
            )
            detect.position = [440, y]
            graph.connect(source.id, "image", detect.id, "image")
            graph.connect(source.id, "image", "evaluate", role + "_image")
            graph.connect(role, "features", "evaluate", role)
        if case == "external_matches":
            matcher = graph.add(
                registry.create(
                    "inspect.match_descriptor_radius",
                    block_id="match",
                    parameters={"max_distance": 64, "max_neighbors": 3},
                )
            )
            matcher.position = [870, 570]
            for role in ("query", "train"):
                graph.connect(role, "features", "match", role)
            graph.connect("match", "matches", "evaluate", "matches")
        if case == "selected_support":
            region = graph.add(
                registry.create(
                    "region.shape",
                    block_id="train_selection",
                    parameters={"rectangle": [0, 0, 250, 320]},
                )
            )
            region.position = [420, 760]
            graph.connect("train_source", "image", region.id, "image")
            graph.connect(region.id, "region", "evaluate", "train_region")
        note = graph.add(registry.create("layout.comment", block_id="attribution"))
        note.name, note.position, note.node_size = (
            "Original independent-transform benchmark",
            [0, -430],
            "large",
        )
        note.description = (
            "Original MIT scenes, projects and matrices by OpenCVDataflowGUI contributors. "
            "No customer/third-party images. build_example.py renders the known image transform "
            "independently of feature extraction/matching. Evaluator never fits a homography. "
            "Center reprojection within three target pixels defines eligible geometric edges; "
            "maximum one-to-one cardinality defines repeatability and recall denominator. "
            "Ranked candidate duplicates consume precision but cannot add true positives. "
            "Legacy native ellipse-overlap evaluation is a separate metric; its functions are "
            "not Python-bound here. Wrong transform and empty features deliberately fail; pair "
            "budget deliberately errors. Native feature counts can vary with build/settings."
        )
        graph.save(HERE / f"{case}.json")
        graph.close()
        truth["cases"][case] = {
            "query": query,
            "train": train,
            "rendering_transform": matrix.tolist(),
            "evaluation_transform": supplied.tolist(),
            "expected_state": "error"
            if case == "pair_budget"
            else "fail"
            if case in ("wrong_transform", "empty_features")
            else "pass",
        }
    (HERE / "truth.json").write_bytes((json.dumps(truth, indent=2) + "\n").encode())


if __name__ == "__main__":
    build()
