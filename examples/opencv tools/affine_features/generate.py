"""Original textured inspection fixture and a known tilted view; no downloaded assets."""

import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def fixture():
    rng = np.random.default_rng(128)
    image = rng.integers(75, 180, (256, 320), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0.6)
    cv2.rectangle(image, (8, 8), (311, 247), 225, 3)
    for row in range(4):
        for column in range(5):
            x, y = 35 + column * 60, 35 + row * 60
            cv2.circle(image, (x, y), 17, 35, -1)
            cv2.circle(image, (x + 3, y - 2), 8, 230, 2)
            for _ in range(6):
                offset = rng.integers(-22, 22, 2)
                end = rng.integers(-22, 22, 2)
                cv2.line(
                    image,
                    tuple(np.array([x, y]) + offset),
                    tuple(np.array([x, y]) + end),
                    int(rng.integers(20, 240)),
                    2,
                )
            cv2.putText(
                image,
                f"{row}{column}",
                (x - 12, y + 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                240,
                1,
                cv2.LINE_AA,
            )
    # X compression plus shear: a fixed affine view, not a fitted reference transform.
    transform = np.array([[0.43, 0.18, 26.0], [0.0, 0.95, 10.0], [0.0, 0.0, 1.0]])
    tilted = cv2.warpPerspective(image, transform, (240, 270), flags=cv2.INTER_LINEAR)
    return image, tilted, transform


def main():
    folder = Path(__file__).resolve().parent
    assets = folder / "assets"
    assets.mkdir(exist_ok=True)
    image, tilted, truth = fixture()
    assert cv2.imwrite(str(assets / "reference.png"), image)
    assert cv2.imwrite(str(assets / "tilted.png"), tilted)
    (assets / "truth.json").write_text(
        json.dumps({"source_to_target": truth.tolist(), "units": "input image pixels"}, indent=2)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    registry, graph = builtin_registry(), Graph()
    try:
        for index, name in enumerate(("reference", "tilted")):
            source = graph.add(
                registry.create(
                    "input.file",
                    block_id=name,
                    parameters={
                        "path": f"assets/{name}.png",
                        "path_base": "project",
                        "color_format": "gray",
                    },
                )
            )
            source.name = f"Original synthetic {name} view"
            source.position = [0.0, 300.0 * index]
            features = graph.add(
                registry.create(
                    "inspect.affine_features",
                    block_id=f"{name}_features",
                    parameters={"max_tilt": 3},
                )
            )
            features.name = f"Affine SIFT — {name}"
            features.position = [330.0, 300.0 * index]
            graph.connect(source.id, "image", features.id, "image")
        matcher = graph.add(
            registry.create("inspect.match_sift", block_id="match", parameters={"ratio": 0.75})
        )
        matcher.position = [660.0, 0.0]
        for source, target in (("reference_features", "query"), ("tilted_features", "train")):
            graph.connect(source, "features", matcher.id, target)
        fit = graph.add(
            registry.create(
                "inspect.homography",
                block_id="fit",
                parameters={"min_inliers": 30, "min_inlier_ratio": 0.8},
            )
        )
        fit.position = [980.0, 0.0]
        graph.connect(matcher.id, "matches", fit.id, "matches")
        review = graph.add(registry.create("inspect.review_matches", block_id="review"))
        review.position = [1300.0, 160.0]
        graph.connect("reference", "image", review.id, "query_image")
        graph.connect("tilted", "image", review.id, "train_image")
        graph.connect("match", "matches", review.id, "matches")
        graph.connect("fit", "transform", review.id, "fit")
        note = graph.add(registry.create("layout.comment", block_id="about"))
        note.name = "Original affine inspection fixture — MIT"
        note.description = (
            "Original synthetic images and generator under the repository MIT license. "
            "Affine SIFT matches an X-compressed/sheared view. Fit uses RANSAC; green pairs "
            "agree with that fit, red pairs disagree. assets/truth.json is the independent "
            "transform used to generate the tilted image. See README for expected accuracy. "
            "This demonstrates synthetic repeatability; it does not qualify real viewpoint changes."
        )
        note.node_size, note.position = "large", [0.0, -310.0]
        graph.save(folder / "project.json")
    finally:
        graph.close()


if __name__ == "__main__":
    main()
