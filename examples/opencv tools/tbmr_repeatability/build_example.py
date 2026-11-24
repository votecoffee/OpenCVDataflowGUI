"""Build an original nested-region target and compare detectors at 1.5×."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

QUERY_TO_TRAIN = [[1.5, 0, 0.25], [0, 1.5, 0.25], [0, 0, 1]]


def target():
    image = np.full((176, 240), 100, np.uint8)
    for row in range(3):
        for col in range(4):
            x, y = 30 + col * 59, 30 + row * 56
            polarity = -1 if (row + col) % 2 else 1
            w, h = 15 + (row * 3 + col) % 6, 11 + (row + col * 2) % 5
            angle = 15 * (col - row)
            for scale, contrast in ((1, 35), (0.75, 63), (0.51, 100)):
                cv2.ellipse(
                    image, (x, y), (round(w * scale), round(h * scale)), angle,
                    0, 360, 100 + polarity * contrast, -1, cv2.LINE_AA,
                )
            left, right = (225, 230) if polarity > 0 else (4, 3)
            cv2.circle(image, (x - 6, y + 2), 4, left, -1)
            cv2.circle(image, (x + 7, y - 3), 3, right, -1)
    return cv2.GaussianBlur(image, (0, 0), 0.7)


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(".png", target(), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise OSError("Could not encode the region target")
    (assets / "regions.png").write_bytes(data.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, name, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = name, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Compare regions after a scale change", 0, -200)
        note.node_size = "large"
        note.description = (
            "Run Once. Both detectors see the same original MIT region target and its 1.5× resize. "
            "Compare one-to-one center matches within 2 target pixels and measured size ratios. "
            "TBMR uses its major axis. MSER uses its area-equivalent diameter."
        )
        add(
            "input.file", "query", "Nested bright and dark regions", 0, 65,
            path="assets/regions.png", path_base="project", color_format="gray",
        )
        add("cv.resize", "train", "Resize by 1.5×", 245, 280, size_mode="scale", scale_x=1.5, scale_y=1.5)
        graph.connect("query", "image", "train", "image")
        for index, (kind, port, tag) in enumerate(
            (("inspect.tbmr", "keypoints", "tbmr"), ("inspect.mser", "regions", "mser"))
        ):
            upper = 20 + 450 * index
            eval_id, review_id = f"{tag}_evaluate", f"{tag}_review"
            add(
                "inspect.evaluate_keypoints", eval_id, f"{tag.upper()} geometry and scale",
                825, upper + 70, reference_transform=QUERY_TO_TRAIN, tolerance=2.0,
                minimum_pairs=20 if tag == "tbmr" else 1,
                minimum_repeatability=0.9 if tag == "tbmr" else 0.0,
            )
            add("inspect.review_matches", review_id, f"Review {tag.upper()} pairs", 1110, upper + 70)
            for role, min_area, max_area, y in (
                ("query", 32, 1056, upper), ("train", 72, 2376, upper + 215),
            ):
                settings = (
                    {"min_area": min_area, "max_area_relative": 0.025}
                    if tag == "tbmr"
                    else {"min_area": min_area, "max_area": max_area, "max_variation": 0.35}
                )
                detect = add(kind, f"{tag}_{role}", f"{role.title()} {tag.upper()}", 530, y, **settings)
                graph.connect(role, "image", detect.id, "image")
                graph.connect(detect.id, port, eval_id, role)
                graph.connect(role, "image", eval_id, f"{role}_image")
                graph.connect(role, "image", review_id, f"{role}_image")
            graph.connect(eval_id, "matches", review_id, "matches")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise RuntimeError(checked["issues"])
        graph.save(output / "tbmr_repeatability.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
