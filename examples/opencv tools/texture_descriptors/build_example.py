"""Generate a matched color texture pair and native descriptor comparison."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

MODES = (
    ("vgg_float", "VGG 120", "inspect.vgg_descriptor", 1.0,
     {"variant": "VGG_120", "scale_factor": .75}),
    ("vgg_byte", "VGG 48 bytes", "inspect.vgg_descriptor", 40.0,
     {"variant": "VGG_48", "scale_factor": .75, "normalize_descriptor": True}),
    ("lucid", "LUCID color", "inspect.lucid_descriptor", 128.0,
     {"patch_radius": 4, "blur_radius": 1}),
)


def make_images():
    height, width = 220, 304
    y, x = np.indices((height, width))
    rng = np.random.default_rng(7500)
    color = cv2.GaussianBlur(rng.integers(10, 242, (height, width, 3), np.uint8), (5, 5), .75)
    blend = np.stack(
        (np.sin(.03 * x + .012 * y), np.cos(.023 * x - .033 * y), np.sin(.015 * x + .033 * y)),
        axis=2,
    ) * 26
    color = np.rint(color.astype(float) + blend).clip(0, 255).astype(np.uint8)
    for index in range(50):
        px, py = map(int, rng.integers([20, 20], [width - 22, height - 22]))
        tone = tuple(map(int, rng.integers(20, 235, 3)))
        if index % 2:
            cv2.circle(color, (px, py), int(rng.integers(3, 10)), tone, -1)
        else:
            cv2.rectangle(color, (px, py), (px + 11, py + 8), tone, -1)
    affine = np.array([[1., 0., 12.], [0., 1., -8.]])
    train = cv2.warpAffine(color, affine, (width, height), borderMode=cv2.BORDER_REFLECT101)
    changed = train[85:143, 180:255]
    changed[:] = np.rint(changed.astype(float) * .84 + 14).clip(0, 255).astype(np.uint8)
    return {"query": color, "train": train}, np.vstack((affine, [0, 0, 1])).tolist()


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    images, transform = make_images()
    for role, image in images.items():
        ok, data = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode the {role} image")
        (assets / f"{role}.png").write_bytes(data.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, position, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, list(position)
        return block

    try:
        note = add("layout.comment", "note", "Compare intensity and color texture", (0, -225))
        note.node_size = "large"
        note.description = (
            "Run Once. Each branch uses the same single-level ORB detections. The second view "
            "shifts the texture and relights one area. VGG uses L2 in both formats. "
            "LUCID uses Hamming on color samples. Review each overlay against the known shift. "
            "Original MIT assets."
        )
        for role, top in (("query", 30), ("train", 680)):
            source = add(
                "input.file", f"{role}_image", f"{role.title()} color texture", (0, top),
                path=f"assets/{role}.png", path_base="project", color_format="bgr",
            )
            detector = add(
                "inspect.orb", f"{role}_orb", f"{role.title()} ORB", (230, top),
                max_features=280, levels=1, edge_threshold=18, fast_threshold=8,
            )
            detector.exposed_outputs = ("features",)
            graph.connect(source.id, "image", detector.id, "image")
            for index, (key, name, kind, _, settings) in enumerate(MODES):
                block = add(
                    kind, f"{key}_{role}", f"{role.title()} {name}",
                    (470, top + 185 * index), **settings,
                )
                block.exposed_outputs = ("image", "features", "count")
                graph.connect(source.id, "image", block.id, "image")
                graph.connect(detector.id, "features", block.id, "keypoints")
        for index, (key, name, _, limit, _) in enumerate(MODES):
            top = 80 + 290 * index
            match = add(
                "inspect.match_descriptor_radius", f"{key}_match", f"Match {name}", (780, top),
                max_distance=limit, max_neighbors=1,
            )
            evaluate = add(
                "inspect.evaluate_features", f"{key}_evaluate", f"Review {name}", (1090, top),
                reference_transform=transform, reprojection_tolerance=2.5, match_threshold=limit,
                minimum_correspondences=150, minimum_precision=0.95, maximum_graphic_matches=120,
            )
            for role in ("query", "train"):
                graph.connect(f"{key}_{role}", "features", match.id, role)
                graph.connect(f"{key}_{role}", "features", evaluate.id, role)
                graph.connect(f"{key}_{role}", "image", evaluate.id, f"{role}_image")
            graph.connect(match.id, "matches", evaluate.id, "matches")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise RuntimeError(checked["issues"])
        graph.save(output / "texture_descriptors.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
