"""Generate the original board images and saved SuperPoint v6 graphs."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
MODEL_SHA = "56eb68c0c01637e1d8368156ffe4bfc6973faf328204458c0928c1d63ca9abde"
MATCHER_SHA = "6ff7040d0a497fc6639337946d7538dae07428c18f77a067a0b5a960e7cc551a"
TRANSLATION = [8, 6]
CASES = ("extract", "translation", "blank", "resized_region")


def images():
    rng = np.random.default_rng(978173)
    h, w = 240, 320
    y, x = np.mgrid[:h, :w]
    reference = np.clip(22 + 0.04 * x + 0.05 * y + rng.normal(0, 1.5, (h, w)), 0, 255).astype(
        np.uint8
    )
    cv2.rectangle(reference, (29, 27), (288, 210), 100, 7, cv2.LINE_AA)
    cv2.rectangle(reference, (29, 27), (288, 210), 149, -1)
    for center in ((46, 44), (269, 47), (48, 190), (268, 190)):
        cv2.circle(reference, center, 8, 227, 2, cv2.LINE_AA)
        cv2.circle(reference, center, 5, 27, -1, cv2.LINE_AA)
    for i in range(10):
        left = 75 + i % 5 * 35
        top = 69 + i // 5 * 54 + int(rng.integers(-5, 6))
        width, height = rng.integers(13, 23, 2).tolist()
        right, bottom = left + width, top + height
        cv2.rectangle(reference, (left, top), (right, bottom), 53, -1, cv2.LINE_AA)
        cv2.rectangle(reference, (left, top), (right, bottom), 207, 1, cv2.LINE_AA)
        cv2.circle(reference, (left + 3 + i % 3, top + 4), 2, 222, -1, cv2.LINE_AA)
        middle = top + height // 2
        cv2.line(reference, (right, middle), (right + 8, middle), 232, 2, cv2.LINE_AA)
        cv2.line(reference, (right + 8, middle), (right + 8, bottom + 10), 232, 1, cv2.LINE_AA)
    cv2.putText(
        reference, "REV A  204", (77, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.81, 47, 2, cv2.LINE_AA
    )
    transform = np.float32([[1, 0, TRANSLATION[0]], [0, 1, TRANSLATION[1]]])
    moved = cv2.warpAffine(reference, transform, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    moved = np.clip(moved.astype(np.float32) * 0.91 + 9, 0, 255).astype(np.uint8)
    return {"reference": reference, "query": moved, "blank": np.zeros_like(reference)}


def saved_graph(case, registry):
    graph = Graph()

    def block(type_id, id_, position, **parameters):
        value = graph.add(registry.create(type_id, block_id=id_, parameters=parameters))
        value.position = position
        return value

    sources = (
        ("reference", "query")
        if case == "translation"
        else (("blank",) if case == "blank" else ("reference",))
    )
    for row, side in enumerate(sources):
        y = row * 310
        block("input.file", side, [0, y], path=f"assets/{side}.png", path_base="project")
        block(
            "ml.superpoint_onnx_features",
            f"{side}_features",
            [360, y],
            model_path="../../../.local/models/superpoint-v6-dense.onnx",
            model_sha256=MODEL_SHA,
            maximum_model_bytes=5_300_000,
            maximum_edge=193 if case == "resized_region" else 640,
        )
        graph.connect(side, "image", f"{side}_features", "image")

    if case == "resized_region":
        block("region.shape", "roi", [0, 280], shape="rectangle", rectangle=[40, 35, 130, 160])
        graph.connect("reference", "image", "roi", "image")
        graph.connect("roi", "region", "reference_features", "region")
    if case == "translation":
        matcher = block(
            "ml.lightglue_superpoint",
            "match",
            [720, 300],
            weights_path="../../../.local/models/superpoint_lightglue.pth",
            weights_sha256=MATCHER_SHA,
            device="cpu",
            flash=False,
            depth_confidence=-1.0,
            width_confidence=-1.0,
            minimum_matches=24,
            minimum_inliers=20,
        )
        matcher.exposed_outputs.append("result_graphics")
        graph.connect("reference_features", "features", "match", "reference_features")
        graph.connect("query_features", "features", "match", "query_features")
        graph.connect("query_features", "image", "match", "image")
        block("cv.overlay_result_graphics", "overlay", [1060, 300])
        graph.connect("query_features", "image", "overlay", "image")
        graph.connect("match", "result_graphics", "overlay", "graphics_1")
        block("output.filmstrip_image", "filmstrip", [1400, 300])
        graph.connect("overlay", "image", "filmstrip", "image")

    comment = block("layout.comment", "credits", [0, -370])
    comment.name, comment.node_size = "SuperPoint v6", "large"
    comment.description = (
        "SuperPoint v6 checkpoint by Remi Pautrat and Paul-Edouard Sarlin (MIT). "
        "Original synthetic board images by OpenCVDataflowGUI contributors (MIT). "
        "The translation graph uses LightGlue by Lindenberger, Sarlin and Pollefeys (Apache-2.0). "
        "Prepare the two model assets explicitly as described in README.md. "
        "The known board shift is +8,+6 pixels with reduced contrast."
    )
    return graph


def build(output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "assets").mkdir(exist_ok=True)
    frames = images()
    filenames = [
        *(output / "assets" / f"{side}.png" for side in frames),
        *(output / f"{case}.json" for case in CASES),
    ]
    existing = [path for path in filenames if path.exists()]
    if existing:
        raise FileExistsError(f"Existing example files preserved: {existing}")
    for name, image in frames.items():
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise ValueError(f"PNG encoding failed: {name}")
        (output / "assets" / f"{name}.png").write_bytes(encoded.tobytes())
    registry = builtin_registry()
    for case in CASES:
        graph = saved_graph(case, registry)
        try:
            graph.save(output / f"{case}.json")
        finally:
            graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    build(args.output)
