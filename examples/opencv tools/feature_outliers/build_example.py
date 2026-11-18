"""Original normal-variation images, held-out edits and portable covariance examples."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import FlowGroup, ValuePacket

HERE = Path(__file__).resolve().parent
TRAINING = [f"train-{i}.png" for i in range(9)]
KNOWN = ["known.png", "known-dark.png", "known-bright.png"]
HOG = {"bins": 4}
CASES = (
    "train",
    "train_l2",
    "known_set",
    "crack",
    "missing_arc",
    "low_contrast_miss",
    "empty_view",
    "normalized_set",
    "diagnostic_only",
    "wrong_descriptor",
    "singular_no_ridge",
    "pca_known",
    "pca_wrong_revision",
)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def part(dx, dy, seed, gain=1):
    image = np.full((32, 32), 20, np.uint8)
    cv2.circle(image, (16 + dx, 16 + dy), 10, 180, -1)
    cv2.rectangle(image, (12 + dx, 12 + dy), (20 + dx, 20 + dy), 30, -1)
    return np.uint8(
        np.clip(
            image.astype(float) * gain + np.random.default_rng(seed).integers(-2, 3, image.shape),
            0,
            255,
        )
    )


def execute(registry, kind, inputs, **parameters):
    block = registry.create(kind, parameters=parameters)
    try:
        result = block.execute(inputs)
        if result is None:
            raise RuntimeError(block.error)
        return result
    finally:
        block.close()


def build():
    assets = HERE / "assets"
    if assets.exists():
        raise FileExistsError("Existing feature outlier assets preserved")
    registry = builtin_registry()
    images = {
        name: part(dx, dy, 100 + i)
        for i, (name, (dx, dy)) in enumerate(
            zip(TRAINING, ((x, y) for x in (-1, 0, 1) for y in (-1, 0, 1)), strict=True)
        )
    }
    images.update(
        {
            "known.png": part(0, 0, 21),
            "known-dark.png": part(0, 0, 22, 0.93),
            "known-bright.png": part(0, 0, 23, 1.07),
        }
    )
    base = images["known.png"]
    crack = base.copy()
    cv2.line(crack, (22, 8), (22, 24), 20, 2)
    arc = base.copy()
    cv2.rectangle(arc, (18, 5), (29, 16), 20, -1)
    subtle = base.copy()
    subtle[24, 16] = max(0, int(subtle[24, 16]) - 10)
    images.update(
        {
            "crack.png": crack,
            "missing-arc.png": arc,
            "low-contrast.png": subtle,
            "empty.png": np.full((32, 32), 20, np.uint8),
        }
    )
    for name in ("crack", "missing-arc", "low-contrast"):
        images[name + "-edit-mask.png"] = np.uint8(images[name + ".png"] != base) * 255
    files = {}
    for name, image in images.items():
        ok, raw = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("Cannot encode " + name)
        files[name] = raw.tobytes()
    packets = []
    for name in TRAINING:
        result = execute(registry, "inspect.hog", {"image": images[name]}, **HOG)
        packets.append(ValuePacket(result.outputs["descriptor"], result.payloads["descriptor"]))
    samples = FlowGroup(tuple(packets), tuple(TRAINING))
    for mode in ("none", "l2"):
        result = execute(
            registry, "cv.feature_covariance", {"samples": samples}, feature_normalization=mode
        )
        files[f"covariance-{mode}.json"] = encoded(result.outputs["model"])
    for mode in ("none", "l2"):
        pca = execute(
            registry, "cv.fit_pca", {"samples": samples}, selection="count", normalization=mode
        )
        files[f"pca-{mode}.json"] = encoded(pca.outputs["model"])
        if mode == "none":
            projected = execute(
                registry, "cv.transform_pca", {"samples": samples, "model": pca.outputs["model"]}
            )
            covariance = execute(
                registry,
                "cv.feature_covariance",
                {
                    "samples": ValuePacket(
                        projected.outputs["samples"], projected.payloads["samples"]
                    )
                },
            )
            files["covariance-pca.json"] = encoded(covariance.outputs["model"])
    assets.mkdir()
    for name, raw in files.items():
        (assets / name).write_bytes(raw)
    cases = {}
    for case in CASES:
        graph = Graph()

        def add(kind, block_id, x, y, **parameters):
            block = graph.add(registry.create(kind, block_id=block_id, parameters=parameters))
            block.position = [x, y]
            return block

        try:
            training = case.startswith("train")
            names = (
                TRAINING
                if training
                else [*KNOWN, "crack.png"]
                if case == "normalized_set"
                else KNOWN
                if case == "known_set"
                else [
                    "crack.png"
                    if case in ("crack", "diagnostic_only")
                    else "missing-arc.png"
                    if case == "missing_arc"
                    else "low-contrast.png"
                    if case == "low_contrast_miss"
                    else "empty.png"
                    if case == "empty_view"
                    else "known.png"
                ]
            )
            add(
                "input.file",
                "source",
                0,
                0,
                source_mode="files",
                path_base="project",
                paths=["assets/" + name for name in names],
                color_format="gray",
                pixel_depth="uint8",
            )
            add(
                "inspect.hog",
                "hog",
                360,
                0,
                **HOG,
                **({"signed_gradient": True} if case == "wrong_descriptor" else {}),
            )
            graph.connect("source", "image", "hog", "image")
            mode = "l2" if case in ("train_l2", "normalized_set") else "none"
            if training:
                add("flow.batch", "batch", 720, 0, size=9)
                add("cv.feature_covariance", "fit", 1080, 0, feature_normalization=mode)
                add(
                    "cv.save_feature_covariance",
                    "save",
                    1440,
                    0,
                    path=f"trained-covariance-{mode}.json",
                )
                for a, b, c, d in (
                    ("hog", "descriptor", "batch", "item_1"),
                    ("batch", "group", "fit", "samples"),
                    ("fit", "model", "save", "model"),
                ):
                    graph.connect(a, b, c, d)
                states = ["inactive"] * 8 + ["pass"]
                target = "save"
            else:
                pca_case = case.startswith("pca")
                add(
                    "cv.load_feature_covariance",
                    "model",
                    360,
                    440,
                    path="assets/covariance-pca.json"
                    if pca_case
                    else f"assets/covariance-{mode}.json",
                )
                inspect = add(
                    "inspect.mahalanobis_distance",
                    "inspect",
                    1080 if pca_case else 720,
                    0,
                    ridge=0 if case == "singular_no_ridge" else 0.0001 if mode == "l2" else 0.001,
                    check_distance=case != "diagnostic_only",
                    distance_limit=4,
                )
                inspect.exposed_outputs.append("result_graphics")
                graph.connect("source", "image", "inspect", "image")
                graph.connect("model", "model", "inspect", "model")
                if pca_case:
                    add(
                        "cv.load_pca",
                        "pca",
                        360,
                        800,
                        path="assets/pca-l2.json"
                        if case == "pca_wrong_revision"
                        else "assets/pca-none.json",
                    )
                    add("cv.transform_pca", "project", 720, 0)
                    graph.connect("hog", "descriptor", "project", "samples")
                    graph.connect("pca", "model", "project", "model")
                    graph.connect("project", "samples", "inspect", "samples")
                else:
                    graph.connect("hog", "descriptor", "inspect", "samples")
                add("cv.overlay_result_graphics", "overlay", 1440, 0)
                add("output.filmstrip_image", "filmstrip", 1800, 0)
                graph.connect("inspect", "image", "overlay", "image")
                graph.connect("inspect", "result_graphics", "overlay", "graphics_1")
                graph.connect("overlay", "image", "filmstrip", "image")
                target = "inspect"
                states = (
                    ["error"]
                    if case in ("wrong_descriptor", "singular_no_ridge", "pca_wrong_revision")
                    else ["pass", "pass", "pass", "fail"]
                    if case == "normalized_set"
                    else ["fail"]
                    if case in ("crack", "missing_arc", "empty_view")
                    else ["pass"] * len(names)
                )
            note = add("layout.comment", "attribution", 0, -480)
            note.name, note.node_size = "Original known-good feature inspection controls", "large"
            note.description = (
                "Original MIT synthetic part images, edit masks, models and graphs by "
                "OpenCVDataflowGUI contributors. Nine training views have +/-1 pixel shifts and "
                "separate noise. Held-out noise, gain changes and edits are excluded from fitting. "
                "Four-bin HOG has 144 values; sample covariance plus explicit Ridge measures "
                "whole-window deviation. A crack, removed arc and empty view fail; a subtle "
                "one-pixel intensity edit passes and remains a documented miss. Whole-window "
                "graphics do not localize defects. Distance is not a probability or physical "
                "measurement. See README.md, SOURCES.md and benchmark.json."
            )
            graph.save(HERE / (case + ".json"))
            cases[case] = {"target": target, "states": states, "images": names}
        finally:
            graph.close()
    (HERE / "truth.json").write_bytes(
        encoded(
            {
                "cases": cases,
                "training": TRAINING,
                "known": KNOWN,
                "asset_sha256": {
                    name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()
                },
                "distance_limit": 4,
                "ridge_none": 0.001,
                "ridge_l2": 0.0001,
                "scope": "Whole-window decisions; low-contrast edit is an intentional known miss",
            }
        )
    )


if __name__ == "__main__":
    build()
