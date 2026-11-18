"""Original HOG inspection training, held-out boxes and explicit false-positive controls."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import FlowGroup, ValuePacket

HERE = Path(__file__).resolve().parent
CASES = (
    "train",
    "single_scale",
    "multiscale",
    "raw_windows",
    "low_threshold",
    "selection",
    "high_depth",
    "count_fail",
    "blank",
    "wrong_model",
    "nonlinear_svm",
    "default_people",
    "daimler_people",
)
TRAINING = [
    f"{kind}-{seed}.png" for kind in ("part", "ring", "cross", "blank") for seed in range(4)
]
LABELS = ["part"] * 4 + ["background"] * 12
BOXES = [[32, 24, 32, 32], [112, 64, 32, 32]]


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def patch(kind, seed):
    image = np.full((32, 32), 30, np.uint8)
    if kind == "part":
        cv2.rectangle(image, (8, 6), (23, 25), 210, -1)
        cv2.rectangle(image, (14, 11), (23, 19), 30, -1)
        cv2.circle(image, (10, 9), 2, 80, -1)
    elif kind == "ring":
        cv2.circle(image, (16, 16), 10, 200, 3)
    elif kind == "cross":
        cv2.line(image, (7, 7), (24, 24), 200, 3)
        cv2.line(image, (7, 24), (24, 7), 200, 3)
    return np.uint8(
        np.clip(
            image.astype(int) + np.random.default_rng(seed).integers(-2, 3, image.shape), 0, 255
        )
    )


def build():
    assets = HERE / "assets"
    if assets.exists():
        raise FileExistsError("Existing HOG detection assets preserved")
    registry = builtin_registry()

    def run(kind, inputs, **parameters):
        block = registry.create(kind, parameters=parameters)
        try:
            result = block.execute(inputs)
            if result is None:
                raise RuntimeError(block.error)
            return result
        finally:
            block.close()

    images = {
        name: patch(name.rsplit("-", 1)[0], 100 + index % 4) for index, name in enumerate(TRAINING)
    }
    scene = np.full((128, 192), 30, np.uint8)
    for x, y, w, h in BOXES:
        scene[y : y + h, x : x + w] = patch("part", 44)
    scene[72:104, 40:72] = patch("ring", 45)
    images.update(
        {
            "scene.png": scene,
            "scene16.png": scene.astype(np.uint16) * 257,
            "empty.png": np.full(scene.shape, 30, np.uint8),
        }
    )
    files = {}
    for name, image in images.items():
        ok, raw = cv2.imencode(".png", image)
        assert ok
        files[name] = raw.tobytes()
    rows = []
    for name in TRAINING:
        result = run("inspect.hog", {"image": images[name]})
        rows.append(ValuePacket(result.outputs["descriptor"], result.payloads["descriptor"]))
    group = FlowGroup(tuple(rows), tuple(TRAINING))
    for kernel in ("linear", "rbf"):
        svm = run("ml.svm_train", {"samples": group}, kernel=kernel, labels=LABELS).outputs["model"]
        files[f"classifier-{kernel}.json"] = encoded(svm)
        if kernel == "linear":
            model = run("ml.hog_detector_model", {"svm": svm}, preset="linear_svm").outputs["model"]
            files["detector.json"] = encoded(model)
            changed = json.loads(files["detector.json"])
            changed["coefficients"][0] += 1
            files["changed-detector.json"] = encoded(changed)
    assets.mkdir()
    for name, raw in files.items():
        (assets / name).write_bytes(raw)
    cases = {}
    for case in CASES:
        graph = Graph()

        def add(kind, name, x, y, **parameters):
            block = graph.add(registry.create(kind, block_id=name, parameters=parameters))
            block.position = [x, y]
            return block

        try:
            fitting = case == "train"
            names = (
                TRAINING
                if fitting
                else ["empty.png"]
                if case in ("blank", "default_people", "daimler_people")
                else ["scene16.png"]
                if case == "high_depth"
                else ["scene.png"]
            )
            add(
                "input.file",
                "source",
                0,
                0,
                source_mode="files",
                paths=["assets/" + name for name in names],
                path_base="project",
                color_format="gray",
                pixel_depth="unchanged",
            )
            if fitting:
                add("inspect.hog", "features", 370, 0)
                add("flow.batch", "batch", 740, 0, size=16)
                add("ml.svm_train", "svm", 1110, 0, kernel="linear", labels=LABELS)
                add("ml.hog_detector_model", "model", 1480, 0, preset="linear_svm")
                add("ml.hog_detector_save", "save", 1850, 0, path="trained-detector.json")
                for a, o, b, i in (
                    ("source", "image", "features", "image"),
                    ("features", "descriptor", "batch", "item_1"),
                    ("batch", "group", "svm", "samples"),
                    ("svm", "model", "model", "svm"),
                    ("model", "model", "save", "model"),
                ):
                    graph.connect(a, o, b, i)
                target, states = "save", ["inactive"] * 15 + ["pass"]
            elif case == "nonlinear_svm":
                add("ml.svm_load", "svm", 370, 0, path="assets/classifier-rbf.json")
                add("ml.hog_detector_model", "model", 740, 0, preset="linear_svm")
                graph.connect("svm", "model", "model", "svm")
                target, states = "model", ["error"]
            else:
                presets = case in ("default_people", "daimler_people")
                if not presets:
                    add(
                        "ml.hog_detector_load",
                        "model",
                        370,
                        420,
                        path="assets/changed-detector.json"
                        if case == "wrong_model"
                        else "assets/detector.json",
                    )
                detector = add(
                    "ml.hog_detect",
                    "detect",
                    740,
                    0,
                    hit_threshold=0 if case == "low_threshold" else 0.5,
                    detector=case if presets else "default_people",
                    pyramid_levels=1
                    if case in ("single_scale", "high_depth", "count_fail", "selection")
                    else 3,
                    scale_factor=1.25,
                    nms_enabled=case != "raw_windows",
                    nms_iou=0.2,
                    minimum_count=0 if presets else 1,
                    maximum_count=1 if case == "count_fail" else 0,
                )
                detector.exposed_outputs.append("result_graphics")
                graph.connect("source", "image", "detect", "image")
                if not presets:
                    graph.connect("model", "model", "detect", "model")
                if case == "selection":
                    add("region.shape", "selection", 370, -420, rectangle=[0, 0, 96, 128])
                    graph.connect("source", "image", "selection", "image")
                    graph.connect("selection", "region", "detect", "region")
                add("cv.overlay_result_graphics", "overlay", 1110, 0)
                add("output.filmstrip_image", "filmstrip", 1480, 0)
                graph.connect("detect", "image", "overlay", "image")
                graph.connect("detect", "result_graphics", "overlay", "graphics_1")
                graph.connect("overlay", "image", "filmstrip", "image")
                target = "model" if case == "wrong_model" else "detect"
                states = (
                    ["error"]
                    if case == "wrong_model"
                    else ["fail"]
                    if case in ("count_fail", "blank")
                    else ["pass"]
                )
            note = add("layout.comment", "attribution", 0, -850)
            note.name, note.node_size = "Original HOG inspection controls - MIT", "large"
            note.description = (
                "Original MIT synthetic images, native SVM training and projects by "
                "OpenCVDataflowGUI contributors. Sixteen noisy training windows exclude the "
                "query noise seeds. Two query parts have exact construction boxes; a ring is "
                "a distractor. Raw margins are not probabilities. The permissive threshold "
                "retains extra detections, while multiscale search yields overlapping windows "
                "removed by explicit IoU suppression. Native people models are exercised only "
                "on a blank control: this is not pedestrian accuracy evidence. See README.md, "
                "SOURCES.md, truth.json and benchmark.json. All image data and saved custom "
                "models are original; no customer captures."
            )
            graph.save(HERE / (case + ".json"))
            cases[case] = {"target": target, "states": states, "images": names}
        finally:
            graph.close()
    (HERE / "truth.json").write_bytes(
        encoded(
            {
                "cases": cases,
                "boxes": BOXES,
                "training": TRAINING,
                "labels": LABELS,
                "evidence": "Original construction boxes; native/independent margin "
                "and window references, not production accuracy",
                "asset_sha256": {
                    name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()
                },
            }
        )
    )


if __name__ == "__main__":
    build()
