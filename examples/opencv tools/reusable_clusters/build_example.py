"""Original synthetic color inspections and HOG vocabulary training/reload graphs."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import FlowGroup, ValuePacket

HERE = Path(__file__).resolve().parent
COLORS = np.uint8([[24, 32, 200], [40, 180, 36], [210, 70, 25]])
CASES = (
    "train_palette",
    "palette_ok",
    "palette_defect",
    "palette_rgb16",
    "palette_lab",
    "palette_selection",
    "train_vocabulary",
    "vocabulary_query",
    "vocabulary_wrong_contract",
)


def image_bytes(image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Fixture PNG encoding failed")
    return encoded.tobytes()


def stripes(vertical, shift):
    image = np.full((32, 32), 25, np.uint8)
    for offset in (6, 14, 22):
        if vertical:
            cv2.line(image, (offset + shift, 3), (offset + shift, 28), 230, 2)
        else:
            cv2.line(image, (3, offset + shift), (28, offset + shift), 230, 2)
    return image


def build():
    registry = builtin_registry()
    target = HERE / "assets"
    if target.exists():
        raise FileExistsError("Existing cluster example assets preserved")
    reference = np.repeat(np.repeat(COLORS[None], 96, axis=0), 48, axis=1)
    query = np.concatenate((reference[:, 96:], reference[:, :96]), axis=1)
    defect = query.copy()
    defect[20:40, 12:28] = [240, 240, 240]
    images = {
        "reference.png": reference,
        "query.png": query,
        "defect.png": defect,
        "query16.png": query.astype(np.uint16) * 257,
    }
    training = []
    for vertical in (True, False):
        for shift in (-1, 0, 1):
            name = f"train-{'vertical' if vertical else 'horizontal'}-{shift + 1}.png"
            training.append(name)
            images[name] = stripes(vertical, shift)
    images["heldout-vertical.png"] = stripes(True, 2)
    images["heldout-horizontal.png"] = stripes(False, -2)
    files = {name: image_bytes(image) for name, image in images.items()}
    models = {}
    for space in ("bgr", "lab"):
        trainer = registry.create(
            "ml.palette_train", parameters={"color_space": space, "attempts": 1}
        )
        try:
            result = trainer.execute({"image": reference})
            if result is None:
                raise RuntimeError(trainer.error)
            models["palette-" + space + ".json"] = result.outputs["model"]
        finally:
            trainer.close()
    hog = registry.create("inspect.hog")
    entries = []
    try:
        for name in training:
            result = hog.execute({"image": images[name]})
            if result is None:
                raise RuntimeError(hog.error)
            entries.append(ValuePacket(result.outputs["descriptor"], result.payloads["descriptor"]))
    finally:
        hog.close()
    trainer = registry.create("ml.kmeans_train", parameters={"clusters": 2, "attempts": 1})
    try:
        result = trainer.execute(
            {
                "samples": FlowGroup(tuple(entries), tuple(training)),
                "initial_labels": [0, 0, 0, 1, 1, 1],
            }
        )
        if result is None:
            raise RuntimeError(trainer.error)
        models["vocabulary.json"] = result.outputs["model"]
    finally:
        trainer.close()
    for name, model in models.items():
        files[name] = (json.dumps(model, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    target.mkdir()
    for name, raw in files.items():
        (target / name).write_bytes(raw)

    truth = {}
    for case in CASES:
        graph = Graph()

        def add(kind, block_id, x, y, **parameters):
            block = graph.add(registry.create(kind, block_id=block_id, parameters=parameters))
            block.position = [x, y]
            return block

        try:
            vocabulary = "vocabulary" in case
            training_case = case.startswith("train_")
            if vocabulary:
                params = {"source_mode": "files", "paths": ["assets/" + n for n in training]}
                if not training_case:
                    params = {
                        "source_mode": "files",
                        "paths": ["assets/heldout-vertical.png", "assets/heldout-horizontal.png"],
                    }
                add(
                    "input.file", "source", 0, 0, path_base="project", color_format="gray", **params
                )
                add(
                    "inspect.hog",
                    "hog",
                    360,
                    0,
                    **({"signed_gradient": True} if case == "vocabulary_wrong_contract" else {}),
                )
                graph.connect("source", "image", "hog", "image")
                if training_case:
                    add("flow.batch", "batch", 720, 0, size=6)
                    graph.connect("hog", "descriptor", "batch", "item_1")
                    add("ml.kmeans_train", "train", 1080, 0, clusters=2, attempts=1)
                    graph.connect("batch", "group", "train", "samples")
                    add("ml.kmeans_save", "save", 1440, 0, path="trained-vocabulary.json")
                    graph.connect("train", "model", "save", "model")
                else:
                    add("ml.kmeans_load", "model", 360, 420, path="assets/vocabulary.json")
                    add("ml.kmeans_predict", "predict", 720, 0)
                    graph.connect("model", "model", "predict", "model")
                    graph.connect("hog", "descriptor", "predict", "samples")
            else:
                name = (
                    "reference.png"
                    if training_case
                    else "query16.png"
                    if case == "palette_rgb16"
                    else "defect.png"
                    if case in ("palette_defect", "palette_selection")
                    else "query.png"
                )
                add(
                    "input.file",
                    "source",
                    0,
                    0,
                    path="assets/" + name,
                    path_base="project",
                    pixel_depth="unchanged",
                    color_format="rgb" if case == "palette_rgb16" else "unchanged",
                )
                if training_case:
                    add("ml.palette_train", "train", 400, 0, color_space="bgr", attempts=1)
                    graph.connect("source", "image", "train", "image")
                    add("ml.kmeans_save", "save", 800, 0, path="trained-palette.json")
                    graph.connect("train", "model", "save", "model")
                else:
                    add(
                        "ml.kmeans_load",
                        "model",
                        0,
                        420,
                        path="assets/palette-lab.json"
                        if case == "palette_lab"
                        else "assets/palette-bgr.json",
                    )
                    block = add("ml.palette_predict", "predict", 400, 0, max_distance=0.05)
                    block.exposed_outputs.append("result_graphics")
                    graph.connect("model", "model", "predict", "model")
                    graph.connect("source", "image", "predict", "image")
                    if case == "palette_selection":
                        add("region.shape", "selection", 0, 840, rectangle=[48, 0, 96, 96])
                        graph.connect("source", "image", "selection", "image")
                        graph.connect("selection", "region", "predict", "region")
                    add("cv.overlay_result_graphics", "overlay", 800, 0)
                    add("output.filmstrip_image", "filmstrip", 1200, 0)
                    graph.connect("predict", "image", "overlay", "image")
                    graph.connect("predict", "result_graphics", "overlay", "graphics_1")
                    graph.connect("overlay", "image", "filmstrip", "image")
            note = add("layout.comment", "attribution", 0, -480)
            note.name, note.node_size = "Original synthetic cluster study", "large"
            note.description = (
                "Original synthetic images and graph code by OpenCVDataflowGUI "
                "contributors, MIT. No external images, models or tutorial content. Fixed palette "
                "IDs persist across other images and depth/encoding changes; descriptor vocabulary "
                "uses native OpenCV HOG on separate stripe arrangements. Training saves new files "
                "and preserves existing ones. Defect intentionally fails; wrong HOG preprocessing "
                "is an error without prediction events. Colors/distances are not probabilities "
                "or production accuracy evidence. See README.md and SOURCES.md."
            )
            graph.save(HERE / (case + ".json"))
            truth[case] = {
                "passed": case not in ("palette_defect", "vocabulary_wrong_contract"),
                "target": "save" if training_case else "predict",
                "runs": 6 if case == "train_vocabulary" else 2 if vocabulary else 1,
                "expected_state": "error"
                if case == "vocabulary_wrong_contract"
                else "fail"
                if case == "palette_defect"
                else "pass",
            }
        finally:
            graph.close()
    (HERE / "truth.json").write_bytes(
        (
            json.dumps(
                {
                    "cases": truth,
                    "asset_sha256": {
                        n: hashlib.sha256(raw).hexdigest() for n, raw in files.items()
                    },
                    "palette_query_label_columns": [2] * 48 + [0] * 48 + [1] * 48,
                    "defect_rectangle": [12, 20, 16, 20],
                },
                indent=2,
            )
            + "\n"
        ).encode()
    )


if __name__ == "__main__":
    build()
