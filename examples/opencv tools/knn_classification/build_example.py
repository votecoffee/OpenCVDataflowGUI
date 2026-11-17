"""Public image-descriptor training, native KNN and deliberate unknown-image controls."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import FlowGroup, ValuePacket

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "reusable_clusters"
TRAINING = [
    f"train-{direction}-{i}.png" for direction in ("vertical", "horizontal") for i in range(3)
]
CLASSES = ["vertical"] * 3 + ["horizontal"] * 3
CASES = (
    "train",
    "train_normalized",
    "classify",
    "single_neighbor",
    "normalized",
    "reject_diagonal",
    "vote_tie",
    "accepted_classes",
    "wrong_contract",
)


def build():
    assets = HERE / "assets"
    if assets.exists():
        raise FileExistsError("Existing K-nearest example assets preserved")
    source_truth = json.loads((SOURCE / "truth.json").read_bytes())
    files = {}
    for name in [*TRAINING, "heldout-vertical.png", "heldout-horizontal.png"]:
        raw = (SOURCE / "assets" / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != source_truth["asset_sha256"][name]:
            raise ValueError("Original public stripe control changed: " + name)
        files[name] = raw
    diagonal = np.full((32, 32), 25, np.uint8)
    for offset in (-8, 2, 12):
        cv2.line(diagonal, (3, max(3, offset + 3)), (28, min(28, offset + 28)), 230, 2)
    ok, encoded = cv2.imencode(".png", diagonal)
    if not ok:
        raise RuntimeError("Cannot encode diagonal control")
    files["heldout-diagonal.png"] = encoded.tobytes()
    registry = builtin_registry()
    hog = registry.create("inspect.hog")
    entries = []
    try:
        for name in TRAINING:
            image = cv2.imdecode(np.frombuffer(files[name], np.uint8), cv2.IMREAD_GRAYSCALE)
            result = hog.execute({"image": image})
            if result is None:
                raise RuntimeError(hog.error)
            entries.append(ValuePacket(result.outputs["descriptor"], result.payloads["descriptor"]))
    finally:
        hog.close()
    for mode in ("none", "l2"):
        trainer = registry.create(
            "ml.knn_train", parameters={"normalization": mode, "labels": CLASSES}
        )
        try:
            result = trainer.execute({"samples": FlowGroup(tuple(entries), tuple(TRAINING))})
            if result is None:
                raise RuntimeError(trainer.error)
            files["classifier-" + mode + ".json"] = (
                json.dumps(result.outputs["model"], sort_keys=True, indent=2, allow_nan=False)
                + "\n"
            ).encode()
        finally:
            trainer.close()
    assets.mkdir()
    for name, raw in files.items():
        (assets / name).write_bytes(raw)
    truth = {}
    for case in CASES:
        graph = Graph()

        def add(kind, block_id, x, y, **parameters):
            block = graph.add(registry.create(kind, block_id=block_id, parameters=parameters))
            block.position = [x, y]
            return block

        try:
            training = case in ("train", "train_normalized")
            paths = (
                TRAINING
                if training
                else ["heldout-diagonal.png"]
                if case == "reject_diagonal"
                else ["heldout-vertical.png"]
                if case == "vote_tie"
                else ["heldout-vertical.png", "heldout-horizontal.png"]
            )
            add(
                "input.file",
                "source",
                0,
                0,
                source_mode="files",
                paths=["assets/" + n for n in paths],
                path_base="project",
                color_format="gray",
                pixel_depth="uint8",
            )
            add(
                "inspect.hog",
                "hog",
                360,
                0,
                **({"signed_gradient": True} if case == "wrong_contract" else {}),
            )
            graph.connect("source", "image", "hog", "image")
            if training:
                mode = "l2" if case == "train_normalized" else "none"
                add("flow.batch", "batch", 720, 0, size=6)
                add("ml.knn_train", "train", 1080, 0, labels=CLASSES, normalization=mode)
                add(
                    "ml.knn_save",
                    "save",
                    1440,
                    0,
                    path="trained-normalized.json" if mode == "l2" else "trained-classifier.json",
                )
                graph.connect("hog", "descriptor", "batch", "item_1")
                graph.connect("batch", "group", "train", "samples")
                graph.connect("train", "model", "save", "model")
                labels, states = [], ["pass"] * 6
            else:
                add(
                    "ml.knn_load",
                    "model",
                    360,
                    420,
                    path="assets/classifier-l2.json"
                    if case == "normalized"
                    else "assets/classifier-none.json",
                )
                settings = {"max_distance": 0.5 if case in ("classify", "reject_diagonal") else 0.0}
                if case == "single_neighbor":
                    settings["k"] = 1
                if case == "vote_tie":
                    settings["k"] = 6
                if case == "accepted_classes":
                    settings["accepted_labels"] = ["vertical"]
                block = add("ml.knn_predict", "predict", 720, 0, **settings)
                block.exposed_outputs.append("result_graphics")
                add("cv.overlay_result_graphics", "overlay", 1080, 0)
                add("output.filmstrip_image", "filmstrip", 1440, 0)
                graph.connect("source", "image", "predict", "image")
                graph.connect("hog", "descriptor", "predict", "samples")
                graph.connect("model", "model", "predict", "model")
                graph.connect("predict", "image", "overlay", "image")
                graph.connect("predict", "result_graphics", "overlay", "graphics_1")
                graph.connect("overlay", "image", "filmstrip", "image")
                labels = (
                    [None]
                    if case in ("reject_diagonal", "vote_tie")
                    else ["vertical", "horizontal"]
                )
                states = (
                    ["error", "error"]
                    if case == "wrong_contract"
                    else ["fail"]
                    if case in ("reject_diagonal", "vote_tie")
                    else ["pass", "fail"]
                    if case == "accepted_classes"
                    else ["pass", "pass"]
                )
            note = add("layout.comment", "attribution", 0, -470)
            note.name, note.node_size = "Original synthetic HOG / K-nearest controls", "large"
            note.description = (
                "Original synthetic stripe images and code by OpenCVDataflowGUI "
                "contributors, MIT. Eight images copied unchanged from reusable_clusters; diagonal "
                "control is new. No external model, image or tutorial content. Training and query "
                "pixel hashes differ. Raw native votes/distances remain visible when recognition "
                "is withheld. Distance and vote agreement are not probabilities. This bounded "
                "orientation fixture is not production classification or OCR accuracy evidence. "
                "Models include the training feature rows. See README.md and SOURCES.md."
            )
            graph.save(HERE / (case + ".json"))
            truth[case] = {
                "target": "save" if training else "predict",
                "runs": len(paths),
                "states": states,
                "labels": labels,
                "passed": all(s == "pass" for s in states),
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
                    "training": TRAINING,
                    "class_labels": CLASSES,
                    "known_distance_limit": 0.5,
                    "expected_diagonal_unknown_reason": "knn_distance",
                },
                indent=2,
            )
            + "\n"
        ).encode()
    )


if __name__ == "__main__":
    build()
