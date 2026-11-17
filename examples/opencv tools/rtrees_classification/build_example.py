"""Native forest controls using original, pinned HOG image fixtures."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import FlowGroup, ValuePacket

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "knn_classification"
SCRIPT = Path(__file__).resolve()
TRAINING = [f"train-{d}-{i}.png" for d in ("vertical", "horizontal") for i in range(3)]
LABELS = ["vertical"] * 3 + ["horizontal"] * 3
CASES = (
    "train",
    "train_importance",
    "train_normalized",
    "train_weighted",
    "train_oob",
    "classify",
    "normalized",
    "vote_threshold",
    "accepted_classes",
    "wrong_contract",
)


def prepare_models(assets):
    """Explicit child-only fixture seeding; runtime training never changes the caller's seed."""
    cv2.setNumThreads(1)
    registry = builtin_registry()
    hog = registry.create("inspect.hog")
    entries = []
    try:
        for name in TRAINING:
            result = hog.execute({"image": cv2.imread(str(assets / name), cv2.IMREAD_GRAYSCALE)})
            if result is None:
                raise RuntimeError(hog.error)
            entries.append(ValuePacket(result.outputs["descriptor"], result.payloads["descriptor"]))
    finally:
        hog.close()
    for index, mode in enumerate(("none", "l2")):
        cv2.setRNGSeed(90113 + index)
        train = registry.create(
            "ml.rtrees_train", parameters={"labels": LABELS, "normalization": mode}
        )
        save = registry.create(
            "ml.rtrees_save", parameters={"path": str(assets / f"forest-{mode}.json")}
        )
        try:
            result = train.execute({"samples": FlowGroup(tuple(entries), tuple(TRAINING))})
            if result is None or save.execute({"model": result.outputs["model"]}) is None:
                raise RuntimeError(train.error or save.error)
        finally:
            train.close()
            save.close()


def build():
    assets = HERE / "assets"
    if assets.exists():
        raise FileExistsError("Existing forest example assets preserved")
    source_truth = json.loads((SOURCE / "truth.json").read_bytes())
    files = {}
    for name in [
        *TRAINING,
        *["heldout-" + d + ".png" for d in ("vertical", "horizontal", "diagonal")],
    ]:
        raw = (SOURCE / "assets" / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != source_truth["asset_sha256"][name]:
            raise ValueError("Pinned original stripe image changed: " + name)
        files[name] = raw
    assets.mkdir()
    for name, raw in files.items():
        (assets / name).write_bytes(raw)
    subprocess.run(
        [sys.executable, str(SCRIPT), "--prepare-models", str(assets)],
        check=True,
        timeout=45,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    files.update({p.name: p.read_bytes() for p in assets.glob("*.json")})
    registry = builtin_registry()
    truth = {}
    for case in CASES:
        graph = Graph()

        def add(kind, block_id, x, y, **parameters):
            block = graph.add(registry.create(kind, block_id=block_id, parameters=parameters))
            block.position = [x, y]
            return block

        try:
            training = case.startswith("train")
            paths = (
                TRAINING
                if training
                else ["heldout-diagonal.png"]
                if case == "vote_threshold"
                else ["heldout-vertical.png", "heldout-horizontal.png"]
            )
            add(
                "input.file",
                "source",
                0,
                0,
                source_mode="files",
                paths=["assets/" + p for p in paths],
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
                settings = {"labels": LABELS}
                if case == "train_importance":
                    settings["calculate_importance"] = True
                if case == "train_normalized":
                    settings["normalization"] = "l2"
                if case == "train_weighted":
                    settings.update(class_priors=[1, 2], sample_weights=[1, 2, 1, 1, 2, 1])
                if case == "train_oob":
                    settings.update(termination="tree_limit_or_oob_error", oob_error=0.05)
                add("flow.batch", "batch", 720, 0, size=6)
                add("ml.rtrees_train", "train", 1080, 0, **settings)
                add("ml.rtrees_save", "save", 1440, 0, path="trained-" + case + ".json")
                graph.connect("hog", "descriptor", "batch", "item_1")
                graph.connect("batch", "group", "train", "samples")
                graph.connect("train", "model", "save", "model")
                labels, states = [], ["pass"] * 6
            else:
                add(
                    "ml.rtrees_load",
                    "model",
                    360,
                    420,
                    path="assets/forest-l2.json"
                    if case == "normalized"
                    else "assets/forest-none.json",
                )
                settings = {
                    "min_vote_fraction": 0.9 if case in ("classify", "vote_threshold") else 0.0
                }
                if case == "accepted_classes":
                    settings["accepted_labels"] = ["vertical"]
                block = add("ml.rtrees_predict", "predict", 720, 0, **settings)
                block.exposed_outputs.append("result_graphics")
                add("cv.overlay_result_graphics", "overlay", 1080, 0)
                add("output.filmstrip_image", "filmstrip", 1440, 0)
                graph.connect("source", "image", "predict", "image")
                graph.connect("hog", "descriptor", "predict", "samples")
                graph.connect("model", "model", "predict", "model")
                graph.connect("predict", "image", "overlay", "image")
                graph.connect("predict", "result_graphics", "overlay", "graphics_1")
                graph.connect("overlay", "image", "filmstrip", "image")
                labels = [None] if case == "vote_threshold" else ["vertical", "horizontal"]
                states = (
                    ["error", "error"]
                    if case == "wrong_contract"
                    else ["fail"]
                    if case == "vote_threshold"
                    else ["pass", "fail"]
                    if case == "accepted_classes"
                    else ["pass", "pass"]
                )
            note = add("layout.comment", "attribution", 0, -480)
            note.name, note.node_size = "Original synthetic forest classification controls", "large"
            note.description = (
                "Original stripe images/code by OpenCVDataflowGUI contributors, MIT. "
                "Images copied unchanged from knn_classification. No external media, tutorial code "
                "or pretrained model. Frozen forests are generated by actual native training in "
                "a separate preparation process with recorded seeds; runtime training never "
                "changes the host RNG seed. Known/diagonal vote behavior is a bounded CPU fixture, "
                "not general novelty detection or production accuracy. Wrong preprocessing "
                "intentionally errors without prediction events. See README.md and SOURCES.md."
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
                    "class_labels": LABELS,
                    "fixture_seeds": {"none": 90113, "l2": 90114},
                    "expected_none_votes": {
                        "vertical": [60, 4],
                        "horizontal": [2, 62],
                        "diagonal": [12, 52],
                    },
                    "vote_threshold": 0.9,
                },
                indent=2,
            )
            + "\n"
        ).encode()
    )


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--prepare-models":
        prepare_models(Path(sys.argv[2]).resolve())
    else:
        build()
