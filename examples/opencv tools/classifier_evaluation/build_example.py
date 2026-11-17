"""Original image-level evaluation fixtures; no external images or model downloads."""

import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import FlowGroup, ValuePacket
from opencv_dataflow.validation import run_set

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "knn_classification"
CASES = ("correct", "rejected", "mislabelled_training", "wrong_contract")


def write_json(path, value):
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def annotate(graph, registry, message):
    for block in list(graph.blocks.values()):
        if block.type_id == "layout.comment":
            graph.remove(block.id)
    note = graph.add(registry.create("layout.comment", block_id="attribution"))
    note.name, note.node_size, note.position = (
        "Original classifier evaluation controls",
        "large",
        [0, -470],
    )
    note.description = (
        "Original MIT stripe images and HOG/KNN controls by OpenCVDataflowGUI contributors. "
        "Copied unchanged from knn_classification with pinned SHA-256. "
        "No outside tutorial, dataset or model content. Training captures 0/1/2 and held-out "
        "capture are separate; these tiny synthetic controls prove software behavior only. "
        + message
        + " See README.md, SOURCES.md and truth.json."
    )


def build():
    assets = HERE / "assets"
    if assets.exists():
        raise FileExistsError("Existing evaluation assets preserved")
    original = json.loads((SOURCE / "truth.json").read_bytes())
    names = [*original["training"], "heldout-vertical.png", "heldout-horizontal.png"]
    names.append("classifier-none.json")
    files = {}
    for name in names:
        raw = (SOURCE / "assets" / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != original["asset_sha256"][name]:
            raise ValueError("Public fixture changed: " + name)
        files[name] = raw
    assets.mkdir()
    for name, raw in files.items():
        (assets / name).write_bytes(raw)
    registry = builtin_registry()
    # Train real native KNN with intentionally swapped training labels. Held-out
    # ground truth is never swapped; this must yield an off-diagonal confusion.
    hog = registry.create("inspect.hog")
    try:
        entries = []
        for name in original["training"]:
            image = cv2.imdecode(np.frombuffer(files[name], np.uint8), cv2.IMREAD_GRAYSCALE)
            result = hog.execute({"image": image})
            if result is None:
                raise RuntimeError(hog.error)
            entries.append(ValuePacket(result.outputs["descriptor"], result.payloads["descriptor"]))
        samples = FlowGroup(tuple(entries), tuple(original["training"]))
        trainer = registry.create(
            "ml.knn_train", parameters={"labels": ["horizontal"] * 3 + ["vertical"] * 3}
        )
        try:
            result = trainer.execute({"samples": samples})
            if result is None:
                raise RuntimeError(trainer.error)
            write_json(assets / "mislabelled.json", result.outputs["model"])
        finally:
            trainer.close()
    finally:
        hog.close()
    image_cases = [
        {
            "id": direction,
            "path": f"assets/heldout-{direction}.png",
            "class_label": direction,
            "capture_group": "heldout",
            "expectation": "unknown",
            "image_sha256": hashlib.sha256(files[f"heldout-{direction}.png"]).hexdigest(),
        }
        for direction in ("vertical", "horizontal")
    ]
    records = None
    truth = {"projects": {}}
    for case in CASES:
        graph = Graph.load(SOURCE / "classify.json", registry)
        try:
            graph.project_dir = HERE
            predictor = graph.blocks["predict"]
            if case == "rejected":
                predictor.parameters["max_distance"] = 0.000001
            if case == "mislabelled_training":
                graph.blocks["model"].parameters["path"] = "assets/mislabelled.json"
            if case == "wrong_contract":
                graph.blocks["hog"].parameters["signed_gradient"] = True
            graph.validation_sets = [
                {
                    "id": "heldout",
                    "name": "Held-out image classes",
                    "source": "source",
                    "state_mode": "sequence",
                    "cases": copy.deepcopy(image_cases),
                    "classification": {
                        "block": "predict",
                        "classes": ["vertical", "horizontal"],
                        "training_groups": ["capture-0", "capture-1", "capture-2"],
                        "score_output": "report",
                        "score_path": "vote_fractions.0",
                        "score_kind": "native_knn_vote_fraction",
                        "thresholds": [0, 0.8, 1],
                    },
                }
            ]
            annotate(
                graph,
                registry,
                "Run Results > Validation image sets > Run image set. "
                + {
                    "correct": "Expected: 2/2 correct.",
                    "rejected": "Expected: both rejected; accuracy/coverage zero.",
                    "mislabelled_training": "Deliberately swapped training labels; both wrong.",
                    "wrong_contract": "Signed HOG violates the saved model contract; both errors.",
                }[case],
            )
            graph.save(HERE / (case + ".json"))
            report = run_set(graph.to_dict(), graph.validation_sets[0], registry, HERE)
            summary = report["classification"]["summary"]
            truth["projects"][case] = {
                "accuracy": summary["accuracy"],
                "coverage": summary["coverage"],
                "status_counts": summary["status_counts"],
                "confusion": report["classification"]["confusion"],
                "passed": report["summary"]["passed"],
            }
            if case == "correct":
                records = report["classification"]["records"]
                # Random run tickets and timing are deliberately not persisted.
        finally:
            graph.close()
    graph = Graph()
    try:
        graph.add(
            registry.create(
                "ml.evaluate_classification",
                block_id="evaluate",
                parameters={
                    "records": records,
                    "expected_cases": image_cases,
                    "classes": ["vertical", "horizontal"],
                    "thresholds": [0, 0.8, 1],
                },
            )
        )
        annotate(
            graph, registry, "Stored actual classifier records; Run once to inspect Report/Plot."
        )
        graph.save(HERE / "metrics.json")
    finally:
        graph.close()
    samples = [
        {
            "id": name.removesuffix(".png"),
            "path": "assets/" + name,
            "class_label": "vertical" if "vertical" in name else "horizontal",
            "capture_group": "heldout" if name.startswith("heldout") else "capture-" + name[-5],
            "image_sha256": hashlib.sha256(files[name]).hexdigest(),
        }
        for name in names
        if name.endswith(".png")
    ]
    graph = Graph()
    try:
        graph.add(
            registry.create(
                "ml.split_image_groups",
                block_id="split",
                parameters={
                    "samples": samples,
                    "test_fraction": 0.25,
                    "validation_fraction": 0.25,
                    "seed": 17,
                },
            )
        )
        annotate(
            graph,
            registry,
            "Four independent capture groups. Split outputs plan whole "
            "groups, not image pixels or feature matrices. Hash claims are not file verification.",
        )
        graph.save(HERE / "split_groups.json")
    finally:
        graph.close()
    truth["asset_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(assets.iterdir())
    }
    truth["training"] = original["training"]
    truth["training_groups"] = ["capture-0", "capture-1", "capture-2"]
    write_json(HERE / "truth.json", truth)


if __name__ == "__main__":
    build()
