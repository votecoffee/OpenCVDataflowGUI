"""Train portable image PCA/classifier fixtures and saved public graphs."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import FlowGroup, ValuePacket

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "knn_classification"
TRAINING = [f"train-{d}-{i}.png" for d in ("vertical", "horizontal") for i in range(3)]
LABELS = ["vertical"] * 3 + ["horizontal"] * 3
CASES = (
    "train",
    "train_l2",
    "classify",
    "normalized",
    "variance",
    "reconstruct",
    "reject_diagonal",
    "wrong_descriptor",
    "wrong_revision",
    "dimensions_only",
)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


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
        raise FileExistsError("Existing image/PCA example assets preserved")
    registry = builtin_registry()
    originals = json.loads((SOURCE / "truth.json").read_bytes())["asset_sha256"]
    files = {}
    for name in [
        *TRAINING,
        "heldout-vertical.png",
        "heldout-horizontal.png",
        "heldout-diagonal.png",
    ]:
        raw = (SOURCE / "assets" / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != originals[name]:
            raise ValueError("Public synthetic image differs: " + name)
        files[name] = raw
    entries = []
    for name in TRAINING:
        image = cv2.imdecode(np.frombuffer(files[name], np.uint8), cv2.IMREAD_GRAYSCALE)
        result = execute(registry, "inspect.hog", {"image": image})
        entries.append(ValuePacket(result.outputs["descriptor"], result.payloads["descriptor"]))
    samples = FlowGroup(tuple(entries), tuple(TRAINING))
    for mode in ("none", "l2", "variance"):
        result = execute(
            registry,
            "cv.fit_pca",
            {"samples": samples},
            selection="variance" if mode == "variance" else "count",
            normalization="l2" if mode == "l2" else "none",
        )
        model = result.outputs["model"]
        files[f"pca-{mode}.json"] = encoded(model)
        projected = execute(registry, "cv.transform_pca", {"samples": samples, "model": model})
        classifier = execute(
            registry,
            "ml.knn_train",
            {"samples": ValuePacket(projected.outputs["samples"], projected.payloads["samples"])},
            labels=LABELS,
            k=1,
        )
        files[f"classifier-{mode}.json"] = encoded(classifier.outputs["model"])
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
            paths = (
                TRAINING
                if training
                else ["heldout-diagonal.png"]
                if case == "reject_diagonal"
                else ["heldout-vertical.png", "heldout-horizontal.png"]
            )
            add(
                "input.file",
                "source",
                0,
                0,
                source_mode="files",
                path_base="project",
                paths=["assets/" + name for name in paths],
                color_format="gray",
                pixel_depth="uint8",
            )
            add(
                "inspect.hog",
                "hog",
                360,
                0,
                **(
                    {"signed_gradient": True}
                    if case in ("wrong_descriptor", "dimensions_only")
                    else {}
                ),
            )
            graph.connect("source", "image", "hog", "image")
            mode = (
                "l2"
                if case in ("train_l2", "normalized", "wrong_revision")
                else "variance"
                if case == "variance"
                else "none"
            )
            if training:
                add("flow.batch", "batch", 720, 0, size=6)
                add("cv.fit_pca", "fit", 1080, 0, selection="count", normalization=mode)
                add("cv.transform_pca", "project", 1440, 0)
                add("ml.knn_train", "train", 1800, 0, labels=LABELS, k=1)
                add("cv.save_pca", "save_pca", 1440, 380, path=f"trained-pca-{mode}.json")
                add("ml.knn_save", "save", 2160, 0, path=f"trained-classifier-{mode}.json")
                for a, b, c, d in (
                    ("hog", "descriptor", "batch", "item_1"),
                    ("batch", "group", "fit", "samples"),
                    ("batch", "group", "project", "samples"),
                    ("fit", "model", "project", "model"),
                    ("fit", "model", "save_pca", "model"),
                    ("project", "samples", "train", "samples"),
                    ("train", "model", "save", "model"),
                ):
                    graph.connect(a, b, c, d)
                target, states, labels = "save", ["inactive"] * 5 + ["pass"], []
            else:
                add("cv.load_pca", "pca", 360, 380, path=f"assets/pca-{mode}.json")
                project = add(
                    "cv.transform_pca",
                    "project",
                    720,
                    0,
                    check_reconstruction_error=case == "reject_diagonal",
                    error_limit=0.03,
                    contract_policy="dimensions_only"
                    if case == "dimensions_only"
                    else "require_match",
                )
                graph.connect("hog", "descriptor", "project", "samples")
                graph.connect("pca", "model", "project", "model")
                graph.connect("source", "image", "project", "image")
                if case in ("reject_diagonal", "wrong_descriptor", "dimensions_only"):
                    terminal = project
                    target, labels = "project", []
                    states = (
                        ["fail"]
                        if case == "reject_diagonal"
                        else ["error"] * 2
                        if case == "wrong_descriptor"
                        else ["pass"] * 2
                    )
                elif case == "reconstruct":
                    terminal = add("cv.transform_pca", "restore", 1080, 0, direction="reconstruct")
                    graph.connect("pca", "model", "restore", "model")
                    graph.connect("project", "samples", "restore", "samples")
                    graph.connect("source", "image", "restore", "image")
                    target, states, labels = "restore", ["pass"] * 2, []
                else:
                    classifier_mode = "none" if case == "wrong_revision" else mode
                    add(
                        "ml.knn_load",
                        "classifier",
                        720,
                        650,
                        path=f"assets/classifier-{classifier_mode}.json",
                    )
                    terminal = add("ml.knn_predict", "predict", 1080, 0)
                    graph.connect("classifier", "model", "predict", "model")
                    graph.connect("project", "samples", "predict", "samples")
                    graph.connect("project", "image", "predict", "image")
                    target = "predict"
                    states = ["error"] * 2 if case == "wrong_revision" else ["pass"] * 2
                    labels = [] if case == "wrong_revision" else ["vertical", "horizontal"]
                terminal.exposed_outputs.append("result_graphics")
                add("cv.overlay_result_graphics", "overlay", 1440, 0)
                add("output.filmstrip_image", "filmstrip", 1800, 0)
                graph.connect(terminal.id, "image", "overlay", "image")
                graph.connect(terminal.id, "result_graphics", "overlay", "graphics_1")
                graph.connect("overlay", "image", "filmstrip", "image")
            note = add("layout.comment", "attribution", 0, -470)
            note.name, note.node_size = "Original image-descriptor PCA controls", "large"
            note.description = (
                "Original MIT synthetic stripe images and code by OpenCVDataflowGUI contributors. "
                "Nine images copied unchanged from knn_classification. "
                "Six training images and three "
                "held-out image hashes are disjoint. PCA and classifier models are trained only on "
                "the six training windows. Training variance is not held-out accuracy; "
                "reconstruction error and nearest-neighbor distance are not probabilities. "
                "These controlled windows do not establish general defect/classification "
                "performance. See README and SOURCES."
            )
            graph.save(HERE / (case + ".json"))
            cases[case] = {"target": target, "runs": len(paths), "states": states, "labels": labels}
        finally:
            graph.close()
    (HERE / "truth.json").write_bytes(
        encoded(
            {
                "cases": cases,
                "training": TRAINING,
                "labels": LABELS,
                "asset_sha256": {
                    name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()
                },
                "reconstruction_limit": 0.03,
            }
        )
    )


if __name__ == "__main__":
    build()
