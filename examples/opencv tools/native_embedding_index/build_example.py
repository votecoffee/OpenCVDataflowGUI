"""Original image-derived galleries and native retrieval/persistence controls."""

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
TRAINING = [
    f"train-{direction}-{i}.png" for direction in ("vertical", "horizontal") for i in range(3)
]
LABELS = ["vertical"] * 3 + ["horizontal"] * 3
CASES = (
    "euclidean",
    "manhattan",
    "angular",
    "dot",
    "hamming",
    "normalized",
    "reject",
    "wrong_contract",
    "train_memory",
    "train_disk",
)


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def build(destination=HERE):
    destination = Path(destination)
    assets = destination / "assets"
    if assets.exists():
        raise FileExistsError("Existing native retrieval assets preserved")
    assets.mkdir(parents=True)
    (destination / ".gitattributes").write_bytes(b"*.json -text\n.gitattributes -text\n")
    source_truth = json.loads((SOURCE / "truth.json").read_bytes())
    for name in [
        *TRAINING,
        "heldout-vertical.png",
        "heldout-horizontal.png",
        "heldout-diagonal.png",
    ]:
        raw = (SOURCE / "assets" / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != source_truth["asset_sha256"][name]:
            raise ValueError("Original public image changed: " + name)
        (assets / name).write_bytes(raw)
    registry = builtin_registry()
    for metric in ("euclidean", "manhattan", "angular", "dot", "hamming", "normalized"):
        descriptor = registry.create(
            "cv.image_hash" if metric == "hamming" else "inspect.hog",
            parameters={"method": "block_mean"} if metric == "hamming" else {},
        )
        values = []
        try:
            for name in TRAINING:
                image = cv2.imdecode(
                    np.frombuffer((assets / name).read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE
                )
                result = descriptor.execute({"image": image})
                if result is None:
                    raise RuntimeError(descriptor.error)
                port = "hash" if metric == "hamming" else "descriptor"
                values.append(ValuePacket(result.outputs[port], result.payloads[port]))
        finally:
            descriptor.close()
        train = registry.create(
            "ml.native_ann_build",
            parameters={
                "metric": "euclidean" if metric == "normalized" else metric,
                "normalization": "l2" if metric == "normalized" else "none",
                "build_mode": "disk" if metric == "normalized" else "memory",
                "trees": 3,
                "seed": 7,
                "item_ids": TRAINING,
                "labels": LABELS,
            },
        )
        try:
            result = train.execute({"samples": FlowGroup(tuple(values), tuple(TRAINING))})
            if result is None:
                raise RuntimeError(train.error)
            (assets / (metric + ".json")).write_bytes(encoded(result.outputs["model"]))
        finally:
            train.close()
    for case in CASES:
        graph = Graph()
        try:

            def add(kind, key, x, y=0, **parameters):
                block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
                block.position = [x, y]
                return block

            training = case.startswith("train_")
            paths = (
                TRAINING
                if training
                else ["heldout-diagonal.png"]
                if case == "reject"
                else ["heldout-vertical.png", "heldout-horizontal.png"]
            )
            add(
                "input.file",
                "source",
                0,
                source_mode="files",
                paths=["assets/" + name for name in paths],
                path_base="project",
                color_format="gray",
            )
            add(
                "cv.image_hash" if case == "hamming" else "inspect.hog",
                "descriptor",
                360,
                **(
                    {"signed_gradient": True}
                    if case == "wrong_contract"
                    else {"method": "block_mean"}
                    if case == "hamming"
                    else {}
                ),
            )
            port = "hash" if case == "hamming" else "descriptor"
            graph.connect("source", "image", "descriptor", "image")
            if training:
                add("flow.batch", "batch", 720, size=6)
                add(
                    "ml.native_ann_build",
                    "index",
                    1080,
                    item_ids=TRAINING,
                    labels=LABELS,
                    trees=3,
                    seed=7,
                    build_mode="disk" if case == "train_disk" else "memory",
                )
                graph.connect("descriptor", port, "batch", "item_1")
                graph.connect("batch", "group", "index", "samples")
            else:
                model_case = (
                    case
                    if case in ("euclidean", "manhattan", "angular", "dot", "hamming", "normalized")
                    else "euclidean"
                )
                add("ml.native_ann_load", "model", 360, 400, path="assets/" + model_case + ".json")
                match = add(
                    "ml.native_ann_search",
                    "search",
                    720,
                    neighbors=3,
                    candidates=6,
                    search_k=10000,
                    filter_results=case == "reject",
                    maximum_distance=0,
                )
                match.exposed_outputs.append("result_graphics")
                graph.connect("descriptor", port, "search", "samples")
                graph.connect("source", "image", "search", "image")
                graph.connect("model", "model", "search", "model")
                add("cv.overlay_result_graphics", "overlay", 1080)
                add("output.filmstrip_image", "filmstrip", 1440)
                graph.connect("search", "image", "overlay", "image")
                graph.connect("search", "result_graphics", "overlay", "graphics_1")
                graph.connect("overlay", "image", "filmstrip", "image")
            note = add("layout.comment", "attribution", 0, -450)
            note.name, note.node_size = "Original native image retrieval controls - MIT", "large"
            note.description = (
                "Original synthetic stripe images copied unchanged from OpenCVDataflowGUI's "
                "K-nearest examples, MIT. Six gallery images and disjoint held-out queries feed "
                "HOG or binary block-mean image hashes. Five metrics use native "
                "ANN candidates with exact "
                "metric reranking; Hamming uses native Euclidean on unpacked bits. Stable IDs are "
                "filenames, labels are orientation classes. This controlled fixture does not "
                "demonstrate general similarity or inspection accuracy. See README.md/SOURCES.md."
            )
            graph.save(destination / (case + ".json"))
        finally:
            graph.close()
    pins = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(assets.iterdir())}
    (destination / "truth.json").write_bytes(
        encoded(
            {
                "cases": list(CASES),
                "asset_sha256": pins,
                "training": TRAINING,
                "labels": LABELS,
                "query_images": ["heldout-vertical.png", "heldout-horizontal.png"],
            }
        )
    )


if __name__ == "__main__":
    build()
