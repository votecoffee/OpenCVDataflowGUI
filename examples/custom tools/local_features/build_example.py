"""Original tensor-interface controls, not a SuperPoint detector or learned feature benchmark."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
MATCHER_SHA = "6ff7040d0a497fc6639337946d7538dae07428c18f77a067a0b5a960e7cc551a"


def assets():
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(9)
    points = rng.uniform([10, 10], [170, 130], (24, 2)).astype(np.float32)
    descriptors = rng.normal(size=(24, 256)).astype(np.float32)
    descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)
    scores = np.linspace(0.4, 0.95, 24, dtype=np.float32)
    nodes = [helper.make_node("GlobalAveragePool", ["image"], ["mean"])]
    initializers, outputs = [], []
    for name, values, gain in (
        ("keypoints", points, 0),
        ("query_keypoints", points + [8, 6], 0),
        ("descriptors", descriptors, 0.0001),
        ("scores", scores, 0),
    ):
        values = values.astype(np.float32)
        shape = [1, *values.shape]
        initializers.extend(
            [
                numpy_helper.from_array(
                    np.full((values.size, 1, 1, 1), gain, np.float32), name + "_w"
                ),
                numpy_helper.from_array(values.ravel(), name + "_b"),
                numpy_helper.from_array(np.array(shape, np.int64), name + "_shape"),
            ]
        )
        nodes.extend(
            [
                helper.make_node(
                    "Conv", ["mean", name + "_w", name + "_b"], [name + "_raw"], kernel_shape=[1, 1]
                ),
                helper.make_node("Reshape", [name + "_raw", name + "_shape"], [name]),
            ]
        )
        outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))
    graph = helper.make_graph(
        nodes,
        "original_feature_contract",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 1, 160, 200])],
        outputs,
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    model.doc_string = "Original prescribed keypoints/descriptors. Not a trained detector."
    onnx.checker.check_model(model)
    # Identical image pair to the existing SIFT +8,+6 native regression.
    image = cv2.GaussianBlur(
        np.random.default_rng(4).integers(0, 255, (160, 200), np.uint8), (3, 3), 0
    )
    query = cv2.warpAffine(image, np.float32([[1, 0, 8], [0, 1, 6]]), (200, 160))
    return model.SerializeToString(), image, query


def build():
    model, reference, query = assets()
    path = HERE / "assets"
    path.mkdir(parents=True, exist_ok=True)
    files = {"feature_heads.onnx": model}
    for name, pixels in (("reference.png", reference), ("query.png", query)):
        ok, encoded = cv2.imencode(".png", pixels)
        assert ok
        files[name] = encoded.tobytes()
    for name in files:
        if (path / name).exists():
            raise FileExistsError("Existing assets preserved: " + name)
    for name, raw in files.items():
        (path / name).write_bytes(raw)
    registry = builtin_registry()
    cases = {}
    for case in (
        "pack_only",
        "translation",
        "masked",
        "score_filter",
        "empty",
        "minimum_matches",
        "comparison",
    ):
        graph = Graph()
        try:
            for side in ("reference", "query"):
                source = graph.add(
                    registry.create(
                        "input.file",
                        block_id=side,
                        parameters={"path": f"assets/{side}.png", "path_base": "project"},
                    )
                )
                source.position = [0, 0 if side == "reference" else 430]
                infer = graph.add(
                    registry.create(
                        "ml.dnn",
                        block_id=side + "_infer",
                        parameters={
                            "model_path": "assets/feature_heads.onnx",
                            "width": 200,
                            "height": 160,
                            "color_format": "gray",
                            "mean": [0.0],
                            "std": [1.0],
                            "input_name": "image",
                            "engine": "classic",
                            "output_names": [
                                "keypoints" if side == "reference" else "query_keypoints",
                                "descriptors",
                                "scores",
                            ],
                        },
                    )
                )
                pack = graph.add(
                    registry.create(
                        "ml.pack_superpoint_features",
                        block_id=side + "_pack",
                        parameters={
                            "points_output": "keypoints"
                            if side == "reference"
                            else "query_keypoints",
                            "minimum_score": 1.0
                            if case == "empty"
                            else 0.75
                            if case == "score_filter"
                            else 0,
                            "minimum_count": 0 if case == "empty" else 4,
                        },
                    )
                )
                infer.position, pack.position = [360, source.position[1]], [720, source.position[1]]
                graph.connect(side, "image", infer.id, "image")
                graph.connect(side, "image", pack.id, "image")
                for port in ("tensors", "preprocessing"):
                    graph.connect(infer.id, port, pack.id, port)
                if case == "masked":
                    roi = graph.add(
                        registry.create(
                            "region.shape",
                            block_id=side + "_roi",
                            parameters={
                                "shape": "rectangle",
                                "rectangle": [0 if side == "reference" else 8, 0, 100, 160],
                            },
                        )
                    )
                    graph.connect(side, "image", roi.id, "image")
                    graph.connect(roi.id, "region", pack.id, "region")
                    roi.position = [360, source.position[1] + 220]
            if case != "pack_only":
                match = graph.add(
                    registry.create(
                        "ml.lightglue_superpoint",
                        block_id="match",
                        parameters={
                            "weights_path": "../../../.local/models/superpoint_lightglue.pth",
                            "weights_sha256": MATCHER_SHA,
                            "minimum_matches": 4096 if case == "minimum_matches" else 4,
                            "minimum_inliers": 4,
                        },
                    )
                )
                match.position = [1100, 430]
                match.exposed_outputs.append("result_graphics")
                graph.connect("query", "image", "match", "image")
                graph.connect("query_pack", "features", "match", "query_features")
                graph.connect("reference_pack", "features", "match", "reference_features")
                overlay = graph.add(
                    registry.create("cv.overlay_result_graphics", block_id="overlay")
                )
                overlay.position = [1460, 430]
                graph.connect("query", "image", "overlay", "image")
                graph.connect("match", "result_graphics", "overlay", "graphics_1")
                film = graph.add(registry.create("output.filmstrip_image", block_id="filmstrip"))
                film.position = [1820, 430]
                graph.connect("overlay", "image", "filmstrip", "image")
            if case == "comparison":
                sift = graph.add(
                    registry.create(
                        "ml.lightglue",
                        block_id="sift",
                        parameters={
                            "weights_path": "../../../.local/models/sift_lightglue.pth",
                            "max_features": 64,
                        },
                    )
                )
                sift.position = [1100, 0]
                graph.connect("reference", "image", "sift", "reference")
                graph.connect("query", "image", "sift", "image")
            note = graph.add(registry.create("layout.comment", block_id="attribution"))
            note.name, note.node_size, note.position = (
                "Local-feature contract controls",
                "large",
                [0, -500],
            )
            note.description = (
                "Original MIT texture pair, prescribed random keypoints/descriptors and ONNX "
                "arithmetic by OpenCVDataflowGUI contributors. Expected translation +8,+6 pixels. "
                "These heads are NOT SuperPoint detection and do not qualify detector accuracy. "
                "Matcher: Lindenberger, Sarlin and Pollefeys, LightGlue (ICCV 2023), Apache-2.0, "
                "downloaded separately. SuperPoint detector/code/weights have separate restrictive "
                "terms and are not included. "
                "See README for sources and scope. Empty/minimum_matches intentionally fail."
            )
            graph.save(HERE / (case + ".json"))
            cases[case] = {
                "passed": case not in ("empty", "minimum_matches"),
                "matcher": case != "pack_only",
                "translation": [8, 6],
            }
        finally:
            graph.close()
    (HERE / "truth.json").write_bytes(
        (
            json.dumps(
                {
                    "license": "MIT - original contract controls, not a SuperPoint detector",
                    "cases": cases,
                    "matcher_sha256": MATCHER_SHA,
                    "asset_sha256": {
                        name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()
                    },
                },
                indent=2,
            )
            + "\n"
        ).encode()
    )


if __name__ == "__main__":
    build()
