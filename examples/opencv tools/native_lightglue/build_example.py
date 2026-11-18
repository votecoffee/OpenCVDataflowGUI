"""Original image-dependent ONNX head controls; these models have no learned weights."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
CASES = ("translation", "score_rejection", "selected", "empty", "high_depth", "pair_budget")


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def models():
    import onnx
    from onnx import TensorProto as T
    from onnx import helper as h
    from onnx import numpy_helper as n

    def save(nodes, inputs, outputs, tensors):
        model = h.make_model(
            h.make_graph(
                nodes,
                "original_native_head_control",
                inputs,
                outputs,
                [n.from_array(v, k) for k, v in tensors.items()],
            ),
            opset_imports=[h.make_opsetid("", 13)],
            ir_version=8,
        )
        onnx.checker.check_model(model)
        return model.SerializeToString()

    def info(name, kind, shape):
        return h.make_tensor_value_info(name, kind, shape)

    points = np.float32(
        [
            [-0.65, -0.6],
            [-0.1, -0.6],
            [0.45, -0.6],
            [-0.65, 0.05],
            [-0.1, 0.05],
            [0.45, 0.05],
            [-0.65, 0.6],
            [0.45, 0.6],
        ]
    )
    descriptors = np.eye(8, 128, dtype=np.float32)
    weights = np.zeros((3, 16 + 1024 + 8), np.float32)
    weights[0, :16:2] = 0.5  # RGB red average moves the observations horizontally.
    bias = np.concatenate([points.ravel(), descriptors.ravel(), np.ones(8, np.float32)])
    feature = save(
        [
            h.make_node("GlobalAveragePool", ["image"], ["pooled"]),
            h.make_node("Flatten", ["pooled"], ["mean"]),
            h.make_node("Gemm", ["mean", "weights", "bias"], ["flat"]),
            h.make_node("Split", ["flat", "lengths"], ["xy", "desc", "score"], axis=1),
            h.make_node("Reshape", ["xy", "xy_shape"], ["keypoints"]),
            h.make_node("Reshape", ["desc", "desc_shape"], ["descriptors"]),
            h.make_node("Reshape", ["score", "score_shape"], ["scores"]),
        ],
        [info("image", T.FLOAT, [1, 3, 32, 32])],
        [
            info("keypoints", T.FLOAT, [8, 2]),
            info("descriptors", T.FLOAT, [8, 128]),
            info("scores", T.FLOAT, [8, 1]),
        ],
        {
            "weights": weights,
            "bias": bias,
            "lengths": np.int64([16, 1024, 8]),
            "xy_shape": np.int64([8, 2]),
            "desc_shape": np.int64([8, 128]),
            "score_shape": np.int64([8, 1]),
        },
    )
    matcher = save(
        [
            h.make_node("Sub", ["kpts0", "kpts1"], ["difference"]),
            h.make_node("Abs", ["difference"], ["absolute"]),
            h.make_node("ReduceSum", ["absolute", "axis"], ["displacement"], keepdims=0),
            h.make_node("Mul", ["desc0", "desc1"], ["products"]),
            h.make_node("ReduceSum", ["products", "axis"], ["similarity"], keepdims=0),
            h.make_node("Mul", ["desc0", "confidence_weights"], ["weighted"]),
            h.make_node("ReduceSum", ["weighted", "axis"], ["base"], keepdims=0),
            h.make_node("Mul", ["similarity", "base"], ["scored"]),
            h.make_node("Mul", ["displacement", "penalty"], ["deduction"]),
            h.make_node("Sub", ["scored", "deduction"], ["confidence"]),
            h.make_node("Reshape", ["confidence", "score_shape"], ["mscores0"]),
            h.make_node("Identity", ["pairs"], ["matches0"]),
        ],
        [
            info("kpts0", T.FLOAT, [1, "K", 2]),
            info("kpts1", T.FLOAT, [1, "K", 2]),
            info("desc0", T.FLOAT, [1, "K", 128]),
            info("desc1", T.FLOAT, [1, "K", 128]),
        ],
        [info("matches0", T.INT64, [8, 2]), info("mscores0", T.FLOAT, ["K", 1])],
        {
            "axis": np.int64([2]),
            "confidence_weights": np.float32(
                [0.98, 0.94, 0.9, 0.85, 0.8, 0.6, 0.3, 0.1] + [0] * 120
            ),
            "penalty": np.float32(0.25),
            "score_shape": np.int64([-1, 1]),
            "pairs": np.int64([[i, i] for i in range(8)]),
        },
    )
    return feature, matcher


def build(destination=HERE):
    destination = Path(destination)
    assets = destination / "assets"
    if assets.exists():
        raise FileExistsError("Existing native LightGlue assets preserved")
    assets.mkdir(parents=True)
    for name, raw in zip(("features.onnx", "matcher.onnx"), models(), strict=True):
        (assets / name).write_bytes(raw)
    reference = np.full((48, 64, 3), [35, 50, 60], np.uint8)
    for x, y in [(12, 10), (30, 10), (48, 10), (12, 25), (30, 25), (48, 25), (12, 38), (48, 38)]:
        cv2.rectangle(reference, (x - 2, y - 2), (x + 2, y + 2), (90, 150, 100), -1)
    query = reference.copy()
    query[..., 2] += 32
    for name, image in (("reference", reference), ("query", query)):
        (assets / (name + ".png")).write_bytes(cv2.imencode(".png", image)[1].tobytes())
    pins = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(assets.iterdir())}
    registry = builtin_registry()
    for case in CASES:
        graph = Graph()
        try:

            def add(kind, key, x, y=350, **p):
                block = graph.add(registry.create(kind, block_id=key, parameters=p))
                block.position = [x, y]
                return block

            for side, y in (("reference", 0), ("query", 350)):
                add(
                    "input.file",
                    side,
                    0,
                    y,
                    path=f"assets/{side}.png",
                    path_base="project",
                    pixel_depth="uint16" if case == "high_depth" else "unchanged",
                    color_format="rgb" if case == "high_depth" else "unchanged",
                )
                add(
                    "ml.native_aliked_features",
                    side + "_features",
                    380,
                    y,
                    model_path="assets/features.onnx",
                    model_sha256=pins["features.onnx"],
                    width=32,
                    height=32,
                    minimum_count=0 if case == "empty" else 4,
                )
                graph.connect(side, "image", side + "_features", "image")
                if case in ("selected", "empty"):
                    add(
                        "region.shape",
                        side + "_region",
                        0,
                        y + 170,
                        shape="rectangle",
                        rectangle=[-10, -10, 1, 1] if case == "empty" else [0, 0, 63, 30],
                    )
                    graph.connect(side, "image", side + "_region", "image")
                    graph.connect(side + "_region", "region", side + "_features", "region")
            match = add(
                "ml.native_lightglue",
                "match",
                780,
                model_path="assets/matcher.onnx",
                model_sha256=pins["matcher.onnx"],
                minimum_matches=4,
                minimum_inliers=4,
                filter_threshold=0.95 if case == "score_rejection" else 0.1,
                maximum_attention_pairs=1 if case == "pair_budget" else 1024,
            )
            match.exposed_outputs.append("result_graphics")
            graph.connect("query", "image", "match", "image")
            for side in ("query", "reference"):
                graph.connect(side + "_features", "features", "match", side + "_features")
            add("cv.overlay_result_graphics", "overlay", 1180)
            add("output.filmstrip_image", "filmstrip", 1580)
            graph.connect("match", "image", "overlay", "image")
            graph.connect("match", "result_graphics", "overlay", "graphics_1")
            graph.connect("overlay", "image", "filmstrip", "image")
            note = add("layout.comment", "attribution", 0, -450)
            note.name, note.node_size = "Original native ONNX controls - MIT", "large"
            note.description = (
                "Original OpenCVDataflowGUI images and untrained arithmetic ONNX models, MIT. "
                "Image red-channel mean moves feature points; descriptor agreement and normalized "
                "point displacement determine match scores. These fixtures test native input, "
                "score, geometry and failure contracts, not learned feature accuracy. "
                "See README.md/SOURCES.md for the separate pinned pretrained qualification."
            )
            graph.save(destination / (case + ".json"))
        finally:
            graph.close()
    (destination / "truth.json").write_bytes(
        encoded(
            {
                "cases": list(CASES),
                "asset_sha256": pins,
                "translation_x": 64 * 0.5 * 0.5 * (32 / 255),
                "translation_y": 0,
                "translation_count": 7,
                "translation_inliers": 7,
            }
        )
    )


if __name__ == "__main__":
    build()
