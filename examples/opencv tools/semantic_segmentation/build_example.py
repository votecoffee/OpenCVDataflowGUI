"""Original per-pixel ONNX controls and independent source-grid label references."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
NAMES = ["background", "red part", "green part", "blue part"]
CASES = {
    "multiclass": {},
    "letterbox": {"resize": "letterbox"},
    "channels_last": {"output": "channels_last", "layout": "NHWC"},
    "probabilities": {"output": "probabilities", "activation": "probability"},
    "binary": {"output": "binary", "format": "binary", "activation": "sigmoid"},
    "class_ids": {"output": "class_ids", "format": "class_ids", "layout": "NHW"},
    "low_resolution": {"output": "low_resolution"},
    "linear_scores": {"resampling": "scores_linear"},
    "selected_roi": {"roi": True},
    "confidence_gate": {"minimum_confidence": 0.8},
    "empty": {"empty": True, "passed": False},
    "area_failure": {"maximum_fraction": 0.05, "passed": False},
}


def model_bytes():
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    weights = np.zeros((4, 3, 1, 1), np.float32)
    for i in range(3):
        weights[i + 1, i, 0, 0] = 4
    constants = {
        "weights": weights,
        "bias": np.array([1, 0, 0, 0], np.float32),
        "red_weights": np.array([[[[8]], [[0]], [[0]]]], np.float32),
        "red_bias": np.array([-4], np.float32),
    }
    nodes = [
        helper.make_node("Conv", ["image", "weights", "bias"], ["logits"], kernel_shape=[1, 1]),
        helper.make_node("Transpose", ["logits"], ["channels_last"], perm=[0, 2, 3, 1]),
        helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
        helper.make_node("ArgMax", ["logits"], ["ids"], axis=1, keepdims=0),
        helper.make_node("Cast", ["ids"], ["class_ids"], to=TensorProto.FLOAT),
        helper.make_node(
            "AveragePool", ["logits"], ["low_resolution"], kernel_shape=[2, 2], strides=[2, 2]
        ),
        helper.make_node(
            "Conv", ["image", "red_weights", "red_bias"], ["binary"], kernel_shape=[1, 1]
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "original_color_segmentation",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 16, 24])],
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
            for name, shape in (
                ("logits", [1, 4, 16, 24]),
                ("channels_last", [1, 16, 24, 4]),
                ("probabilities", [1, 4, 16, 24]),
                ("class_ids", [1, 16, 24]),
                ("low_resolution", [1, 4, 8, 12]),
                ("binary", [1, 1, 16, 24]),
            )
        ],
        [numpy_helper.from_array(value, name) for name, value in constants.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    model.doc_string = "Original MIT color arithmetic, no trained weights or recognition claims."
    onnx.checker.check_model(model)
    return model.SerializeToString()


def asset(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw:
        raise FileExistsError("Different existing assets preserved; build in a fresh folder")
    path.write_bytes(raw)


def png(path, image):
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    asset(path, encoded.tobytes())


def reference(image, options):
    """Color equations and scalar sampling, without calling inference or the decoder."""
    h, w = image.shape[:2]
    sx, sy, left, top = 24 / w, 16 / h, 0, 0
    rgb = image[..., ::-1].astype(np.float32)
    if options.get("resize") == "letterbox":
        rw, rh = round(w * min(sx, sy)), round(h * min(sx, sy))
        left, top = (24 - rw) // 2, (16 - rh) // 2
        network = np.zeros((16, 24, 3), np.float32)
        network[top : top + rh, left : left + rw] = cv2.resize(rgb, (rw, rh))
        sx, sy = rw / w, rh / h
    else:
        network = cv2.resize(rgb, (24, 16))
    network *= np.float32(1 / 255)
    values = np.concatenate((np.ones((1, 16, 24), np.float32), 4 * network.transpose(2, 0, 1)))
    binary = options.get("format") == "binary"
    if binary:
        values = (8 * network[..., 0] - 4)[None]
    if options.get("output") == "low_resolution":
        values = values.reshape(4, 8, 2, 12, 2).mean(axis=(2, 4))
    oh, ow = values.shape[1:]
    x = ((np.arange(w) + 0.5) * sx + left) * ow / 24
    y = ((np.arange(h) + 0.5) * sy + top) * oh / 16
    if options.get("resampling") == "scores_linear":
        sampled = np.empty((values.shape[0], h, w), np.float32)
        # Scalar bilinear reference for the installed CPU float-map implementation.
        for row in range(h):
            yy = float(np.float32(y[row] - 0.5))
            y0 = int(np.floor(yy))
            fy = yy - y0
            for col in range(w):
                xx = float(np.float32(x[col] - 0.5))
                x0 = int(np.floor(xx))
                fx = xx - x0
                sampled[:, row, col] = sum(
                    weight * values[:, min(max(iy, 0), oh - 1), min(max(ix, 0), ow - 1)]
                    for iy, ix, weight in (
                        (y0, x0, (1 - fy) * (1 - fx)),
                        (y0, x0 + 1, (1 - fy) * fx),
                        (y0 + 1, x0, fy * (1 - fx)),
                        (y0 + 1, x0 + 1, fy * fx),
                    )
                )
        values = sampled
    else:
        xi, yi = np.floor(x).astype(int), np.floor(y).astype(int)
        values = values[:, yi[:, None], xi]
    if binary:
        foreground = 1 / (1 + np.exp(-values[0]))
        labels = (foreground >= 0.5).astype(np.int32)
        confidence = np.where(labels == 1, foreground, 1 - foreground)
    else:
        labels = np.argmax(values, axis=0).astype(np.int32)
        scores = np.exp(values.astype(np.float64) - values.max(axis=0))
        confidence = (scores / scores.sum(axis=0)).max(axis=0)
    roi = np.ones((h, w), bool)
    if options.get("roi"):
        roi[:] = False
        roi[5:45, 32:88] = True
    labels[~roi | (labels == 0)] = -1
    labels[confidence < options.get("minimum_confidence", 0)] = -1
    return labels, int(roi.sum())


def build():
    assets = HERE / "assets"
    asset(assets / "color_segmentation.onnx", model_bytes())
    image = np.full((53, 96, 3), 8, np.uint8)
    image[5:45, 4:28] = [20, 20, 240]
    image[8:40, 36:60] = [20, 230, 20]
    image[14:49, 68:92] = [235, 20, 20]
    image[18:25, 12:20] = 8  # A hole tests outlines and downstream blobs.
    image[1:4, 37:58] = 90  # Ambiguous gray patch tests confidence gating.
    png(assets / "colors.png", image)
    png(assets / "empty.png", np.full_like(image, 8))
    registry, truth = builtin_registry(), {}
    for case, options in CASES.items():
        graph = Graph()
        try:
            source = graph.add(
                registry.create(
                    "input.file",
                    block_id="source",
                    parameters={
                        "path": "assets/empty.png" if options.get("empty") else "assets/colors.png",
                        "path_base": "project",
                    },
                )
            )
            inference = graph.add(
                registry.create(
                    "ml.dnn",
                    block_id="infer",
                    parameters={
                        "model_path": "assets/color_segmentation.onnx",
                        "width": 24,
                        "height": 16,
                        "input_name": "image",
                        "output_names": [options.get("output", "logits")],
                        "engine": "classic",
                        "resize_mode": options.get("resize", "stretch"),
                    },
                )
            )
            binary = options.get("format") == "binary"
            decode = graph.add(
                registry.create(
                    "ml.decode_segmentation",
                    block_id="decode",
                    parameters={
                        "output_name": options.get("output", "logits"),
                        "format": options.get("format", "multiclass"),
                        "layout": options.get("layout", "NCHW"),
                        "score_activation": options.get("activation", "softmax"),
                        "resampling": options.get("resampling", "labels_nearest"),
                        "labels": ["background", "red part"] if binary else NAMES,
                        "class_count": 4 if options.get("format") == "class_ids" else 0,
                        "selected_class_ids": [1] if binary else [1, 2, 3],
                        "ignored_class_ids": [0],
                        "minimum_confidence": options.get("minimum_confidence", 0),
                        "maximum_fraction": options.get("maximum_fraction", 1),
                    },
                )
            )
            decode.exposed_outputs.append("result_graphics")
            overlay = graph.add(registry.create("cv.overlay_result_graphics", block_id="overlay"))
            filmstrip = graph.add(registry.create("output.filmstrip_image", block_id="filmstrip"))
            for a, o, b, i in (
                ("source", "image", "infer", "image"),
                ("source", "image", "decode", "image"),
                ("infer", "tensors", "decode", "tensors"),
                ("infer", "preprocessing", "decode", "preprocessing"),
                ("decode", "image", "overlay", "image"),
                ("decode", "result_graphics", "overlay", "graphics_1"),
                ("overlay", "image", "filmstrip", "image"),
            ):
                graph.connect(a, o, b, i)
            for index, block in enumerate((source, inference, decode, overlay, filmstrip)):
                block.position = [index * 410, 0]
            decode.node_size = "large"
            if options.get("roi"):
                region = graph.add(
                    registry.create(
                        "region.shape",
                        block_id="roi",
                        parameters={
                            "rectangle": [32, 5, 56, 40],
                        },
                    )
                )
                region.position = [410, 430]
                graph.connect("source", "image", "roi", "image")
                graph.connect("roi", "region", "decode", "region")
            if case == "multiclass":
                blobs = graph.add(registry.create("inspect.blobs", block_id="blobs"))
                blobs.position = [1220, 430]
                graph.connect("decode", "mask", "blobs", "image")
            note = graph.add(registry.create("layout.comment", block_id="attribution"))
            note.name, note.node_size, note.position = (
                "Original segmentation controls - MIT",
                "large",
                [0, -500],
            )
            note.description = (
                "Original MIT synthetic images, color arithmetic ONNX, graphs and independent "
                "pixel references by OpenCVDataflowGUI contributors. No learned recognition or "
                "customer data. Labels use -1 for ignored/unknown; class zero is ignored here. "
                "Nearest labels and linear scores are distinct boundary policies. Empty and area "
                "failure intentionally fail inspection. Current outputs remain; see README "
                "and truth.json."
            )
            graph.save(HERE / (case + ".json"))
            expected, roi_pixels = reference(
                np.full_like(image, 8) if options.get("empty") else image, options
            )
            filename = "expected_" + case + ".png"
            png(assets / filename, np.where(expected < 0, 255, expected).astype(np.uint8))
            truth[case] = {
                **options,
                "passed": options.get("passed", True),
                "expected_labels": filename,
                "selected_pixels": int(np.count_nonzero(expected >= 0)),
                "roi_pixels": roi_pixels,
                "class_counts": [
                    int(np.count_nonzero(expected == i)) for i in range(2 if binary else 4)
                ],
            }
        finally:
            graph.close()
    (HERE / "truth.json").write_bytes(
        (
            json.dumps(
                {
                    "license": "MIT - original color arithmetic and numerical references",
                    "cases": truth,
                    "asset_sha256": {
                        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(assets.iterdir())
                    },
                },
                indent=2,
            )
            + "\n"
        ).encode()
    )


if __name__ == "__main__":
    build()
