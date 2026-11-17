"""Original named image/image, image/gain, NHWC and four-branch ONNX controls."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
CASES = {
    "paired_images": {},
    "two_sources": {},
    "reference_input": {},
    "numeric_aux": {},
    "nhwc": {},
    "primary_second": {},
    "four_inputs": {},
    "wrong_shape": {"error": "shape"},
    "mismatched_tickets": {"error": "different tickets"},
}


def models():
    import onnx
    from onnx import TensorProto, helper

    result = {}
    for kind in ("pair", "nhwc", "gains", "four"):
        names = ["left", "right", "aux1", "aux2"] if kind == "four" else ["left", "right"]
        dimensions = [[1, 3, 8, 8] for _ in names]
        if kind in ("pair", "nhwc"):
            dimensions[1] = [1, 1, 8, 8]
        if kind == "gains":
            dimensions[1] = [1, 3, 1, 1]
        if kind == "nhwc":
            dimensions = [[1, 8, 8, 3], [1, 8, 8, 1]]
        nodes = []
        working = names.copy()
        if kind == "nhwc":
            working = [name + "_nchw" for name in names]
            nodes.extend(
                helper.make_node("Transpose", [name], [out], perm=[0, 3, 1, 2])
                for name, out in zip(names, working, strict=True)
            )
        if kind == "four":
            nodes.extend(
                (
                    helper.make_node("Add", working[:2], ["first"]),
                    helper.make_node("Add", working[2:], ["second"]),
                    helper.make_node("Add", ["first", "second"], ["combined"]),
                )
            )
        else:
            nodes.append(
                helper.make_node("Mul" if kind == "gains" else "Sub", working, ["combined"])
            )
        nodes.extend(
            (
                helper.make_node("GlobalAveragePool", ["combined"], ["pooled"]),
                helper.make_node("Flatten", ["pooled"], ["scores"], axis=1),
                helper.make_node("GlobalAveragePool", [working[0]], ["left_pool"]),
                helper.make_node("Flatten", ["left_pool"], ["left_mean"], axis=1),
            )
        )
        graph = helper.make_graph(
            nodes,
            "original_multi_input_" + kind,
            [
                helper.make_tensor_value_info(name, TensorProto.FLOAT, dims)
                for name, dims in zip(names, dimensions, strict=True)
            ],
            [
                helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 3])
                for name in ("scores", "left_mean")
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
        model.doc_string = "Original MIT numerical multi-input fixture; no learned parameters."
        onnx.checker.check_model(model)
        result[kind + ".onnx"] = model.SerializeToString()
    return result


def write_asset(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise FileExistsError("Existing different assets preserved; regenerate in a fresh folder")
    path.write_bytes(data)


def build():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    for name, data in models().items():
        write_asset(assets / name, data)
    for name, value in (("left", [20, 80, 200]), ("right", [200, 40, 20])):
        ok, encoded = cv2.imencode(".png", np.full((24, 40, 3), value, np.uint8))
        assert ok
        write_asset(assets / (name + ".png"), encoded.tobytes())
        if name == "right":
            write_asset(assets / "second_view/left.png", encoded.tobytes())
    registry, truth = builtin_registry(), {}
    for case, options in CASES.items():
        graph = Graph()
        try:
            source = graph.add(
                registry.create(
                    "input.file",
                    block_id="source",
                    parameters={"path": "assets/left.png", "path_base": "project"},
                )
            )
            left = graph.add(
                registry.create(
                    "ml.image_tensor",
                    block_id="left",
                    parameters={
                        "width": 8,
                        "height": 8,
                        "layout": "nhwc" if case == "nhwc" else "nchw",
                    },
                )
            )
            graph.connect("source", "image", "left", "image")
            model = (
                "four"
                if case == "four_inputs"
                else "gains"
                if case in ("numeric_aux", "wrong_shape")
                else "nhwc"
                if case == "nhwc"
                else "pair"
            )
            p = {
                "model_path": f"assets/{model}.onnx",
                "engine": "classic",
                "output_names": ["scores"],
                "input_1_shape": [1, 8, 8, 3] if case == "nhwc" else [1, 3, 8, 8],
                "input_2_shape": [1, 8, 8, 1] if case == "nhwc" else [1, 1, 8, 8],
            }
            if case in ("numeric_aux", "wrong_shape"):
                right = graph.add(
                    registry.create(
                        "ml.make_tensor",
                        block_id="right",
                        parameters={
                            "shape": [1, 3] if case == "wrong_shape" else [1, 3, 1, 1],
                            "values": [0.5, 2, 1],
                        },
                    )
                )
                p.update(input_2_role="reference", input_2_shape=[1, 3, 1, 1])
            else:
                right = graph.add(
                    registry.create(
                        "ml.image_tensor",
                        block_id="right",
                        parameters={
                            "width": 8,
                            "height": 8,
                            "color_format": "rgb" if case == "four_inputs" else "gray",
                            "mean": [0, 0, 0] if case == "four_inputs" else [0],
                            "std": [1, 1, 1] if case == "four_inputs" else [1],
                            "layout": "nhwc" if case == "nhwc" else "nchw",
                        },
                    )
                )
                second_source = source
                if case in ("two_sources", "reference_input", "mismatched_tickets"):
                    second_source = graph.add(
                        registry.create(
                            "input.file",
                            block_id="other",
                            parameters={
                                "path": "assets/second_view/left.png"
                                if case == "two_sources"
                                else "assets/right.png",
                                "path_base": "project",
                            },
                        )
                    )
                    second_source.position = [0, 360]
                    if case == "two_sources":
                        p["ticket_policy"] = "same_value"
                    if case == "reference_input":
                        p["input_2_role"] = "reference"
                graph.connect(second_source.id, "image", "right", "image")
            if case == "four_inputs":
                p.update(
                    active_inputs=4,
                    input_2_shape=[1, 3, 8, 8],
                    input_3_shape=[1, 3, 8, 8],
                    input_4_shape=[1, 3, 8, 8],
                    input_3_role="current",
                    input_4_role="current",
                )
            if case == "primary_second":
                p.update(
                    input_1_name="right",
                    input_2_name="left",
                    input_1_shape=[1, 1, 8, 8],
                    input_2_shape=[1, 3, 8, 8],
                    primary_input="input_2",
                )
            multi = graph.add(registry.create("ml.dnn_multi", block_id="multi", parameters=p))
            first, second = (right, left) if case == "primary_second" else (left, right)
            graph.connect(first.id, "tensor", "multi", "input_1")
            graph.connect(second.id, "tensor", "multi", "input_2")
            if case == "four_inputs":
                graph.connect("left", "tensor", "multi", "input_3")
                graph.connect("right", "tensor", "multi", "input_4")
            labels = ["red", "green", "blue"]
            classify = graph.add(
                registry.create(
                    "ml.classify",
                    block_id="classify",
                    parameters={
                        "output_name": "scores",
                        "labels": labels,
                        "accepted_labels": ["green" if case == "numeric_aux" else "red"],
                    },
                )
            )
            graph.connect("multi", "tensors", "classify", "tensors")
            for block, position in (
                (left, [420, 0]),
                (right, [420, 360]),
                (multi, [880, 0]),
                (classify, [1320, 0]),
            ):
                block.position = position
            multi.node_size = "large"
            note = graph.add(registry.create("layout.comment", block_id="attribution"))
            note.name, note.position, note.node_size = (
                "Original multi-input controls - MIT",
                [0, -510],
                "large",
            )
            note.description = (
                "Original MIT images, ONNX controls, analytic references and graphs by "
                "OpenCVDataflowGUI contributors. No learned models, customer data or third-party "
                "assets. Independent image preparation supplies named float32 tensors. The "
                "models subtract RGB/gray, multiply channel gains or sum four branches. "
                "Cached constants/references explicitly bypass live-ticket matching. Two-source "
                "pairing uses matching filename values, without proving exposure synchronization. "
                "Wrong-shape and mismatched-ticket projects intentionally contain an invalid "
                "inference activation and must emit no predictions. See truth.json and README."
            )
            graph.save(HERE / (case + ".json"))
            rgb = np.array([200, 80, 20], float) / 255
            gray = (
                np.array([0.299, 0.587, 0.114])
                @ (
                    np.array([20, 40, 200])
                    if case in ("two_sources", "reference_input", "mismatched_tickets")
                    else np.array([200, 80, 20])
                )
            ) / 255
            expected = (
                rgb * [0.5, 2, 1]
                if model == "gains"
                else rgb * 4
                if model == "four"
                else rgb - gray
            )
            truth[case] = {
                **options,
                "scores": expected.tolist(),
                "label": "green" if case == "numeric_aux" else "red",
            }
        finally:
            graph.close()
    manifest = {
        "license": "MIT - original numerical fixtures",
        "cases": truth,
        "asset_sha256": {
            p.relative_to(assets).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(assets.rglob("*"))
            if p.is_file()
        },
    }
    (HERE / "truth.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode())


if __name__ == "__main__":
    build()
