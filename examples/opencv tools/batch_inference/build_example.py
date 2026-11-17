"""Original image-batch controls; no trained weights or third-party image data."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
CASES = {
    "dynamic_tail": {},
    "padded_fixed": {},
    "source_size": {},
    "mixed_sizes": {},
    "grayscale": {},
    "serial_results": {},
    "reject_tail": {"error": "complete multiple"},
    "wrong_signature": {"error": "declared Input shape"},
    "mixed_source_sizes": {"error": "common source size"},
    "wrong_output_axis": {"error": "batch axis"},
}
COLORS = [[20, 40, 200], [30, 180, 50], [220, 60, 10]]


def models():
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    result = {}
    for name in ("dynamic", "fixed", "collapsed", "gray"):
        channels = 1 if name == "gray" else 3
        nodes = [helper.make_node("GlobalAveragePool", ["image"], ["pool"])]
        initializers = []
        if name == "fixed":
            initializers = [numpy_helper.from_array(np.array([2, 3], np.int64), "fixed_shape")]
            nodes.append(helper.make_node("Reshape", ["pool", "fixed_shape"], ["scores"]))
        else:
            nodes.append(helper.make_node("Flatten", ["pool"], ["flat"], axis=1))
            nodes.append(
                helper.make_node("ReduceMean", ["flat"], ["scores"], axes=[0], keepdims=1)
                if name == "collapsed"
                else helper.make_node("Identity", ["flat"], ["scores"])
            )
        nodes.append(helper.make_node("Mul", ["scores", "scores"], ["squares"]))
        graph = helper.make_graph(
            nodes,
            "original_batch_" + name,
            [
                helper.make_tensor_value_info(
                    "image",
                    TensorProto.FLOAT,
                    [2, 3, 8, 8] if name == "fixed" else ["N", channels, "H", "W"],
                )
            ],
            [
                helper.make_tensor_value_info(
                    output,
                    TensorProto.FLOAT,
                    [2 if name == "fixed" else 1 if name == "collapsed" else "N", channels],
                )
                for output in ("scores", "squares")
            ],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
        model.doc_string = "Original MIT arithmetic batch control; no learned weights."
        onnx.checker.check_model(model)
        result[name + ".onnx"] = model.SerializeToString()
    return result


def write_asset(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw:
        raise FileExistsError("Different existing assets preserved; regenerate in a fresh folder")
    path.write_bytes(raw)


def build():
    assets = HERE / "assets"
    for name, raw in models().items():
        write_asset(assets / name, raw)
    for i, color in enumerate(COLORS):
        for mixed in (False, True):
            h, w = (24 + i * 4, 40 + i * 8) if mixed else (24, 40)
            ok, encoded = cv2.imencode(".png", np.full((h, w, 3), color, np.uint8))
            assert ok
            write_asset(assets / f"{'mixed_' if mixed else ''}{i}.png", encoded.tobytes())
    truth, registry = {}, builtin_registry()
    for case, options in CASES.items():
        graph = Graph()
        try:
            graph.add(registry.create("flow.batch", block_id="collect", parameters={"size": 3}))
            for i in range(3):
                mixed = case in ("mixed_sizes", "mixed_source_sizes")
                source = graph.add(
                    registry.create(
                        "input.file",
                        block_id=f"source_{i}",
                        parameters={
                            "path": f"assets/{'mixed_' if mixed else ''}{i}.png",
                            "path_base": "project",
                        },
                    )
                )
                source.position = [0, i * 270]
                graph.connect(source.id, "image", "collect", f"item_{i + 1}")
            parameters = {
                "model_path": "assets/"
                + (
                    "fixed.onnx"
                    if case == "padded_fixed"
                    else "collapsed.onnx"
                    if case == "wrong_output_axis"
                    else "gray.onnx"
                    if case == "grayscale"
                    else "dynamic.onnx"
                ),
                "input_shape": [2, 3, 8, 8]
                if case in ("padded_fixed", "wrong_signature")
                else [-1, 3, -1, -1],
                "width": 8,
                "height": 8,
                "batch_size": 2,
                "output_names": ["scores"],
                "tail_policy": "pad_repeat"
                if case == "padded_fixed"
                else "reject"
                if case == "reject_tail"
                else "variable",
                "spatial_mode": "source"
                if case in ("source_size", "mixed_source_sizes")
                else "configured",
                "engine": "classic",
            }
            if case == "grayscale":
                parameters.update(
                    color_format="gray", mean=[0], std=[1], input_shape=[-1, 1, -1, -1]
                )
            graph.add(registry.create("ml.dnn_batch", block_id="infer", parameters=parameters))
            graph.add(
                registry.create(
                    "flow.unbatch" if case == "serial_results" else "flow.select_entry",
                    block_id="select",
                    parameters={} if case == "serial_results" else {"index": 1},
                )
            )
            graph.add(registry.create("ml.unpack_inference", block_id="unpack"))
            graph.add(
                registry.create(
                    "ml.classify",
                    block_id="classify",
                    parameters={
                        "output_name": "scores",
                        "labels": ["gray"] if case == "grayscale" else ["red", "green", "blue"],
                        "accepted_labels": []
                        if case == "serial_results"
                        else ["gray" if case == "grayscale" else "green"],
                    },
                )
            )
            graph.connect("collect", "group", "infer", "images")
            graph.connect("infer", "results", "select", "group")
            graph.connect("select", "value", "unpack", "result")
            graph.connect("unpack", "tensors", "classify", "tensors")
            for i, name in enumerate(("collect", "infer", "select", "unpack", "classify")):
                graph.blocks[name].position = [400 + i * 360, 0]
            note = graph.add(registry.create("layout.comment", block_id="attribution"))
            note.name, note.node_size, note.position = (
                "Original batch controls - MIT",
                "large",
                [0, -500],
            )
            note.description = (
                "Original MIT images, ONNX arithmetic models, graphs and independent references "
                "by OpenCVDataflowGUI contributors. No customer/third-party assets or learned "
                "weights. Three captures have distinct tickets and grids. Results retain stored "
                "order, source images and per-image preprocessing. Default Select shows green "
                "row 1; Grayscale uses one channel and Serial emits all three rows. "
                "Variable tail submits 2+1 rows; fixed padding submits 2+2 but returns only 3. "
                "Reject-tail, wrong-signature, mixed-source-size and collapsed-output projects "
                "intentionally emit no predictions. See README and truth.json."
            )
            graph.save(HERE / (case + ".json"))
            truth[case] = {
                **options,
                "scores": [[(0.299 * c[2] + 0.587 * c[1] + 0.114 * c[0]) / 255] for c in COLORS]
                if case == "grayscale"
                else [[c[2] / 255, c[1] / 255, c[0] / 255] for c in COLORS],
                "tolerance": 2e-5 if case == "source_size" else 2e-6,
                "selected_label": "gray"
                if case == "grayscale"
                else "blue"
                if case == "serial_results"
                else "green",
                "padding_rows": 1 if case == "padded_fixed" else 0,
            }
        finally:
            graph.close()
    (HERE / "truth.json").write_bytes(
        (
            json.dumps(
                {
                    "license": "MIT - original numerical controls",
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
