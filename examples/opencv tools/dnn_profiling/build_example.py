"""Original pinned image CNNs with fusion, disabled observation and native-call controls."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.model_contracts import export_contract

HERE = Path(__file__).resolve().parent
CASES = (
    "classic_fused",
    "classic_unfused",
    "new_cpu",
    "auto_cpu",
    "disabled",
    "receipt_off",
    "plot_off",
    "optional_budget",
    "required_budget",
    "multi_input",
    "batch_tail",
    "sealed",
    "high_depth",
)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def model(multi=False):
    import onnx
    from onnx import TensorProto as T
    from onnx import helper as h
    from onnx import numpy_helper as n

    inputs = [h.make_tensor_value_info("left" if multi else "image", T.FLOAT, ["N", 3, 16, 16])]
    nodes = []
    if multi:
        inputs.append(h.make_tensor_value_info("right", T.FLOAT, ["N", 3, 16, 16]))
        nodes.append(h.make_node("Add", ["left", "right"], ["image"], name="paired_views"))
    nodes.extend(
        [
            h.make_node(
                "Conv",
                ["image", "weights", "bias"],
                ["convolved"],
                name="color_conv",
                kernel_shape=[1, 1],
            ),
            h.make_node("Relu", ["convolved"], ["activated"], name="activation"),
            h.make_node("GlobalAveragePool", ["activated"], ["pooled"], name="spatial_mean"),
            h.make_node("Flatten", ["pooled"], ["scores"], name="output_rows", axis=1),
        ]
    )
    graph = h.make_graph(
        nodes,
        "original_profile_control",
        inputs,
        [h.make_tensor_value_info("scores", T.FLOAT, ["N", 3])],
        [
            n.from_array(np.eye(3, dtype=np.float32).reshape(3, 3, 1, 1), "weights"),
            n.from_array(np.float32([0.15, -0.1, 0.05]), "bias"),
        ],
    )
    result = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)], ir_version=8)
    onnx.checker.check_model(result)
    return result.SerializeToString()


def preparation():
    return {
        "width": 16,
        "height": 16,
        "color_format": "rgb",
        "mean": [127.5] * 3,
        "std": [1.0] * 3,
        "scale": 1 / 127.5,
    }


def build(destination=HERE):
    destination = Path(destination)
    assets = destination / "assets"
    if assets.exists():
        raise FileExistsError("Existing profiling example assets preserved")
    assets.mkdir(parents=True)
    (destination / ".gitattributes").write_bytes(b"*.json -text\n.gitattributes -text\n")
    for name, raw in (("single.onnx", model()), ("multi.onnx", model(True))):
        (assets / name).write_bytes(raw)
    for index in range(3):
        y, x = np.mgrid[:24, :32]
        image = np.dstack(
            (
                (x * 7 + index * 19) % 256,
                (y * 9 + index * 41) % 256,
                ((x + y) * 5 + index * 67) % 256,
            )
        ).astype(np.uint8)
        (assets / f"{index}.png").write_bytes(cv2.imencode(".png", image)[1].tobytes())
    registry = builtin_registry()
    for case in CASES:
        graph = Graph()
        try:

            def add(kind, key, x, y=0, **params):
                value = graph.add(registry.create(kind, block_id=key, parameters=params))
                value.position = [x, y]
                return value

            for i in range(3 if case == "batch_tail" else 1):
                add(
                    "input.file",
                    "source" if i == 0 else f"source_{i}",
                    0,
                    i * 250,
                    path=f"assets/{i}.png",
                    source_mode="files" if case == "batch_tail" else "single",
                    paths=[f"assets/{i}.png"] if case == "batch_tail" else [],
                    path_base="project",
                    pixel_depth="uint16" if case == "high_depth" else "unchanged",
                )
            common = {
                "model_path": "assets/single.onnx",
                "input_name": "image",
                "output_names": ["scores"],
                **preparation(),
                "engine": "classic" if case.startswith("classic_") else "new",
            }
            if case == "auto_cpu":
                common["engine"] = "auto"
            if case == "classic_unfused":
                common["fusion"] = False
            if case == "disabled":
                common["profile_mode"] = "off"
            if case == "receipt_off":
                common["receipt_mode"] = "off"
            if case == "plot_off":
                common["profile_plot_layers"] = 0
            if case in ("optional_budget", "required_budget"):
                common.update(
                    maximum_profile_layers=1,
                    profile_mode="required" if case == "required_budget" else "optional",
                )
            if case == "high_depth":
                common.update(mean=[32767.5] * 3, scale=1 / 32767.5)
            if case == "multi_input":
                for side, y in (("left", 0), ("right", 300)):
                    add("ml.image_tensor", side, 370, y, **preparation())
                    graph.connect("source", "image", side, "image")
                block = add(
                    "ml.dnn_multi_profile",
                    "profile",
                    740,
                    model_path="assets/multi.onnx",
                    engine="new",
                    output_names=["scores"],
                    input_1_shape=[1, 3, 16, 16],
                    input_2_shape=[1, 3, 16, 16],
                )
                graph.connect("left", "tensor", "profile", "input_1")
                graph.connect("right", "tensor", "profile", "input_2")
            elif case == "batch_tail":
                add("flow.batch", "group", 370, size=3)
                for i in range(3):
                    graph.connect(
                        "source" if i == 0 else f"source_{i}", "image", "group", f"item_{i + 1}"
                    )
                block = add(
                    "ml.dnn_batch_profile",
                    "profile",
                    740,
                    **common,
                    batch_size=2,
                    input_shape=[-1, 3, 16, 16],
                )
                graph.connect("group", "group", "profile", "images")
            else:
                block = add("ml.dnn_profile", "profile", 400, **common)
                graph.connect("source", "image", "profile", "image")
            if case == "sealed":
                block.parameters["model_path"] = str((assets / "single.onnx").resolve())
                export_contract(block, destination / "bundle")
                block.parameters.update(model_path="", contract_path="bundle/contract.json")
            note = add("layout.comment", "attribution", 0, -460)
            note.name, note.node_size = "Original DNN profiling controls - MIT", "large"
            note.description = (
                "Original OpenCVDataflowGUI images, untrained 1x1 CNNs and projects, MIT. "
                "Known channel transforms and pooling test outputs while native CPU layer timings "
                "are collected from the same forward. Timings vary by machine and warmup. "
                "Zero ticks may mean fusion, skipping or clock resolution; unavailable timings "
                "are explicit. Disabled collection does not call native timing APIs. "
                "See README.md/SOURCES.md. No learned accuracy or accelerator claim."
            )
            graph.save(destination / (case + ".json"))
        finally:
            graph.close()
    files = [*assets.iterdir(), *(destination / "bundle").iterdir()]
    (destination / "truth.json").write_bytes(
        encoded(
            {
                "cases": list(CASES),
                "asset_sha256": {
                    p.relative_to(destination).as_posix(): hashlib.sha256(
                        p.read_bytes()
                    ).hexdigest()
                    for p in sorted(files)
                },
            }
        )
    )


if __name__ == "__main__":
    build()
