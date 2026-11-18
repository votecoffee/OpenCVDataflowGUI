"""Original dynamic CTC arithmetic, ordered source crops and native CRNN controls."""

import hashlib
import json
import runpy
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
_preparation = runpy.run_path(str(Path(__file__).with_name("prepare_model.py")))
PARAMETERS, NATIVE_QUADS = _preparation["PARAMETERS"], _preparation["NATIVE_QUADS"]
SOURCE = HERE.parent / "pinned_text_detection/assets/labels.png"
PATTERNS = [
    [255, 255, 128, 0, 255],
    [0, 0, 128, 255, 255],
    [255, 128, 255, 128, 255],
    [128, 128, 128, 128, 128],
    [0, 128, 0, 128, 0],
]
TEXTS = ["ABA", "BA", "AAA", "", "BBB"]
QUADS = [
    [[20, 10 + i * 32], [69, 10 + i * 32], [69, 24 + i * 32], [20, 24 + i * 32]] for i in range(5)
]
CASES = {
    "dynamic_batch": {},
    "padded_tail": {"tail_policy": "pad_repeat"},
    "blank_failure": {"blank": True, "passed": False},
    "minimum_reads": {"blank": True, "aggregate": "minimum_reads", "minimum_reads": 4},
    "confidence_gate": {"minimum_confidence": 0.99999, "passed": False},
    "selected_roi": {"roi": True},
    "empty_selection": {"empty_roi": True, "passed": False},
    "native_serial": {"native": True},
}


def asset(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw:
        raise FileExistsError("Different existing assets preserved; generate in a fresh directory")
    path.write_bytes(raw)


def model_bytes(layout="TNC"):
    from onnx import TensorProto, checker, helper, numpy_helper

    nodes = [
        helper.make_node("Conv", ["input", "weights", "bias"], ["logits"], kernel_shape=[1, 1]),
        helper.make_node("ReduceMean", ["logits"], ["columns"], axes=[2], keepdims=0),
        helper.make_node(
            "Transpose", ["columns"], ["output"], perm=[2, 0, 1] if layout == "TNC" else [0, 2, 1]
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "original_dynamic_ctc_control",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 1, 3, 5])],
        [
            helper.make_tensor_value_info(
                "output", TensorProto.FLOAT, [5, "batch", 3] if layout == "TNC" else ["batch", 5, 3]
            )
        ],
        [
            numpy_helper.from_array(
                np.array([0, 20, -20], np.float32).reshape(3, 1, 1, 1), "weights"
            ),
            numpy_helper.from_array(np.array([13, 0, 20], np.float32), "bias"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    model.doc_string = "Original MIT pixel arithmetic; not learned OCR or a recognition benchmark."
    checker.check_model(model)
    return model.SerializeToString()


def build():
    image = np.full((180, 300), 255, np.uint8)
    for index, values in enumerate(PATTERNS):
        image[10 + index * 32 : 25 + index * 32, 20:70] = np.repeat(np.array(values, np.uint8), 10)[
            None
        ]
    ok, raw = cv2.imencode(".png", image)
    assert ok
    asset(HERE / "assets/stripes.png", raw.tobytes())
    asset(HERE / "assets/native_labels.png", SOURCE.read_bytes())
    asset(HERE / "assets/dynamic_tnc.onnx", model_bytes())
    asset(HERE / "assets/dynamic_ntc.onnx", model_bytes("NTC"))
    registry = builtin_registry()
    for name, options in CASES.items():
        graph = Graph()
        try:
            graph.program["assets"] = ["MODEL-LICENSE.txt", "SOURCES.md"]

            def add(kind, identity, pos, **parameters):
                block = graph.add(registry.create(kind, block_id=identity, parameters=parameters))
                block.position = list(pos)
                return block

            native = options.get("native")
            add(
                "input.file",
                "source",
                (0, 0),
                path="assets/native_labels.png" if native else "assets/stripes.png",
                path_base="project",
            )
            indices = list(range(5)) if options.get("blank") else [0, 1, 2, 4]
            parameters = (
                {**PARAMETERS, "contract_path": ".local/crnn_en/contract.json"}
                if native
                else {
                    "model_path": "assets/dynamic_tnc.onnx",
                    "engine": "classic",
                    "batch_size": 3,
                    "width": 5,
                    "height": 3,
                    "input_shape": [-1, 1, 3, 5],
                    "mean": [0.0],
                    "scale": 1 / 255,
                    "output_names": ["output"],
                    "vocabulary": ["A", "B"],
                    "quadrilaterals": [QUADS[i] for i in indices],
                }
            )
            parameters.update(
                {
                    key: value
                    for key, value in options.items()
                    if key in ("tail_policy", "aggregate", "minimum_reads", "minimum_confidence")
                }
            )
            block = add("ml.recognize_text_batch", "ocr", (430, 0), **parameters)
            block.node_size = "large"
            block.exposed_outputs.append("result_graphics")
            add("cv.overlay_result_graphics", "overlay", (900, 0))
            add("output.filmstrip_image", "filmstrip", (1350, 0))
            for a, o, b, i in (
                ("source", "image", "ocr", "image"),
                ("ocr", "image", "overlay", "image"),
                ("ocr", "result_graphics", "overlay", "graphics_1"),
                ("overlay", "image", "filmstrip", "image"),
            ):
                graph.connect(a, o, b, i)
            if options.get("roi") or options.get("empty_roi"):
                add(
                    "region.shape",
                    "roi",
                    (430, 450),
                    rectangle=[290, 170, 5, 5] if options.get("empty_roi") else [10, 0, 70, 75],
                )
                graph.connect("source", "image", "roi", "image")
                graph.connect("roi", "region", "ocr", "region")
            note = add("layout.comment", "attribution", (0, -500))
            note.name = "OCR evidence / attribution"
            note.node_size = "large"
            note.description = (
                "Original MIT source stripes, dynamic CTC equations and glyph labels by "
                "OpenCVDataflowGUI contributors. Stripes verify native N=3 batching, "
                "not learned OCR. "
                "Native labels use the unchanged Apache-2.0 CRNN English export from OpenCV Zoo "
                "revision 48b10fa505aeb51fbd127fc60a5c660040e823b7, explicitly limited to N=1. "
                "Confidence is a greedy-path metric, not a calibrated posterior. "
                "See README.md/SOURCES.md; prepare_model.py is explicit and optional "
                "for dynamic cases."
            )
            graph.save(HERE / (name + ".json"))
        finally:
            graph.close()
    asset(
        HERE / "truth.json",
        (
            json.dumps(
                {
                    "cases": CASES,
                    "patterns": PATTERNS,
                    "texts": TEXTS,
                    "quads": QUADS,
                    "native_quads": NATIVE_QUADS,
                    "native_texts": ["lot2048", "pnax12", "pass"],
                    "asset_sha256": {
                        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted((HERE / "assets").iterdir())
                    },
                },
                indent=2,
            )
            + "\n"
        ).encode(),
    )


if __name__ == "__main__":
    build()
