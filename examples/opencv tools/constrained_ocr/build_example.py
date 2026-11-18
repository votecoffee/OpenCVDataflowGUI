"""Original image-dependent CTC probability controls and independent alignment sums."""

import hashlib
import itertools
import json
import runpy
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "ocr_batch"
PARAMETERS = runpy.run_path(str(Path(__file__).with_name("prepare_model.py")))["PARAMETERS"]
CASES = {
    "beam_recovers": {},
    "greedy_blank": {"decoder": "CTC-greedy", "passed": False},
    "repeat_blank": {"image": "repeat", "lexicon": ["AA"]},
    "blank_dominant": {"image": "faint", "passed": False},
    "unreachable_word": {"lexicon": ["AAAA"], "passed": False},
    "confidence_limit": {"minimum_confidence": 0.9, "passed": False},
    "lexicon_file": {"file": True},
    "alphabet_check": {
        "arithmetic": True,
        "decoder": "CTC-greedy",
        "allowed_characters": "A",
        "passed": False,
    },
    "lexicon_check": {
        "arithmetic": True,
        "decoder": "CTC-greedy",
        "lexicon": ["BA"],
        "passed": False,
    },
    "ratio_guard": {
        "arithmetic": True,
        "allowed_characters": "A",
        "minimum_greedy_ratio": 0.1,
        "passed": False,
    },
    "native_lexicon": {"native": True},
}


def asset(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw:
        raise FileExistsError(
            "Different existing assets preserved; generate into a fresh directory"
        )
    path.write_bytes(raw)


def model_bytes():
    from onnx import TensorProto, checker, helper, numpy_helper

    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["input", "weights", "bias"], ["planes"], kernel_shape=[1, 1]),
            helper.make_node("ReduceMean", ["planes"], ["columns"], axes=[2], keepdims=0),
            helper.make_node("Transpose", ["columns"], ["output"], perm=[2, 0, 1]),
        ],
        "original_ctc_probability_control",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 1, 3, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, "batch", 2])],
        [
            numpy_helper.from_array(np.array([-1, 1], np.float32).reshape(2, 1, 1, 1), "weights"),
            numpy_helper.from_array(np.array([1, 0], np.float32), "bias"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    model.doc_string = "Original MIT pixel arithmetic, not learned OCR or an accuracy benchmark."
    checker.check_model(model)
    return model.SerializeToString()


def reference(intensities):
    probabilities = [[1 - value / 255, value / 255] for value in intensities]
    result = {}
    for path in itertools.product((0, 1), repeat=3):
        merged = [v for i, v in enumerate(path) if i == 0 or v != path[i - 1]]
        text = "".join("A" for v in merged if v)
        mass = float(np.prod([probabilities[t][value] for t, value in enumerate(path)]))
        result[text] = result.get(text, 0) + mass
    return result


def build():
    for name, values in (
        ("ambiguous", [102, 102, 102]),
        ("repeat", [255, 0, 255]),
        ("faint", [13, 13, 13]),
    ):
        image = np.repeat(np.repeat(np.array(values, np.uint8)[None], 30, axis=0), 10, axis=1)
        ok, raw = cv2.imencode(".png", image)
        assert ok
        asset(HERE / "assets" / (name + ".png"), raw.tobytes())
    for name in ("stripes.png", "native_labels.png", "dynamic_tnc.onnx"):
        asset(HERE / "assets" / name, (SOURCE / "assets" / name).read_bytes())
    asset(HERE / "assets/probability_tnc.onnx", model_bytes())
    asset(HERE / "assets/words.txt", b"A\nAA\n")
    registry = builtin_registry()
    for name, options in CASES.items():
        graph = Graph()
        try:
            graph.program["assets"] = ["MODEL-LICENSE.txt", "SOURCES.md"]

            def add(kind, identity, pos, **parameters):
                b = graph.add(registry.create(kind, block_id=identity, parameters=parameters))
                b.position = list(pos)
                return b

            arithmetic = options.get("arithmetic")
            native = options.get("native")
            image = (
                "native_labels"
                if native
                else "stripes"
                if arithmetic
                else options.get("image", "ambiguous")
            )
            add("input.file", "source", (0, 0), path=f"assets/{image}.png", path_base="project")
            parameters = (
                {**PARAMETERS, "contract_path": ".local/crnn_en/contract.json"}
                if native
                else {
                    "model_path": "assets/dynamic_tnc.onnx"
                    if arithmetic
                    else "assets/probability_tnc.onnx",
                    "engine": "classic",
                    "input_shape": [-1, 1, 3, 5 if arithmetic else 3],
                    "width": 5 if arithmetic else 3,
                    "height": 3,
                    "mean": [0.0],
                    "std": [1.0],
                    "scale": 1 / 255,
                    "output_names": ["output"],
                    "vocabulary": ["A", "B"] if arithmetic else ["A"],
                    "score_activation": "softmax" if arithmetic else "probability",
                    "lexicon": [] if arithmetic else ["A"],
                    "quadrilaterals": [[[20, 10], [69, 10], [69, 24], [20, 24]]]
                    if arithmetic
                    else [],
                }
            )
            parameters.update(
                {
                    k: v
                    for k, v in options.items()
                    if k
                    in (
                        "decoder",
                        "lexicon",
                        "allowed_characters",
                        "minimum_greedy_ratio",
                        "minimum_confidence",
                    )
                }
            )
            if options.get("file"):
                parameters.update(
                    lexicon=[],
                    lexicon_path="assets/words.txt",
                    lexicon_sha256=hashlib.sha256(b"A\nAA\n").hexdigest(),
                )
            block = add("ml.recognize_constrained_text", "ocr", (430, 0), **parameters)
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
            note = add("layout.comment", "attribution", (0, -500))
            note.name = "CTC controls / model attribution"
            note.node_size = "large"
            note.description = (
                "Original MIT probability/stripe controls and glyph labels by OpenCVDataflowGUI "
                "contributors. Tiny models are image-dependent arithmetic, not learned OCR. "
                "The optional unchanged Apache CRNN comes from OpenCV Zoo revision "
                "48b10fa505aeb51fbd127fc60a5c660040e823b7 and uses fixed N=1. "
                "Exact label-sequence mass is not calibrated correctness. Dictionaries retain "
                "original model mass; blank dominance and relative likelihood can reject "
                "forced words. "
                "See README.md/SOURCES.md. Native model preparation is explicit."
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
                    "alignment_mass": {
                        "ambiguous": reference([102, 102, 102]),
                        "repeat": reference([255, 0, 255]),
                        "faint": reference([13, 13, 13]),
                    },
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
