"""Pinned original Q/DQ CNN controls and explicitly prepared MobileNet comparisons."""

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
    "signed_new",
    "unsigned_new",
    "signed_classic",
    "unsigned_classic",
    "signed_auto",
    "strict_tolerance",
    "top1_change",
    "saturation",
    "high_depth",
    "selected",
    "sealed",
    "raw_output",
    "unsupported_op",
)
REAL_CASES = tuple(
    f"mobilenet_{kind}_{engine}"
    for kind in ("qdq", "qoperator")
    for engine in ("classic", "new", "auto")
)
COEFFICIENTS = np.float32([0.83, 0.78, 0.91])
WEIGHT_SCALES = np.float32([0.02, 0.03, 0.025])
WEIGHT_ZEROS = np.int8([-2, 1, 3])
BIAS = np.float32([0.1, -0.03, 0.02])


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def model(kind):
    import onnx
    from onnx import TensorProto as T
    from onnx import helper as h
    from onnx import numpy_helper as n

    weights = np.diag(COEFFICIENTS).reshape(3, 3, 1, 1)
    initial = [n.from_array(BIAS, "bias")]
    nodes = []
    image = "image"
    w = "weights"
    if kind == "float":
        initial.append(n.from_array(weights, w))
    else:
        signed = kind != "unsigned"
        initial += [
            n.from_array(np.float32(0.05), "x_scale"),
            n.from_array(
                np.array(-3 if signed else 123, np.int8 if signed else np.uint8), "x_zero"
            ),
        ]
        nodes += [
            h.make_node("QuantizeLinear", ["image", "x_scale", "x_zero"], ["codes"]),
            h.make_node("DequantizeLinear", ["codes", "x_scale", "x_zero"], ["dequantized"]),
        ]
        image = "dequantized"
        q = (
            np.rint(weights / WEIGHT_SCALES[:, None, None, None])
            + WEIGHT_ZEROS[:, None, None, None]
        )
        initial += [
            n.from_array(q.astype(np.int8), "weights_q"),
            n.from_array(WEIGHT_SCALES, "weights_scale"),
            n.from_array(WEIGHT_ZEROS, "weights_zero"),
        ]
        nodes += [
            h.make_node(
                "DequantizeLinear", ["weights_q", "weights_scale", "weights_zero"], [w], axis=0
            )
        ]
    nodes += [
        h.make_node("Conv", [image, w, "bias"], ["convolved"], kernel_shape=[1, 1]),
        h.make_node("Relu", ["convolved"], ["features"]),
        h.make_node("GlobalAveragePool", ["features"], ["pool"]),
        h.make_node("Flatten", ["pool"], ["scores"], axis=1),
    ]
    output = "scores"
    dtype = T.FLOAT
    shape = ["N", 3]
    if kind == "raw":
        nodes = nodes[:1]
        output = "codes"
        dtype = T.INT8
        shape = ["N", 3, 16, 16]
    if kind == "unsupported":
        nodes += [
            h.make_node(
                "UnsupportedQuantizedInspection",
                ["scores"],
                ["unsupported"],
                domain="org.opencvdataflow.negative",
            )
        ]
        output = "unsupported"
    m = h.make_model(
        h.make_graph(
            nodes,
            "original_qdq_image_cnn",
            [h.make_tensor_value_info("image", T.FLOAT, ["N", 3, 16, 16])],
            [h.make_tensor_value_info(output, dtype, shape)],
            initial,
        ),
        opset_imports=[h.make_opsetid("", 13), h.make_opsetid("org.opencvdataflow.negative", 1)],
        ir_version=8,
    )
    onnx.checker.check_model(m)
    return m.SerializeToString()


def make_graph(case, settings, destination):
    registry = builtin_registry()
    graph = Graph()
    real = case.startswith("mobilenet_")
    try:

        def add(kind, key, x, y=0, **params):
            value = graph.add(registry.create(kind, block_id=key, parameters=params))
            value.position = [x, y]
            return value

        add(
            "input.file",
            "source",
            0,
            path="../../assets/coffee.png"
            if real
            else "assets/tie.png"
            if case == "top1_change"
            else "assets/parts.png",
            path_base="project",
            pixel_depth="uint16" if case == "high_depth" else "unchanged",
        )
        if real:
            kind, engine = settings["kind"], settings["engine"]
            a = add(
                "ml.dnn",
                "reference",
                380,
                contract_path=f".local/reference_{engine}/contract.json",
                **pretrained_parameters(engine),
            )
            b = add(
                "ml.dnn",
                "candidate",
                380,
                330,
                contract_path=f".local/{kind}_{engine}/contract.json",
                **pretrained_parameters(engine),
            )
        else:
            common = {
                "width": 16,
                "height": 16,
                "mean": [127.5] * 3,
                "scale": 1 / 127.5,
                "input_name": "image",
                "output_names": ["scores"],
                "engine": "classic"
                if "classic" in case
                else "auto"
                if case == "signed_auto"
                else "new",
            }
            if case == "high_depth":
                common.update(mean=[32767.5] * 3, scale=1 / 32767.5)
            if case == "saturation":
                common["scale"] = 0.5
            a = add("ml.dnn", "reference", 380, model_path="assets/float.onnx", **common)
            kind = (
                "unsigned"
                if case.startswith("unsigned")
                else "raw"
                if case == "raw_output"
                else "unsupported"
                if case == "unsupported_op"
                else "signed"
            )
            if kind == "raw":
                common["output_names"] = ["codes"]
            if kind == "unsupported":
                common["output_names"] = ["unsupported"]
            b = add("ml.dnn", "candidate", 380, 330, model_path=f"assets/{kind}.onnx", **common)
            if case == "sealed":
                for block in (a, b):
                    block.parameters["model_path"] = str(
                        (destination / block.parameters["model_path"]).resolve()
                    )
                    export_contract(block, destination / ("bundle_" + block.id))
                    block.parameters.update(
                        model_path="", contract_path=f"bundle_{block.id}/contract.json"
                    )
        compare = add(
            "ml.compare_image_tensors",
            "compare",
            820,
            reference_output_name="output" if real else "scores",
            candidate_output_name="output" if real else "scores",
            absolute_tolerance=2.5 if real else 0 if case == "strict_tolerance" else 0.04,
            relative_tolerance=0,
            top_k=5 if real else 3,
            minimum_topk_overlap=0.6 if real else 1.0,
        )
        compare.exposed_outputs.append("result_graphics")
        add("cv.overlay_result_graphics", "overlay", 1200)
        add("output.filmstrip_image", "filmstrip", 1580)
        for a, o, b, i in [
            ("source", "image", "reference", "image"),
            ("source", "image", "candidate", "image"),
            ("source", "image", "compare", "image"),
            ("reference", "tensors", "compare", "reference"),
            ("candidate", "tensors", "compare", "candidate"),
            ("compare", "image", "overlay", "image"),
            ("compare", "result_graphics", "overlay", "graphics_1"),
            ("overlay", "image", "filmstrip", "image"),
        ]:
            graph.connect(a, o, b, i)
        if case == "selected":
            add("region.shape", "selection", 0, 500, rectangle=[6, 4, 20, 16])
            graph.connect("source", "image", "selection", "image")
            for key in ("reference", "candidate"):
                graph.connect("selection", "region", key, "region")
        note = add("layout.comment", "attribution", 0, -470)
        note.name, note.node_size = "Quantized model comparison and sources", "large"
        note.description = (
            "OpenCVDataflowGUI comparison graph, MIT. "
            + (
                "Unmodified ONNX Model Zoo MobileNetV2-12 FP32/QDQ/QOperator weights, Apache 2.0, "
                "revision c5cae0f1942c8d7727372c7b1f6b485aa39e4421. "
                "Coffee photo by Rachel Michetti, "
                "courtesy Pikolo Espresso Bar, CC0 via scikit-image. "
                "Prepare model bundles explicitly; "
                "no runtime downloads. This is one-image fidelity, "
                "not ImageNet accuracy or a speed claim. "
                if real
                else "Original synthetic image and untrained 1x1 CNNs. "
                "Q/DQ uses nonzero signed/unsigned "
                "activation zero points and per-output-channel signed weight scales/zero points. "
                "Independent rounding/clipping arithmetic checks pixels and scores. "
                "No learned accuracy claim. "
            )
            + "See README.md/SOURCES.md."
        )
        graph.save(destination / (case + ".json"))
    finally:
        graph.close()


def pretrained_parameters(engine):
    return {
        "width": 224,
        "height": 224,
        "input_name": "input",
        "output_names": ["output"],
        "engine": engine,
        "mean": [0.485 * 255, 0.456 * 255, 0.406 * 255],
        "scale": 1 / 255,
        "std": [0.229, 0.224, 0.225],
        "receipt_mode": "detailed",
    }


def build(destination=HERE):
    destination = Path(destination)
    assets = destination / "assets"
    if assets.exists():
        raise FileExistsError("Existing quantized inference assets preserved")
    assets.mkdir(parents=True)
    (destination / ".gitattributes").write_bytes(b"*.json -text\n.gitattributes -text\n")
    for kind in ("float", "signed", "unsigned", "raw", "unsupported"):
        (assets / (kind + ".onnx")).write_bytes(model(kind))
    y, x = np.mgrid[:24, :32]
    image = np.dstack(((x * 7 + y * 3) % 256, (x * 2 + y * 9) % 256, (x * 5 + y * 5) % 256)).astype(
        np.uint8
    )
    (assets / "tie.png").write_bytes(cv2.imencode(".png", image)[1].tobytes())
    image[..., 2] = np.minimum(image[..., 2].astype(np.int16) + 20, 255).astype(np.uint8)
    (assets / "parts.png").write_bytes(cv2.imencode(".png", image)[1].tobytes())
    for case in CASES:
        make_graph(case, {}, destination)
    for case in REAL_CASES:
        _, kind, engine = case.split("_")
        make_graph(case, {"kind": kind, "engine": engine}, destination)
    files = [
        *assets.iterdir(),
        *(destination / "bundle_reference").iterdir(),
        *(destination / "bundle_candidate").iterdir(),
    ]
    (destination / "truth.json").write_bytes(
        encoded(
            {
                "cases": list(CASES),
                "real_cases": list(REAL_CASES),
                "coefficient": COEFFICIENTS.tolist(),
                "bias": BIAS.tolist(),
                "weight_scale": WEIGHT_SCALES.tolist(),
                "weight_zero_point": WEIGHT_ZEROS.tolist(),
                "activation_scale": 0.05,
                "signed_zero_point": -3,
                "unsigned_zero_point": 123,
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
