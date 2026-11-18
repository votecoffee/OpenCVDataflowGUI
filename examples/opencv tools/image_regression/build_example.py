"""Original continuous bead-width proxy, native EPS-SVR export and error controls."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.model_contracts import export_contract

HERE = Path(__file__).resolve().parent
TRAIN_WIDTHS = [4, 8, 12, 16, 20, 24, 28]
TEST_WIDTHS = [6.5, 13.25, 21.75]
CASES = (
    "width_small",
    "width_middle",
    "width_large",
    "high_depth",
    "brightness",
    "nchw_head",
    "no_reference",
    "low_contrast",
    "range_fail",
    "wrong_layout",
    "bad_model",
    "evaluation",
    "evaluation_lighting",
    "evaluation_mixed",
    "batch_rows",
)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def bead(width, seed, contrast=200):
    y, x = np.indices((32, 64), dtype=float)
    center = 31.5 + ((seed % 5) - 2) * 0.37
    left, right = center - width / 2, center + width / 2
    coverage = np.maximum(0, np.minimum(x + 0.5, right) - np.maximum(x - 0.5, left))
    noise = np.random.default_rng(seed).normal(0, 0.45, x.shape)
    return np.uint8(np.clip(np.rint(20 + contrast * coverage + noise), 0, 255))


def models(assets):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    images = [bead(width, index) for index, width in enumerate(TRAIN_WIDTHS)]
    X = np.array([[image.astype(np.float32).mean() / 255] for image in images], np.float32)
    targets = np.float32(TRAIN_WIDTHS) * 0.1
    native = cv2.ml.SVM_create()
    native.setType(cv2.ml.SVM_EPS_SVR)
    native.setKernel(cv2.ml.SVM_LINEAR)
    native.setC(1000)
    native.setP(1e-5)
    native.setTermCriteria((cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10000, 1e-8))
    assert native.train(X, cv2.ml.ROW_SAMPLE, targets)
    rho, alpha, indices = native.getDecisionFunction(0)
    weight = float((alpha.reshape(-1) @ native.getSupportVectors()[indices.reshape(-1)])[0])
    bias = -float(rho)
    fs = cv2.FileStorage(
        "", cv2.FILE_STORAGE_WRITE | cv2.FILE_STORAGE_MEMORY | cv2.FILE_STORAGE_FORMAT_YAML
    )
    native.write(fs, "svm")
    (assets / "native-eps-svr.yml").write_text(
        fs.releaseAndGetString(), encoding="utf-8", newline="\n"
    )
    for index, (width, image) in enumerate(zip(TRAIN_WIDTHS, images, strict=True)):
        (assets / f"train-{index}.png").write_bytes(cv2.imencode(".png", image)[1].tobytes())
    for kind in ("vector", "nchw", "changed", "batch"):
        nodes = [
            helper.make_node("GlobalAveragePool", ["image"], ["pool"]),
            helper.make_node("Flatten", ["pool"], ["flat"], axis=1),
            helper.make_node("Gemm", ["flat", "weights", "bias"], ["vector"]),
        ]
        output = "measurements"
        shape = ["N" if kind == "batch" else 1, 2]
        initializers = [
            numpy_helper.from_array(
                np.float32([[weight * (1.05 if kind == "changed" else 1), 1.0]]), "weights"
            ),
            numpy_helper.from_array(np.float32([bias, 0]), "bias"),
        ]
        if kind == "nchw":
            initializers.append(numpy_helper.from_array(np.int64([1, 2, 1, 1]), "shape"))
            nodes.append(helper.make_node("Reshape", ["vector", "shape"], [output]))
            shape = [1, 2, 1, 1]
        else:
            nodes.append(helper.make_node("Identity", ["vector"], [output]))
        graph = helper.make_graph(
            nodes,
            "original_bead_width_regression",
            [
                helper.make_tensor_value_info(
                    "image", TensorProto.FLOAT, ["N" if kind == "batch" else 1, 1, 32, 64]
                )
            ],
            [helper.make_tensor_value_info(output, TensorProto.FLOAT, shape)],
            initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8
        onnx.checker.check_model(model)
        (assets / (kind + ".onnx")).write_bytes(model.SerializeToString())
    return {
        "weight": weight,
        "bias": bias,
        "training_widths_pixels": TRAIN_WIDTHS,
        "targets_mm": targets.astype(float).tolist(),
        "pixel_pitch_mm": 0.1,
        "kernel": "linear",
        "svm_type": "EPS_SVR",
        "c": 1000,
        "p": 1e-5,
    }


def inference_parameters():
    return {
        "width": 64,
        "height": 32,
        "color_format": "gray",
        "input_name": "image",
        "output_names": ["measurements"],
        "mean": [0.0],
        "std": [1.0],
        "engine": "classic",
        "receipt_mode": "detailed",
    }


def build(destination=HERE):
    destination = Path(destination)
    assets = destination / "assets"
    if assets.exists():
        raise FileExistsError("Existing image-regression assets preserved")
    assets.mkdir(parents=True)
    training = models(assets)
    images = {f"test-{i}.png": bead(width, 100 + i) for i, width in enumerate(TEST_WIDTHS)}
    images["dim.png"] = bead(TEST_WIDTHS[1], 101, contrast=130)
    images["test16.png"] = images["test-1.png"].astype(np.uint16) * 257
    images["batch-a.png"] = bead(TEST_WIDTHS[1], 110)
    images["batch-b.png"] = bead(TEST_WIDTHS[1], 111)
    images["batch-c.png"] = bead(TEST_WIDTHS[1], 112)
    for name, image in images.items():
        (assets / name).write_bytes(cv2.imencode(".png", image)[1].tobytes())
    registry = builtin_registry()
    for kind in ("vector", "nchw", "changed", "batch"):
        params = {
            **inference_parameters(),
            "model_path": str((assets / (kind + ".onnx")).resolve()),
        }
        if kind == "batch":
            params.update(input_shape=[-1, 1, 32, 64], batch_size=2)
        block = registry.create("ml.dnn_batch" if kind == "batch" else "ml.dnn", parameters=params)
        try:
            export_contract(block, destination / "bundles" / kind)
        finally:
            block.close()
    bad = destination / "bundles/bad"
    bad.mkdir()
    for name in ("contract.json", "model.onnx"):
        (bad / name).write_bytes((destination / "bundles/vector" / name).read_bytes())
    raw = (bad / "model.onnx").read_bytes()
    (bad / "model.onnx").write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    cases = {}
    for case in CASES:
        graph = Graph()

        def add(kind, name, x, y, **params):
            block = graph.add(registry.create(kind, block_id=name, parameters=params))
            block.position = [x, y]
            return block

        def pipeline(suffix, image_name, reference, model="vector", **settings):
            source = "source" + suffix
            convert = "convert" + suffix
            net = "net" + suffix
            decode = "decode" + suffix
            y = 500 * len([k for k in graph.blocks if k.startswith("source")])
            add(
                "input.file",
                source,
                0,
                y,
                source_mode="files",
                paths=["assets/" + image_name],
                path_base="project",
                color_format="unchanged",
                pixel_depth="unchanged",
            )
            add("cv.convert", convert, 350, y, color_format="gray", pixel_depth="uint8")
            inference = add(
                "ml.dnn",
                net,
                700,
                y,
                **{
                    **inference_parameters(),
                    "contract_path": "bundles/" + model + "/contract.json",
                },
            )
            inference.exposed_outputs.append("receipt")
            decoder = add(
                "ml.decode_regression",
                decode,
                1050,
                y,
                output_name="measurements",
                measurement_name="bead_width",
                units="mm",
                reference_enabled=True,
                reference_value=reference,
                check_absolute_error=True,
                maximum_absolute_error=0.02,
                **settings,
            )
            decoder.exposed_outputs.append("result_graphics")
            for a, o, b, i in [
                (source, "image", convert, "image"),
                (source, "image", decode, "image"),
                (convert, "image", net, "image"),
                (net, "tensors", decode, "tensors"),
                (net, "preprocessing", decode, "preprocessing"),
                (net, "receipt", decode, "receipt"),
            ]:
                graph.connect(a, o, b, i)
            return source, net, decode

        try:
            if case.startswith("evaluation"):
                for index, width in enumerate(TEST_WIDTHS):
                    name = (
                        "dim.png"
                        if case == "evaluation_lighting" and index == 1
                        else f"test-{index}.png"
                    )
                    _, _, decode = pipeline(
                        str(index),
                        name,
                        width * 0.1,
                        model="changed" if case == "evaluation_mixed" and index == 1 else "vector",
                    )
                    if index == 0:
                        add("flow.batch", "batch", 1400, 700, size=3)
                    graph.connect(decode, "measurement", "batch", f"item_{index + 1}")
                add(
                    "ml.regression_metrics",
                    "metrics",
                    1750,
                    700,
                    limit_metric="mae",
                    maximum_error=0.02,
                )
                graph.connect("batch", "group", "metrics", "measurements")
                target = "metrics"
            elif case == "batch_rows":
                add(
                    "input.file",
                    "source",
                    0,
                    0,
                    source_mode="files",
                    paths=["assets/batch-a.png", "assets/batch-b.png", "assets/batch-c.png"],
                    path_base="project",
                    color_format="unchanged",
                    pixel_depth="unchanged",
                )
                add("flow.batch", "images", 350, 0, size=3)
                graph.connect("source", "image", "images", "item_1")
                net = add(
                    "ml.dnn_batch",
                    "net",
                    700,
                    0,
                    **{
                        **inference_parameters(),
                        "contract_path": "bundles/batch/contract.json",
                        "input_shape": [-1, 1, 32, 64],
                        "batch_size": 2,
                    },
                )
                net.exposed_outputs.append("receipt")
                add("flow.unbatch", "unbatch", 1050, 0)
                add("ml.unpack_inference", "unpack", 1400, 0)
                graph.blocks["unpack"].exposed_outputs.append("receipt")
                add(
                    "ml.decode_regression",
                    "decode",
                    1750,
                    0,
                    output_name="measurements",
                    measurement_name="bead_width",
                    units="mm",
                    reference_enabled=True,
                    reference_value=1.325,
                    check_absolute_error=True,
                    maximum_absolute_error=0.02,
                )
                add("flow.batch", "measurements", 2100, 0, size=3)
                add(
                    "ml.regression_metrics",
                    "metrics",
                    2450,
                    0,
                    limit_metric="mae",
                    maximum_error=0.02,
                )
                for a, o, b, i in [
                    ("images", "group", "net", "images"),
                    ("net", "results", "unbatch", "group"),
                    ("unbatch", "value", "unpack", "result"),
                    ("unpack", "image", "decode", "image"),
                    ("unpack", "tensors", "decode", "tensors"),
                    ("unpack", "preprocessing", "decode", "preprocessing"),
                    ("unpack", "receipt", "decode", "receipt"),
                    ("decode", "measurement", "measurements", "item_1"),
                    ("measurements", "group", "metrics", "measurements"),
                ]:
                    graph.connect(a, o, b, i)
                target = "metrics"
            else:
                index = 0 if case == "width_small" else 2 if case == "width_large" else 1
                name = (
                    "test16.png"
                    if case == "high_depth"
                    else "dim.png"
                    if case == "low_contrast"
                    else f"test-{index}.png"
                )
                model = (
                    "bad"
                    if case == "bad_model"
                    else "nchw"
                    if case in ("nchw_head", "wrong_layout")
                    else "vector"
                )
                source, net, decode = pipeline("", name, TEST_WIDTHS[index] * 0.1, model=model)
                block = graph.blocks[decode]
                if case == "brightness":
                    block.parameters.update(
                        output_index=1,
                        measurement_name="average_level",
                        units="gray",
                        value_scale=255.0,
                        reference_enabled=False,
                        check_absolute_error=False,
                    )
                if case == "nchw_head":
                    block.parameters["layout"] = "nchw_scalar"
                if case == "no_reference":
                    block.parameters.update(reference_enabled=False, check_absolute_error=False)
                if case == "range_fail":
                    block.parameters.update(check_range=True, minimum_value=1.4, maximum_value=2.0)
                add("cv.overlay_result_graphics", "overlay", 1400, 0)
                add("output.filmstrip_image", "filmstrip", 1750, 0)
                graph.connect(decode, "image", "overlay", "image")
                graph.connect(decode, "result_graphics", "overlay", "graphics_1")
                graph.connect("overlay", "image", "filmstrip", "image")
                target = net if case == "bad_model" else decode
            note = add("layout.comment", "attribution", 0, -850)
            note.name, note.node_size = (
                "Original continuous-image regression controls - MIT",
                "large",
            )
            note.description = (
                "Original MIT synthetic bead images, native EPS-SVR "
                "fit, ONNX exports and projects by OpenCVDataflowGUI "
                "contributors. Seven training widths and separate "
                "held-out widths/noise seeds. "
                "Width labels use an assumed 0.1 mm/pixel; this is not "
                "camera calibration or production bead measurement. "
                "Global brightness estimates coverage only under the "
                "trained lighting/background conditions. "
                "The dim control deliberately violates that condition. "
                "Continuous measurements are not confidence scores. See "
                "README.md, SOURCES.md, truth.json and benchmark.json."
            )
            graph.save(destination / (case + ".json"))
            cases[case] = {"target": target}
        finally:
            graph.close()
    files = list(assets.iterdir()) + [
        p for p in (destination / "bundles").rglob("*") if p.is_file()
    ]
    (destination / "truth.json").write_bytes(
        encoded(
            {
                "cases": cases,
                "training": training,
                "test_widths_pixels": TEST_WIDTHS,
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
