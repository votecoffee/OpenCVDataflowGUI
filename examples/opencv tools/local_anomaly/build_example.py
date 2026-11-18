"""Original patch-feature ONNX, known-good references and localized defect controls."""

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
    "normal",
    "defect",
    "missing_texture",
    "weak_defect",
    "bright_shift",
    "selected_region",
    "high_depth",
    "allowed_area",
    "wrong_features",
    "corrupt_reference",
    "train_then_inspect",
    "sampled_reference",
)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def surface(seed):
    y, x = np.indices((32, 32))
    noise = np.random.default_rng(seed).integers(-2, 3, (32, 32))
    return np.uint8(70 + 80 * (x % 2) + 3 * ((y // 2) % 3) + noise)


def params():
    return {
        "width": 32,
        "height": 32,
        "color_format": "gray",
        "input_name": "image",
        "output_names": ["local"],
        "mean": [0.0],
        "std": [1.0],
        "engine": "classic",
        "receipt_mode": "detailed",
    }


def add(graph, registry, kind, name, x, y=0, **parameters):
    block = graph.add(registry.create(kind, block_id=name, parameters=parameters))
    block.position = [x, y]
    return block


def pipeline(graph, registry, name, image, *, normalization="none", y=0):
    source = add(
        graph,
        registry,
        "input.file",
        name + "source",
        0,
        y,
        source_mode="files",
        paths=["assets/" + image],
        path_base="project",
        color_format="unchanged",
        pixel_depth="unchanged",
    )
    convert = add(
        graph,
        registry,
        "cv.convert",
        name + "convert",
        350,
        y,
        color_format="gray",
        pixel_depth="uint8",
    )
    net = add(
        graph,
        registry,
        "ml.dnn",
        name + "net",
        700,
        y,
        **{**params(), "contract_path": "bundle/contract.json"},
    )
    net.exposed_outputs.append("receipt")
    features = add(
        graph,
        registry,
        "ml.local_features",
        name + "features",
        1050,
        y,
        output_name="local",
        normalization=normalization,
    )
    for a, out, b, port in [
        (source, "image", convert, "image"),
        (source, "image", features, "image"),
        (convert, "image", net, "image"),
        (net, "tensors", features, "tensors"),
        (net, "preprocessing", features, "preprocessing"),
        (net, "receipt", features, "receipt"),
    ]:
        graph.connect(a.id, out, b.id, port)
    return features


def train_pipeline(graph, registry, *, sampled=False):
    batch = add(graph, registry, "flow.batch", "batch", 1400, 800, size=3)
    train = add(
        graph,
        registry,
        "ml.train_local_anomaly",
        "train",
        1750,
        800,
        sampling="even_stride" if sampled else "all",
        bank_limit=16 if sampled else 2048,
    )
    for index in range(3):
        feature = pipeline(
            graph, registry, f"train{index}_", f"train-{index}.png", y=500 * (index + 1)
        )
        graph.connect(feature.id, "features", batch.id, f"item_{index + 1}")
    graph.connect(batch.id, "group", train.id, "features")
    return train


def build(destination=HERE):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    destination = Path(destination)
    assets = destination / "assets"
    if assets.exists():
        raise FileExistsError("Existing example assets preserved")
    assets.mkdir(parents=True)
    images = {f"train-{i}.png": surface(i) for i in range(3)}
    images["normal.png"] = surface(100)
    images["defect.png"] = surface(101)
    images["defect.png"][10:18, 12:20] = 0
    images["missing.png"] = surface(102)
    images["missing.png"][10:18, 12:20] = 110
    images["weak.png"] = surface(103)
    images["weak.png"][10, 12] += 30
    images["bright.png"] = np.uint8(surface(104) + 55)
    images["high16.png"] = images["defect.png"].astype(np.uint16) * 257
    truth_mask = np.zeros((32, 32), np.uint8)
    truth_mask[10:18, 12:20] = 255
    images["truth-defect.png"] = truth_mask
    for name, image in images.items():
        (assets / name).write_bytes(cv2.imencode(".png", image)[1].tobytes())
    weights = np.zeros((4, 1, 2, 2), np.float32)
    for index in range(4):
        weights[index, 0, index // 2, index % 2] = 1
    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv", ["image", "weights"], ["local"], kernel_shape=[2, 2], strides=[2, 2]
            )
        ],
        "original_local_pixel_embedding",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 1, 32, 32])],
        [helper.make_tensor_value_info("local", TensorProto.FLOAT, [1, 4, 16, 16])],
        [numpy_helper.from_array(weights, "weights")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    (assets / "embedding.onnx").write_bytes(model.SerializeToString())
    registry = builtin_registry()
    net = registry.create(
        "ml.dnn", parameters={**params(), "model_path": str((assets / "embedding.onnx").resolve())}
    )
    try:
        export_contract(net, destination / "bundle")
    finally:
        net.close()
    graph = Graph()
    graph.project_dir = destination
    try:
        train = train_pipeline(graph, registry)
        graph.run()
        assert train.last_result is not None, [(k, b.error) for k, b in graph.blocks.items()]
        bank = train.last_result.outputs["model"]
        (assets / "reference.json").write_bytes(encoded(bank))
        corrupt = json.loads(encoded(bank))
        corrupt["samples"][0][0] += 0.125
        (assets / "corrupt.json").write_bytes(encoded(corrupt))
    finally:
        graph.close()
    cases = {}
    for case in CASES:
        graph = Graph()
        graph.project_dir = destination
        try:
            image = (
                "normal.png"
                if case in ("normal", "wrong_features", "corrupt_reference")
                else "missing.png"
                if case == "missing_texture"
                else "weak.png"
                if case == "weak_defect"
                else "bright.png"
                if case == "bright_shift"
                else "high16.png"
                if case == "high_depth"
                else "defect.png"
            )
            feature = pipeline(
                graph,
                registry,
                "",
                image,
                normalization="l2" if case == "wrong_features" else "none",
            )
            inspect = add(
                graph,
                registry,
                "ml.local_anomaly",
                "inspect",
                1750,
                allowed_anomaly_pixels=64 if case == "allowed_area" else 0,
                reject_training_image=True,
            )
            inspect.exposed_outputs.append("result_graphics")
            if case in ("train_then_inspect", "sampled_reference"):
                reference = train_pipeline(graph, registry, sampled=case == "sampled_reference")
            else:
                reference = add(
                    graph,
                    registry,
                    "ml.load_local_anomaly",
                    "reference",
                    1400,
                    400,
                    path="assets/"
                    + ("corrupt.json" if case == "corrupt_reference" else "reference.json"),
                )
            graph.connect(reference.id, "model", inspect.id, "model")
            graph.connect(feature.id, "image", inspect.id, "image")
            graph.connect(feature.id, "features", inspect.id, "features")
            if case == "selected_region":
                region = add(
                    graph,
                    registry,
                    "region.shape",
                    "selection",
                    1400,
                    600,
                    shape="rectangle",
                    rectangle=[0, 0, 10, 32],
                )
                graph.connect("source", "image", region.id, "image")
                graph.connect(region.id, "region", inspect.id, "region")
            add(graph, registry, "cv.overlay_result_graphics", "overlay", 2100)
            add(graph, registry, "output.filmstrip_image", "filmstrip", 2450)
            graph.connect(inspect.id, "image", "overlay", "image")
            graph.connect(inspect.id, "result_graphics", "overlay", "graphics_1")
            graph.connect("overlay", "image", "filmstrip", "image")
            note = add(graph, registry, "layout.comment", "attribution", 0, -600)
            note.name, note.node_size = "Original local anomaly controls - MIT", "large"
            note.description = (
                "Original synthetic images, ONNX patch encoder, reference bank and graphs by "
                "OpenCVDataflowGUI contributors, MIT. Three training seeds "
                "and separate query seeds. "
                "Four raw pixel values form each 2x2 embedding. No pretrained network or PatchCore "
                "accuracy claim. Weak defect intentionally falls below the distance threshold; "
                "the global bank can miss rearranged known textures. See README.md/SOURCES.md."
            )
            graph.save(destination / (case + ".json"))
            cases[case] = {
                "target": "reference" if case == "corrupt_reference" else "inspect",
                "image": image,
            }
        finally:
            graph.close()
    (destination / "truth.json").write_bytes(
        encoded(
            {
                "cases": cases,
                "training_seeds": [0, 1, 2],
                "heldout_seeds": [100, 101, 102, 103, 104],
                "threshold": 0.2,
                "asset_sha256": {
                    p.relative_to(destination).as_posix(): hashlib.sha256(
                        p.read_bytes()
                    ).hexdigest()
                    for p in sorted([*assets.iterdir(), *(destination / "bundle").iterdir()])
                },
            }
        )
    )


if __name__ == "__main__":
    build()
