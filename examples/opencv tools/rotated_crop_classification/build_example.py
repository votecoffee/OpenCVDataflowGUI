"""Original measured-pose inspection and independently sampled texture classification."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.model_contracts import export_contract
from opencv_dataflow.parallel_inspection import ParallelInspect
from opencv_dataflow.toolblocks import Binding, ToolBlock

HERE = Path(__file__).resolve().parent
CASES = (
    "parallel_inspection",
    "serial_inspection",
    "all_known",
    "single_part",
    "accepted_horizontal",
    "no_alignment",
    "high_depth",
    "selected_parts",
    "empty",
    "coverage_pad",
    "coverage_strict",
)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def texture(kind, u, v):
    return (
        170 + 60 * np.cos(2 * np.pi * (v if kind == "horizontal" else u) / 8)
        if kind != "unknown"
        else np.zeros_like(u) + 170
    )


def window(kind, seed):
    y, x = np.indices((32, 64), dtype=float)
    rng = np.random.default_rng(seed)
    values = texture(kind, x - 31.5, y - 15.5) + rng.normal(0, 2, x.shape)
    return np.uint8(np.clip(np.rint(values), 0, 255))


def train(destination):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    images = [window(kind, seed) for kind in ["horizontal", "vertical"] for seed in range(12)]
    X = np.array(
        [cv2.resize(img.astype(np.float32), (32, 16)).ravel() / 255 for img in images], np.float32
    )
    labels = np.repeat([0, 1], 12).astype(np.int32)
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(1)
    svm.train(X, cv2.ml.ROW_SAMPLE, labels)
    rho, alpha, indices = svm.getDecisionFunction(0)
    w = (alpha.ravel() @ svm.getSupportVectors()[indices.ravel()]).astype(np.float32)
    bias = -float(rho)
    assert np.all(svm.predict(X)[1].ravel() == labels)
    W = np.stack([3 * w, -3 * w], axis=1)
    b = np.float32([3 * bias, -3 * bias])
    nodes = [
        helper.make_node("Flatten", ["image"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "weights", "bias"], ["scores"]),
    ]
    graph = helper.make_graph(
        nodes,
        "original_trained_stripe_svm",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 1, 16, 32])],
        [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 2])],
        [numpy_helper.from_array(W, "weights"), numpy_helper.from_array(b, "bias")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    (destination / "classifier.onnx").write_bytes(model.SerializeToString())
    (destination / "classifier-weights.json").write_bytes(
        encoded(
            {
                "weights": W.tolist(),
                "bias": b.tolist(),
                "native_svm_rho": rho,
                "raw_margin_multiplier": 3.0,
                "training_rows": 24,
                "classes": ["horizontal", "vertical"],
            }
        )
    )


def scene(all_known=False, one=False):
    image = np.full((320, 500), 20, np.uint8)
    rows = []
    y, x = np.indices(image.shape, dtype=float)
    for index, (cx, cy, angle, kind) in enumerate(
        [
            (90, 80, -55, "horizontal"),
            (250, 80, -20, "vertical"),
            (410, 80, 25, "horizontal"),
            (90, 230, 70, "vertical"),
            (250, 230, 110, "horizontal"),
            (410, 230, 160, "unknown"),
        ]
    ):
        if one and index:
            break
        if all_known and kind == "unknown":
            kind = "vertical"
        a = np.deg2rad(angle)
        u = np.cos(a) * (x - cx) - np.sin(a) * (y - cy)
        v = np.sin(a) * (x - cx) + np.cos(a) * (y - cy)
        support = (np.abs(u) < 44) & (np.abs(v) < 24)
        values = texture(kind, u, v) + np.random.default_rng(100 + index).normal(0, 2, x.shape)
        image[support] = np.uint8(np.clip(np.rint(values[support]), 0, 255))
        rows.append(dict(center=[cx, cy], angle=angle, label=kind))
    return image, rows


def inspection(registry, *, accepted=None, crop_parameters=None):
    graph = Graph()

    def add(kind, name, **p):
        return graph.add(registry.create(kind, block_id=name, parameters=p))

    add("flow.split", "image", branches=3, workers=1)
    add("pose.offset", "pose")
    crop = add("cv.crop_pose", "crop", height=32, local_height=32.0, **(crop_parameters or {}))
    crop.exposed_outputs.append("result_graphics")
    add("cv.convert", "convert", color_format="gray", pixel_depth="uint8")
    add(
        "ml.dnn",
        "net",
        contract_path="bundle/contract.json",
        width=32,
        height=16,
        color_format="gray",
        input_name="image",
        output_names=["scores"],
        mean=[0.0],
        std=[1.0],
        engine="classic",
    )
    add(
        "ml.classify",
        "classify",
        labels=["horizontal", "vertical"],
        top_k=2,
        minimum_confidence=0.9,
        accepted_labels=accepted or [],
    )
    add("cv.annotation_layer", "text", shape="label", font_size=12)
    add("cv.overlay_result_graphics", "overlay")
    for a, o, b, i in [
        ("image", "branch_1", "crop", "image"),
        ("image", "branch_2", "text", "image"),
        ("image", "branch_3", "overlay", "image"),
        ("pose", "pose", "crop", "pose"),
        ("pose", "points", "text", "points"),
        ("crop", "image", "convert", "image"),
        ("convert", "image", "net", "image"),
        ("net", "tensors", "classify", "tensors"),
        ("classify", "label", "text", "text"),
        ("crop", "result_graphics", "overlay", "graphics_1"),
        ("text", "result_graphics", "overlay", "graphics_2"),
    ]:
        graph.connect(a, o, b, i)
    return ToolBlock(
        graph,
        {"image": Binding("image", "value"), "pose": Binding("pose", "pose")},
        {
            "crop": Binding("crop", "image"),
            "geometry": Binding("crop", "geometry"),
            "overlay": Binding("overlay", "image"),
            "label": Binding("classify", "label"),
            "confidence": Binding("classify", "confidence"),
        },
    )


def build(destination=HERE):
    destination = Path(destination)
    assets = destination / "assets"
    if assets.exists() or (destination / "bundle").exists():
        raise FileExistsError("Existing pose-classification assets preserved")
    assets.mkdir(parents=True)
    train(assets)
    pictures = {
        "scene.png": scene()[0],
        "all-known.png": scene(all_known=True)[0],
        "one-part.png": scene(one=True)[0],
    }
    pictures["scene16.png"] = pictures["scene.png"].astype(np.uint16) * 257
    pictures["blank.png"] = np.full_like(pictures["scene.png"], 20)
    training = []
    for kind in ("horizontal", "vertical"):
        for seed in range(12):
            name = f"train-{kind}-{seed:02}.png"
            pictures[name] = window(kind, seed)
            training.append({"file": name, "label": kind})
    for name, image in pictures.items():
        (assets / name).write_bytes(cv2.imencode(".png", image)[1].tobytes())
    registry = builtin_registry()
    model = registry.create(
        "ml.dnn",
        parameters={
            "model_path": str((assets / "classifier.onnx").resolve()),
            "width": 32,
            "height": 16,
            "color_format": "gray",
            "input_name": "image",
            "output_names": ["scores"],
            "mean": [0.0],
            "std": [1.0],
            "engine": "classic",
        },
    )
    try:
        export_contract(model, destination / "bundle")
    finally:
        model.close()
    cases = {}
    for case in CASES:
        graph = Graph()

        def add(kind, name, layout_x, layout_y, **p):
            block = graph.add(registry.create(kind, block_id=name, parameters=p))
            block.position = [layout_x, layout_y]
            return block

        try:
            name = (
                "all-known.png"
                if case == "all_known"
                else "one-part.png"
                if case == "single_part"
                else "blank.png"
                if case == "empty"
                else "scene16.png"
                if case == "high_depth"
                else "scene.png"
            )
            add(
                "input.file",
                "source",
                0,
                0,
                source_mode="files",
                paths=["assets/" + name],
                path_base="project",
                color_format="unchanged",
                pixel_depth="unchanged",
            )
            if case in ("coverage_pad", "coverage_strict"):
                add("pose.define", "pose", 360, 350, x=8.0, y=8.0, angle_degrees=25.0)
                add(
                    "cv.crop_pose",
                    "crop",
                    720,
                    0,
                    width=64,
                    height=32,
                    bounds_policy="pad" if case == "coverage_pad" else "strict",
                )
                graph.connect("source", "image", "pose", "image")
                graph.connect("source", "image", "crop", "image")
                graph.connect("pose", "pose", "crop", "pose")
                add("output.filmstrip_image", "filmstrip", 1080, 0)
                graph.connect("crop", "image", "filmstrip", "image")
                target = "crop"
            else:
                add("cv.convert", "gray", 350, 0, color_format="gray", pixel_depth="uint8")
                add("cv.threshold", "threshold", 700, 0, value=80.0)
                add("inspect.blobs", "parts", 1050, 0, min_area=1000)
                for a, o, b, i in [
                    ("source", "image", "gray", "image"),
                    ("gray", "image", "threshold", "image"),
                    ("threshold", "image", "parts", "image"),
                ]:
                    graph.connect(a, o, b, i)
                if case == "selected_parts":
                    add("region.shape", "selection", 700, -400, rectangle=[0, 0, 170, 320])
                    graph.connect("threshold", "image", "selection", "image")
                    graph.connect("selection", "region", "parts", "region")
                tool = ParallelInspect(
                    registry=registry,
                    inspection=inspection(
                        registry, accepted=["horizontal"] if case == "accepted_horizontal" else []
                    ),
                    parameters={
                        "workers": 1 if case == "serial_inspection" else 3,
                        "feature_angle": "fixed" if case == "no_alignment" else "feature",
                        "image_output": "crop",
                    },
                    block_id="inspect",
                )
                tool.position = [1400, 0]
                graph.add(tool)
                graph.connect("source", "image", "inspect", "image")
                graph.connect("parts", "collection", "inspect", "features")
                add("output.filmstrip_image", "filmstrip", 1800, 0)
                graph.connect("inspect", "overlay", "filmstrip", "image")
                target = "inspect"
            note = add("layout.comment", "attribution", 0, -850)
            note.name, note.node_size = (
                "Original pose-normalized inspection controls - MIT",
                "large",
            )
            note.description = (
                "Original MIT synthetic parts, training windows, native linear SVM "
                "weights, ONNX export and projects by OpenCVDataflowGUI "
                "contributors. "
                "Scenes are drawn analytically with separate noise seeds. Blob "
                "moments measure center and an unoriented 180-degree axis; both "
                "texture classes are unchanged by that ambiguity. "
                "One unmarked part is an explicit low-confidence rejection control. "
                "Five known parts passing does not establish production accuracy or "
                "calibrated confidence. "
                "The existing parallel inspection ToolBlock owns per-part jobs and "
                "child tickets; each crop retains its mapping to the original parent "
                "image. See README.md, SOURCES.md and truth.json."
            )
            graph.save(destination / (case + ".json"))
            cases[case] = {"target": target, "image": name}
        finally:
            graph.close()
    rows = scene()[1]
    files = list(assets.iterdir()) + list((destination / "bundle").iterdir())
    (destination / "truth.json").write_bytes(
        encoded(
            {
                "cases": cases,
                "parts": rows,
                "training": training,
                "reference_scope": "Analytic construction centers/angles/classes; "
                "noisy held-out scenes are separate from training, not industrial accuracy data",
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
