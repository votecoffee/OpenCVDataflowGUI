"""Original stereo arithmetic controls and public-image known-shift native projects."""

import hashlib
import json
import runpy
import tempfile
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.learned_stereo import preprocessing
from opencv_dataflow.model_contracts import export_contract

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "monocular_depth/assets/tless_color.png"
PREPARATION = runpy.run_path(str(Path(__file__).with_name("prepare_model.py")))
CASES = {
    "plane": {},
    "subpixel": {},
    "concatenated": {"input_mode": "concatenated"},
    "letterbox": {"width": 16, "height": 16, "resize_mode": "letterbox", "head_scale": 16 / 31},
    "head_units": {"width": 16, "height": 16, "head_units": "head_pixels", "head_scale": 8 / 31},
    "selected": {},
    "right_mask": {},
    "empty": {},
    "coverage_alarm": {"minimum_valid_fraction": 0.99},
    "native_8": {},
    "native_16": {},
    "native_zero": {},
}


def asset(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw:
        raise FileExistsError("Different existing assets preserved")
    path.write_bytes(raw)


def fixtures():
    yy, xx = np.indices((23, 31))
    red = (40 + xx * 4 + yy * 2).astype(np.uint8)
    left = np.dstack((30 + yy * 5, 220 - xx * 3, red)).astype(np.uint8)
    right = left.copy()
    right[..., 2] += 12
    subpixel = left.copy()
    subpixel[..., 2] += 6
    mask = np.ones((23, 31), np.uint8) * 255
    mask[:, 8:12] = 0
    raw = SOURCE.read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "04b19c87d60aaabe98c6be9655a58c5cc620ac31dc356ede55e89dc02cd2a166"
    )
    native = cv2.resize(
        cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR),
        (320, 240),
        interpolation=cv2.INTER_AREA,
    )
    images = {
        "left": left,
        "right": right,
        "subpixel": subpixel,
        "right_mask": mask,
        "native_left": native,
    }
    for name, d in (("native_8", 8), ("native_16", 16), ("native_zero", 0)):
        image = np.zeros_like(native)
        if d:
            image[:, :-d] = native[:, d:]
        else:
            image = native.copy()
        images[name] = image
    return images


def model_bytes(pooled=False, concatenated=False):
    from onnx import TensorProto, checker, helper, numpy_helper

    nodes = []
    if concatenated:
        inputs = [
            helper.make_tensor_value_info("left", TensorProto.FLOAT, [1, 6, "height", "width"])
        ]
        weights = np.array([-255 / 4, 0, 0, 255 / 4, 0, 0], np.float32).reshape(1, 6, 1, 1)
        value = "left"
    else:
        inputs = [
            helper.make_tensor_value_info(k, TensorProto.FLOAT, [1, 3, "height", "width"])
            for k in ("left", "right")
        ]
        nodes.append(helper.make_node("Sub", ["right", "left"], ["delta"]))
        weights = np.array([255 / 4, 0, 0], np.float32).reshape(1, 3, 1, 1)
        value = "delta"
    nodes.append(helper.make_node("Conv", [value, "weights"], ["quantity"], kernel_shape=[1, 1]))
    value = "quantity"
    if pooled:
        nodes.append(
            helper.make_node("AveragePool", [value], ["small"], kernel_shape=[2, 2], strides=[2, 2])
        )
        value = "small"
    nodes.append(helper.make_node("Squeeze", [value, "axis"], ["disparity"]))
    graph = helper.make_graph(
        nodes,
        "original_stereo_arithmetic",
        inputs,
        [
            helper.make_tensor_value_info(
                "disparity", TensorProto.FLOAT, [1, "head_height", "head_width"]
            )
        ],
        [
            numpy_helper.from_array(weights, "weights"),
            numpy_helper.from_array(np.array([1], np.int64), "axis"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    model.doc_string = "Original MIT paired arithmetic control, not a learned matching model."
    checker.check_model(model)
    return model.SerializeToString()


def seal(block, raw, case):
    with tempfile.TemporaryDirectory(prefix="dataflow-stereo-") as temporary:
        root = Path(temporary)
        path = root / "input.onnx"
        path.write_bytes(raw)
        block.parameters["model_path"], block.parameters["contract_path"] = str(path), ""
        export_contract(block, root / "bundle", preprocessing=preprocessing(block.parameters))
        for item in (root / "bundle").iterdir():
            asset(HERE / "models" / case / item.name, item.read_bytes())
    block.parameters["model_path"], block.parameters["contract_path"] = (
        "",
        f"models/{case}/contract.json",
    )


def build():
    for name, image in fixtures().items():
        ok, data = cv2.imencode(".png", image)
        assert ok
        asset(HERE / "assets" / (name + ".png"), data.tobytes())
    registry = builtin_registry()
    for case, changes in CASES.items():
        graph = Graph()
        try:
            graph.program["assets"] = ["SOURCES.md", "MODEL-LICENSE.txt"]

            def add(kind, identity, position, **parameters):
                block = graph.add(registry.create(kind, block_id=identity, parameters=parameters))
                block.position = list(position)
                return block

            native = case.startswith("native_")
            for name in ("left", "right"):
                image = (
                    "native_left"
                    if native and name == "left"
                    else case
                    if native
                    else "subpixel"
                    if case == "subpixel" and name == "right"
                    else name
                )
                b = add(
                    "input.file",
                    name,
                    (0, 0 if name == "left" else 360),
                    path=f"assets/{image}.png",
                    path_base="project",
                    filename_ticket=False,
                )
                b.ticket_settings = {"mode": "fixed", "value": "paired-image"}
            p = (
                dict(PREPARATION["PARAMETERS"])
                if native
                else {"width": 31, "height": 23, "engine": "classic"}
            )
            p.update(changes)
            if native:
                p["contract_path"] = ".local/hitnet/contract.json"
            block = add("ml.stereo_disparity", "estimate", (520, 0), **p)
            block.node_size = "large"
            block.exposed_outputs = list(dict.fromkeys([*block.exposed_outputs, "preview"]))
            if not native:
                seal(
                    block,
                    model_bytes(case in ("letterbox", "head_units"), case == "concatenated"),
                    case,
                )
            add("output.filmstrip_image", "filmstrip", (1000, 0))
            graph.connect("estimate", "preview", "filmstrip", "image")
            if case == "plane":
                for name in ("left", "right"):
                    add(
                        "3d.camera",
                        name + "_camera",
                        (-800, 0 if name == "left" else 340),
                        fx=80.0,
                        fy=80.0,
                        cx=15.0,
                        cy=11.0,
                        width=31,
                        height=23,
                        frame=name,
                    )
                add(
                    "3d.pose",
                    "baseline",
                    (-800, 680),
                    translation=[-0.1, 0.0, 0.0],
                    source_frame="left",
                    target_frame="right",
                )
                add("3d.stereo_rig", "rig", (-400, 0))
                add("3d.stereo_rectify", "rectify", (0, -400), interpolation="nearest")
                for name in ("left", "right"):
                    graph.connect(name + "_camera", "camera", "rig", name + "_camera")
                    graph.connect(name, "image", "rectify", name)
                    graph.connect("rectify", name, "estimate", name)
                graph.connect("baseline", "transform", "rig", "transform")
                graph.connect("rig", "rig", "rectify", "rig")
                graph.connect("rectify", "rectification", "estimate", "rectification")
                add("3d.reproject", "cloud", (1000, 460), sample_step=1)
                graph.connect("estimate", "disparity", "cloud", "disparity")
                graph.connect("estimate", "rectification", "cloud", "rectification")
                graph.connect("estimate", "image", "cloud", "image")
            else:
                for name in ("left", "right"):
                    graph.connect(name, "image", "estimate", name)
            if case in ("selected", "empty"):
                add(
                    "region.shape",
                    "region",
                    (0, 720),
                    shape="circle",
                    center=[15, 11] if case == "selected" else [-20, -20],
                    radius=7,
                )
                graph.connect("left", "image", "region", "image")
                graph.connect("region", "region", "estimate", "region")
            if case == "right_mask":
                add(
                    "region.shape",
                    "mask",
                    (0, 720),
                    rectangle=[8, 0, 4, 23],
                    invert=True,
                )
                graph.connect("right", "image", "mask", "image")
                graph.connect("mask", "mask", "estimate", "right_mask")
            note = add("layout.comment", "attribution", (0, -900))
            note.node_size = "large"
            note.name = "Stereo sources and pixel units"
            note.description = (
                "Original MIT paired arithmetic controls by OpenCVDataflowGUI contributors. "
                "Optional HITNET ETH3D uses unchanged Apache-2.0 Google Research / PINTO0309 "
                "ONNX weights; see MODEL-LICENSE.txt and SOURCES.md. T-LESS scene 6/frame 2 image: "
                "T. Hodan et al., WACV 2017, CC BY 4.0, https://cmp.felk.cvut.cz/t-less/ . "
                "Native pairs are resized and shifted copies with known artificial disparity, "
                "not real stereo acquisitions or measured depth. Disparity is x_left - x_right "
                "in original left pixels. Coverage is not matching confidence. The plane "
                "project's 0.1 m baseline and camera are synthetic controls."
            )
            graph.save(HERE / (case + ".json"))
        finally:
            graph.close()
    truth = {
        "cases": list(CASES),
        "native_cases": [k for k in CASES if k.startswith("native_")],
        "source_size": [31, 23],
        "plane_disparity": 3.0,
        "subpixel_disparity": 1.5,
        "hitnet_sha256": PREPARATION["SHA256"],
        "asset_sha256": {
            p.relative_to(HERE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in (HERE / "assets", HERE / "models")
            for p in sorted(folder.rglob("*"))
            if p.is_file()
        },
    }
    asset(HERE / "truth.json", (json.dumps(truth, indent=2) + "\n").encode())


if __name__ == "__main__":
    build()
