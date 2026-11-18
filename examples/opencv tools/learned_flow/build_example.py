"""Known paired affine ramps and licensed artificial-motion RAFT controls."""

import hashlib
import json
import runpy
import tempfile
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.learned_flow import preprocessing
from opencv_dataflow.model_contracts import export_contract

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "monocular_depth/assets/tless_color.png"
PREPARATION = runpy.run_path(str(Path(__file__).with_name("prepare_model.py")))
CASES = {
    "plane": {},
    "subpixel": {},
    "concatenated": {"input_mode": "concatenated"},
    "letterbox": {
        "width": 16,
        "height": 16,
        "resize_mode": "letterbox",
        "head_scale": [16 / 31, 12 / 23],
    },
    "head_units": {
        "width": 16,
        "height": 16,
        "head_units": "head_pixels",
        "head_scale": [8 / 31, 8 / 23],
    },
    "selected": {},
    "second_mask": {},
    "empty": {},
    "coverage_alarm": {"minimum_valid_fraction": 0.99},
    "native_positive": {},
    "native_negative": {},
    "native_zero": {},
}
MOTIONS = {"native_positive": [4, -2], "native_negative": [-3, 2], "native_zero": [0, 0]}


def asset(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw:
        raise FileExistsError("Different existing assets preserved")
    path.write_bytes(raw)


def fixtures():
    y, x = np.indices((23, 31))
    first = np.dstack((30 + y * 5, 220 - x * 3, 40 + 4 * x + 2 * y)).astype(np.uint8)
    second = (first.astype(np.int16) + [5, 6, -6]).astype(np.uint8)
    subpixel = (first.astype(np.int16) + [-5, 4, -8]).astype(np.uint8)
    raw = SOURCE.read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "04b19c87d60aaabe98c6be9655a58c5cc620ac31dc356ede55e89dc02cd2a166"
    )
    native = cv2.resize(
        cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR),
        (240, 180),
        interpolation=cv2.INTER_AREA,
    )
    images = {"first": first, "second": second, "subpixel": subpixel, "native_first": native}
    for name, (dx, dy) in MOTIONS.items():
        images[name] = cv2.warpAffine(
            native,
            np.array([[1, 0, dx], [0, 1, dy]], np.float32),
            (240, 180),
            flags=cv2.INTER_NEAREST,
        )
    return images


def model_bytes(pooled=False, concatenated=False):
    from onnx import TensorProto, checker, helper, numpy_helper

    weights = np.array([[-255 / 4, 0, 255 / 10], [0, 0, -255 / 5]], np.float32)
    nodes = []
    if concatenated:
        inputs = [
            helper.make_tensor_value_info("first", TensorProto.FLOAT, [1, 6, "height", "width"])
        ]
        weights = np.concatenate((-weights, weights), axis=1)
        value = "first"
    else:
        inputs = [
            helper.make_tensor_value_info(k, TensorProto.FLOAT, [1, 3, "height", "width"])
            for k in ("first", "second")
        ]
        nodes.append(helper.make_node("Sub", ["second", "first"], ["delta"]))
        value = "delta"
    nodes.append(helper.make_node("Conv", [value, "weights"], ["field"], kernel_shape=[1, 1]))
    value = "field"
    if pooled:
        nodes.append(
            helper.make_node("AveragePool", [value], ["small"], kernel_shape=[2, 2], strides=[2, 2])
        )
        value = "small"
    nodes.append(helper.make_node("Identity", [value], ["flow"]))
    graph = helper.make_graph(
        nodes,
        "original_affine_ramp_flow",
        inputs,
        [helper.make_tensor_value_info("flow", TensorProto.FLOAT, [1, 2, "head_h", "head_w"])],
        [numpy_helper.from_array(weights.reshape(2, -1, 1, 1), "weights")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    model.doc_string = "Original MIT affine-ramp arithmetic control, not a learned estimator."
    checker.check_model(model)
    return model.SerializeToString()


def seal(block, raw, case):
    with tempfile.TemporaryDirectory(prefix="dataflow-flow-") as temporary:
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
            graph.program["assets"] = ["SOURCES.md", "BSD-3-LICENSE.txt", "MITLICENSE.txt"]

            def add(kind, identity, position, **parameters):
                block = graph.add(registry.create(kind, block_id=identity, parameters=parameters))
                block.position = list(position)
                return block

            native = case.startswith("native_")
            for name in ("first", "second"):
                image = (
                    "native_first"
                    if native and name == "first"
                    else case
                    if native
                    else "subpixel"
                    if case == "subpixel" and name == "second"
                    else name
                )
                add(
                    "input.file",
                    name,
                    (0, 0 if name == "first" else 360),
                    path=f"assets/{image}.png",
                    path_base="project",
                )
            p = dict(PREPARATION["PARAMETERS"]) if native else {"width": 31, "height": 23}
            p.update(changes)
            if native:
                p["contract_path"] = ".local/raft/contract.json"
            block = add("ml.optical_flow", "estimate", (500, 0), **p)
            block.node_size = "large"
            block.exposed_outputs = list(dict.fromkeys([*block.exposed_outputs, "preview"]))
            if not native:
                seal(
                    block,
                    model_bytes(case in ("letterbox", "head_units"), case == "concatenated"),
                    case,
                )
            for name in ("first", "second"):
                graph.connect(name, "image", "estimate", name)
            add("output.filmstrip_image", "filmstrip", (1000, 0))
            graph.connect("estimate", "preview", "filmstrip", "image")
            if case in ("plane", "native_positive"):
                warp = add(
                    "cv.warp_flow",
                    "warp",
                    (1000, 450),
                    direction="source_to_target",
                    consistency="off",
                )
                warp.exposed_inputs = list(
                    dict.fromkeys([*warp.exposed_inputs, "target", "flow_support"])
                )
                for a, out, inp in (
                    ("first", "image", "image"),
                    ("second", "image", "target"),
                    ("estimate", "flow", "flow"),
                    ("estimate", "valid_mask", "flow_support"),
                ):
                    graph.connect(a, out, "warp", inp)
            if case in ("selected", "empty", "second_mask"):
                name = "second" if case == "second_mask" else "first"
                p = (
                    {"rectangle": [8, 4, 4, 9], "invert": True}
                    if case == "second_mask"
                    else {
                        "shape": "circle",
                        "center": [15, 11] if case == "selected" else [-20, -20],
                        "radius": 7,
                    }
                )
                add("region.shape", "region", (0, 720), **p)
                graph.connect(name, "image", "region", "image")
                graph.connect(
                    "region",
                    "mask" if case == "second_mask" else "region",
                    "estimate",
                    "second_mask" if case == "second_mask" else "region",
                )
            note = add("layout.comment", "attribution", (0, -900))
            note.node_size = "large"
            note.name = "Motion direction / sources"
            note.description = (
                "Original MIT affine-ramp controls by OpenCVDataflowGUI contributors. "
                "Optional RAFT: Zachary Teed/Jia Deng, Princeton, BSD-3-Clause; ONNX conversion "
                "Jeong-gi Kwak/PINTO0309, MIT, via OpenCV Zoo. See SOURCES.md and both notices. "
                "T-LESS texture: T. Hodan et al., WACV 2017, CC BY 4.0, "
                "https://cmp.felk.cvut.cz/t-less/ . Native pairs are resized/translated copies, "
                "not actual recorded motion or occlusion ground truth. Vectors are "
                "first-to-second dx,dy in original first-image pixels. Swap inputs for "
                "reverse direction. Coverage is not accuracy confidence."
            )
            graph.save(HERE / (case + ".json"))
        finally:
            graph.close()
    truth = {
        "cases": list(CASES),
        "native_motions": MOTIONS,
        "source_size": [31, 23],
        "plane_motion": [2.0, -1.0],
        "subpixel_motion": [1.5, 1.0],
        "raft_sha256": PREPARATION["SHA256"],
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
