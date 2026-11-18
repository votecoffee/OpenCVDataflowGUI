"""Known metric/relative controls and licensed real RGB-D examples, without plugins."""

import hashlib
import json
import runpy
import tempfile
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.model_contracts import export_contract
from opencv_dataflow.monocular_depth import preprocessing

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "Simple Plugins/gdrnpp/preprocessing"
PREPARATION = runpy.run_path(str(Path(__file__).with_name("prepare_model.py")))
CASES = {
    "relative": {},
    "metric": {"convention": "metric_depth"},
    "inverse_metric": {"convention": "metric_inverse_depth"},
    "millimeters": {
        "convention": "metric_depth",
        "units": "mm",
        "head_scale": 1000.0,
        "maximum_depth": 5000.0,
    },
    "letterbox": {"width": 16, "height": 16, "resize_mode": "letterbox"},
    "invalid_hole": {"convention": "metric_inverse_depth"},
    "selected": {},
    "empty": {},
    "coverage_alarm": {"minimum_valid_pixels": 714},
    **PREPARATION["VARIANTS"],
}


def asset(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw:
        raise FileExistsError("Different existing assets preserved")
    path.write_bytes(raw)


def fixtures():
    yy, xx = np.indices((23, 31))
    red = (40 + xx * 4 + yy * 2).astype(np.uint8)
    bgr = np.dstack((30 + yy * 5, 220 - xx * 3, red)).astype(np.uint8)
    hole = bgr.copy()
    hole[8:12, 13:17, 2] = 0
    return {"ramp": bgr, "hole": hole}


def model_bytes(hole=False, small=False):
    from onnx import TensorProto, checker, helper, numpy_helper

    weights = np.array([1 if hole else 2, 0, 0], np.float32).reshape(1, 3, 1, 1)
    bias = np.array([0 if hole else 1], np.float32)
    nodes = [helper.make_node("Conv", ["input", "weights", "bias"], ["q"], kernel_shape=[1, 1])]
    out = "q"
    if small:
        nodes.append(
            helper.make_node("AveragePool", [out], ["small"], kernel_shape=[2, 2], strides=[2, 2])
        )
        out = "small"
    nodes.append(helper.make_node("Squeeze", [out, "axis"], ["output"]))
    graph = helper.make_graph(
        nodes,
        "original_depth_control",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, "height", "width"])],
        [
            helper.make_tensor_value_info(
                "output", TensorProto.FLOAT, [1, "head_height", "head_width"]
            )
        ],
        [
            numpy_helper.from_array(weights, "weights"),
            numpy_helper.from_array(bias, "bias"),
            numpy_helper.from_array(np.array([1], np.int64), "axis"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    model.doc_string = (
        "Original MIT image-dependent depth arithmetic; no learned geometry or accuracy claim."
    )
    checker.check_model(model)
    return model.SerializeToString()


def seal(block, raw, case):
    with tempfile.TemporaryDirectory(prefix="dataflow-depth-fixture-") as temporary:
        root = Path(temporary)
        path = root / "input.onnx"
        path.write_bytes(raw)
        block.parameters["model_path"], block.parameters["contract_path"] = str(path), ""
        export_contract(
            block, root / "bundle", preprocessing={"input": preprocessing(block.parameters)}
        )
        for item in (root / "bundle").iterdir():
            asset(HERE / "models" / case / item.name, item.read_bytes())
    block.parameters["model_path"] = ""
    block.parameters["contract_path"] = f"models/{case}/contract.json"


def build():
    for name, image in fixtures().items():
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        asset(HERE / "assets" / (name + ".png"), encoded.tobytes())
    provenance = json.loads((SOURCE / "provenance.json").read_bytes())
    for source, destination in (
        ("source.png", "tless_color.png"),
        ("depth.png", "tless_depth.png"),
    ):
        raw = (SOURCE / source).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == provenance["sha256"][source]
        asset(HERE / "assets" / destination, raw)
    capture = json.loads((SOURCE / "capture.json").read_bytes())
    asset(
        HERE / "assets/tless_camera.json",
        (
            json.dumps(
                {
                    "dataset": "T-LESS",
                    "scene": 6,
                    "frame": 2,
                    "camera_matrix": capture["camera_matrix"],
                    "depth_scale_to_m": capture["depth_scale_to_m"],
                },
                indent=2,
            )
            + "\n"
        ).encode(),
    )
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
            image = "tless_color" if native else "hole" if case == "invalid_hole" else "ramp"
            add("input.file", "source", (0, 0), path=f"assets/{image}.png", path_base="project")
            parameters = (
                dict(PREPARATION["PARAMETERS"])
                if native
                else {"width": 31, "height": 23, "input_interpolation": "linear"}
            )
            parameters.update(changes)
            if native:
                parameters["contract_path"] = f".local/{case}/contract.json"
            block = add("ml.monocular_depth", "estimate", (450, 0), **parameters)
            block.node_size = "large"
            block.exposed_outputs = list(
                dict.fromkeys([*block.exposed_outputs, "preview", "estimate", "depth"])
            )
            if not native:
                seal(block, model_bytes(case == "invalid_hole", case == "letterbox"), case)
            add("output.filmstrip_image", "filmstrip", (920, 0))
            graph.connect("source", "image", "estimate", "image")
            graph.connect("estimate", "preview", "filmstrip", "image")
            if case == "metric":
                add(
                    "3d.camera",
                    "camera",
                    (0, 480),
                    fx=80.0,
                    fy=90.0,
                    cx=15.0,
                    cy=11.0,
                    width=31,
                    height=23,
                )
                add("3d.depth_cloud", "cloud", (920, 400), sample_step=1)
                graph.connect("camera", "camera", "cloud", "camera")
                graph.connect("estimate", "depth", "cloud", "depth")
                graph.connect("source", "image", "cloud", "image")
            if case in ("selected", "empty"):
                add(
                    "region.shape",
                    "region",
                    (0, 360),
                    shape="circle",
                    center=[15, 11] if case == "selected" else [-20, -20],
                    radius=7,
                )
                graph.connect("source", "image", "region", "image")
                graph.connect("region", "region", "estimate", "region")
            if native:
                add(
                    "input.depth_file",
                    "reference",
                    (0, -370),
                    path="assets/tless_depth.png",
                    path_base="project",
                    units_per_sample=0.0001,
                    display_min=0.4,
                    display_max=1.3,
                )
            note = add("layout.comment", "attribution", (0, -850))
            note.name = "Depth sources / units and scale"
            note.node_size = "large"
            note.description = (
                "Original MIT image-dependent arithmetic controls by OpenCVDataflowGUI "
                "contributors. These controls are not learned models. Optional MiDaS v2.1 "
                "Small uses unchanged MIT Intel ISL weights. Real RGB/first-return depth: "
                "T-LESS, T. Hodan et al., WACV 2017, CC BY 4.0; scene 6/frame 2. "
                "https://cmp.felk.cvut.cz/t-less/ ; https://creativecommons.org/licenses/by/4.0/ . "
                "See SOURCES.md and MODEL-LICENSE.txt. Relative inverse values are arbitrary, "
                "not meters. Only explicitly declared metric models/fixed calibrations emit "
                "Depth. Display stretching does not change numbers or identify physical "
                "scale. Reference depths are evaluation-only."
            )
            graph.save(HERE / (case + ".json"))
        finally:
            graph.close()
    truth = {
        "cases": list(CASES),
        "native_cases": list(PREPARATION["VARIANTS"]),
        "source_size": [31, 23],
        "quantity_formula": "q = 1 + 2 * source_red / 255; inverse metric z = 1 / q",
        "hole_rectangle": [13, 8, 4, 4],
        "midas_sha256": PREPARATION["SHA256"],
        "dataset": {
            "name": "T-LESS",
            "license": "CC-BY-4.0",
            "scene": 6,
            "frame": 2,
            "depth_scale_to_m": 0.0001,
        },
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
