"""Original clean/noisy images, analytic controls and optional pinned DnCNN projects."""

import hashlib
import json
import runpy
import tempfile
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.learned_restoration import preprocessing
from opencv_dataflow.model_contracts import export_contract

HERE = Path(__file__).resolve().parent
PREPARATION = runpy.run_path(str(Path(__file__).with_name("prepare_model.py")))
CASES = (
    "whole",
    "tiles",
    "residual",
    "rgb_alpha",
    "gray16",
    "nhwc",
    "selected",
    "empty",
    "fidelity_alarm",
    *PREPARATION["VARIANTS"],
)


def asset(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw:
        raise FileExistsError("Different existing assets preserved; build into a fresh folder")
    path.write_bytes(raw)


def fixtures():
    reference = np.full((96, 128), 160, np.uint8)
    cv2.circle(reference, (30, 45), 17, 60, -1)
    reference[15:29, 68:115] = 95
    reference[48:77, 78] = 145
    reference[48:77, 90] = 50
    reference[65:71, 100:120] = 80
    reference[65:71, 109] = 160
    reference[82:90, 4:64] = np.tile([156, 164], 30)
    random = np.random.default_rng(842)
    noisy = np.rint(np.clip(reference + random.normal(0, 25, reference.shape), 0, 255)).astype(
        np.uint8
    )
    low = np.rint(np.clip(reference + random.normal(0, 2, reference.shape), 0, 255)).astype(
        np.uint8
    )
    yy, xx = np.indices(reference.shape)
    color = np.dstack((noisy, noisy // 2, 255 - noisy))
    alpha = np.dstack((color, (xx * 2).astype(np.uint8)))
    return {
        "reference": reference,
        "noisy": noisy,
        "low_noise": low,
        "alpha": alpha,
        "gray16": noisy.astype(np.uint16) * 257,
    }


def model_bytes(channels=1, layout="NCHW", residual=False):
    from onnx import TensorProto, checker, helper, numpy_helper

    nodes, into = [], "input"
    if layout == "NHWC":
        nodes.append(helper.make_node("Transpose", ["input"], ["planar"], perm=[0, 3, 1, 2]))
        into = "planar"
    nodes.append(
        helper.make_node(
            "Conv",
            [into, "weights"],
            ["average"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            group=channels,
        )
    )
    out = "average"
    if residual:
        nodes.append(helper.make_node("Sub", [into, out], ["noise"]))
        out = "noise"
    nodes.append(
        helper.make_node(
            "Transpose" if layout == "NHWC" else "Identity",
            [out],
            ["output"],
            **({"perm": [0, 2, 3, 1]} if layout == "NHWC" else {}),
        )
    )
    shape = (
        [1, channels, "height", "width"] if layout == "NCHW" else [1, "height", "width", channels]
    )
    graph = helper.make_graph(
        nodes,
        "original_average_control",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
        [numpy_helper.from_array(np.full((channels, 1, 3, 3), 1 / 9, np.float32), "weights")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    model.doc_string = (
        "Original MIT arithmetic averaging control; no learned weights or accuracy claim."
    )
    checker.check_model(model)
    return model.SerializeToString()


def seal(block, raw, case):
    with tempfile.TemporaryDirectory(prefix="dataflow-restoration-fixture-") as temporary:
        root = Path(temporary)
        model = root / "source.onnx"
        model.write_bytes(raw)
        block.parameters["model_path"], block.parameters["contract_path"] = str(model), ""
        export_contract(
            block, root / "bundle", preprocessing={"input": preprocessing(block.parameters)}
        )
        for path in (root / "bundle").iterdir():
            asset(HERE / "models" / case / path.name, path.read_bytes())
    block.parameters["model_path"] = ""
    block.parameters["contract_path"] = f"models/{case}/contract.json"


def build():
    for name, image in fixtures().items():
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        asset(HERE / "assets" / (name + ".png"), encoded.tobytes())
    registry = builtin_registry()
    for case in CASES:
        graph = Graph()
        try:
            graph.program["assets"] = ["MODEL-LICENSE.txt", "SOURCES.md"]

            def add(kind, identity, position, **parameters):
                block = graph.add(registry.create(kind, block_id=identity, parameters=parameters))
                block.position = list(position)
                return block

            native = case.startswith("native_")
            image = (
                "alpha"
                if case == "rgb_alpha"
                else "gray16"
                if case == "gray16"
                else "low_noise"
                if case == "native_low_noise"
                else "noisy"
            )
            add(
                "input.file",
                "source",
                (0, 0),
                path=f"assets/{image}.png",
                path_base="project",
                color_format="unchanged",
                pixel_depth="unchanged",
            )
            parameters = (
                dict(PREPARATION["PARAMETERS"])
                if native
                else {"tile_width": 32, "tile_height": 28, "halo": 1}
            )
            if native:
                parameters.update(PREPARATION["VARIANTS"][case])
                parameters["contract_path"] = f".local/{case}/contract.json"
            else:
                parameters.update(
                    processing_mode="halo_tiles" if case in ("tiles", "selected") else "whole",
                    output_mode="noise_residual" if case == "residual" else "restored",
                    color_format="rgb" if case == "rgb_alpha" else "gray",
                    input_layout="NHWC" if case == "nhwc" else "NCHW",
                    output_layout="NHWC" if case == "nhwc" else "NCHW",
                )
            block = add("ml.restore_image", "restore", (440, 0), **parameters)
            block.node_size = "large"
            if not native:
                seal(
                    block,
                    model_bytes(
                        3 if case == "rgb_alpha" else 1,
                        parameters["input_layout"],
                        case == "residual",
                    ),
                    case,
                )
            add("output.filmstrip_image", "filmstrip", (900, 0))
            graph.connect("source", "image", "restore", "image")
            graph.connect("restore", "image", "filmstrip", "image")
            if case not in ("rgb_alpha", "gray16"):
                add(
                    "input.file",
                    "reference",
                    (0, -340),
                    path="assets/reference.png",
                    path_base="project",
                    color_format="gray",
                    pixel_depth="uint8",
                )
                for identity, source_id in (
                    ("input_error", "source"),
                    ("restored_error", "restore"),
                ):
                    add(
                        "inspect.image_quality",
                        identity,
                        (920, -350 if source_id == "source" else 300),
                        metric="mse",
                        check_threshold=(case == "fidelity_alarm" or case == "native_low_noise")
                        and source_id == "restore",
                        threshold=5.0 if case == "native_low_noise" else 0.0,
                    )
                    graph.connect(source_id, "image", identity, "image_a")
                    graph.connect("reference", "image", identity, "image_b")
            if case in ("selected", "empty"):
                add(
                    "region.shape",
                    "region",
                    (0, 360),
                    shape="rectangle",
                    rectangle=[24, 24, 50, 40] if case == "selected" else [-20, -20, 1, 1],
                )
                graph.connect("source", "image", "region", "image")
                graph.connect("region", "region", "restore", "region")
            if case in ("native_tiles", "native_no_halo"):
                p = {
                    **PREPARATION["PARAMETERS"],
                    "contract_path": ".local/native_whole/contract.json",
                }
                add("ml.restore_image", "whole_reference", (450, 460), **p)
                add(
                    "inspect.image_quality",
                    "seam_error",
                    (920, 680),
                    metric="max_absolute_error",
                    check_threshold=True,
                    threshold=1.0,
                )
                graph.connect("source", "image", "whole_reference", "image")
                graph.connect("restore", "image", "seam_error", "image_a")
                graph.connect("whole_reference", "image", "seam_error", "image_b")
            note = add("layout.comment", "attribution", (0, -850))
            note.name = "Restoration sources / inspection fidelity"
            note.node_size = "large"
            note.description = (
                "Original MIT clean/noisy geometric and arithmetic controls by "
                "OpenCVDataflowGUI contributors. Averaging fixtures are not learned denoisers. "
                "Optional native examples use unchanged MIT Kai Zhang KAIR DnCNN-25 weights, "
                "release v1.0, with an explicit deterministic ONNX export. See SOURCES.md and "
                "MODEL-LICENSE.txt. Clean data remains an independent reference. Pixel removal, "
                "false detail and halo seams are measured, not assumed safe. Passing inference "
                "does not imply passing inspection. Prepare optional models explicitly."
            )
            graph.save(HERE / (case + ".json"))
        finally:
            graph.close()
    truth = {
        "cases": list(CASES),
        "native_cases": list(PREPARATION["VARIANTS"]),
        "source_size": [128, 96],
        "noise_sigma_samples": 25,
        "low_noise_sigma_samples": 2,
        "flat_roi": [4, 4, 24, 16],
        "fine_texture_roi": [4, 82, 60, 8],
        "faint_crack_roi": [75, 48, 7, 29],
        "strong_crack_roi": [87, 48, 7, 29],
        "circle_center": [30, 45],
        "circle_radius": 17,
        "native_weights_sha256": PREPARATION["WEIGHTS_SHA256"],
        "native_onnx_sha256": PREPARATION["ONNX_SHA256"],
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
