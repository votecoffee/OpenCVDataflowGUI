"""Original image-dependent ONNX proposal heads and a separately prepared Zoo detector."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
COCO = (
    "person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant "
    "stop_sign parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe "
    "backpack umbrella handbag tie suitcase frisbee skis snowboard sports_ball kite baseball_bat "
    "baseball_glove skateboard surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl "
    "banana apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch "
    "potted_plant bed dining_table toilet tv laptop mouse remote keyboard cell_phone microwave "
    "oven toaster sink refrigerator book clock vase scissors teddy_bear hair_drier toothbrush"
).split()
CASES = {
    "stretch": {"count": 3},
    "letterbox": {"count": 3, "resize": "letterbox"},
    "channels": {"count": 3, "output": "channels", "layout": "channels_first"},
    "objectness": {"count": 3, "output": "objectness", "format": "xywh_objectness"},
    "ssd": {"count": 3, "output": "ssd", "format": "ssd", "units": "normalized", "offset": -1},
    "agnostic": {"count": 2, "agnostic": True},
    "selected": {"count": 1},
    "empty": {"count": 0, "passed": False},
    "count_failure": {"count": 3, "minimum": 4, "passed": False},
}


def model_bytes():
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    boxes = np.array(
        [[[20, 24, 20, 16], [21, 24, 20, 16], [20, 24, 20, 16], [46, 44, 12, 12]]], np.float32
    )
    weights = np.zeros((3, 8), np.float32)
    weights[0, [0, 2, 6]], weights[1, 5] = [0.9, 0.8, 0.6], 0.9
    initial = {
        "boxes": boxes,
        "weights": weights,
        "score_shape": np.array([1, 4, 2], np.int64),
        "ones": np.ones((1, 4, 1), np.float32),
        "norm_xyxy": np.array(
            [[[10, 16, 30, 32], [11, 16, 31, 32], [10, 16, 30, 32], [40, 38, 52, 50]]], np.float32
        )
        / 64,
        "score_weights": np.array([[0.9, 0.8, 0, 0.6], [0, 0, 0.9, 0], [0, 0, 0, 0]], np.float32),
        "score_column": np.array([1, 4, 1], np.int64),
        "image_ids": np.zeros((1, 4, 1), np.float32),
        "class_ids": np.array([[[1], [1], [2], [1]]], np.float32),
        "ssd_shape": np.array([1, 1, 4, 7], np.int64),
    }
    nodes = [
        helper.make_node("GlobalAveragePool", ["image"], ["pool"]),
        helper.make_node("Flatten", ["pool"], ["rgb"], axis=1),
        helper.make_node("MatMul", ["rgb", "weights"], ["flat_scores"]),
        helper.make_node("Reshape", ["flat_scores", "score_shape"], ["class_scores"]),
        helper.make_node("Concat", ["boxes", "class_scores"], ["rows"], axis=2),
        helper.make_node("Transpose", ["rows"], ["channels"], perm=[0, 2, 1]),
        helper.make_node("Concat", ["boxes", "ones", "class_scores"], ["objectness"], axis=2),
        helper.make_node("MatMul", ["rgb", "score_weights"], ["flat_confidence"]),
        helper.make_node("Reshape", ["flat_confidence", "score_column"], ["confidence"]),
        helper.make_node(
            "Concat", ["image_ids", "class_ids", "confidence", "norm_xyxy"], ["ssd_rows"], axis=2
        ),
        helper.make_node("Reshape", ["ssd_rows", "ssd_shape"], ["ssd"]),
    ]
    graph = helper.make_graph(
        nodes,
        "original_image_dependent_proposal_fixture",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 64, 64])],
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
            for name, shape in (
                ("rows", [1, 4, 6]),
                ("channels", [1, 6, 4]),
                ("objectness", [1, 4, 7]),
                ("ssd", [1, 1, 4, 7]),
            )
        ],
        [numpy_helper.from_array(value, name) for name, value in initial.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    model.doc_string = (
        "Original MIT wiring fixture; fixed proposals, image-dependent RGB scores; "
        "not a trained recognizer."
    )
    onnx.checker.check_model(model)
    return model.SerializeToString()


def build():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    data = model_bytes()
    model = assets / "proposal_heads.onnx"
    if model.exists() and model.read_bytes() != data:
        raise FileExistsError("Different existing model preserved; regenerate in a fresh folder")
    model.write_bytes(data)
    for name, bgr in (("color", [30, 180, 220]), ("black", [0, 0, 0])):
        pixels = np.full((80, 160, 3), bgr, np.uint8)
        ok, encoded = cv2.imencode(".png", pixels)
        assert ok
        path = assets / (name + ".png")
        if path.exists() and path.read_bytes() != encoded.tobytes():
            raise FileExistsError(
                "Different existing image preserved; regenerate in a fresh folder"
            )
        path.write_bytes(encoded.tobytes())
    registry = builtin_registry()
    for name, options in {**CASES, "yolox_coffee": {"count": 1}}.items():
        trained = name == "yolox_coffee"
        graph = Graph()
        try:
            source = graph.add(
                registry.create(
                    "input.file",
                    block_id="source",
                    parameters={
                        "path": "../../assets/coffee.png"
                        if trained
                        else "assets/black.png"
                        if name == "empty"
                        else "assets/color.png",
                        "path_base": "project",
                    },
                )
            )
            inference = graph.add(
                registry.create(
                    "ml.dnn",
                    block_id="infer",
                    parameters={
                        "model_path": "../../../.local/models/yolox_s.onnx"
                        if trained
                        else "assets/proposal_heads.onnx",
                        "width": 640 if trained else 64,
                        "height": 640 if trained else 64,
                        "resize_mode": "letterbox" if trained else options.get("resize", "stretch"),
                        "padding_value": 114 if trained else 0,
                        "scale": 1 if trained else 1 / 255,
                        "engine": "classic",
                    },
                )
            )
            decoder = graph.add(
                registry.create(
                    "ml.decode_detections",
                    block_id="decode",
                    parameters={
                        "output_name": "output" if trained else options.get("output", "rows"),
                        "labels": [label.replace("_", " ") for label in COCO]
                        if trained
                        else ["red", "green"],
                        "format": "xywh_objectness"
                        if trained
                        else options.get("format", "xywh_scores"),
                        "grid_decode": "yolox" if trained else "none",
                        "confidence_threshold": 0.35 if trained else 0.25,
                        "layout": options.get("layout", "rows"),
                        "coordinate_units": options.get("units", "network_pixels"),
                        "class_id_offset": options.get("offset", 0),
                        "class_agnostic_nms": options.get("agnostic", False),
                        "minimum_count": options.get("minimum", 1),
                        "accepted_class_ids": [41] if trained else [],
                    },
                )
            )
            decoder.exposed_outputs.append("result_graphics")
            overlay = graph.add(registry.create("cv.overlay_result_graphics", block_id="overlay"))
            filmstrip = graph.add(registry.create("output.filmstrip_image", block_id="filmstrip"))
            for b, pos in zip(
                (source, inference, decoder, overlay, filmstrip),
                ([0, 0], [420, 220], [840, 0], [1260, 0], [1680, 0]),
                strict=True,
            ):
                b.position = pos
            decoder.node_size = "large"
            for a, o, b, i in (
                ("source", "image", "infer", "image"),
                ("source", "image", "decode", "image"),
                ("infer", "tensors", "decode", "tensors"),
                ("infer", "preprocessing", "decode", "preprocessing"),
                ("decode", "image", "overlay", "image"),
                ("decode", "result_graphics", "overlay", "graphics_1"),
                ("overlay", "image", "filmstrip", "image"),
            ):
                graph.connect(a, o, b, i)
            if name == "selected":
                region = graph.add(
                    registry.create(
                        "region.shape",
                        block_id="selection",
                        parameters={"rectangle": [90, 45, 60, 30]},
                    )
                )
                region.position = [420, 580]
                graph.connect("source", "image", "selection", "image")
                graph.connect("selection", "region", "decode", "region")
            note = graph.add(registry.create("layout.comment", block_id="attribution"))
            note.name, note.position, note.node_size = (
                "Detection fixture and source credits",
                [0, -510],
                "large",
            )
            note.description = "Original MIT pipeline by OpenCVDataflowGUI contributors. "
            note.description += (
                "OpenCV Zoo YOLOX-S, Apache 2.0, revision "
                "48b10fa505aeb51fbd127fc60a5c660040e823b7; "
                "Megvii YOLOX and OpenCV Zoo contributors. "
                "Coffee photograph by Rachel Michetti, courtesy of Pikolo Espresso Bar; "
                "distributed by scikit-image under CC0. Original/credits in examples/assets. "
                "Prepare the pinned model explicitly; graph runs never download. "
                if trained
                else "Original MIT control images and ONNX proposal model. Fixed box proposals use "
                "RGB image means for scores. This is a wiring/geometry fixture without learned "
                "recognition. Expected boxes/counts and file hashes are in truth.json. "
            )
            note.description += (
                "Boxes are source pixel-edge rectangles, without measured part orientation."
            )
            graph.save(HERE / (name + ".json"))
        finally:
            graph.close()
    truth = {
        "license": "MIT - original control model/images",
        "model_sha256": hashlib.sha256(data).hexdigest(),
        "cases": CASES,
        "source_size": [160, 80],
        "network_size": [64, 64],
        "network_xyxy": [[10, 16, 30, 32], [11, 16, 31, 32], [10, 16, 30, 32], [40, 38, 52, 50]],
        "asset_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(assets.iterdir())
        },
    }
    (HERE / "truth.json").write_bytes((json.dumps(truth, indent=2) + "\n").encode())


if __name__ == "__main__":
    build()
