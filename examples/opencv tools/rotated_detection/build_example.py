"""Original oriented-proposal controls with independent geometric ground truth."""

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
RECTANGLES = [
    [20, 26, 24, 10, 30],
    [20.2, 26, 24, 10, 31],
    [20, 26, 24, 10, 30],
    [48, 44, 18, 6, -35],
    [20, 26, 6, 2, 30],
    [48, 18, 20, 3, 45],
    [48, 18, 20, 3, 135],
]
CASES = {
    "stretch": {"ids": [0, 2, 3, 4, 5, 6]},
    "letterbox": {"ids": [0, 2, 3, 4, 5, 6], "resize": "letterbox"},
    "channels": {"ids": [0, 2, 3, 4, 5, 6], "output": "channels", "layout": "channels_first"},
    "radians": {"ids": [0, 2, 3, 4, 5, 6], "output": "radians", "unit": "radians"},
    "objectness": {"ids": [0, 2, 3, 4, 5, 6], "output": "objectness", "format": "xywha_objectness"},
    "explicit_ids": {
        "ids": [0, 2, 3, 4, 5, 6],
        "output": "explicit",
        "format": "xywha_score_class",
    },
    "native_nms": {"ids": [0, 2, 3, 5, 6], "method": "opencv"},
    "agnostic": {"ids": [0, 3, 4, 5, 6], "agnostic": True},
    "selected": {"ids": [3]},
    "pose_region": {"ids": [0, 2, 3, 4, 5, 6]},
    "count_failure": {"ids": [0, 2, 3, 4, 5, 6], "minimum": 7, "passed": False},
    "empty": {"ids": [], "passed": False},
}


def vertices(rectangle):
    x, y, w, h, theta = rectangle
    a = math.radians(theta)
    return np.array(
        [
            [x + u * math.cos(a) - v * math.sin(a), y + u * math.sin(a) + v * math.cos(a)]
            for u, v in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))
        ]
    )


def model_bytes():
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    n = len(RECTANGLES)
    rects = np.array(RECTANGLES, np.float32)[None]
    weights = np.zeros((3, n * 2), np.float32)
    confidence = np.zeros((3, n), np.float32)
    for i, multiplier in enumerate((0.9, 0.8, 0.85, 0.7, 0.6, 0.65, 0.62)):
        channel, cls = (1, 1) if i == 2 else (0, 0)
        weights[channel, i * 2 + cls] = multiplier
        confidence[channel, i] = multiplier
    constants = {
        "xywh": rects[:, :, :4],
        "angles": rects[:, :, 4:],
        "radians": np.deg2rad(rects[:, :, 4:]),
        "weights": weights,
        "confidence_weights": confidence,
        "classes": np.array([[[0], [0], [1], [0], [0], [0], [0]]], np.float32),
        "ones": np.ones((1, n, 1), np.float32),
        "score_shape": np.array([1, n, 2], np.int64),
        "column_shape": np.array([1, n, 1], np.int64),
    }
    nodes = [
        helper.make_node("GlobalAveragePool", ["image"], ["pool"]),
        helper.make_node("Flatten", ["pool"], ["rgb"], axis=1),
        helper.make_node("MatMul", ["rgb", "weights"], ["flat_scores"]),
        helper.make_node("Reshape", ["flat_scores", "score_shape"], ["scores"]),
        helper.make_node("Concat", ["xywh", "scores", "angles"], ["rows"], axis=2),
        helper.make_node("Transpose", ["rows"], ["channels"], perm=[0, 2, 1]),
        helper.make_node("Concat", ["xywh", "scores", "radians"], ["radian_rows"], axis=2),
        helper.make_node("Concat", ["xywh", "angles", "ones", "scores"], ["objectness"], axis=2),
        helper.make_node("MatMul", ["rgb", "confidence_weights"], ["flat_confidence"]),
        helper.make_node("Reshape", ["flat_confidence", "column_shape"], ["confidence"]),
        helper.make_node(
            "Concat", ["xywh", "angles", "confidence", "classes"], ["explicit"], axis=2
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "original_rotated_proposals",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 64, 64])],
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
            for name, shape in (
                ("rows", [1, n, 7]),
                ("channels", [1, 7, n]),
                ("radian_rows", [1, n, 7]),
                ("objectness", [1, n, 8]),
                ("explicit", [1, n, 7]),
            )
        ],
        [numpy_helper.from_array(value, name) for name, value in constants.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    model.doc_string = (
        "Original MIT fixed rotated proposals with image-dependent scores; no learned weights."
    )
    onnx.checker.check_model(model)
    return model.SerializeToString()


def asset(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw:
        raise FileExistsError("Different original assets preserved; regenerate in a fresh folder")
    path.write_bytes(raw)


def build():
    assets = HERE / "assets"
    asset(assets / "oriented_heads.onnx", model_bytes())
    for mode in ("stretch", "letterbox", "black"):
        image = (
            np.zeros((96, 160, 3), np.uint8)
            if mode == "black"
            else np.full((96, 160, 3), [30, 180, 220], np.uint8)
        )
        if mode != "black":
            for i in (0, 3, 4, 5, 6):
                polygon = vertices(RECTANGLES[i])
                polygon = (
                    (polygon - [0, 13]) * [2.5, 96 / 38]
                    if mode == "letterbox"
                    else polygon * [2.5, 1.5]
                )
                cv2.polylines(image, [np.rint(polygon).astype(np.int32)], True, (230, 230, 230), 1)
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        asset(assets / (mode + ".png"), encoded.tobytes())
    registry, truth = builtin_registry(), {}
    for case, options in CASES.items():
        graph = Graph()
        try:
            source = graph.add(
                registry.create(
                    "input.file",
                    block_id="source",
                    parameters={
                        "path": "assets/"
                        + ("black" if case == "empty" else options.get("resize", "stretch"))
                        + ".png",
                        "path_base": "project",
                    },
                )
            )
            output = options.get("output", "rows")
            output = "radian_rows" if output == "radians" else output
            inference = graph.add(
                registry.create(
                    "ml.dnn",
                    block_id="infer",
                    parameters={
                        "model_path": "assets/oriented_heads.onnx",
                        "width": 64,
                        "height": 64,
                        "input_name": "image",
                        "output_names": [output],
                        "engine": "classic",
                        "resize_mode": options.get("resize", "stretch"),
                    },
                )
            )
            decode = graph.add(
                registry.create(
                    "ml.decode_rotated_detections",
                    block_id="decode",
                    parameters={
                        "output_name": output,
                        "format": options.get("format", "xywh_scores_angle"),
                        "labels": ["part", "fixture"],
                        "layout": options.get("layout", "rows"),
                        "angle_unit": options.get("unit", "degrees"),
                        "nms_method": options.get("method", "polygon_iou"),
                        "class_agnostic_nms": options.get("agnostic", False),
                        "nms_iou": 0.3,
                        "minimum_count": options.get("minimum", 1),
                    },
                )
            )
            decode.exposed_outputs.append("result_graphics")
            overlay = graph.add(registry.create("cv.overlay_result_graphics", block_id="overlay"))
            filmstrip = graph.add(registry.create("output.filmstrip_image", block_id="filmstrip"))
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
            for i, block in enumerate((source, inference, decode, overlay, filmstrip)):
                block.position = [i * 410, 0]
            decode.node_size = "large"
            if case == "selected":
                graph.add(
                    registry.create(
                        "region.shape",
                        block_id="selection",
                        parameters={"rectangle": [90, 50, 60, 40]},
                    )
                )
                graph.connect("source", "image", "selection", "image")
                graph.connect("selection", "region", "decode", "region")
                graph.blocks["selection"].position = [410, 450]
            if case == "pose_region":
                graph.add(
                    registry.create(
                        "collection.sort",
                        block_id="sort",
                        parameters={"metric": "custom", "custom_metric": "score"},
                    )
                )
                graph.add(
                    registry.create(
                        "pose.from_feature", block_id="pose", parameters={"angle_mode": "feature"}
                    )
                )
                region = graph.add(
                    registry.create(
                        "region.pose", block_id="roi", parameters={"rectangle": [-8, -3, 16, 6]}
                    )
                )
                region.exposed_outputs.append("result_graphics")
                for a, o, b, i in (
                    ("decode", "collection", "sort", "collection"),
                    ("sort", "collection", "pose", "features"),
                    ("source", "image", "pose", "image"),
                    ("pose", "pose", "roi", "pose"),
                    ("source", "image", "roi", "image"),
                    ("roi", "result_graphics", "overlay", "graphics_2"),
                ):
                    graph.connect(a, o, b, i)
                for i, name in enumerate(("sort", "pose", "roi")):
                    graph.blocks[name].position = [800 + i * 410, 460]
            note = graph.add(registry.create("layout.comment", block_id="attribution"))
            note.name, note.node_size, note.position = (
                "Original rotated controls - MIT",
                "large",
                [0, -500],
            )
            note.description = (
                "Original MIT images, ONNX controls, graphs and analytic geometry by "
                "OpenCVDataflowGUI contributors. Fixed oriented proposals use RGB image means "
                "for scores; no trained recognition or customer/third-party data. Default polygon "
                "IoU preserves small contained parts; this build's native NMS suppresses them. "
                "Crossing thin rectangles test oriented overlap. Exact source polygons/affine "
                "poses retain non-uniform resize. Count-failure and empty projects intentionally "
                "fail inspection while retaining current outputs. See README/truth.json."
            )
            graph.save(HERE / (case + ".json"))
            truth[case] = {
                **options,
                "passed": options.get("passed", True),
                "network_rectangles": RECTANGLES,
                "source_unclipped_points": {
                    str(i): (
                        (vertices(RECTANGLES[i]) - [0, 13]) * [2.5, 96 / 38]
                        if options.get("resize") == "letterbox"
                        else vertices(RECTANGLES[i]) * [2.5, 1.5]
                    ).tolist()
                    for i in options["ids"]
                },
            }
        finally:
            graph.close()
    (HERE / "truth.json").write_bytes(
        (
            json.dumps(
                {
                    "license": "MIT - original numerical controls",
                    "cases": truth,
                    "asset_sha256": {
                        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(assets.iterdir())
                    },
                },
                indent=2,
            )
            + "\n"
        ).encode()
    )


if __name__ == "__main__":
    build()
