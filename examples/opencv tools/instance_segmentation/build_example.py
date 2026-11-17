"""Original image-dependent instance heads and independently calculated mask pixels."""

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
BOXES = np.array(
    [[10, 12, 16, 16], [10, 12, 16, 16], [16, 12, 16, 16], [1, 27, 10, 12]], np.float32
)
CASES = {
    "roi": {},
    "letterbox": {"resize": "letterbox"},
    "per_class": {"mask_mode": "roi_per_class", "mask_output": "class_masks"},
    "channels_last": {"mask_layout": "NHWC", "mask_output": "channels_last"},
    "logits_after": {
        "mask_output": "mask_logits",
        "mask_activation": "sigmoid",
        "activation_order": "after_resampling",
        "mask_interpolation": "linear",
    },
    "prototypes": {
        "mask_mode": "prototypes",
        "mask_output": "prototypes",
        "output": "packed",
        "mask_activation": "sigmoid",
        "activation_order": "after_resampling",
    },
    "prototype_channels": {
        "mask_mode": "prototypes",
        "mask_output": "prototype_hwc",
        "output": "packed_channels",
        "layout": "channels_first",
        "mask_layout": "NHWC",
        "mask_activation": "sigmoid",
    },
    "prototype_no_crop": {
        "mask_mode": "prototypes",
        "mask_output": "prototypes",
        "output": "packed",
        "mask_activation": "sigmoid",
        "crop_to_box": False,
    },
    "separate_heads": {"separate": True},
    "selected_roi": {"selected": True},
    "empty": {"empty": True, "passed": False},
    "count_failure": {"minimum_count": 4, "passed": False},
}


def templates():
    masks = np.full((4, 1, 8, 8), 0.1, np.float32)
    masks[0, 0, :, :6] = 0.9
    masks[0, 0, 3:5, 2:4] = 0.1
    masks[1] = 0.9
    masks[2] = 0.9
    masks[2, 0, 2:5, 2:5] = 0.1
    masks[3, 0, :, 4:] = 0.9
    classes = np.zeros((4, 2, 8, 8), np.float32)
    classes[[0, 1, 3], 0] = masks[[0, 1, 3], 0]
    classes[2, 1] = masks[2, 0]
    return masks, classes


def model_bytes():
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    masks, classes = templates()
    weights = np.zeros((3, 8), np.float32)
    confidence = np.zeros((3, 4), np.float32)
    for i, factor in enumerate((0.95, 0.8, 0.85, 0.7)):
        channel, cls = (1, 1) if i == 2 else (0, 0)
        weights[channel, i * 2 + cls], confidence[channel, i] = factor, factor
    xyxy = BOXES.copy()
    xyxy[:, :2] -= xyxy[:, 2:] / 2
    xyxy[:, 2:] += xyxy[:, :2]
    values = {
        "xywh": BOXES[None],
        "xyxy_values": xyxy,
        "score_weights": weights,
        "confidence_weights": confidence,
        "gain_weights": np.array([[0.6], [0], [0]], np.float32),
        "gain_bias": np.array([[0.4]], np.float32),
        "zero_weights": np.zeros((3, 1), np.float32),
        "mask_template": masks,
        "class_template": classes,
        "logit_template": np.where(masks > 0.5, 2, -2).astype(np.float32),
        "coefficients": np.array([[[1, 0.8], [-1, 0], [-1, 1.4], [0, 2]]], np.float32),
        "proto_x": np.tile(np.linspace(-2, 2, 8), (8, 1)).astype(np.float32)[None, None],
        "proto_one": np.ones((1, 1, 8, 8), np.float32),
        "class_values": np.array([0, 0, 1, 0], np.float32),
        "score_shape": np.array([1, 4, 2], np.int64),
        "scalar_shape": np.array([1, 1, 1, 1], np.int64),
        "vector_shape": np.array([4], np.int64),
        "zero_shape": np.array([1], np.int64),
        "roi_repeats": np.array([4, 1, 8, 8], np.int64),
        "class_repeats": np.array([4, 2, 8, 8], np.int64),
        "proto_repeats": np.array([1, 1, 8, 8], np.int64),
        "box_repeats": np.array([4, 4], np.int64),
        "label_repeats": np.array([4], np.int64),
    }
    nodes = [
        helper.make_node("GlobalAveragePool", ["image"], ["pool"]),
        helper.make_node("Flatten", ["pool"], ["rgb"], axis=1),
        helper.make_node("MatMul", ["rgb", "score_weights"], ["flat_scores"]),
        helper.make_node("Reshape", ["flat_scores", "score_shape"], ["scores_3d"]),
        helper.make_node("Concat", ["xywh", "scores_3d"], ["rows"], axis=2),
        helper.make_node("Concat", ["rows", "coefficients"], ["packed"], axis=2),
        helper.make_node("Transpose", ["packed"], ["packed_channels"], perm=[0, 2, 1]),
        helper.make_node("MatMul", ["rgb", "gain_weights"], ["gain_product"]),
        helper.make_node("Add", ["gain_product", "gain_bias"], ["gain"]),
        helper.make_node("Reshape", ["gain", "scalar_shape"], ["gain4"]),
        helper.make_node("Tile", ["gain4", "roi_repeats"], ["roi_gain"]),
        helper.make_node("Tile", ["gain4", "class_repeats"], ["class_gain"]),
        helper.make_node("Tile", ["gain4", "proto_repeats"], ["proto_gain"]),
        helper.make_node("Mul", ["mask_template", "roi_gain"], ["masks"]),
        helper.make_node("Mul", ["class_template", "class_gain"], ["class_masks"]),
        helper.make_node("Mul", ["logit_template", "roi_gain"], ["mask_logits"]),
        helper.make_node("Transpose", ["masks"], ["channels_last"], perm=[0, 2, 3, 1]),
        helper.make_node("MatMul", ["rgb", "zero_weights"], ["zero"]),
        helper.make_node("Reshape", ["zero", "scalar_shape"], ["zero4"]),
        helper.make_node("Tile", ["zero4", "proto_repeats"], ["proto_zero"]),
        helper.make_node("Add", ["proto_x", "proto_zero"], ["basis_x"]),
        helper.make_node("Mul", ["proto_one", "proto_gain"], ["basis_bias"]),
        helper.make_node("Concat", ["basis_x", "basis_bias"], ["prototypes"], axis=1),
        helper.make_node("Transpose", ["prototypes"], ["prototype_hwc"], perm=[0, 2, 3, 1]),
        helper.make_node("Tile", ["zero", "box_repeats"], ["box_zero"]),
        helper.make_node("Add", ["xyxy_values", "box_zero"], ["xyxy"]),
        helper.make_node("Reshape", ["zero", "zero_shape"], ["zero_vector"]),
        helper.make_node("Tile", ["zero_vector", "label_repeats"], ["label_zero"]),
        helper.make_node("Add", ["class_values", "label_zero"], ["labels"]),
        helper.make_node("MatMul", ["rgb", "confidence_weights"], ["confidence"]),
        helper.make_node("Reshape", ["confidence", "vector_shape"], ["scores"]),
    ]
    shapes = {
        "rows": [1, 4, 6],
        "packed": [1, 4, 8],
        "packed_channels": [1, 8, 4],
        "masks": [4, 1, 8, 8],
        "class_masks": [4, 2, 8, 8],
        "mask_logits": [4, 1, 8, 8],
        "channels_last": [4, 8, 8, 1],
        "prototypes": [1, 2, 8, 8],
        "prototype_hwc": [1, 8, 8, 2],
        "xyxy": [4, 4],
        "labels": [4],
        "scores": [4],
    }
    graph = helper.make_graph(
        nodes,
        "original_instance_heads",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 32, 32])],
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
            for name, shape in shapes.items()
        ],
        [numpy_helper.from_array(value, name) for name, value in values.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    model.doc_string = (
        "Original MIT fixed proposals/mask templates with image-dependent scores/gain. "
        "No learned weights."
    )
    onnx.checker.check_model(model)
    return model.SerializeToString()


def asset(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw:
        raise FileExistsError("Different existing assets preserved; regenerate in a fresh folder")
    path.write_bytes(raw)


def sample(plane, x, y, linear):
    if not linear:
        return float(plane[min(int(y), 7), min(int(x), 7)])
    xx, yy = (
        round(float(np.float32(x - 0.5)) * 32) / 32,
        round(float(np.float32(y - 0.5)) * 32) / 32,
    )
    ix, iy = math.floor(xx), math.floor(yy)
    fx, fy = xx - ix, yy - iy
    return sum(
        weight * float(plane[min(max(row, 0), 7), min(max(col, 0), 7)])
        for col, row, weight in (
            (ix, iy, (1 - fx) * (1 - fy)),
            (ix + 1, iy, fx * (1 - fy)),
            (ix, iy + 1, (1 - fx) * fy),
            (ix + 1, iy + 1, fx * fy),
        )
    )


def reference(image, options):
    rgb = image[..., ::-1].astype(np.float32)
    sx, sy, tx, ty = 32 / 80, 32 / 48, 0, 0
    if options.get("resize") == "letterbox":
        network = np.zeros((32, 32, 3), np.float32)
        network[6:25] = cv2.resize(rgb, (32, 19))
        sy, ty = 19 / 48, 6
    else:
        network = cv2.resize(rgb, (32, 32))
    means = (network * np.float32(1 / 255)).mean(axis=(0, 1), dtype=np.float64)
    gain = np.float32(np.float32(means[0]) * np.float32(0.6) + np.float32(0.4))
    scores = [
        float(np.float32(means[0 if i != 2 else 1] * factor))
        for i, factor in enumerate((0.95, 0.8, 0.85, 0.7))
    ]
    # Rows 0 and 1 have identical same-class boxes; native box NMS retains row 0.
    ids = sorted((i for i in (0, 2, 3) if scores[i] > 0.25), key=lambda i: -scores[i])
    template, _ = templates()
    coefficients = [[1, 0.8], [-1, 0], [-1, 1.4], [0, 2]]
    prototype = np.stack((np.tile(np.linspace(-2, 2, 8), (8, 1)), np.full((8, 8), gain))).astype(
        np.float32
    )
    result = {}
    for identity in ids:
        mode = options.get("mask_mode", "roi")
        proto = mode == "prototypes"
        activation = options.get("mask_activation", "probability")
        after = options.get("activation_order") == "after_resampling"
        if proto:
            plane = sum(
                float(np.float32(coefficients[identity][k])) * prototype[k].astype(np.float64)
                for k in range(2)
            )
        elif activation == "sigmoid":
            plane = (np.where(template[identity, 0] > 0.5, 2, -2).astype(np.float32) * gain).astype(
                np.float64
            )
        else:
            plane = (template[identity, 0] * gain).astype(np.float64)
        if activation == "sigmoid" and not after:
            plane = 1 / (1 + np.exp(-plane))
        mask = np.zeros((48, 80), np.uint8)
        cx, cy, bw, bh = BOXES[identity]
        left, top = float(cx - bw / 2), float(cy - bh / 2)
        for y in range(48):
            for x in range(80):
                nx, ny = (x + 0.5) * sx + tx, (y + 0.5) * sy + ty
                if not proto or options.get("crop_to_box", True):
                    if not (left <= nx < left + bw and top <= ny < top + bh):
                        continue
                if options.get("selected") and not (20 <= x < 60 and 5 <= y < 40):
                    continue
                xx, yy = (nx / 4, ny / 4) if proto else ((nx - left) * 8 / bw, (ny - top) * 8 / bh)
                value = sample(plane, xx, yy, options.get("mask_interpolation") == "linear")
                if activation == "sigmoid" and after:
                    value = 1 / (1 + math.exp(-value))
                mask[y, x] = 255 if value >= 0.5 else 0
        if mask.any():
            result[str(identity)] = {
                "packed_bits_hex": np.packbits(mask != 0).tobytes().hex(),
                "pixels": int(np.count_nonzero(mask)),
                "score": scores[identity],
            }
    return result


def build():
    assets = HERE / "assets"
    asset(assets / "instance_heads.onnx", model_bytes())
    image = np.full((48, 80, 3), [40, 180, 230], np.uint8)
    cv2.rectangle(image, (6, 7), (44, 33), (150, 180, 220), 1)
    cv2.circle(image, (39, 19), 12, (200, 230, 250), 1)
    for name, pixels in (("parts.png", image), ("empty.png", np.zeros_like(image))):
        ok, encoded = cv2.imencode(".png", pixels)
        assert ok
        asset(assets / name, encoded.tobytes())
    registry, truth = builtin_registry(), {}
    for case, options in CASES.items():
        graph = Graph()
        try:
            source = graph.add(
                registry.create(
                    "input.file",
                    block_id="source",
                    parameters={
                        "path": "assets/empty.png" if options.get("empty") else "assets/parts.png",
                        "path_base": "project",
                    },
                )
            )
            output, mask_name = options.get("output", "rows"), options.get("mask_output", "masks")
            inference = graph.add(
                registry.create(
                    "ml.dnn",
                    block_id="infer",
                    parameters={
                        "model_path": "assets/instance_heads.onnx",
                        "width": 32,
                        "height": 32,
                        "input_name": "image",
                        "output_names": ["xyxy", "scores", "labels", mask_name]
                        if options.get("separate")
                        else [output, mask_name],
                        "engine": "classic",
                        "resize_mode": options.get("resize", "stretch"),
                    },
                )
            )
            parameters = {
                "output_name": "packed_boxes" if options.get("separate") else output,
                "mask_output_name": mask_name,
                "labels": ["part A", "part B"],
                "mask_interpolation": options.get("mask_interpolation", "nearest"),
                "minimum_count": options.get("minimum_count", 1),
            }
            for key in (
                "mask_mode",
                "mask_layout",
                "mask_activation",
                "activation_order",
                "layout",
                "crop_to_box",
            ):
                if key in options:
                    parameters[key] = options[key]
            if options.get("separate"):
                parameters["format"] = "xyxy_score_class"
            decode = graph.add(
                registry.create("ml.decode_instances", block_id="decode", parameters=parameters)
            )
            decode.exposed_outputs.append("result_graphics")
            overlay = graph.add(registry.create("cv.overlay_result_graphics", block_id="overlay"))
            filmstrip = graph.add(registry.create("output.filmstrip_image", block_id="filmstrip"))
            for a, o, b, i in (
                ("source", "image", "infer", "image"),
                ("source", "image", "decode", "image"),
                ("infer", "preprocessing", "decode", "preprocessing"),
                ("decode", "image", "overlay", "image"),
                ("decode", "result_graphics", "overlay", "graphics_1"),
                ("overlay", "image", "filmstrip", "image"),
            ):
                graph.connect(a, o, b, i)
            if options.get("separate"):
                pack = graph.add(
                    registry.create(
                        "ml.pack_detection_heads",
                        block_id="pack",
                        parameters={"boxes_output": "xyxy"},
                    )
                )
                pack.position = [410, 400]
                graph.connect("infer", "tensors", "pack", "tensors")
                graph.connect("pack", "tensors", "decode", "tensors")
            else:
                graph.connect("infer", "tensors", "decode", "tensors")
            if options.get("selected"):
                roi = graph.add(
                    registry.create(
                        "region.shape", block_id="roi", parameters={"rectangle": [20, 5, 40, 35]}
                    )
                )
                sort = graph.add(
                    registry.create(
                        "collection.sort",
                        block_id="sort",
                        parameters={"metric": "custom", "custom_metric": "mask_area"},
                    )
                )
                select = graph.add(registry.create("ml.select_instance_mask", block_id="select"))
                blobs = graph.add(registry.create("inspect.blobs", block_id="blobs"))
                for a, o, b, i in (
                    ("source", "image", "roi", "image"),
                    ("roi", "region", "decode", "region"),
                    ("decode", "collection", "sort", "collection"),
                    ("sort", "collection", "select", "collection"),
                    ("source", "image", "select", "image"),
                    ("select", "mask", "blobs", "image"),
                ):
                    graph.connect(a, o, b, i)
                for j, block in enumerate((roi, sort, select, blobs)):
                    block.position = [410 + j * 410, 450]
            for j, block in enumerate((source, inference, decode, overlay, filmstrip)):
                block.position = [j * 410, 0]
            decode.node_size = "large"
            note = graph.add(registry.create("layout.comment", block_id="attribution"))
            note.name, note.node_size, note.position = (
                "Original instance-mask controls - MIT",
                "large",
                [0, -500],
            )
            note.description = (
                "Original MIT synthetic images, image-dependent arithmetic ONNX and independent "
                "mask references by OpenCVDataflowGUI contributors. Fixed proposals and mask "
                "templates are numerical controls, not learned recognition. Per-instance "
                "scanline masks preserve overlap and original row identity after NMS. Clipped "
                "ROI masks keep original box support. Empty/count-failure projects "
                "intentionally fail inspection. See README/truth.json."
            )
            graph.save(HERE / (case + ".json"))
            masks = reference(np.zeros_like(image) if options.get("empty") else image, options)
            truth[case] = {
                **options,
                "passed": options.get("passed", True),
                "count": len(masks),
                "image_size": [80, 48],
                "instances": masks,
            }
        finally:
            graph.close()
    (HERE / "truth.json").write_bytes(
        (
            json.dumps(
                {
                    "license": "MIT - original numerical controls",
                    "encoding": "row-major np.packbits, big bit order; foreground=1",
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
