"""Original pixel-dependent ONNX fixture and seam-crossing tile projects (MIT)."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
NAMES = ["background", "red part", "green part", "blue part"]
CASES = {
    "probability_mean": {},
    "feather": {"weighting": "feather"},
    "label_vote": {"mode": "label_vote"},
    "resized_tile": {"resized": True},
    "selected_roi": {"roi": True},
    "coverage_failure": {"gap": True, "passed": False},
    "failed_tile": {"failed_tile": True, "passed": False},
    "area_failure": {"maximum_fraction": 0.01, "passed": False},
    "empty": {"empty": True, "passed": False},
}


def model_bytes():
    from onnx import TensorProto, checker, helper, numpy_helper

    weights = np.zeros((4, 3, 1, 1), np.float32)
    for index in range(3):
        weights[index + 1, index, 0, 0] = 4
    graph = helper.make_graph(
        [helper.make_node("Conv", ["image", "weights", "bias"], ["logits"], kernel_shape=[1, 1])],
        "original_pixel_color_control",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, "height", "width"])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 4, "height", "width"])],
        [
            numpy_helper.from_array(weights, "weights"),
            numpy_helper.from_array(np.array([1, 0, 0, 0], np.float32), "bias"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    model.doc_string = "Original MIT pixel arithmetic, no learned weights or recognition claim."
    checker.check_model(model)
    return model.SerializeToString()


def asset(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw:
        raise FileExistsError("Different existing asset preserved; generate into a fresh folder")
    path.write_bytes(raw)


def reference(image):
    """Independent class equations; no native inference, decoder or fusion calls."""
    rgb = image[..., ::-1].astype(np.float64) / 255
    logits = np.concatenate((np.ones((*image.shape[:2], 1)), 4 * rgb), axis=2)
    return logits.argmax(axis=2).astype(np.int32)


def build():
    image = np.zeros((32, 96, 3), np.uint8)
    image[6:14, 20:76] = [0, 0, 255]  # crosses both overlap boundaries
    image[18:28, 40:56] = [0, 255, 0]
    image[18:28, 72:90] = [255, 0, 0]
    for name, value in (("parts", image), ("black", np.zeros_like(image))):
        ok, raw = cv2.imencode(".png", value)
        assert ok
        asset(HERE / "assets" / (name + ".png"), raw.tobytes())
    asset(HERE / "assets/pixel_color.onnx", model_bytes())
    truth = reference(image)
    # Lossless signed labels travel as class ID +1 in an openly inspectable PNG.
    ok, raw = cv2.imencode(".png", (truth + 1).astype(np.uint8))
    assert ok
    asset(HERE / "assets/truth_labels_plus_one.png", raw.tobytes())
    registry = builtin_registry()
    for name, options in CASES.items():
        graph = Graph()
        try:
            graph.set_branch_workers(2)

            def add(kind, identity, pos, **parameters):
                block = graph.add(registry.create(kind, block_id=identity, parameters=parameters))
                block.position = list(pos)
                return block

            add(
                "input.file",
                "source",
                (0, 0),
                path="assets/black.png" if options.get("empty") else "assets/parts.png",
                path_base="project",
            )

            def infer(source, suffix, width, height, pos, export=False):
                add(
                    "ml.dnn",
                    "infer_" + suffix,
                    pos,
                    model_path="assets/pixel_color.onnx",
                    input_name="image",
                    output_names=["logits"],
                    engine="classic",
                    width=width,
                    height=height,
                )
                decode = add(
                    "ml.decode_segmentation",
                    "decode_" + suffix,
                    (pos[0] + 400, pos[1]),
                    output_name="logits",
                    labels=NAMES,
                    selected_class_ids=[1, 2, 3],
                    minimum_pixels=10000 if options.get("failed_tile") and suffix == "right" else 0,
                    export_segmentation=export,
                )
                for a, o, b, i in (
                    (source, "image", "infer_" + suffix, "image"),
                    (source, "image", decode.id, "image"),
                    ("infer_" + suffix, "tensors", decode.id, "tensors"),
                    ("infer_" + suffix, "preprocessing", decode.id, "preprocessing"),
                ):
                    graph.connect(a, o, b, i)

            for index, (side, x) in enumerate(
                (("left", 0), ("right", 48 if options.get("gap") else 32))
            ):
                width = 32 if options.get("gap") else 64
                add("cv.crop", side, (400, index * 500), x=x, y=0, width=width, height=32)
                graph.connect("source", "image", side, "image")
                source, height = side, 32
                if options.get("resized") and side == "right":
                    width, height = 32, 16
                    add("cv.resize", "resize", (400, 750), width=width, height=height)
                    graph.connect(side, "image", "resize", "image")
                    source = "resize"
                infer(source, side, width, height, (800, index * 500), export=True)
            infer("source", "full", 96, 32, (400, 1100))
            add("flow.join", "join", (1600, 400))
            for index, side in enumerate(("left", "right"), 1):
                graph.connect("decode_" + side, "segmentation", "join", f"branch_{index}")
            fuse = add(
                "ml.fuse_tile_segmentation",
                "fuse",
                (2000, 0),
                expected_tiles=2,
                selected_class_ids=[1, 2, 3],
                mode=options.get("mode", "probability_mean"),
                weighting=options.get("weighting", "uniform"),
                feather_width=8,
                maximum_fraction=options.get("maximum_fraction", 1),
            )
            fuse.node_size = "large"
            fuse.exposed_outputs.append("result_graphics")
            add("cv.overlay_result_graphics", "overlay", (2450, 0))
            add("output.filmstrip_image", "filmstrip", (2850, 0))
            for a, o, b, i in (
                ("source", "image", "fuse", "image"),
                ("join", "group", "fuse", "tiles"),
                ("fuse", "image", "overlay", "image"),
                ("fuse", "result_graphics", "overlay", "graphics_1"),
                ("overlay", "image", "filmstrip", "image"),
            ):
                graph.connect(a, o, b, i)
            if options.get("roi"):
                add("region.shape", "roi", (1600, -350), rectangle=[20, 4, 56, 12])
                graph.connect("source", "image", "roi", "image")
                graph.connect("roi", "region", "fuse", "region")
            note = add("layout.comment", "attribution", (0, -500))
            note.name, note.node_size = "Original MIT seam-crossing controls", "large"
            note.description = (
                "Original MIT synthetic images, pixel-dependent ONNX equations and independent "
                "labels by OpenCVDataflowGUI contributors. No pretrained model or customer images. "
                "Red part crosses both overlap boundaries. Full decode is an independent "
                "whole-image execution. Coverage, failed tile, area and empty controls "
                "intentionally fail. Resized tiles use "
                "bilinear probabilities and may change edge pixels. See README.md."
            )
            graph.save(HERE / (name + ".json"))
        finally:
            graph.close()
    asset(
        HERE / "truth.json",
        (
            json.dumps(
                {
                    "license": "MIT",
                    "cases": CASES,
                    "class_names": NAMES,
                    "class_pixels": [int((truth == i).sum()) for i in range(4)],
                    "asset_sha256": {
                        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted((HERE / "assets").iterdir())
                    },
                },
                indent=2,
            )
            + "\n"
        ).encode(),
    )


if __name__ == "__main__":
    build()
