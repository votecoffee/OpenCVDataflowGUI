"""Original lossless part sequences and an image-dependent two-channel ONNX mask model."""

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
    "crossing",
    "occlusion",
    "expiry",
    "transparent_overlap",
    "confirmation",
    "high_depth",
    "gray_appearance",
    "decode_budget",
)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def frames(kind):
    count = 11 if kind == "expiry" else 9
    result, truth = [], []
    for index in range(count):
        image = np.zeros((40, 64, 3), np.uint8)
        rx, bx = 6 + 4 * index, 46 - 4 * index
        red, blue = np.zeros((40, 64), bool), np.zeros((40, 64), bool)
        red[12:24, rx : rx + 12] = True
        blue[12:24, bx : bx + 12] = True
        if kind in ("occlusion", "expiry"):
            red[:] = False
            red[26:36, 5:15] = True
            if index in (3, 4) or kind == "expiry" and 2 <= index <= 7:
                blue[:] = False
        if kind == "transparent_overlap":
            image[..., 0][blue] = 220
            image[..., 2][red] = 220
        else:
            blue &= ~red
            image[blue] = [220, 0, 0]
            image[red] = [0, 0, 220]
        result.append(image)
        truth.append({"index": index, "red_runs": runs(red), "blue_runs": runs(blue)})
    return result, truth


def runs(mask):
    records = []
    for y, row in enumerate(mask):
        edges = np.flatnonzero(np.diff(np.pad(row, 1)))
        records.extend(
            [y, int(a), int(b - a)] for a, b in zip(edges[::2], edges[1::2], strict=True)
        )
    return records


def native_parameters():
    return {
        "width": 64,
        "height": 40,
        "color_format": "bgr",
        "input_name": "image",
        "output_names": ["detections", "masks"],
        "mean": [0.0] * 3,
        "std": [1.0] * 3,
        "engine": "classic",
        "receipt_mode": "detailed",
    }


def build(destination=HERE):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    destination = Path(destination)
    assets = destination / "assets"
    if assets.exists():
        raise FileExistsError("Existing example assets preserved")
    assets.mkdir(parents=True)
    sequences = {}
    for kind in ("crossing", "occlusion", "expiry", "transparent_overlap"):
        images, truth = frames(kind)
        writer = cv2.VideoWriter(
            str(assets / (kind + ".avi")), cv2.VideoWriter_fourcc(*"FFV1"), 10, (64, 40)
        )
        assert writer.isOpened()
        try:
            for image in images:
                writer.write(image)
        finally:
            writer.release()
        capture = cv2.VideoCapture(str(assets / (kind + ".avi")))
        try:
            for image in images:
                ok, actual = capture.read()
                assert ok and np.array_equal(image, actual)
        finally:
            capture.release()
        sequences[kind] = truth
    boxes = np.float32([0, 0, 64, 40, 1, 0] * 2)
    nodes = [
        helper.make_node("Gather", ["image", "channels"], ["planes"], axis=1),
        helper.make_node("Transpose", ["planes"], ["masks"], perm=[1, 0, 2, 3]),
        helper.make_node("GlobalAveragePool", ["image"], ["average"]),
        helper.make_node("Flatten", ["average"], ["flat"]),
        helper.make_node("Gemm", ["flat", "weights", "bias"], ["head"]),
        helper.make_node("Reshape", ["head", "shape"], ["detections"]),
    ]
    graph = helper.make_graph(
        nodes,
        "original_color_instance_masks",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 40, 64])],
        [
            helper.make_tensor_value_info("detections", TensorProto.FLOAT, [1, 2, 6]),
            helper.make_tensor_value_info("masks", TensorProto.FLOAT, [2, 1, 40, 64]),
        ],
        [
            numpy_helper.from_array(np.int64([2, 0]), "channels"),
            numpy_helper.from_array(np.zeros((3, 12), np.float32), "weights"),
            numpy_helper.from_array(boxes, "bias"),
            numpy_helper.from_array(np.int64([1, 2, 6]), "shape"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    (assets / "color_masks.onnx").write_bytes(model.SerializeToString())
    registry = builtin_registry()
    native = registry.create(
        "ml.dnn",
        parameters={
            **native_parameters(),
            "model_path": str((assets / "color_masks.onnx").resolve()),
        },
    )
    try:
        export_contract(native, destination / "bundle")
    finally:
        native.close()
    cases = {}
    for case in CASES:
        kind = case if case in sequences else "crossing"
        graph = Graph()
        graph.project_dir = destination

        def add(kind, name, x, y=0, **p):
            b = graph.add(registry.create(kind, block_id=name, parameters=p))
            b.position = [x, y]
            return b

        try:
            add(
                "input.video_file",
                "source",
                0,
                path="assets/" + kind + ".avi",
                path_base="project",
                pixel_depth="uint16" if case == "high_depth" else "uint8",
            )
            add("cv.convert", "working", 350, color_format="bgr", pixel_depth="uint8")
            net = add(
                "ml.dnn",
                "net",
                700,
                **{**native_parameters(), "contract_path": "bundle/contract.json"},
            )
            net.exposed_outputs.append("receipt")
            add(
                "ml.decode_instances",
                "instances",
                1050,
                output_name="detections",
                format="xyxy_score_class",
                labels=["part"],
                nms_enabled=False,
                minimum_count=0,
                mask_output_name="masks",
                mask_interpolation="nearest",
                mask_threshold=0.5,
            )
            parameters = {}
            if case == "confirmation":
                parameters["minimum_observations"] = 2
            if case == "gray_appearance":
                parameters["appearance"] = "gray"
            if case == "transparent_overlap":
                parameters.update(appearance="off")
            if case == "decode_budget":
                parameters["maximum_decoded_pixels"] = 1
            track = add("sequence.track_instances", "track", 1400, **parameters)
            track.exposed_outputs.append("result_graphics")
            add("cv.overlay_result_graphics", "overlay", 1750)
            add("output.filmstrip_image", "filmstrip", 2100)
            add("ml.select_instance_mask", "selected", 1750, 400, selection="id", instance_id=1)
            for a, o, b, i in [
                ("source", "image", "working", "image"),
                ("working", "image", "net", "image"),
                ("source", "image", "instances", "image"),
                ("net", "tensors", "instances", "tensors"),
                ("net", "preprocessing", "instances", "preprocessing"),
                ("source", "image", "track", "image"),
                ("instances", "collection", "track", "instances"),
                ("net", "receipt", "track", "receipt"),
                ("track", "image", "overlay", "image"),
                ("track", "result_graphics", "overlay", "graphics_1"),
                ("overlay", "image", "filmstrip", "image"),
                ("track", "image", "selected", "image"),
                ("track", "tracks", "selected", "collection"),
            ]:
                graph.connect(a, o, b, i)
            note = add("layout.comment", "attribution", 0, -600)
            note.name, note.node_size = "Original instance-mask tracking controls - MIT", "large"
            note.description = (
                "Original OpenCVDataflowGUI synthetic lossless videos, ONNX masks and "
                "visible-mask truth, MIT. "
                "Same-class red/blue objects cross, overlap or disappear. Masks depend on "
                "current image channels; "
                "no pretrained backbone accuracy is claimed. Missing/ambiguous tracks have "
                "report predictions "
                "but no emitted current masks. Expiry intentionally creates a new ID. See "
                "README.md/SOURCES.md."
            )
            graph.save(destination / (case + ".json"))
            cases[case] = {"sequence": kind, "frames": len(sequences[kind])}
        finally:
            graph.close()
    (destination / "truth.json").write_bytes(
        encoded(
            {
                "sequences": sequences,
                "cases": cases,
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
