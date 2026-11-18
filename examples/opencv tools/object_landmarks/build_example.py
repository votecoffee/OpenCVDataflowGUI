"""Pinned pretrained body-landmark graphs, including explicit model failure limits."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.model_contracts import export_contract, image_preprocessing

HERE = Path(__file__).resolve().parent
SOURCE = HERE
MODEL_SHA = "9d89c599319a18fb7d2e28451a883476164543182bafca5f09eb2cf767ed2f3f"
IMAGE_SHA = "8828e3f97f7a12f2b08a24840dae2636c818e2a5745fc676fc03715b6c8b653f"
NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "left_mouth",
    "right_mouth",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot",
    "right_foot",
]
LINKS = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 7],
    [0, 4],
    [4, 5],
    [5, 6],
    [6, 8],
    [9, 10],
    [11, 12],
    [11, 13],
    [13, 15],
    [15, 17],
    [15, 19],
    [15, 21],
    [17, 19],
    [12, 14],
    [14, 16],
    [16, 18],
    [16, 20],
    [16, 22],
    [18, 20],
    [11, 23],
    [12, 24],
    [23, 24],
    [23, 25],
    [25, 27],
    [27, 29],
    [27, 31],
    [29, 31],
    [24, 26],
    [26, 28],
    [28, 30],
    [28, 32],
    [30, 32],
]
CASES = (
    "portrait",
    "upper_body",
    "require_all",
    "presence_only",
    "show_rejected",
    "object_reject",
    "blank",
    "reordered",
    "missing_pose",
)
PREPARE = {"width": 256, "height": 256, "layout": "nhwc"}
INFERENCE = {
    "active_inputs": 1,
    "input_1_name": "input_1",
    "input_1_shape": [1, 256, 256, 3],
    "output_names": ["Identity", "Identity_1"],
    "engine": "classic",
}


def decoder_settings(case):
    channels = (
        list(range(17))
        if case == "upper_body"
        else [16, 14, 12, 11, 13, 15]
        if case == "reordered"
        else list(range(33))
    )
    order = {channel: index for index, channel in enumerate(channels)}
    links = [[order[a], order[b]] for a, b in LINKS if a in order and b in order]
    return {
        "format": "xy_score",
        "output_name": "Identity",
        "coordinate_layout": "flat_rows",
        "coordinate_columns": 5,
        "coordinate_score_columns": [4] if case == "presence_only" else [3, 4],
        "score_activation": "sigmoid",
        "coordinate_origin": "scaled_indices",
        "channels": channels,
        "names": [NAMES[i] for i in channels],
        "connections": links,
        "object_score_name": "Identity_1",
        "minimum_object_confidence": 1.0 if case == "object_reject" else 0.5,
        "minimum_confidence": 0.8,
        "require_all": case == "require_all",
        "minimum_count": 3 if case == "reordered" else 8,
        "rejected_graphics": "show" if case == "show_rejected" else "hide",
    }


def build(destination=None, source=None):
    destination = HERE if destination is None else Path(destination)
    source = SOURCE if source is None else Path(source)
    assets = destination / "assets"
    if assets.exists() or (destination / "bundle").exists():
        raise FileExistsError("Existing object-landmark assets preserved")
    model, image = source / "bundle/model.onnx", source / "assets/astronaut.png"
    assert hashlib.sha256(model.read_bytes()).hexdigest() == MODEL_SHA
    assert hashlib.sha256(image.read_bytes()).hexdigest() == IMAGE_SHA
    assets.mkdir(parents=True)
    (assets / "astronaut.png").write_bytes(image.read_bytes())
    ok, raw = cv2.imencode(".png", np.zeros((512, 512, 3), np.uint8))
    assert ok
    (assets / "blank.png").write_bytes(raw.tobytes())
    (destination / "LICENSE.opencv-zoo").write_bytes((source / "LICENSE.opencv-zoo").read_bytes())
    registry = builtin_registry()
    prepare = registry.create("ml.image_tensor", parameters=PREPARE)
    infer = registry.create("ml.dnn_multi", parameters={**INFERENCE, "model_path": str(model)})
    try:
        export_contract(
            infer,
            destination / "bundle",
            preprocessing={"input_1": image_preprocessing(prepare.parameters)},
        )
    finally:
        prepare.close()
        infer.close()
    cases = {}
    for case in CASES:
        graph = Graph()

        def add(kind, name, x, y, **parameters):
            block = graph.add(registry.create(kind, block_id=name, parameters=parameters))
            block.position = [x, y]
            return block

        try:
            add(
                "input.file",
                "source",
                0,
                0,
                path="assets/blank.png" if case == "blank" else "assets/astronaut.png",
                color_format="bgr",
                pixel_depth="uint8",
                path_base="project",
            )
            add("ml.image_tensor", "prepare", 390, 0, **PREPARE)
            add("ml.dnn_multi", "infer", 780, 0, **INFERENCE, contract_path="bundle/contract.json")
            decode = add("ml.decode_landmarks", "decode", 1170, 0, **decoder_settings(case))
            decode.node_size = "large"
            decode.exposed_outputs.append("result_graphics")
            add("cv.overlay_result_graphics", "overlay", 1560, 0)
            add("output.filmstrip_image", "filmstrip", 1950, 0)
            for a, o, b, i in (
                ("source", "image", "prepare", "image"),
                ("prepare", "tensor", "infer", "input_1"),
                ("source", "image", "decode", "image"),
                ("infer", "tensors", "decode", "tensors"),
                ("infer", "preprocessing", "decode", "preprocessing"),
                ("decode", "image", "overlay", "image"),
                ("decode", "result_graphics", "overlay", "graphics_1"),
                ("overlay", "image", "filmstrip", "image"),
            ):
                graph.connect(a, o, b, i)
            if case == "missing_pose":
                add("ml.landmark_pose", "pose", 1170, 500, origin_index=16, direction_index=14)
                graph.connect("source", "image", "pose", "image")
                graph.connect("decode", "landmarks", "pose", "landmarks")
            note = add("layout.comment", "attribution", 0, -460)
            note.name, note.node_size = "NASA portrait / OpenCV Zoo MediaPipe Pose", "large"
            note.description = (
                "NASA Eileen Collins portrait, public domain via scikit-image. "
                "Unchanged OpenCV Zoo MediaPipe Pose ONNX at "
                "48b10fa505aeb51fbd127fc60a5c660040e823b7, "
                "Apache 2.0; see LICENSE.opencv-zoo and SOURCES.md. Original adapter graphs by "
                "OpenCVDataflowGUI contributors (MIT). Supplied single-object full view: no person "
                "detector or rotation/crop alignment. Visibility/presence are model estimates, "
                "not ground truth: this cropped portrait can produce confident unsupported body "
                "landmarks. Blank/strict/object-reject and missing pose demonstrate valid "
                "failures. The 2D decoder does not output the model's Z/world coordinates "
                "or auxiliary points."
            )
            graph.save(destination / (case + ".json"))
            cases[case] = {
                "image": "blank.png" if case == "blank" else "astronaut.png",
                "channels": decoder_settings(case)["channels"],
            }
        finally:
            graph.close()
    files = [
        *assets.iterdir(),
        *(destination / "bundle").iterdir(),
        destination / "LICENSE.opencv-zoo",
    ]
    truth = {
        "cases": cases,
        "model_sha256": MODEL_SHA,
        "evidence": "Native parity and declared confidence rules; "
        "not annotated accuracy/visibility truth",
        "asset_sha256": {
            path.relative_to(destination).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
    }
    (destination / "truth.json").write_bytes(
        (json.dumps(truth, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )


if __name__ == "__main__":
    build()
