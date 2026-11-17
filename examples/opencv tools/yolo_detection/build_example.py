"""Reproduce pinned raw/decoded YOLO inspection projects and an attributed photo crop."""

import hashlib
import json
from pathlib import Path

import cv2
from prepare_models import PARAMETERS

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
SOURCE = HERE
PHOTO_SHA256 = "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7"
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
    "raw_coffee": {"count": 1, "passed": True},
    "decoded_rows": {"count": 1, "passed": True, "kind": "decoded"},
    "decoded_channels": {"count": 1, "passed": True, "kind": "decoded", "layout": "channels_first"},
    "portrait": {"count": 1, "passed": True},
    "duplicates": {"count": 8, "passed": True, "nms_enabled": False},
    "class_aware": {"count": 2, "passed": True, "classes": [], "nms_iou": 0.0},
    "class_agnostic": {
        "count": 1,
        "passed": True,
        "classes": [],
        "nms_iou": 0.0,
        "class_agnostic_nms": True,
    },
    "selected_out": {"count": 0, "passed": False},
    "confidence_fail": {"count": 0, "passed": False, "confidence_threshold": 0.95},
    "objectness_fail": {"count": 0, "passed": False, "objectness_threshold": 1.0},
    "empty": {"count": 0, "passed": False},
    "wrong_layout": {"count": None, "passed": False, "kind": "decoded"},
}


def build():
    if any((HERE / (name + ".json")).exists() for name in CASES):
        raise FileExistsError("Existing YOLO projects preserved; generate into a fresh folder")
    photo = (SOURCE / "../../assets/coffee.png").read_bytes()
    if hashlib.sha256(photo).hexdigest() != PHOTO_SHA256:
        raise ValueError("Source photo changed; review its attribution and annotation first")
    image = cv2.imread(str(SOURCE / "../../assets/coffee.png"))
    ok, encoded = cv2.imencode(".png", image[:399, 145:454])
    assert ok
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    portrait = assets / "coffee_portrait.png"
    if portrait.exists() and portrait.read_bytes() != encoded.tobytes():
        raise FileExistsError("Different existing portrait preserved")
    portrait.write_bytes(encoded.tobytes())
    registry = builtin_registry()
    for name, options in CASES.items():
        kind = options.get("kind", "raw")
        photo_path = (
            "assets/coffee_portrait.png"
            if name == "portrait"
            else "../dnn_detection/assets/black.png"
            if name == "empty"
            else "../../assets/coffee.png"
        )
        known = (
            {"size": [160, 80], "objects": []}
            if name == "empty"
            else {
                "size": [309, 399] if name == "portrait" else [600, 400],
                "objects": [
                    {
                        "id": "cup",
                        "label": "cup",
                        "bbox": [25, 18, 245, 292] if name == "portrait" else [170, 18, 245, 292],
                    }
                ],
            }
        )
        if name in ("class_aware", "class_agnostic"):
            # The wooden table extends to all image edges; this reference comes
            # from the photograph's frame, not from the model's predicted box.
            known["objects"].append(
                {"id": "table", "label": "dining table", "bbox": [0, 0, 600, 400]}
            )
        graph = Graph()
        try:
            graph.program["assets"] = ["SOURCES.md", "MODEL-LICENSE.txt"]
            graph.add(
                registry.create(
                    "input.file",
                    block_id="source",
                    parameters={"path": photo_path, "path_base": "project"},
                )
            )
            infer = graph.add(
                registry.create(
                    "ml.dnn",
                    block_id="infer",
                    parameters={
                        **PARAMETERS[kind],
                        "contract_path": f".local/yolox_{kind}/contract.json",
                    },
                )
            )
            infer.exposed_outputs.append("receipt")
            layout = options.get("layout", "rows")
            head = (
                "output"
                if kind == "raw"
                else (
                    "decoded_channels"
                    if layout == "channels_first" or name == "wrong_layout"
                    else "decoded_rows"
                )
            )
            decoder = graph.add(
                registry.create(
                    "ml.decode_detections",
                    block_id="decode",
                    parameters={
                        "format": "xywh_objectness" if kind == "raw" else "xywh_scores",
                        "grid_decode": "yolox" if kind == "raw" else "none",
                        "layout": layout,
                        "output_name": head,
                        "labels": [label.replace("_", " ") for label in COCO],
                        "accepted_class_ids": options.get("classes", [41]),
                        **{
                            key: options[key]
                            for key in (
                                "nms_enabled",
                                "nms_iou",
                                "class_agnostic_nms",
                                "objectness_threshold",
                            )
                            if key in options
                        },
                        "confidence_threshold": options.get("confidence_threshold", 0.35),
                    },
                )
            )
            decoder.exposed_outputs.append("result_graphics")
            decoder.node_size = "large"
            graph.add(registry.create("cv.overlay_result_graphics", block_id="overlay"))
            graph.add(registry.create("output.filmstrip_image", block_id="filmstrip"))
            for index, block in enumerate(graph.blocks.values()):
                block.position = [index * 420, 180 if block.id == "infer" else 0]
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
            if name == "selected_out":
                region = graph.add(
                    registry.create(
                        "region.shape",
                        block_id="selection",
                        parameters={"rectangle": [460, 0, 130, 100]},
                    )
                )
                region.position = [420, 650]
                graph.connect("source", "image", "selection", "image")
                graph.connect("selection", "region", "decode", "region")
            note = graph.add(registry.create("layout.comment", block_id="attribution"))
            note.name, note.position, note.node_size = (
                "YOLO model and photograph credits",
                [0, -510],
                "large",
            )
            note.description = (
                "Coffee photograph: Rachel Michetti, courtesy of Pikolo Espresso Bar; "
                "scikit-image CC0. Portrait is a crop [145,0,309,399] from the same image. "
                "YOLOX-S: Megvii YOLOX / OpenCV Zoo contributors, Apache 2.0, revision "
                "48b10fa505aeb51fbd127fc60a5c660040e823b7. Raw weights unchanged. "
                "Decoded export is modified locally by OpenCVDataflowGUI contributors: "
                "grid/size decode, objectness-times-class scores and axis transpose appended; "
                "learned weights unchanged. It is derived YOLOX, not a YOLOv8 model. "
                "Original MIT pipeline and independent hand-marked cup annotation. "
                "Prepare models explicitly; no runtime downloads. Source boxes use exact rounded "
                "centered letterboxing. Cup is hand marked; class controls also label the "
                "full-image wooden table extent. This is not an accuracy benchmark. "
                "See README.md and SOURCES.md."
            )
            source_asset = HERE / photo_path if name == "portrait" else SOURCE / photo_path
            graph.validation_sets = [
                {
                    "id": "cup",
                    "name": "Hand-marked cup / detector controls",
                    "source": "source",
                    "state_mode": "isolated",
                    "cases": [
                        {
                            "id": name,
                            "path": photo_path,
                            "expectation": "must_pass" if options["passed"] else "must_fail",
                            "capture_group": "black-control" if name == "empty" else "coffee-photo",
                            "image_sha256": hashlib.sha256(source_asset.read_bytes()).hexdigest(),
                            "detection_truth": known,
                        }
                    ],
                    "detection": {
                        "block": "decode",
                        "classes": ["cup", "dining table"]
                        if name in ("class_aware", "class_agnostic")
                        else ["cup"],
                        "iou_threshold": 0.6,
                        "score_thresholds": [0, 0.35, 0.7, 0.9, 1],
                        "min_precision": 1,
                        "min_recall": 1,
                    },
                }
            ]
            graph.save(HERE / (name + ".json"))
        finally:
            graph.close()
    truth = {
        "cases": CASES,
        "photo_sha256": PHOTO_SHA256,
        "portrait_sha256": hashlib.sha256(portrait.read_bytes()).hexdigest(),
        "cup_xywh": [170, 18, 245, 292],
        "portrait_crop_xywh": [145, 0, 309, 399],
        "annotation_method": "Independent hand-marked outer cup including handle, excluding saucer",
        "letterbox_references": {
            "coffee": {"resized": [640, 427], "left_top": [0, 106]},
            "portrait": {"resized": [496, 640], "left_top": [72, 0]},
        },
    }
    (HERE / "truth.json").write_bytes(
        (json.dumps(truth, indent=2, allow_nan=False) + "\n").encode()
    )


if __name__ == "__main__":
    build()
