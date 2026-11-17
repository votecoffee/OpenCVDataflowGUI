"""Reproduce overlapping real-photo tile projects without replacing existing projects."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from prepare_tile_model import PARAMETERS

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
    "coffee": {"count": 1, "passed": True},
    "duplicates": {"count": 2, "passed": True, "nms_enabled": False},
    "resized": {"count": 1, "passed": True},
    "selected_out": {"count": 0, "passed": False},
    "count_fail": {"count": 1, "passed": False, "minimum_count": 2},
    "failed_tile": {"count": 1, "passed": False},
    "empty": {"count": 0, "passed": False},
    "missing_tile": {"count": None, "passed": False, "expected_tiles": 3},
}
CROPS = {"left": [80, 0, 440, 380], "right": [140, 0, 420, 400]}


def build():
    if any((HERE / (name + ".json")).exists() for name in (*CASES, "truth")):
        raise FileExistsError("Existing tile projects preserved; generate into a fresh folder")
    photo = (SOURCE / "../../assets/coffee.png").read_bytes()
    if hashlib.sha256(photo).hexdigest() != PHOTO_SHA256:
        raise ValueError("Source photo changed; review attribution and annotation first")
    ok, black = cv2.imencode(".png", np.zeros((400, 600, 3), np.uint8))
    assert ok
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    negative = assets / "black.png"
    if negative.exists() and negative.read_bytes() != black.tobytes():
        raise FileExistsError("Different existing negative preserved")
    negative.write_bytes(black.tobytes())
    registry = builtin_registry()
    for name, options in CASES.items():
        photo_path = "assets/black.png" if name == "empty" else "../../assets/coffee.png"
        graph = Graph()
        try:
            graph.set_branch_workers(2)
            graph.program["assets"] = ["SOURCES.md", "MODEL-LICENSE.txt"]

            def add(type_id, block_id, position, **parameters):
                tool = graph.add(registry.create(type_id, block_id=block_id, parameters=parameters))
                tool.position = list(position)
                return tool

            add("input.file", "source", (0, 0), path=photo_path, path_base="project")
            for index, (side, (x, y, width, height)) in enumerate(CROPS.items()):
                yy = index * 540
                add("cv.crop", side, (400, yy), x=x, y=y, width=width, height=height)
                graph.connect("source", "image", side, "image")
                image_source = side
                if name == "resized" and side == "right":
                    add("cv.resize", "resize", (400, yy + 250), width=210, height=200)
                    graph.connect(side, "image", "resize", "image")
                    image_source = "resize"
                add(
                    "ml.dnn",
                    "infer_" + side,
                    (800, yy + 170),
                    **PARAMETERS,
                    contract_path=".local/yolox_raw/contract.json",
                )
                add(
                    "ml.decode_detections",
                    "decode_" + side,
                    (1200, yy),
                    format="xywh_objectness",
                    grid_decode="yolox",
                    output_name="output",
                    labels=[label.replace("_", " ") for label in COCO],
                    accepted_class_ids=[41],
                    confidence_threshold=0.35,
                    minimum_count=2 if name == "failed_tile" and side == "right" else 0,
                )
                for a, o, b, i in (
                    (image_source, "image", "infer_" + side, "image"),
                    (image_source, "image", "decode_" + side, "image"),
                    ("infer_" + side, "tensors", "decode_" + side, "tensors"),
                    ("infer_" + side, "preprocessing", "decode_" + side, "preprocessing"),
                ):
                    graph.connect(a, o, b, i)
            add("flow.join", "join", (1600, 400))
            for index, side in enumerate(CROPS, 1):
                graph.connect("decode_" + side, "collection", "join", f"branch_{index}")
            fuse = add(
                "ml.fuse_tile_detections",
                "fuse",
                (2000, 0),
                expected_tiles=options.get("expected_tiles", 2),
                nms_enabled=options.get("nms_enabled", True),
                minimum_count=options.get("minimum_count", 1),
            )
            fuse.node_size = "large"
            fuse.exposed_outputs.append("result_graphics")
            add("cv.overlay_result_graphics", "overlay", (2450, 0))
            add("output.filmstrip_image", "filmstrip", (2850, 0))
            for a, o, b, i in (
                ("source", "image", "fuse", "image"),
                ("join", "group", "fuse", "detections"),
                ("fuse", "image", "overlay", "image"),
                ("fuse", "result_graphics", "overlay", "graphics_1"),
                ("overlay", "image", "filmstrip", "image"),
            ):
                graph.connect(a, o, b, i)
            if name == "selected_out":
                add("region.shape", "selection", (1600, -300), rectangle=[460, 0, 130, 100])
                graph.connect("source", "image", "selection", "image")
                graph.connect("selection", "region", "fuse", "region")
            note = add("layout.comment", "attribution", (0, -510))
            note.name, note.node_size = "Model / photograph attribution", "large"
            note.description = (
                "Coffee: Rachel Michetti, courtesy of Pikolo Espresso Bar; scikit-image CC0. "
                "Runtime crops are [80,0,440,380] and [140,0,420,400]. YOLOX-S: Megvii YOLOX / "
                "OpenCV Zoo contributors, Apache 2.0, revision "
                "48b10fa505aeb51fbd127fc60a5c660040e823b7; "
                "weights unchanged. Original MIT pipeline, independent cup annotation "
                "and black control. This is one shared photograph, not an accuracy benchmark. "
                "Prepare weights explicitly. "
                "See README.md and SOURCES.md."
            )
            source_asset = HERE / photo_path if name == "empty" else SOURCE / photo_path
            graph.validation_sets = [
                {
                    "id": "cup",
                    "name": "Independent cup annotation / duplicate controls",
                    "source": "source",
                    "state_mode": "isolated",
                    "cases": [
                        {
                            "id": name,
                            "path": photo_path,
                            "expectation": "must_pass" if options["passed"] else "must_fail",
                            "capture_group": "black-control" if name == "empty" else "coffee-photo",
                            "image_sha256": hashlib.sha256(source_asset.read_bytes()).hexdigest(),
                            "detection_truth": {
                                "size": [600, 400],
                                "objects": []
                                if name == "empty"
                                else [
                                    {"id": "cup", "label": "cup", "bbox": [170, 18, 245, 292]},
                                ],
                            },
                        }
                    ],
                    "detection": {
                        "block": "fuse",
                        "classes": ["cup"],
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
        "crops_xywh": CROPS,
        "black_sha256": hashlib.sha256(negative.read_bytes()).hexdigest(),
        "cup_xywh": [170, 18, 245, 292],
        "annotation_method": "Independent hand-marked outer cup including handle, excluding saucer",
    }
    (HERE / "truth.json").write_bytes(
        (json.dumps(truth, indent=2, allow_nan=False) + "\n").encode()
    )


if __name__ == "__main__":
    build()
