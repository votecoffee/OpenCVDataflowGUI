"""Build real-photo SSD workflows without inference, downloads or model conversion."""

import hashlib
import json
from pathlib import Path

from prepare_model import PARAMETERS

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
SOURCE = HERE
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
    "coffee": (1, True),
    "selected_out": (0, False),
    "confidence_fail": (0, False),
    "count_fail": (1, False),
    "empty": (0, False),
}
TRUTH = {
    "size": [600, 400],
    "objects": [{"id": "cup", "label": "cup", "bbox": [170, 18, 245, 292]}],
}


def build():
    if any((HERE / (name + ".json")).exists() for name in CASES):
        raise FileExistsError("Existing SSD projects preserved; generate into a fresh folder")
    registry = builtin_registry()
    for name in CASES:
        photo_path = (
            "../dnn_detection/assets/black.png" if name == "empty" else "../../assets/coffee.png"
        )
        known = {"size": [160, 80], "objects": []} if name == "empty" else TRUTH
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
                    parameters={**PARAMETERS, "contract_path": ".local/ssd_resnet34/contract.json"},
                )
            )
            infer.exposed_outputs.append("receipt")
            decode = graph.add(
                registry.create(
                    "ml.decode_detections",
                    block_id="decode",
                    parameters={
                        "format": "ssd_heads",
                        "coordinate_units": "normalized",
                        "class_id_offset": -1,
                        "labels": [label.replace("_", " ") for label in COCO],
                        "accepted_class_ids": [41],
                        "confidence_threshold": 0.95 if name == "confidence_fail" else 0.4,
                        # This model already performs suppression in its ONNX graph.
                        "nms_enabled": False,
                        "minimum_count": 2 if name == "count_fail" else 1,
                    },
                )
            )
            decode.exposed_outputs.append("result_graphics")
            graph.add(registry.create("cv.overlay_result_graphics", block_id="overlay"))
            graph.add(registry.create("output.filmstrip_image", block_id="filmstrip"))
            for index, block in enumerate(graph.blocks.values()):
                block.position = [index * 420, 180 if block.id == "infer" else 0]
            decode.node_size = "large"
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
                "SSD model and photograph credits",
                [0, -510],
                "large",
            )
            note.description = (
                "Coffee photograph: Rachel Michetti, courtesy of Pikolo Espresso Bar; "
                "scikit-image CC0. Unchanged examples/assets/coffee.png. "
                "Empty-image control: original MIT black.png from dnn_detection. "
                "SSD ResNet-34: ONNX Model Zoo / MLPerf contributors, Apache 2.0, revision "
                "c5cae0f1942c8d7727372c7b1f6b485aa39e4421. "
                "Model not included; run prepare_model.py. "
                "Original MIT pipeline and hand-marked cup box by OpenCVDataflowGUI contributors. "
                "1200x1200 RGB stretch; separate normalized XYXY boxes, one-based contiguous "
                "COCO IDs and scores. Uses OpenCV 5 New CPU; Classic rejects this export. "
                "Only cup is judged. Rectangles do not measure orientation or foreground area. "
                "See README.md and SOURCES.md. "
                "This is an integration control, not an accuracy study."
            )
            graph.validation_sets = [
                {
                    "id": "cup",
                    "name": "Hand-marked coffee cup / empty control",
                    "source": "source",
                    "state_mode": "isolated",
                    "cases": [
                        {
                            "id": name,
                            "path": photo_path,
                            "expectation": "must_pass" if name == "coffee" else "must_fail",
                            "capture_group": "black-control" if name == "empty" else "coffee-photo",
                            "image_sha256": hashlib.sha256(
                                (SOURCE / photo_path).read_bytes()
                            ).hexdigest(),
                            "detection_truth": known,
                        }
                    ],
                    "detection": {
                        "block": "decode",
                        "classes": ["cup"],
                        "iou_threshold": 0.6,
                        "score_thresholds": [0, 0.4, 0.6, 1],
                        "min_precision": 1,
                        "min_recall": 1,
                    },
                }
            ]
            graph.save(HERE / (name + ".json"))
        finally:
            graph.close()
    (HERE / "truth.json").write_bytes(
        (
            json.dumps(
                {
                    "annotation": TRUTH,
                    "cases": CASES,
                    "annotation_method": (
                        "Independently hand-marked outer cup including handle, excluding saucer"
                    ),
                    "iou_threshold": 0.6,
                },
                indent=2,
            )
            + "\n"
        ).encode()
    )


if __name__ == "__main__":
    build()
