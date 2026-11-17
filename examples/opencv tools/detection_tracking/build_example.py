"""Controlled crossings/gaps plus translated credited photographs and actual DNN detections."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from prepare_tracking_model import PARAMETERS

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
SOURCE = HERE
CASES = {
    "crossing": {"sequence": "crossing", "appearance_policy": "required"},
    "occluded_swap": {"sequence": "swap", "appearance_policy": "required"},
    "motion_only": {"sequence": "swap", "appearance_policy": "off"},
    "detection_gap": {"sequence": "gap", "appearance_policy": "required"},
    "expire_reentry": {
        "sequence": "gap",
        "appearance_policy": "required",
        "maximum_missed_frames": 1,
    },
    "yolox_coffee": {"sequence": "coffee", "appearance_policy": "required"},
}


def build():
    if any((HERE / (name + ".json")).exists() for name in (*CASES, "truth")):
        raise FileExistsError("Existing tracking examples preserved; generate into a fresh folder")
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    sequences = {}

    def save(name, index, image, records):
        filename = f"assets/{name}_{index:02d}.png"
        ok, raw = cv2.imencode(".png", image)
        assert ok
        path = HERE / filename
        if path.exists() and path.read_bytes() != raw.tobytes():
            raise FileExistsError("Different existing fixture preserved")
        path.write_bytes(raw.tobytes())
        sequences.setdefault(name, []).append(
            {"path": filename, "truth": records, "sha256": hashlib.sha256(raw).hexdigest()}
        )

    for name in ("crossing", "swap", "gap"):
        for index in range(9):
            image = np.zeros((128, 220, 3), np.uint8)
            records = []
            if name == "crossing":
                placements = [
                    ("red", 30 + 18 * index, 35, (0, 0, 255)),
                    ("blue", 190 - 18 * index, 90, (255, 0, 0)),
                ]
            elif name == "swap":
                placements = (
                    []
                    if index in (3, 4)
                    else [
                        ("red", 60 if index < 3 else 160, 64, (0, 0, 255)),
                        ("blue", 160 if index < 3 else 60, 64, (255, 0, 0)),
                    ]
                )
            else:
                placements = [("red", 80, 64, (0, 0, 40) if index in (3, 4, 5) else (0, 0, 255))]
            for identity, x, y, color in placements:
                bounds = [x - 16, y - 12, 32, 24]
                image[y - 12 : y + 12, x - 16 : x + 16] = color
                records.append({"id": identity, "bbox": bounds, "label": "part"})
            save(name, index, image, records)
    photo = (SOURCE / "../../assets/coffee.png").read_bytes()
    photo_hash = hashlib.sha256(photo).hexdigest()
    if photo_hash != "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7":
        raise ValueError("Photo changed; review attribution and annotation")
    original = cv2.imdecode(np.frombuffer(photo, np.uint8), cv2.IMREAD_COLOR)
    for index, dx in enumerate((0, 8, 16, None, 32, 40)):
        image = (
            np.zeros_like(original)
            if dx is None
            else cv2.warpAffine(original, np.float64([[1, 0, dx], [0, 1, 0]]), (600, 400))
        )
        records = (
            [] if dx is None else [{"id": "cup", "label": "cup", "bbox": [170 + dx, 18, 245, 292]}]
        )
        save("coffee", index, image, records)
    registry = builtin_registry()
    for name, options in CASES.items():
        paths = [row["path"] for row in sequences[options["sequence"]]]
        graph = Graph()
        try:
            graph.program["assets"] = ["SOURCES.md", "MODEL-LICENSE.txt"]

            def add(type_id, identity, position, **parameters):
                tool = graph.add(registry.create(type_id, block_id=identity, parameters=parameters))
                tool.position = list(position)
                return tool

            add(
                "input.file",
                "source",
                (0, 0),
                path=paths[0],
                paths=paths,
                source_mode="files",
                path_base="project",
                repeat="stop",
            )
            detection_source = "source"
            if name == "yolox_coffee":
                add(
                    "ml.dnn",
                    "infer",
                    (380, 260),
                    **PARAMETERS,
                    contract_path=".local/yolox_raw/contract.json",
                )
                # Single-class filtering uses original COCO zero-based cup index.
                labels = [str(i) for i in range(80)]
                labels[41] = "cup"
                add(
                    "ml.decode_detections",
                    "detect",
                    (780, 0),
                    format="xywh_objectness",
                    grid_decode="yolox",
                    output_name="output",
                    labels=labels,
                    accepted_class_ids=[41],
                    confidence_threshold=0.35,
                    minimum_count=0,
                )
                for a, out, b, inp in (
                    ("source", "image", "infer", "image"),
                    ("infer", "tensors", "detect", "tensors"),
                    ("infer", "preprocessing", "detect", "preprocessing"),
                ):
                    graph.connect(a, out, b, inp)
            else:
                add("cv.threshold", "threshold", (380, 260), value=20)
                graph.connect("source", "image", "threshold", "image")
                detection_source = "threshold"
                add("inspect.blobs", "detect", (780, 0), min_area=100)
            add("ml.box_appearance", "appearance", (1150, 0))
            track = add(
                "sequence.track_image",
                "track",
                (1550, 0),
                clock="nominal",
                minimum_observations=1,
                emit_predictions=True,
                minimum_confidence=0,
                appearance_policy=options["appearance_policy"],
                maximum_appearance_distance=0.5,
                appearance_weight=0.9,
                maximum_missed_frames=options.get("maximum_missed_frames", 4),
                maximum_unobserved_ms=2000,
                association_distance=200,
                innovation_gate=1e6,
                position_noise=0.05,
                position_acceleration=500,
                maximum_position_sigma=10000,
                angle_period="180",
                maximum_angle_difference=180,
            )
            track.exposed_outputs.append("result_graphics")
            track.node_size = "large"
            add("cv.overlay_result_graphics", "overlay", (2000, 0))
            add("output.filmstrip_image", "filmstrip", (2400, 0))
            for a, out, b, inp in (
                (detection_source, "image", "detect", "image"),
                ("source", "image", "appearance", "image"),
                ("detect", "collection", "appearance", "detections"),
                ("source", "image", "track", "image"),
                ("appearance", "collection", "track", "detections"),
                ("source", "image", "overlay", "image"),
                ("track", "result_graphics", "overlay", "graphics_1"),
                ("overlay", "image", "filmstrip", "image"),
            ):
                graph.connect(a, out, b, inp)
            note = add("layout.comment", "attribution", (0, -450))
            note.name, note.node_size = "Tracking controls / source credits", "large"
            note.description = (
                "Original MIT colored-part controls and independent annotations. The dim red part "
                "remains visible but falls below the blob threshold, producing a genuine "
                "detection gap. Coffee: Rachel Michetti, courtesy Pikolo Espresso Bar, "
                "scikit-image CC0; known integer translations and a black occlusion control. "
                "YOLOX-S: Megvii / OpenCV Zoo contributors, "
                "Apache 2.0, revision 48b10fa505aeb51fbd127fc60a5c660040e823b7; weights unchanged. "
                "Run All; green observed, amber predicted. Histograms are appearance evidence, "
                "not learned re-identification or proof of identity. See README.md, SOURCES.md "
                "and evaluate_examples.py."
            )
            graph.save(HERE / (name + ".json"))
        finally:
            graph.close()
    (HERE / "truth.json").write_bytes(
        (
            json.dumps(
                {
                    "cases": CASES,
                    "sequences": sequences,
                    "source_photo_sha256": photo_hash,
                    "scoring": (
                        "Observed output tracks only, source-grid IoU >= 0.6; "
                        "predictions reported separately."
                    ),
                },
                indent=2,
            )
            + "\n"
        ).encode()
    )


if __name__ == "__main__":
    build()
