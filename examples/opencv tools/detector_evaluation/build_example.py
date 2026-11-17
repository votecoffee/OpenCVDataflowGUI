"""Original fixed-proposal, image-score controls with independently derived source boxes."""

import hashlib
import json
from pathlib import Path

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.validation import run_set

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "dnn_detection"
CASES = (
    "correct",
    "duplicates",
    "low_recall",
    "localization",
    "wrong_class",
    "negative_false_positives",
    "decoder_error",
)


def write_json(path, value):
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def labels():
    # Fixture proposals are declared network-pixel XYXY edges, not inferred from
    # decoder output. Stretch maps 64x64 -> 160x80; duplicate row 1 is suppressed.
    declarations = (
        ("red-a", "red", [10, 16, 30, 32]),
        ("green-a", "green", [10, 16, 30, 32]),
        ("red-b", "red", [40, 38, 52, 50]),
    )
    return {
        "size": [160, 80],
        "objects": [
            {
                "id": identity,
                "label": label,
                "bbox": [left * 2.5, top * 1.25, (right - left) * 2.5, (bottom - top) * 1.25],
            }
            for identity, label, (left, top, right, bottom) in declarations
        ],
    }


def annotate(graph, registry, message):
    if "attribution" in graph.blocks:
        graph.remove("attribution")
    note = graph.add(registry.create("layout.comment", block_id="attribution"))
    note.name = "Original MIT detector evaluation controls"
    note.node_size, note.position = "large", [0, -510]
    note.description = (
        "Original MIT ONNX proposal fixture and images by OpenCVDataflowGUI contributors; "
        "copied unchanged from dnn_detection with pinned hashes. Fixed proposal boxes and "
        "image-dependent scores, not a trained part detector or accuracy benchmark. "
        "Truth comes from declared network geometry and an independent scale calculation. "
        "Run Results > Validation image sets > Run image set. "
        + message
        + " See README.md, SOURCES.md and truth.json."
    )


def build():
    assets = HERE / "assets"
    if assets.exists():
        raise FileExistsError("Existing detector evaluation assets preserved")
    original = json.loads((SOURCE / "truth.json").read_bytes())
    files = {}
    for name in ("color.png", "black.png", "proposal_heads.onnx"):
        raw = (SOURCE / "assets" / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != original["asset_sha256"][name]:
            raise ValueError("Original fixture changed: " + name)
        files[name] = raw
    assets.mkdir()
    for name, raw in files.items():
        (assets / name).write_bytes(raw)
    registry = builtin_registry()
    summaries, duplicate_records = {}, None
    for case in CASES:
        graph = Graph.load(SOURCE / "stretch.json", registry)
        try:
            graph.project_dir = HERE
            graph.blocks["infer"].parameters["engine"] = "new"
            graph.blocks["decode"].parameters["minimum_count"] = 0
            known = labels()
            options = {
                "block": "decode",
                "classes": ["red", "green"],
                "score_thresholds": [0, 0.6, 0.75, 1],
            }
            if case == "duplicates":
                graph.blocks["decode"].parameters["nms_enabled"] = False
            if case == "low_recall":
                graph.blocks["decode"].parameters["confidence_threshold"] = 0.7
            if case == "localization":
                known["objects"][0]["bbox"][0] += 10
                options["iou_threshold"] = 0.75
            if case == "wrong_class":
                known["objects"][0]["label"] = "blue"
            if case == "negative_false_positives":
                known["objects"] = []
            if case == "decoder_error":
                graph.blocks["decode"].parameters["output_name"] = "unavailable"
            cases = [
                {
                    "id": name,
                    "path": "assets/" + name + ".png",
                    "expectation": "unknown",
                    "capture_group": "control-" + name,
                    "image_sha256": hashlib.sha256(files[name + ".png"]).hexdigest(),
                    "detection_truth": known
                    if name == "color"
                    else {"size": [160, 80], "objects": []},
                }
                for name in ("color", "black")
            ]
            graph.validation_sets = [
                {
                    "id": "objects",
                    "name": "Labeled detector controls",
                    "source": "source",
                    "state_mode": "sequence",
                    "cases": cases,
                    "detection": options,
                }
            ]
            annotate(
                graph,
                registry,
                {
                    "correct": "Expected 3 TP, no FP/misses; clean negative included.",
                    "duplicates": "NMS off: expected 3 TP, 1 duplicate FP, mAP 11/12 at IoU .5.",
                    "low_recall": "Upstream threshold .7 removes two objects: 1 TP, 2 misses.",
                    "localization": "Known 10-pixel truth offset gives IoU 2/3; rejected at .75.",
                    "wrong_class": "One explicit class mismatch gives one FP and one miss.",
                    "negative_false_positives": "Fully labeled negative; all proposals are FP.",
                    "decoder_error": "Unavailable output: both images Error, "
                    "not clean empty results.",
                }[case],
            )
            graph.save(HERE / (case + ".json"))
            report = run_set(graph.to_dict(), graph.validation_sets[0], registry, HERE)
            summaries[case] = report["detection"]["summary"]
            if case == "duplicates":
                duplicate_records = report["detection"]["records"]
        finally:
            graph.close()
    graph = Graph()
    try:
        graph.add(
            registry.create(
                "ml.evaluate_detections",
                block_id="evaluate",
                parameters={
                    "records": duplicate_records,
                    "classes": ["red", "green"],
                    "score_thresholds": [0, 0.6, 0.75, 1],
                },
            )
        )
        annotate(
            graph, registry, "Stored actual duplicate-control records. Run once for Report/PR plot."
        )
        graph.save(HERE / "metrics.json")
    finally:
        graph.close()
    write_json(
        HERE / "truth.json",
        {
            "projects": summaries,
            "asset_sha256": {n: hashlib.sha256(raw).hexdigest() for n, raw in files.items()},
            "source_truth": labels(),
            "duplicate_expected_red_ap": 5 / 6,
            "duplicate_expected_map": 11 / 12,
        },
    )


if __name__ == "__main__":
    build()
