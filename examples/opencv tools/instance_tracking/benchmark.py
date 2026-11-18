"""Direct native mask/pixel reference and independent visible-mask/identity scoring."""

import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def mask_from_runs(runs):
    mask = np.zeros((40, 64), bool)
    for y, x, length in runs:
        mask[y, x : x + length] = True
    return mask


def native_reference():
    truth = json.loads((HERE / "truth.json").read_bytes())
    net = cv2.dnn.readNetFromONNX(str(HERE / "bundle/model.onnx"), cv2.dnn.ENGINE_CLASSIC)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    count, maximum_difference = 0, 0.0
    for name, sequence in truth["sequences"].items():
        capture = cv2.VideoCapture(str(HERE / "assets" / (name + ".avi")))
        try:
            for row in sequence:
                ok, image = capture.read()
                assert ok
                normalized = image.astype(np.float32) * np.float32(1 / 255)
                net.setInput(normalized.transpose(2, 0, 1)[None], "image")
                masks = net.forward("masks")[:, 0]
                expected = normalized[..., [2, 0]].transpose(2, 0, 1)
                maximum_difference = max(maximum_difference, float(np.abs(masks - expected).max()))
                for index, color in enumerate(("red", "blue")):
                    assert np.array_equal(masks[index] > 0.5, mask_from_runs(row[color + "_runs"]))
                count += 1
        finally:
            capture.release()
    return {"native_frames": count, "maximum_mask_difference": maximum_difference}


def evaluate(case, profile="debug"):
    from opencv_dataflow.blocks import builtin_registry
    from opencv_dataflow.core import Graph

    truth = json.loads((HERE / "truth.json").read_bytes())
    definition = truth["cases"][case]
    sequence = truth["sequences"][definition["sequence"]]
    graph = Graph.load(HERE / (case + ".json"), builtin_registry())
    graph.set_run_profile(profile)
    history, records, totals = (
        {},
        [],
        {"tp": 0, "fp": 0, "fn": 0, "id_switches": 0, "execution_errors": 0},
    )
    try:
        for row in sequence:
            graph.run()
            block = graph.blocks["track"]
            result = block.last_result
            if result is None:
                totals["execution_errors"] += 1
                records.append({"index": row["index"], "error": block.error})
                continue
            expected = {color: mask_from_runs(row[color + "_runs"]) for color in ("red", "blue")}
            expected = {color: mask for color, mask in expected.items() if mask.any()}
            found = []
            for item in result.outputs["tracks"]["items"]:
                found.append((item["id"], mask_from_runs(item["mask_region"]["runs"])))
            pairs = []
            for index, (_, mask) in enumerate(found):
                for color, reference in expected.items():
                    union = int(np.count_nonzero(mask | reference))
                    overlap = float(np.count_nonzero(mask & reference) / union)
                    if overlap >= 0.5:
                        pairs.append((overlap, index, color))
            used_predictions, used_truth, matches = set(), set(), []
            for overlap, index, color in sorted(pairs, reverse=True):
                if index in used_predictions or color in used_truth:
                    continue
                used_predictions.add(index)
                used_truth.add(color)
                identity = found[index][0]
                totals["id_switches"] += int(color in history and history[color] != identity)
                history[color] = identity
                matches.append({"truth": color, "track_id": identity, "iou": overlap})
            totals["tp"] += len(matches)
            totals["fp"] += len(found) - len(matches)
            totals["fn"] += len(expected) - len(matches)
            records.append(
                {
                    "index": row["index"],
                    "matches": sorted(matches, key=lambda m: m["truth"]),
                    "observed": len(found),
                    "active": len(result.outputs["report"]["active"]),
                    "predicted": sum(
                        t["status"] == "predicted" for t in result.outputs["report"]["active"]
                    ),
                    "ambiguous": len(result.outputs["report"]["ambiguous_instance_ids"]),
                    "overlap_pixels": result.outputs["report"]["overlap_pixels"],
                }
            )
    finally:
        graph.close()
    return {
        "summary": totals,
        "frames": records,
        "claim": "Original CPU mask-association controls; "
        "no pretrained video segmentation accuracy claim",
    }


if __name__ == "__main__":
    truth = json.loads((HERE / "truth.json").read_bytes())
    print(
        json.dumps(
            {
                "native": native_reference(),
                "cases": {case: evaluate(case) for case in truth["cases"]},
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
