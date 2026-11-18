"""Direct native tracker and independent annotated box/localization references."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def iou(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    intersection = np.prod(
        np.maximum(0, np.minimum(a[:2] + a[2:], b[:2] + b[2:]) - np.maximum(a[:2], b[:2]))
    )
    return float(intersection / (np.prod(a[2:]) + np.prod(b[2:]) - intersection))


def benchmark():
    truth = json.loads((HERE / "truth.json").read_bytes())
    path = HERE / "assets/vit.onnx"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == truth["model_sha256"]
    net = cv2.dnn.readNetFromONNX(str(path), engine=cv2.dnn.ENGINE_CLASSIC)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    tracker = cv2.TrackerVit.create(net, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 0.2)
    first = cv2.imread(str(HERE / "assets/frame_00.png"))
    tracker.init(first, tuple(truth["initial_bounds"]))
    records = []
    for index, entry in enumerate(truth["annotations"][1:], 1):
        located, bounds = tracker.update(cv2.imread(str(HERE / "assets" / entry["file"])))
        expected = np.array(entry["bounds"], np.float64)
        actual = np.array(bounds, np.float64)
        center_error = float(
            np.linalg.norm(
                actual[:2] + (actual[2:] - 1) / 2 - expected[:2] - (expected[2:] - 1) / 2
            )
        )
        records.append(
            {
                "frame": index,
                "located": bool(located),
                "bounds": list(map(int, bounds)),
                "score": float(tracker.getTrackingScore()),
                "iou": iou(bounds, entry["bounds"]),
                "center_error_pixels": center_error,
            }
        )
    located, bounds = tracker.update(np.zeros_like(first))
    return {
        "opencv": cv2.__version__,
        "model_sha256": truth["model_sha256"],
        "source_size": [360, 270],
        "engine": "classic",
        "target": "cpu",
        "scope": "artificial translated single public image; not recorded motion or physical pose",
        "frames": records,
        "minimum_iou": min(r["iou"] for r in records),
        "maximum_center_error_pixels": max(r["center_error_pixels"] for r in records),
        "blank_located": bool(located),
        "blank_score": float(tracker.getTrackingScore()),
        "threshold": 0.2,
        "score_is_probability": False,
    }


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
