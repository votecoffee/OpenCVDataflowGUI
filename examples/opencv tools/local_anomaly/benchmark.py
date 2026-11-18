"""Direct native ONNX and independent pixel-patch/Euclidean localization references."""

import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def patches(image):
    image = image.astype(np.float32) * np.float32(1 / 255)
    return np.stack([image[y::2, x::2] for y in range(2) for x in range(2)], axis=-1).reshape(-1, 4)


def measure():
    truth = json.loads((HERE / "truth.json").read_bytes())
    images = {
        name: cv2.imread(str(HERE / "assets" / name), cv2.IMREAD_GRAYSCALE)
        for name in [
            *(f"train-{i}.png" for i in range(3)),
            "normal.png",
            "defect.png",
            "missing.png",
            "weak.png",
            "bright.png",
        ]
    }
    reference = np.concatenate([patches(images[f"train-{i}.png"]) for i in range(3)])
    net = cv2.dnn.readNetFromONNX(str(HERE / "bundle/model.onnx"), cv2.dnn.ENGINE_CLASSIC)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    native_error, rows = 0.0, {}
    for case, details in truth["cases"].items():
        if case in ("wrong_features", "corrupt_reference"):
            continue
        name = "defect.png" if case == "high_depth" else details["image"]
        image = images[name]
        query = patches(image)
        net.setInput((image.astype(np.float32) * np.float32(1 / 255))[None, None], "image")
        native = net.forward("local")[0].transpose(1, 2, 0).reshape(-1, 4)
        native_error = max(native_error, float(np.max(np.abs(native - query))))
        bank = (
            reference[np.linspace(0, len(reference) - 1, 16).astype(int)]
            if case == "sampled_reference"
            else reference
        )
        errors = query.astype(np.float64)[:, None] - bank.astype(np.float64)[None]
        distances = np.sqrt(np.sum(errors * errors, axis=-1)).min(axis=1)
        mapped = np.repeat(np.repeat(distances.reshape(16, 16), 2, axis=0), 2, axis=1)
        selected = np.ones((32, 32), bool)
        if case == "selected_region":
            selected[:, 10:] = False
        mask = (mapped > 0.2) & selected
        expected = np.zeros((32, 32), bool)
        if name in ("defect.png", "missing.png"):
            expected[10:18, 12:20] = True
        elif name == "weak.png":
            expected[10, 12] = True
        expected &= selected
        union = int(np.count_nonzero(mask | expected))
        pixels = int(mask.sum())
        rows[case] = {
            "maximum_distance": float(mapped[selected].max()),
            "anomaly_pixels": pixels,
            "selected_pixels": int(selected.sum()),
            "ground_truth_defect_pixels": int(expected.sum()),
            "mask_iou": float(np.count_nonzero(mask & expected) / union) if union else None,
            "passed": pixels <= (64 if case == "allowed_area" else 0),
        }
    return {
        "reference_rows": len(reference),
        "maximum_native_patch_difference": native_error,
        "cases": rows,
        "claim": "Original CPU controls, not learned-backbone or industrial accuracy",
    }


if __name__ == "__main__":
    print(json.dumps(measure(), indent=2, sort_keys=True, allow_nan=False))
