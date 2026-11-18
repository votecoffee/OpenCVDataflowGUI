"""Direct-native heads against independently specified native-resolution image truth."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def measurements(image, reference, scale):
    # The truth disk's measured native center is [70,88] in the original 256x192 image.
    fraction = scale / 8
    x, y, w, h = [round(v * fraction) for v in (30, 45, 82, 86)]
    ys, xs = np.nonzero(image[y : y + h, x : x + w] < 110)
    center = [
        float((xs.mean() + x + 0.5) / fraction - 0.5),
        float((ys.mean() + y + 0.5) / fraction - 0.5),
    ]
    delta = image.astype(np.float64) - reference
    small_roi = (
        slice(round(138 * fraction), round(176 * fraction)),
        slice(round(128 * fraction), round(240 * fraction)),
    )
    predicted = image[small_roi] < 110
    known = reference[small_roi] < 110
    return {
        "mae": float(np.abs(delta).mean()),
        "rmse": float(np.sqrt((delta * delta).mean())),
        "circle_center_in_reference": center,
        "circle_error_reference_pixels": float(np.linalg.norm(np.array(center) - [70, 88])),
        "erased_dark_detail_pixels": int((known & ~predicted).sum()),
        "false_dark_detail_pixels": int((predicted & ~known).sum()),
    }


def benchmark():
    truth = json.loads((HERE / "truth.json").read_bytes())
    model = HERE / "assets/LapSRN_x8.pb"
    assert hashlib.sha256(model.read_bytes()).hexdigest() == truth["model_sha256"]
    cv2.dnn_superres.DnnSuperResImpl_create()  # Register the native pixel-shuffle layer.
    network = cv2.dnn.readNetFromTensorflow(str(model), "", cv2.dnn.ENGINE_CLASSIC)
    network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    clean = cv2.imread(str(HERE / "assets/reference.png"), cv2.IMREAD_GRAYSCALE)
    results = {}
    for case in ("reference", "noise"):
        image = cv2.imread(str(HERE / f"assets/{case}.png"), cv2.IMREAD_GRAYSCALE)
        low = cv2.resize(image, (32, 24), interpolation=cv2.INTER_AREA)
        prepared = cv2.addWeighted(low, 1 / 255, low, 0, 0, dtype=cv2.CV_32F)
        network.setInput(cv2.dnn.blobFromImage(prepared))
        outputs = network.forward([f"NCHW_output_{s}x" for s in (2, 4, 8)])
        results[case] = {}
        for scale, tensor in zip((2, 4, 8), outputs, strict=True):
            size = (32 * scale, 24 * scale)
            reference = cv2.resize(clean, size, interpolation=cv2.INTER_AREA)
            learned = cv2.addWeighted(tensor[0, 0], 255, tensor[0, 0], 0, 0, dtype=cv2.CV_8U)
            images = {
                "learned": learned,
                "linear": cv2.resize(low, size, interpolation=cv2.INTER_LINEAR),
                "cubic": cv2.resize(low, size, interpolation=cv2.INTER_CUBIC),
            }
            results[case][str(scale)] = {
                name: measurements(value, reference, scale) for name, value in images.items()
            }
    return {
        "model_sha256": truth["model_sha256"],
        "native_forwards": 2,
        "source_cases": ["reference", "noise"],
        "scales": [2, 4, 8],
        "reference_size": [256, 192],
        "results": results,
    }


if __name__ == "__main__":
    (HERE / "benchmark.json").write_bytes(
        (json.dumps(benchmark(), sort_keys=True, indent=2) + "\n").encode()
    )
