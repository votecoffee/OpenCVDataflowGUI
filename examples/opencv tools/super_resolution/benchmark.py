"""Native CPU reconstruction vs independent known pixels, features and flat-field noise."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def crop(image, rectangle):
    x, y, w, h = rectangle
    return image[y : y + h, x : x + w]


def measurements(image, reference, truth):
    difference = image.astype(np.float64) - reference
    circle = crop(image, truth["circle_roi"]) < truth["dark_threshold"]
    y, x = np.nonzero(circle)
    detail = crop(image, truth["detail_roi"]) < truth["dark_threshold"]
    detail_truth = crop(reference, truth["detail_roi"]) < truth["dark_threshold"]
    gap = crop(image, truth["gap_roi"]) < truth["dark_threshold"]
    widths = []
    for position in (140, 164, 188, 212):
        widths.append(int(np.count_nonzero(image[157, position - 5 : position + 10] < 110)))
    return {
        "mae": float(np.abs(difference).mean()),
        "rmse": float(np.sqrt(np.square(difference).mean())),
        "maximum_absolute_error": float(np.abs(difference).max()),
        "circle_width_height": [int(x.max() - x.min() + 1), int(y.max() - y.min() + 1)],
        "circle_center": [
            float(x.mean() + truth["circle_roi"][0]),
            float(y.mean() + truth["circle_roi"][1]),
        ],
        "circle_area": int(circle.sum()),
        "crack_widths_at_y157": widths,
        "false_dark_detail_pixels": int((detail & ~detail_truth).sum()),
        "erased_dark_detail_pixels": int((~detail & detail_truth).sum()),
        "bead_gap_columns": int((~gap.any(axis=0)).sum()),
        "flat_std": float(crop(image, truth["flat_roi"]).std(dtype=np.float64)),
        "flat_changed_pixels": int(np.count_nonzero(crop(image, truth["flat_roi"]) != 180)),
        "flat_rmse_to_180": float(
            np.sqrt(np.square(crop(image, truth["flat_roi"]).astype(np.float64) - 180).mean())
        ),
    }


def benchmark():
    truth = json.loads((HERE / "truth.json").read_bytes())
    model_path = HERE / "assets/FSRCNN_x2.pb"
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == truth["model_sha256"]
    model = cv2.dnn_superres.DnnSuperResImpl_create()
    model.readModel(str(model_path))
    model.setModel("fsrcnn", 2)
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    clean = cv2.imread(str(HERE / "assets/reference.png"), cv2.IMREAD_GRAYSCALE)
    cases = {}
    for case in ("reference", "noise"):
        current = cv2.imread(str(HERE / "assets" / (case + ".png")), cv2.IMREAD_GRAYSCALE)
        reduced = cv2.resize(current, (128, 96), interpolation=cv2.INTER_AREA)
        outputs = {
            "learned": model.upsample(reduced),
            "linear": cv2.resize(reduced, (256, 192), interpolation=cv2.INTER_LINEAR),
            "cubic": cv2.resize(reduced, (256, 192), interpolation=cv2.INTER_CUBIC),
            "lanczos4": cv2.resize(reduced, (256, 192), interpolation=cv2.INTER_LANCZOS4),
        }
        cases[case] = {name: measurements(image, clean, truth) for name, image in outputs.items()}
    return {
        "model_sha256": truth["model_sha256"],
        "opencv": cv2.__version__,
        "reference": measurements(clean, clean, truth),
        "cases": cases,
        "scope": "known synthetic pixels; nominal grid; no physical dimensional accuracy",
    }


if __name__ == "__main__":
    path = HERE / "benchmark.json"
    content = (json.dumps(benchmark(), indent=2, allow_nan=False) + "\n").encode()
    if path.exists() and path.read_bytes() != content:
        raise FileExistsError("Different retained benchmark preserved; review before replacing")
    path.write_bytes(content)
    print(path)
