"""Independent native SVM/ONNX predictions and NumPy residual statistics."""

import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def measure():
    truth = json.loads((HERE / "truth.json").read_bytes())
    native = cv2.ml.SVM_load(str(HERE / "assets/native-eps-svr.yml"))
    nets = {
        kind: cv2.dnn.readNetFromONNX(
            str(HERE / "bundles" / kind / "model.onnx"), cv2.dnn.ENGINE_CLASSIC
        )
        for kind in ("vector", "nchw", "batch")
    }
    rows = {}
    for name in [
        "test-0.png",
        "test-1.png",
        "test-2.png",
        "dim.png",
        "batch-a.png",
        "batch-b.png",
        "batch-c.png",
    ]:
        image = cv2.imread(str(HERE / "assets" / name), cv2.IMREAD_GRAYSCALE)
        normalized = image.astype(np.float32) / 255
        scalar = float(np.mean(normalized, dtype=np.float64))
        svm_value = float(native.predict(np.float32([[scalar]]))[1][0, 0])
        expected = truth["training"]["weight"] * scalar + truth["training"]["bias"]
        record = {
            "numpy_value": expected,
            "native_svm_value": svm_value,
            "mean_level": float(image.mean()),
        }
        for kind in ("vector", "nchw"):
            nets[kind].setInput(normalized[None, None], "image")
            output = nets[kind].forward(["measurements"])[0].reshape(-1)
            record[kind + "_value"] = float(output[0])
            record[kind + "_brightness"] = float(output[1]) * 255
        record["reference_mm"] = (
            1.325
            if name.startswith("batch") or name == "dim.png"
            else truth["test_widths_pixels"][int(name[5])] * 0.1
        )
        record["residual_mm"] = record["vector_value"] - record["reference_mm"]
        rows[name] = record
    groups = {}
    for name, images in {
        "evaluation": ["test-0.png", "test-1.png", "test-2.png"],
        "evaluation_lighting": ["test-0.png", "dim.png", "test-2.png"],
        "batch_rows": ["batch-a.png", "batch-b.png", "batch-c.png"],
    }.items():
        errors = np.array([rows[image]["residual_mm"] for image in images])
        targets = np.array([rows[image]["reference_mm"] for image in images])
        denominator = float(np.sum((targets - targets.mean()) ** 2))
        groups[name] = {
            "mae": float(np.abs(errors).mean()),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "bias": float(errors.mean()),
            "maximum_absolute_error": float(np.max(np.abs(errors))),
            "r2": None if denominator == 0 else float(1 - np.sum(errors**2) / denominator),
        }
    batch = np.stack(
        [
            cv2.imread(str(HERE / "assets" / name), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
            for name in ("batch-a.png", "batch-b.png", "batch-c.png")
        ]
    )[:, None]
    calls = []
    for start in (0, 2):
        blob = batch[start : start + 2]
        nets["batch"].setInput(blob, "image")
        values = nets["batch"].forward(["measurements"])[0]
        calls.append({"shape": list(blob.shape), "values": values.astype(float).tolist()})
    return {
        "opencv": cv2.__version__,
        "scope": "Native EPS-SVR, direct ONNX and independent "
        "mean/linear/error arithmetic on original held-out "
        "synthetic images; not physical accuracy qualification",
        "images": rows,
        "groups": groups,
        "batch_calls": calls,
        "maximum_onnx_numpy_difference": max(
            abs(row["vector_value"] - row["numpy_value"]) for row in rows.values()
        ),
        "maximum_svm_numpy_difference": max(
            abs(row["native_svm_value"] - row["numpy_value"]) for row in rows.values()
        ),
    }


if __name__ == "__main__":
    print(json.dumps(measure(), sort_keys=True, indent=2, allow_nan=False))
