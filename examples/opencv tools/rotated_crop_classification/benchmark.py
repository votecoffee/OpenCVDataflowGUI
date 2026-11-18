"""Independent component moments, bilinear sampling and native/NumPy inference."""

import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def bilinear(image, x, y):
    ix = np.floor(x).astype(int)
    iy = np.floor(y).astype(int)
    ax, ay = x - ix, y - iy
    if (
        np.any(ix < 0)
        or np.any(iy < 0)
        or np.any(ix + 1 >= image.shape[1])
        or np.any(iy + 1 >= image.shape[0])
    ):
        raise ValueError("This reference only samples fully supported fixture crops")
    return (
        image[iy, ix] * (1 - ax) * (1 - ay)
        + image[iy, ix + 1] * ax * (1 - ay)
        + image[iy + 1, ix] * (1 - ax) * ay
        + image[iy + 1, ix + 1] * ax * ay
    )


def measure():
    truth = json.loads((HERE / "truth.json").read_bytes())
    weights = json.loads((HERE / "assets/classifier-weights.json").read_bytes())
    net = cv2.dnn.readNetFromONNX(str(HERE / "bundle/model.onnx"), cv2.dnn.ENGINE_CLASSIC)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    W, b = np.array(weights["weights"]), np.array(weights["bias"])
    cases = {}
    yy, xx = np.indices((32, 64), dtype=float)
    u, v = xx - 31.5, yy - 15.5
    for case, spec in truth["cases"].items():
        if case.startswith("coverage_"):
            continue
        image = cv2.imread(str(HERE / "assets" / spec["image"]), cv2.IMREAD_UNCHANGED)
        unit = 257 if image.dtype == np.uint16 else 1
        gray = np.uint8(np.rint(image.astype(float) / unit))
        mask = np.uint8(gray > 80)
        count, components, _, _ = cv2.connectedComponentsWithStats(mask, 8)
        rows = []
        for label in range(1, count):
            ys, xs = np.where(components == label)
            cx, cy = float(xs.mean()), float(ys.mean())
            if case == "selected_parts" and cx >= 170:
                continue
            dx, dy = xs - cx, ys - cy
            angle = float(
                0.5 * np.arctan2(2 * np.sum(dx * dy), np.sum(dx * dx) - np.sum(dy * dy)) % np.pi
            )
            used = 0 if case == "no_alignment" else angle
            c, s = np.cos(used), np.sin(used)
            sx, sy = cx + c * u - s * v, cy + s * u + c * v
            matrix = np.array([[c, -s, cx - c * 31.5 + s * 15.5], [s, c, cy - s * 31.5 - c * 15.5]])
            crop = cv2.warpAffine(
                image, matrix, (64, 32), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
            )
            analytic = bilinear(image.astype(float), sx, sy)
            error = float(np.max(np.abs(crop.astype(float) - analytic)) / unit)
            converted = np.uint8(np.rint(crop.astype(float) / unit))
            blob = cv2.resize(converted.astype(np.float32), (32, 16))[None, None] / 255
            logits = blob.reshape(1, -1).astype(float) @ W + b
            net.setInput(blob, "image")
            native = net.forward(["scores"])[0].reshape(-1)
            native_error = float(np.max(np.abs(native - logits[0])))
            probabilities = np.exp(native.astype(float) - native.max())
            probabilities /= probabilities.sum()
            index = int(probabilities.argmax())
            name = weights["classes"][index]
            confidence = float(probabilities[index])
            passed = confidence >= 0.9 and (case != "accepted_horizontal" or name == "horizontal")
            rows.append(
                {
                    "feature_id": label,
                    "center": [cx, cy],
                    "axis_clockwise_degrees": float(np.degrees(angle)),
                    "source_from_crop": [*matrix.tolist(), [0.0, 0.0, 1.0]],
                    "label": name,
                    "confidence": confidence,
                    "passed": passed,
                    "maximum_bilinear_error_in_uint8_units": error,
                    "maximum_logit_error": native_error,
                }
            )
        cases[case] = {
            "rows": rows,
            "pass_count": sum(r["passed"] for r in rows),
            "fail_count": sum(not r["passed"] for r in rows),
        }
    return {
        "opencv": cv2.__version__,
        "scope": "Independent moments/sampling/math plus direct native ONNX parity on "
        "original controlled fixtures; no field accuracy or calibrated "
        "confidence",
        "cases": cases,
    }


if __name__ == "__main__":
    print(json.dumps(measure(), sort_keys=True, indent=2, allow_nan=False))
