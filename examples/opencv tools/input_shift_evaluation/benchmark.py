"""Independent integer pose, separable Gaussian, patch-distance and paired mask references."""

import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def moved(array, dx):
    result = np.zeros_like(array)
    if 0 <= dx < array.shape[1]:
        result[:, dx:] = array[:, : array.shape[1] - dx]
    return result


def patches(image):
    a = image.astype(np.float32) * np.float32(1 / 255)
    return np.stack([a[y::2, x::2] for y in range(2) for x in range(2)], -1).reshape(-1, 4)


def intersection(a, b, support):
    union = np.count_nonzero((a | b) & support)
    return float(np.count_nonzero(a & b & support) / union) if union else None


def measure():
    truth = json.loads((HERE / "truth.json").read_bytes())
    reference = np.concatenate(
        [patches(cv2.imread(str(HERE / f"assets/train-{i}.png"), 0)) for i in range(3)]
    )
    net = cv2.dnn.readNetFromONNX(str(HERE / "bundle/model.onnx"), cv2.dnn.ENGINE_CLASSIC)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    output, native_error = {}, 0.0
    for case, definition in truth["cases"].items():
        if case in ("missing_variant", "wrong_identity", "cropped_truth"):
            continue
        records = {}
        for index, name in enumerate(definition["subjects"]):
            source = cv2.imread(str(HERE / "assets" / name), cv2.IMREAD_UNCHANGED)
            maximum = np.iinfo(source.dtype).max
            known = np.zeros((32, 32), bool)
            if name != "normal.png":
                known[10:18, 12:20] = True
            computed = {}
            for variant, options in definition["variants"].items():
                dx = int(options.get("translation_x", 0))
                normalized = moved(source.astype(float) / maximum, dx)
                normalized = np.clip(normalized + options.get("offset", 0), 0, 1)
                support = moved(np.ones((32, 32), bool), dx)
                if options.get("blur_kernel"):
                    axis = np.arange(-1, 2, dtype=float)
                    kernel = np.exp(-axis * axis / 2)
                    kernel /= kernel.sum()
                    full = kernel[:, None] * kernel[None]
                    padded = np.pad(normalized, 1)
                    normalized = sum(
                        full[y, x] * padded[y : y + 32, x : x + 32]
                        for y in range(3)
                        for x in range(3)
                    )
                    support = np.zeros((32, 32), bool)
                    support[1:-1, 1:-1] = True
                shifted = np.rint(normalized * maximum).astype(source.dtype)
                image = (
                    shifted
                    if source.dtype == np.uint8
                    else np.rint(shifted.astype(float) / 65535 * 255).astype(np.uint8)
                )
                query = patches(image)
                net.setInput((image.astype(np.float32) * np.float32(1 / 255))[None, None], "image")
                native = net.forward("local")[0].transpose(1, 2, 0).reshape(-1, 4)
                native_error = max(native_error, float(np.max(np.abs(native - query))))
                difference = query.astype(float)[:, None] - reference.astype(float)[None]
                distance = (
                    np.sqrt(np.sum(difference * difference, axis=-1)).min(axis=1).reshape(16, 16)
                )
                distance = np.repeat(np.repeat(distance, 2, 0), 2, 1)
                mask = (distance > (0.95 if case == "loose_threshold" else 0.2)) & support
                current_truth = moved(known, dx)
                computed[variant] = {
                    "mask": mask,
                    "truth": current_truth,
                    "support": support,
                    "score": float(distance[support].max()),
                    "passed": not bool(mask.any()),
                    "dx": dx,
                }
            baseline = computed["baseline"]
            for variant, current in computed.items():
                common = moved(baseline["support"], current["dx"]) & current["support"]
                records[f"{index}:{variant}"] = {
                    "score": current["score"],
                    "score_delta": current["score"] - baseline["score"],
                    "passed": current["passed"],
                    "expected_pass": name == "normal.png",
                    "truth_iou": intersection(
                        current["mask"], current["truth"], current["support"]
                    ),
                    "baseline_agreement_iou": intersection(
                        moved(baseline["mask"], current["dx"]), current["mask"], common
                    ),
                    "shared_support_fraction": float(common.mean()),
                    "anomaly_pixels": int(current["mask"].sum()),
                }
        normal = sum(row["expected_pass"] for row in records.values())
        defects = len(records) - normal
        false_rejects = sum(row["expected_pass"] and not row["passed"] for row in records.values())
        false_accepts = sum(not row["expected_pass"] and row["passed"] for row in records.values())
        output[case] = {
            "records": records,
            "false_reject_rate": false_rejects / normal if normal else None,
            "false_accept_rate": false_accepts / defects if defects else None,
        }
    return {
        "cases": output,
        "maximum_native_patch_difference": native_error,
        "limits": "Controlled original synthetic perturbations; "
        "paired views are not independent captures",
    }


if __name__ == "__main__":
    print(json.dumps(measure(), indent=2, sort_keys=True, allow_nan=False))
