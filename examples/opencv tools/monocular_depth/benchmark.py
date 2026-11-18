"""Direct-native relative depth and evaluation-only alignment against measured T-LESS depth."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def bilinear_reference(head, size):
    width, height = size
    hh, hw = head.shape
    # Independent bilinear sum: this OpenCV build uses full float-map coordinates,
    # unlike the separately quantized convertMaps representation.
    x = ((np.arange(width) + 0.5) * hw / width - 0.5).astype(np.float32).astype(np.float64)
    y = ((np.arange(height) + 0.5) * hh / height - 0.5).astype(np.float32).astype(np.float64)
    ix = np.floor(x).astype(int)
    iy = np.floor(y).astype(int)
    fx = x - ix
    fy = y - iy
    x0, x1 = np.clip(ix, 0, hw - 1), np.clip(ix + 1, 0, hw - 1)
    y0, y1 = np.clip(iy, 0, hh - 1), np.clip(iy + 1, 0, hh - 1)
    low = head[y0[:, None], x0] * (1 - fx) + head[y0[:, None], x1] * fx
    high = head[y1[:, None], x0] * (1 - fx) + head[y1[:, None], x1] * fx
    return low * (1 - fy[:, None]) + high * fy[:, None]


def errors(prediction, reference):
    valid = np.isfinite(prediction) & (prediction > 0)
    if not valid.any():
        return {"valid_fraction": 0.0, "rmse_m": None, "absolute_relative": None, "delta1": None}
    p, t = prediction[valid], reference[valid]
    return {
        "valid_fraction": float(valid.mean()),
        "rmse_m": float(np.sqrt(np.square(p - t).mean())),
        "absolute_relative": float((np.abs(p - t) / t).mean()),
        "delta1": float((np.maximum(p / t, t / p) < 1.25).mean()),
    }


def benchmark():
    truth = json.loads((HERE / "truth.json").read_bytes())
    model = HERE / ".local/native_relative/model.onnx"
    assert hashlib.sha256(model.read_bytes()).hexdigest() == truth["midas_sha256"]
    image = cv2.imread(str(HERE / "assets/tless_color.png"))
    h, w = image.shape[:2]
    tensor = cv2.resize(
        image[..., ::-1].astype(np.float32) / 255, (256, 256), interpolation=cv2.INTER_CUBIC
    )
    net = cv2.dnn.readNetFromONNX(str(model), cv2.dnn.ENGINE_CLASSIC)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    net.enableFusion(True)
    net.setInput(np.ascontiguousarray(tensor.transpose(2, 0, 1)[None]), "0")
    head = net.forward(["797"])[0][0]
    prediction = bilinear_reference(head, [w, h])
    raw = cv2.imread(str(HERE / "assets/tless_depth.png"), cv2.IMREAD_UNCHANGED)
    depth = raw.astype(np.float64) * 0.0001
    yy, xx = np.indices((h, w))
    valid = (depth > 0) & np.isfinite(prediction) & (prediction > 0)
    fit = valid & (xx < w // 2) & (xx % 8 == 0) & (yy % 8 == 0)
    held = valid & (xx >= w // 2)
    center = float(prediction[fit].mean())
    scale = float(prediction[fit].std())
    a, b = np.linalg.lstsq(
        np.column_stack(((prediction[fit] - center) / scale, np.ones(int(fit.sum())))),
        1 / depth[fit],
        rcond=None,
    )[0]
    slope = float(a / scale)
    offset = float(b - slope * center)
    inverse = slope * prediction[held] + offset
    fitted = np.full(inverse.shape, np.nan)
    np.divide(1, inverse, out=fitted, where=inverse > 0)
    baseline = np.full(int(held.sum()), float(np.median(depth[fit])))
    random = np.random.default_rng(530)
    indices = np.flatnonzero(held)
    left, right = random.choice(indices, (2, 5000), replace=True)
    z = depth.ravel()
    q = prediction.ravel()
    separated = np.abs(z[left] - z[right]) >= 0.02
    correct = (q[left] - q[right]) * (z[left] - z[right]) < 0
    points = [
        [0, 0],
        [w - 1, 0],
        [0, h - 1],
        [w - 1, h - 1],
        [w // 2, h // 2],
        [123, 147],
        [501, 326],
    ]
    return {
        "opencv": cv2.__version__,
        "artifact_sha256": truth["midas_sha256"],
        "source_size": [w, h],
        "head_size": [256, 256],
        "convention": "relative_inverse_depth",
        "metric_output": False,
        "head_minimum": float(head.min()),
        "head_maximum": float(head.max()),
        "source_samples": [{"pixel": p, "value": float(prediction[p[1], p[0]])} for p in points],
        "known_depth_pixels": int(valid.sum()),
        "fit_pixels": int(fit.sum()),
        "heldout_pixels": int(held.sum()),
        "ordering_pairs": int(separated.sum()),
        "near_far_order_accuracy": float(correct[separated].mean()),
        "evaluation_only_alignment": {
            "family": "inverse_meters = a * relative_value + b",
            "fit": "left image half, 8-pixel stride; measured depths used only for evaluation",
            "validation": "right image half; same captured frame, not another acquisition",
            "slope": slope,
            "offset": offset,
            "positive_inverse_slope": slope > 0,
            "aligned": errors(fitted, depth[held]),
            "constant_calibration_median": errors(baseline, depth[held]),
            "runtime_calibration": False,
        },
        "scope": "single licensed real frame; scale/shift alignment is diagnostic, "
        "not physical model calibration",
    }


if __name__ == "__main__":
    path = HERE / "benchmark.json"
    raw = (json.dumps(benchmark(), indent=2, allow_nan=False) + "\n").encode()
    if path.exists() and path.read_bytes() != raw:
        raise FileExistsError("Different retained benchmark preserved")
    path.write_bytes(raw)
    print(path)
