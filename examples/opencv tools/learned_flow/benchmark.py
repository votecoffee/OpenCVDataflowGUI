"""Direct named-head RAFT and independent four-contributor source-vector reference."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def benchmark():
    model = HERE / ".local/raft/model.onnx"
    raw = model.read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "3b479e0447894245b822bf0b7392155d31ccd22bb9b0ee55503066bf6f6f869a"
    )
    # Path overload preserves Classic dispatch on this OpenCV build.
    net = cv2.dnn.readNetFromONNX(str(model), engine=cv2.dnn.ENGINE_CLASSIC)
    first = cv2.imread(str(HERE / "assets/native_first.png"))
    result = {
        "opencv": cv2.__version__,
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "source_size": [240, 180],
        "network_size": [480, 360],
        "direction": "first_to_second",
        "units": "original_first_pixels",
        "selected_output": "12006",
        "scope": "known translations of one licensed image; not recorded optical motion",
        "cases": {},
    }
    for name, motion in (
        ("native_positive", [4, -2]),
        ("native_negative", [-3, 2]),
        ("native_zero", [0, 0]),
    ):
        second = cv2.imread(str(HERE / "assets" / (name + ".png")))
        for binding, image in zip(("0", "1"), (first, second)):
            rgb = image[..., ::-1].astype(np.float32) / 255
            blob = cv2.resize(rgb, (480, 360), interpolation=cv2.INTER_LINEAR) / np.float32(1 / 255)
            net.setInput(np.ascontiguousarray(blob.transpose(2, 0, 1)[None]), binding)
        names = net.getUnconnectedOutLayersNames()
        heads = dict(zip(names, net.forward(names), strict=True))
        head = heads["12006"][0].transpose(1, 2, 0)
        vectors = (
            head[0::2, 0::2].astype(np.float64)
            + head[0::2, 1::2]
            + head[1::2, 0::2]
            + head[1::2, 1::2]
        ) / 8
        x = np.arange(240)[None, :] + vectors[..., 0]
        y = np.arange(180)[:, None] + vectors[..., 1]
        valid = (x >= 0) & (x <= 239) & (y >= 0) & (y <= 179)
        error = np.linalg.norm(vectors[20:-20, 20:-20] - motion, axis=2)
        support = valid[20:-20, 20:-20]
        result["cases"][name] = {
            "truth_motion": motion,
            "evaluation_rectangle": [20, 20, 200, 140],
            "pixels": int(error.size),
            "mean_endpoint_error": float(error.mean()),
            "bad1_fraction": float((error > 1).mean()),
            "retained_pixels": int(support.sum()),
            "retained_fraction": float(support.mean()),
            "retained_mean_endpoint_error": float(error[support].mean()),
            "source_samples": [
                {"pixel": [x, y], "vector": vectors[y, x].tolist()}
                for x, y in [(0, 0), (239, 179), (120, 90), (47, 62), (195, 141)]
            ],
        }
    return result


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
