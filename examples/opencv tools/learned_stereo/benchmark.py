"""Independent direct native inference and source-pixel error on fixed artificial planes."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def benchmark():
    model = HERE / ".local/hitnet/model.onnx"
    raw = model.read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "8c5505ab77f83c7e9aee6901b2a6647fa1dcfc57e5775c5a903c6b555c210cb2"
    )
    net = cv2.dnn.readNetFromONNX(np.frombuffer(raw, np.uint8), engine=cv2.dnn.ENGINE_NEW)
    left = cv2.imread(str(HERE / "assets/native_left.png"))
    result = {
        "opencv": cv2.__version__,
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "source_size": [320, 240],
        "network_size": [640, 480],
        "units": "original_left_pixels",
        "geometry": "artificial horizontal shifts of a single licensed image; not physical stereo",
        "cases": {},
    }
    for name, d in (("native_zero", 0), ("native_8", 8), ("native_16", 16)):
        right = cv2.imread(str(HERE / "assets" / (name + ".png")))
        blobs = []
        for im in (left, right):
            gray = cv2.cvtColor(im.astype(np.float32) / 255, cv2.COLOR_BGR2GRAY)
            blobs.append(cv2.resize(gray, (640, 480), interpolation=cv2.INTER_LINEAR))
        blob = np.stack(blobs)[None]
        net.setInput(blob, "input")
        head = net.forward(["reference_output_disparity"])[0][0, :, :, 0]
        # Source centers map to 2*x+0.5,2*y+0.5: exact four equal contributors.
        source = (
            head[0::2, 0::2].astype(np.float64)
            + head[0::2, 1::2]
            + head[1::2, 0::2]
            + head[1::2, 1::2]
        ) / 8
        start = max(24, d + 24)
        error = np.abs(source[24:-24, start:-24] - d)
        support = (head[0::2, 0::2] >= 0) & (head[0::2, 1::2] >= 0)
        support &= (head[1::2, 0::2] >= 0) & (head[1::2, 1::2] >= 0)
        right_x = np.arange(320)[None, :] - source
        support &= (right_x >= 0) & (right_x <= 319)
        retained = support[24:-24, start:-24]
        result["cases"][name] = {
            "truth_disparity": d,
            "evaluation_rectangle": [start, 24, 320 - 24 - start, 192],
            "pixels": int(error.size),
            "mean_absolute_error": float(error.mean()),
            "median_disparity": float(np.median(source[24:-24, start:-24])),
            "bad1_fraction": float((error > 1).mean()),
            "retained_pixels": int(retained.sum()),
            "retained_fraction": float(retained.mean()),
            "retained_mean_absolute_error": float(error[retained].mean()),
            "source_samples": [
                {"pixel": [x, y], "value": float(source[y, x])}
                for x, y in [(0, 0), (319, 239), (160, 120), (72, 89), (250, 190)]
            ],
        }
    return result


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
