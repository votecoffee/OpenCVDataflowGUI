"""Exhaustive descriptor-dot-product search, independent IoU suppression and known boxes."""

import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def measure():
    model = json.loads((HERE / "assets/detector.json").read_bytes())
    source = cv2.imread(str(HERE / "assets/scene.png"), cv2.IMREAD_GRAYSCALE)
    g = model["geometry"]
    hog = cv2.HOGDescriptor(
        tuple(g["window_size"]),
        tuple(g["block_size"]),
        tuple(g["block_stride"]),
        tuple(g["cell_size"]),
        g["bins"],
        _winSigma=g["win_sigma"],
        _L2HysThreshold=g["l2_hys_threshold"],
        _gammaCorrection=g["gamma_correction"],
        _signedGradient=g["signed_gradient"],
        _nlevels=3,
    )
    coefficients = np.array(model["coefficients"], np.float32)
    hog.setSVMDetector(coefficients)
    expected = []
    window_count = 0
    for level in range(3):
        scale = 1.25**level
        size = (round(source.shape[1] / scale), round(source.shape[0] / scale))
        image = (
            source if level == 0 else cv2.resize(source, size, interpolation=cv2.INTER_LINEAR_EXACT)
        )
        for y in range(0, size[1] - 32 + 1, 8):
            for x in range(0, size[0] - 32 + 1, 8):
                descriptor = hog.compute(image, locations=[(x, y)]).reshape(-1).astype(float)
                margin = float(descriptor @ coefficients[:-1].astype(float) + coefficients[-1])
                window_count += 1
                expected.append(
                    {
                        "box": [
                            round(x * scale),
                            round(y * scale),
                            round(32 * scale),
                            round(32 * scale),
                        ],
                        "margin": margin,
                    }
                )
    cases = {}
    for threshold in (0.0, 0.5, 0.8):
        native_boxes, native_weights = hog.detectMultiScale(
            source,
            hitThreshold=threshold,
            winStride=(8, 8),
            padding=(0, 0),
            scale=1.25,
            groupThreshold=0,
        )
        native = {
            tuple(map(int, b)): float(s)
            for b, s in zip(native_boxes, np.asarray(native_weights).reshape(-1), strict=True)
        }
        rows = [row for row in expected if row["margin"] >= threshold]
        independent = {tuple(row["box"]): row["margin"] for row in rows}
        pending = sorted(rows, key=lambda row: (-row["margin"], *row["box"]))
        retained = []
        while pending:
            first = pending.pop(0)
            retained.append(first)
            a = first["box"]

            def overlap(row):
                b = row["box"]
                intersection = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])) * max(
                    0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
                )
                return intersection / (a[2] * a[3] + b[2] * b[3] - intersection)

            pending = [row for row in pending if overlap(row) <= 0.2]
        cases[str(threshold)] = {
            "independent_candidates": len(rows),
            "native_candidates": len(native_boxes),
            "differing_boxes": len(set(native) ^ set(independent)),
            "maximum_margin_error": max(
                (abs(native[key] - independent[key]) for key in set(native) & set(independent)),
                default=0,
            ),
            "retained": retained,
        }
    return {
        "opencv": cv2.__version__,
        "windows": window_count,
        "reference": "Native HOG descriptors with independent float64 dot products "
        "and scalar IoU suppression",
        "scope": "One original controlled query, separate training noise; "
        "thresholds illustrate behavior, not unbiased field accuracy",
        "cases": cases,
    }


if __name__ == "__main__":
    print(json.dumps(measure(), indent=2, allow_nan=False))
