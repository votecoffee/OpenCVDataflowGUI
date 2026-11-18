"""Independent native HOG, NumPy covariance and quadratic-distance references."""

import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def measure():
    truth = json.loads((HERE / "truth.json").read_bytes())
    hog = cv2.HOGDescriptor(
        _winSize=(32, 32),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=4,
        _winSigma=-1,
        _L2HysThreshold=0.2,
        _gammaCorrection=True,
        _signedGradient=False,
    )

    def descriptor(name):
        raw = (HERE / "assets" / name).read_bytes()
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), 0)
        return hog.compute(image).ravel().astype(np.float64)

    samples = np.array([descriptor(name) for name in truth["training"]])
    cases = {}
    for mode in ("none", "l2"):
        values = (
            samples / np.linalg.norm(samples, axis=1, keepdims=True) if mode == "l2" else samples
        )
        mean = values.mean(0)
        centered = values - mean
        covariance = centered.T @ centered / (len(values) - 1)
        saved = json.loads((HERE / "assets" / f"covariance-{mode}.json").read_bytes())
        matrix_error = float(np.max(np.abs(covariance - np.asarray(saved["covariance"]))))
        ridge = truth["ridge_l2"] if mode == "l2" else truth["ridge_none"]
        effective = covariance + np.eye(values.shape[1]) * ridge
        inverse = np.linalg.inv(effective)
        _, native_inverse = cv2.invert(effective, flags=cv2.DECOMP_SVD)
        results = []
        for name in [
            *truth["known"],
            "crack.png",
            "missing-arc.png",
            "low-contrast.png",
            "empty.png",
        ]:
            query = descriptor(name)
            if mode == "l2":
                if not query.any():
                    results.append(
                        {"image": name, "supported": False, "reason": "zero L2 descriptor"}
                    )
                    continue
                query /= np.linalg.norm(query)
            delta = query - mean
            distance = float(np.sqrt(delta @ inverse @ delta))
            native = float(cv2.Mahalanobis(query, mean, native_inverse))
            results.append(
                {
                    "image": name,
                    "supported": True,
                    "numpy_distance": distance,
                    "native_distance": native,
                    "native_numpy_error": abs(native - distance),
                    "passed": distance <= truth["distance_limit"],
                }
            )
        cases[mode] = {
            "training_images": len(values),
            "dimensions": values.shape[1],
            "covariance_max_error": matrix_error,
            "ridge": ridge,
            "maximum_native_numpy_error": max(r.get("native_numpy_error", 0) for r in results),
            "heldout": results,
        }
    return {
        "opencv": cv2.__version__,
        "reference": "native HOG and independent NumPy covariance/quadratic form",
        "scope": "Original controlled windows; distances are not probabilities or defect masks",
        "modes": cases,
    }


if __name__ == "__main__":
    print(json.dumps(measure(), indent=2, allow_nan=False))
