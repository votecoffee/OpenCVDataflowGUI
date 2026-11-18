"""Independent native HOG, NumPy SVD and nearest-neighbor image comparison."""

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
        _nbins=9,
        _winSigma=-1,
        _L2HysThreshold=0.2,
        _gammaCorrection=True,
        _signedGradient=False,
    )

    def read(name):
        image = cv2.imdecode(np.frombuffer((HERE / "assets" / name).read_bytes(), np.uint8), 0)
        return hog.compute(image).ravel().astype(np.float64)

    training = np.array([read(name) for name in truth["training"]])
    queries = {
        name: read(name)
        for name in ("heldout-vertical.png", "heldout-horizontal.png", "heldout-diagonal.png")
    }
    results = {}
    for mode in ("none", "l2", "variance"):
        values = training.copy()
        if mode == "l2":
            values /= np.linalg.norm(values, axis=1, keepdims=True)
        mean = values.mean(axis=0)
        _, singular, vectors = np.linalg.svd(values - mean, full_matrices=False)
        eigenvalues = singular**2 / len(values)
        ratios = eigenvalues / eigenvalues.sum()
        count = int(np.searchsorted(ratios.cumsum(), 0.95)) + 1 if mode == "variance" else 2
        basis = vectors[:count]
        reduced = (values - mean) @ basis.T
        saved = json.loads((HERE / "assets" / f"pca-{mode}.json").read_bytes())
        actual_basis = np.asarray(saved["basis"])
        projector_error = float(np.max(np.abs(basis.T @ basis - actual_basis.T @ actual_basis)))
        rows = []
        for name, raw in queries.items():
            query = raw / np.linalg.norm(raw) if mode == "l2" else raw
            scores = (query - mean) @ basis.T
            restored = scores @ basis + mean
            nearest = np.linalg.norm(reduced - scores, axis=1).argmin()
            raw_nearest = np.linalg.norm(values - query, axis=1).argmin()
            rows.append(
                {
                    "image": name,
                    "raw_label": truth["labels"][raw_nearest],
                    "reduced_label": truth["labels"][nearest],
                    "reconstruction_rmse": float(np.sqrt(np.mean((query - restored) ** 2))),
                    "nearest_reduced_distance": float(np.linalg.norm(reduced[nearest] - scores)),
                }
            )
        results[mode] = {
            "training_rows": len(values),
            "input_dimensions": values.shape[1],
            "components": count,
            "retained_training_variance": float(ratios[:count].sum()),
            "saved_projector_max_error": projector_error,
            "heldout": rows,
        }
    return {
        "opencv": cv2.__version__,
        "reference": "native HOG and independent NumPy SVD",
        "scope": "Controlled disjoint stripe windows, not general classification accuracy",
        "modes": results,
    }


if __name__ == "__main__":
    print(json.dumps(measure(), indent=2, allow_nan=False))
