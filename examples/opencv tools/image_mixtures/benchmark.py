"""Independent pixel/texture features, native predictions and Gaussian references."""

import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def measure():
    output = {}
    for case, model_name, filename in (
        ("color_known", "color", "query-color.png"),
        ("color_defect", "color", "defect-color.png"),
        ("spherical_color", "spherical", "query-color.png"),
        ("texture_known", "texture", "query-texture.png"),
        ("texture_defect", "texture", "defect-texture.png"),
    ):
        model = json.loads((HERE / "assets" / f"model-{model_name}.json").read_bytes())
        raw = cv2.imdecode(np.frombuffer((HERE / "assets" / filename).read_bytes(), np.uint8), -1)
        unit = raw.astype(np.float32) / np.float32(255)
        support = np.ones(unit.shape[:2], bool)
        if model_name == "texture":
            requested = np.zeros(raw.shape, bool)
            requested[4:60, 4:44] = True
            requested[4:60, 52:92] = True
            support[:] = False
            rows = []
            for y in range(3, 61):
                for x in range(3, 93):
                    if requested[y - 3 : y + 4, x - 3 : x + 4].all():
                        window = unit[y - 3 : y + 4, x - 3 : x + 4].astype(np.float64)
                        rows.append([window.mean(), window.std()])
                        support[y, x] = True
            values = np.asarray(rows)
        else:
            values = unit[support].astype(np.float64)
        spec = model["native"]
        means = np.array(spec["means"]["data"]).reshape(spec["means"]["rows"], -1)
        weights = np.array(spec["weights"]["data"])
        covs = [np.array(c["data"]).reshape(c["rows"], c["cols"]) for c in spec["covs"]]
        storage = cv2.FileStorage(json.dumps(spec), cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
        try:
            native = cv2.ml.EM_create()
            native.read(storage.root())
        finally:
            storage.release()
        results = [native.predict2(row[None]) for row in values]
        native_density = np.array([row[0][0] for row in results])
        posterior = np.concatenate([row[1] for row in results])
        terms = []
        for weight, mean, covariance in zip(weights, means, covs, strict=True):
            centered = values - mean
            distance = np.einsum("ni,ij,nj->n", centered, np.linalg.inv(covariance), centered)
            terms.append(
                np.log(weight)
                - 0.5
                * (
                    distance
                    + np.linalg.slogdet(covariance)[1]
                    + values.shape[1] * np.log(2 * np.pi)
                )
            )
        terms = np.column_stack(terms)
        high = terms.max(1)
        reference_density = high + np.log(np.exp(terms - high[:, None]).sum(1))
        threshold = 2 if model_name == "texture" else -12
        unknown = np.zeros(support.shape, bool)
        unknown[support] = native_density < threshold
        record = {
            "supported_pixels": len(values),
            "unknown_pixels": int(unknown.sum()),
            "minimum_log_density": float(native_density.min()),
            "native_gaussian_max_error": float(np.max(np.abs(native_density - reference_density))),
            "minimum_winning_posterior": float(posterior.max(1).min()),
            "fitted_covariance": model["training"]["covariance_type"],
            "prediction_covariance": spec["training_params"]["cov_mat_type"],
        }
        if case.endswith("defect"):
            truth_name = "texture-truth.png" if model_name == "texture" else "color-truth.png"
            truth = (
                cv2.imdecode(
                    np.frombuffer((HERE / "assets" / truth_name).read_bytes(), np.uint8), 0
                )
                != 0
            )
            record.update(
                edit_pixels=int(truth.sum()),
                true_positive=int((unknown & truth).sum()),
                false_positive=int((unknown & ~truth).sum()),
                false_negative=int((~unknown & truth).sum()),
            )
        output[case] = record
    return {
        "opencv": cv2.__version__,
        "reference": "independent features and native predict2 versus Gaussian formula",
        "scope": "Original controlled surfaces; texture outputs describe neighborhood centers",
        "cases": output,
    }


if __name__ == "__main__":
    print(json.dumps(measure(), indent=2, allow_nan=False))
