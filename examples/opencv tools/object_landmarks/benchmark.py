"""Direct native inference and independent 2D coordinate/confidence/topology rules."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def measure():
    model = HERE / "bundle/model.onnx"
    network = cv2.dnn.readNetFromONNX(str(model), cv2.dnn.ENGINE_CLASSIC)
    network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    truth = json.loads((HERE / "truth.json").read_bytes())
    results = {}
    for case, expected in truth["cases"].items():
        project = json.loads((HERE / (case + ".json")).read_bytes())
        parameters = next(b["parameters"] for b in project["blocks"] if b["id"] == "decode")
        bgr = cv2.imread(str(HERE / "assets" / expected["image"]))
        # Float32 interpolation precedes normalization, matching the declared project settings.
        rgb = cv2.resize(bgr[..., ::-1].astype(np.float32), (256, 256))
        blob = np.ascontiguousarray((rgb * np.float32(1 / 255))[None])
        network.setInput(blob, "input_1")
        head, object_head = network.forward(["Identity", "Identity_1"])
        rows = head.reshape(39, 5).astype(np.float64)
        channels = np.asarray(parameters["channels"], int)
        selected = rows[channels]
        positions = selected[:, :2] * [bgr.shape[1] / 256, bgr.shape[0] / 256]
        raw_scores = selected[:, parameters["coordinate_score_columns"]]
        activated = np.exp(-np.logaddexp(0, -raw_scores))
        scores = activated.min(axis=1)
        object_confidence = float(object_head.flat[0])
        object_ok = object_confidence >= parameters["minimum_object_confidence"]
        accepted = (
            (positions >= 0).all(1)
            & (positions <= np.array([bgr.shape[1] - 1, bgr.shape[0] - 1])).all(1)
            & (scores >= parameters["minimum_confidence"])
            & object_ok
        )
        indices = np.flatnonzero(accepted).tolist()
        links = [[a, b] for a, b in parameters["connections"] if accepted[a] and accepted[b]]
        passed = bool(
            object_ok
            and len(indices) >= parameters["minimum_count"]
            and (not parameters["require_all"] or accepted.all())
        )
        results[case] = {
            "positions": positions.tolist(),
            "confidence_components": activated.tolist(),
            "confidence": scores.tolist(),
            "object_confidence": object_confidence,
            "accepted_indices": indices,
            "links": links,
            "passed": passed,
        }
    return {
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "opencv": cv2.__version__,
        "reference": "Direct classic CPU inference plus independent coordinate, "
        "sigmoid/minimum and skeleton filtering",
        "scope": "Configured adapter parity; confidence/visibility are model predictions, "
        "not annotated accuracy",
        "cases": results,
    }


if __name__ == "__main__":
    print(json.dumps(measure(), indent=2, allow_nan=False))
