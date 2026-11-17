"""Exact native K-nearest image classifiers with explicit unknown-result policies."""

import json

import cv2
import numpy as np

from .classifier_common import (
    canonical_json,
    descriptor_matrix,
    feature_contract,
    label_strings,
    model_revision,
    normalize_features,
    prediction_drawing,
    validate_prediction_image,
)
from .core import Block, Port, Result
from .ml_common import local_path
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, FlowGroup

MAX_MODEL_BYTES = 8 * 1024**2
MAX_WORK = 64_000_000
DISTANCE_POLICIES = ("winning_class_nearest", "nearest", "mean", "kth")
MODEL_KEYS = {
    "model",
    "algorithm",
    "default_k",
    "normalization",
    "samples",
    "responses",
    "class_labels",
    "dimensions",
    "feature_contract",
    "opencv_major",
    "revision",
}


def parse_knn(model):
    if (
        not isinstance(model, dict)
        or set(model) != MODEL_KEYS
        or model.get("model") != "opencv_knn_classifier_v1"
    ):
        raise ValidationError("Connect a portable K-nearest classifier model")
    if len(canonical_json(model).encode()) > MAX_MODEL_BYTES:
        raise ValidationError("K-nearest model exceeds 8 MiB")
    if model["revision"] != model_revision(model):
        raise ValidationError("K-nearest model revision differs")
    if (
        model["algorithm"] != "brute_force"
        or model["normalization"] not in ("none", "l2")
        or model["opencv_major"] != int(cv2.__version__.split(".")[0])
    ):
        raise ValidationError("K-nearest algorithm, normalization or OpenCV major version differs")
    values, _ = descriptor_matrix(model["samples"], max_abs=1e15)
    if not np.array_equal(np.asarray(model["samples"], np.float64), values.astype(np.float64)):
        raise ValidationError("Model samples must retain exact float32 values")
    if type(model["dimensions"]) is not int or model["dimensions"] != values.shape[1]:
        raise ValidationError("K-nearest model dimensions differ from its sample rows")
    classes = label_strings(model["class_labels"], "Class mapping", maximum=64)
    if not classes or len(set(classes)) != len(classes):
        raise ValidationError("K-nearest needs 1..64 distinct class labels")
    ids = model["responses"]
    if (
        not isinstance(ids, list)
        or len(ids) != len(values)
        or any(type(i) is not int or not 0 <= i < len(classes) for i in ids)
        or set(ids) != set(range(len(classes)))
    ):
        raise ValidationError("Model responses must map every row and stored class")
    k = model["default_k"]
    if type(k) is not int or not 1 <= k <= min(64, len(values)):
        raise ValidationError("Stored K must be in [1,min(64,training rows)]")
    feature_contract(model["feature_contract"])
    if model["normalization"] == "l2" and not np.allclose(
        np.linalg.norm(values.astype(np.float64), axis=1), 1, rtol=0, atol=2e-6
    ):
        raise ValidationError("Stored L2-normalized samples must have unit length")
    return values, np.float32(ids)


def native_knn(values, ids, k):
    native = cv2.ml.KNearest_create()
    native.setAlgorithmType(cv2.ml.KNearest_BRUTE_FORCE)
    native.setIsClassifier(True)
    native.setDefaultK(k)
    if not native.train(values, cv2.ml.ROW_SAMPLE, ids) or not native.isTrained():
        raise ValidationError("OpenCV could not train the K-nearest classifier")
    return native


class _KNNBlock(Block):
    category, block_type, subtype = "ML", "ML", "Classical classification"
    process_inputs_readonly = True
    preview_safe = True
    parallel_safe = True


class TrainKNN(_KNNBlock):
    type_id, title = "ml.knn_train", "Train OpenCV K-nearest classifier"
    inputs = (
        Port("samples", "any"),
        Port("labels", "data", required=False, required_when_connected=True),
    )
    outputs = (Port("model", "data"), Port("report", "data"))
    defaults = {"k": 3, "normalization": "none", "labels": []}
    parameter_choices = {"normalization": ("none", "l2")}
    parameter_ranges = {"k": NumericRange(1, 64, whole=True)}
    parameter_help = {
        "k": "Stored default neighbor count; must not exceed the number of training rows. "
        "Prediction can explicitly override it. No silent native clamping.",
        "normalization": "None or per-row L2; saved rows are already normalized and the same "
        "normalization is applied to incoming descriptors. Zero vectors cannot use L2.",
        "labels": "One string per row, in training/batch order. Wired Labels overrides this "
        "list. Native vote ties prefer the first-appearing encoded class; prediction rejects "
        "ties by default. Stored models include the descriptor rows, not just a summary.",
    }
    input_help = {
        "samples": "Numeric feature rows or a batch of image descriptor vectors. "
        "Known preprocessing contracts must agree."
    }
    payload_reference = "samples"

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        label_strings(p["labels"], "Training labels")

    def process(self, inputs):
        p = self.parameters
        values, contract = descriptor_matrix(inputs["samples"], max_abs=1e15)
        if not isinstance(inputs["samples"], FlowGroup):
            contract = self.input_payloads["samples"].get("feature_contract")
        labels = label_strings(inputs.get("labels", p["labels"]), "Training labels")
        if len(labels) != len(values):
            raise ValidationError("Supply exactly one class label per training row", "labels")
        classes = list(dict.fromkeys(labels))
        if not 1 <= len(classes) <= 64:
            raise ValidationError("K-nearest training requires 1..64 classes", "labels")
        if p["k"] > len(values):
            raise ValidationError("K exceeds the training row count", "k")
        values = normalize_features(values, p["normalization"])
        ids = [classes.index(label) for label in labels]
        model = {
            "model": "opencv_knn_classifier_v1",
            "algorithm": "brute_force",
            "default_k": int(p["k"]),
            "normalization": p["normalization"],
            "samples": values.tolist(),
            "responses": ids,
            "class_labels": classes,
            "dimensions": values.shape[1],
            "feature_contract": feature_contract(contract),
            "opencv_major": int(cv2.__version__.split(".")[0]),
        }
        model["revision"] = model_revision(model)
        parse_knn(model)
        native_knn(values, np.float32(ids), int(p["k"]))
        return Result(
            {
                "model": model,
                "report": {
                    "samples": len(values),
                    "dimensions": values.shape[1],
                    "classes": classes,
                    "class_counts": [labels.count(c) for c in classes],
                    "k": int(p["k"]),
                    "algorithm": "brute_force",
                    "revision": model["revision"],
                },
            },
            message=f"Trained {len(classes)} classes on {len(values)} descriptors",
        )


class PredictKNN(_KNNBlock):
    type_id, title = "ml.knn_predict", "Predict OpenCV K-nearest classes"
    inputs = (
        Port("model", "data"),
        Port("samples", "data"),
        Port("image", required=False, required_when_connected=True),
    )
    outputs = (
        Port("labels", "data"),
        Port("label", "data", required=False),
        Port("candidate_labels", "data"),
        Port("distances", "data"),
        Port("neighbors", "data"),
        Port("accepted", "data"),
        Port("image", required=False),
        Port("report", "data"),
    )
    payload_reference = "samples"
    defaults = {
        "k": 0,
        "contract_policy": "require_match",
        "max_distance": 0.0,
        "distance_policy": "winning_class_nearest",
        "min_vote_fraction": 0.0,
        "reject_vote_ties": True,
        "accepted_labels": [],
    }
    parameter_choices = {
        "contract_policy": ("require_match", "dimensions_only"),
        "distance_policy": DISTANCE_POLICIES,
    }
    parameter_ranges = {
        "k": NumericRange(0, 64, whole=True),
        "max_distance": NumericRange(0, 1e18),
        "min_vote_fraction": NumericRange(0, 1),
    }
    parameter_help = {
        "k": "0 uses the stored model default; 1..64 overrides it. Must not exceed training rows.",
        "contract_policy": "Require exact descriptor preprocessing, or explicitly check "
        "dimensions only. Stored normalization always applies.",
        "max_distance": "Inclusive Euclidean limit for the selected neighbor-distance statistic. "
        "0 disables distance rejection. Native squared distances are also reported.",
        "distance_policy": "Nearest neighbor of the voted class, nearest of all neighbors, "
        "mean Euclidean distance of all K, or Kth/farthest returned neighbor.",
        "min_vote_fraction": "Minimum fraction of K neighbors voting for the native candidate. "
        "This is vote agreement, not a calibrated probability.",
        "reject_vote_ties": "Withhold recognized labels when two or more classes tie for most "
        "votes. Raw candidate and neighbor diagnostics remain available.",
        "accepted_labels": "Optional allowed trained classes for inspection pass/fail; [] "
        "accepts all recognized classes. A disallowed class is still a recognized label.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        label_strings(p["accepted_labels"], "Accepted labels", maximum=64)

    def process(self, inputs):
        model = inputs["model"]
        training, ids = parse_knn(model)
        values, _ = descriptor_matrix(inputs["samples"], max_abs=1e15)
        contract = feature_contract(self.input_payloads["samples"].get("feature_contract"))
        p = self.parameters
        if p["contract_policy"] == "require_match" and contract != model["feature_contract"]:
            raise ValidationError(
                "Prediction preprocessing differs from the trained feature contract"
            )
        if values.shape[1] != training.shape[1]:
            raise ValidationError("Prediction dimensions differ from the trained model")
        k = int(p["k"] or model["default_k"])
        if k > len(training):
            raise ValidationError("K exceeds the training row count", "k")
        if values.size * len(training) > MAX_WORK:
            raise ValidationError("K-nearest work exceeds 64 million feature comparisons")
        if not set(p["accepted_labels"]) <= set(model["class_labels"]):
            raise ValidationError(
                "Accepted labels must be in the trained class mapping", "accepted_labels"
            )
        values = normalize_features(values, model["normalization"])
        image = validate_prediction_image(self, inputs, len(values))
        native = native_knn(training, ids, k)
        _, predicted, neighbors, squared = native.findNearest(values, k)
        classes = model["class_labels"]
        if (
            predicted is None
            or neighbors is None
            or squared is None
            or predicted.shape != (len(values), 1)
            or neighbors.shape != (len(values), k)
            or squared.shape != neighbors.shape
            or not np.isfinite(squared).all()
            or np.any(squared < 0)
        ):
            raise ValidationError("OpenCV returned invalid K-nearest output shapes/distances")
        for rows in (predicted, neighbors):
            if (
                not np.isfinite(rows).all()
                or np.any(rows != np.rint(rows))
                or np.any(rows < 0)
                or np.any(rows >= len(classes))
            ):
                raise ValidationError("OpenCV returned an invalid K-nearest class ID")
        if np.any(np.diff(squared, axis=1) < 0):
            raise ValidationError("OpenCV neighbors are not in distance order")
        candidates = predicted.ravel().astype(int)
        neighbor_ids = neighbors.astype(int)
        euclidean = np.sqrt(squared.astype(np.float64))
        votes = np.array([np.bincount(row, minlength=len(classes)) for row in neighbor_ids])
        maxima = votes.max(axis=1)
        if np.any(votes[np.arange(len(values)), candidates] != maxima):
            raise ValidationError("OpenCV candidate disagrees with neighbor majority votes")
        fractions = maxima / k
        ties = np.count_nonzero(votes == maxima[:, None], axis=1) > 1
        if p["distance_policy"] == "nearest":
            distances = euclidean[:, 0]
        elif p["distance_policy"] == "kth":
            distances = euclidean[:, -1]
        elif p["distance_policy"] == "mean":
            distances = euclidean.mean(axis=1)
        else:
            distances = np.where(neighbor_ids == candidates[:, None], euclidean, np.inf).min(axis=1)
        known = np.ones(len(values), bool)
        reasons = [[] for _ in values]
        for index in range(len(values)):
            if p["max_distance"] > 0 and distances[index] > p["max_distance"]:
                reasons[index].append("knn_distance")
            if fractions[index] < p["min_vote_fraction"]:
                reasons[index].append("knn_vote_fraction")
            if p["reject_vote_ties"] and ties[index]:
                reasons[index].append("knn_vote_tie")
            known[index] = not reasons[index]
        candidate_labels = [classes[i] for i in candidates]
        labels = [label if known[index] else None for index, label in enumerate(candidate_labels)]
        accepted = known & np.array(
            [
                not p["accepted_labels"] or label in p["accepted_labels"]
                for label in candidate_labels
            ]
        )
        passed = bool(accepted.all())
        text = f"{labels[0] or 'Unknown'} · d={distances[0]:.3g} · votes={fractions[0]:.0%}"
        drawings = prediction_drawing(self, image, text, passed, "K-nearest class")
        report = {
            "count": len(values),
            "k": k,
            "revision": model["revision"],
            "class_mapping": classes,
            "vote_counts": votes.tolist(),
            "vote_fractions": fractions.tolist(),
            "vote_ties": ties.tolist(),
            "unknown_reasons": reasons,
            "unknown_count": int(np.count_nonzero(~known)),
            "distance_policy": p["distance_policy"],
            "probabilities_available": False,
            "judgment_settings": p.copy(),
        }
        fail_reasons = []
        if not known.all():
            fail_reasons.append("knn_unknown_class")
        if np.any(known & ~accepted):
            fail_reasons.append("knn_unaccepted_label")
        return Result(
            {
                "labels": labels,
                "label": labels[0] if len(labels) == 1 and labels[0] is not None else NO_VALUE,
                "candidate_labels": candidate_labels,
                "distances": distances.tolist(),
                "neighbors": [
                    {
                        "labels": [classes[i] for i in row],
                        "distances": distance.tolist(),
                        "squared_distances": square.tolist(),
                    }
                    for row, distance, square in zip(neighbor_ids, euclidean, squared, strict=True)
                ],
                "accepted": accepted.tolist(),
                "image": image,
                "report": report,
            },
            drawings=drawings,
            passed=passed,
            fail_reasons=fail_reasons,
            message=text
            if len(values) == 1
            else f"{len(values)} predictions · {report['unknown_count']} unknown",
        )


class SaveKNN(_KNNBlock):
    type_id, title = "ml.knn_save", "Save K-nearest classifier"
    category, block_type, subtype = "Output", "Output", "ML models"
    inputs, outputs = (Port("model", "data"),), (Port("path", "data"),)
    defaults = {"path": "knn-classifier.json"}
    preview_safe = False

    def process(self, inputs):
        model = inputs["model"]
        parse_knn(model)
        path = local_path(self, self.parameters["path"])
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical_json(model) + "\n")
        return Result(
            {"path": str(path)}, message="Saved K-nearest model; existing files preserved"
        )


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValidationError("Duplicate K-nearest model JSON key: " + key)
        result[key] = value
    return result


def _constant(token):
    raise ValidationError("Nonfinite K-nearest model JSON number: " + token)


class LoadKNN(_KNNBlock):
    type_id, title = "ml.knn_load", "Load K-nearest classifier"
    outputs = (Port("model", "data"),)
    defaults = {"path": ""}
    project_asset_parameters = (*Block.project_asset_parameters, "path")

    def process(self, inputs):
        path = local_path(self, self.parameters["path"])
        with path.open("rb") as stream:
            raw = stream.read(MAX_MODEL_BYTES + 1)
        if len(raw) > MAX_MODEL_BYTES:
            raise ValidationError("K-nearest model exceeds 8 MiB")
        model = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant)
        parse_knn(model)
        return Result(
            {"model": model}, message=f"Loaded {len(model['class_labels'])} K-nearest classes"
        )


KNN_BLOCKS = (TrainKNN, PredictKNN, SaveKNN, LoadKNN)
