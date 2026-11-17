"""Portable native image-feature forests, explicit votes and checked tree structure."""

import copy
import hashlib
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
MAX_NODES = 50_000
MAX_WORK = 64_000_000
TRAINING = {
    "max_depth": 10,
    "min_sample_count": 2,
    "active_variables": 0,
    "max_trees": 64,
    "termination": "tree_limit",
    "oob_error": 0.01,
    "calculate_importance": False,
    "class_priors": [],
}
RANGES = {
    "max_depth": NumericRange(1, 16, whole=True),
    "min_sample_count": NumericRange(1, 2048, whole=True),
    "active_variables": NumericRange(0, 65_536, whole=True),
    "max_trees": NumericRange(1, 256, whole=True),
    "oob_error": NumericRange(0, 1),
}
TERMINATION = ("tree_limit", "tree_limit_or_oob_error")
MODEL_KEYS = {
    "model",
    "opencv_major",
    "dimensions",
    "class_labels",
    "normalization",
    "feature_contract",
    "training_parameters",
    "training_summary",
    "forest_yaml",
    "revision",
}


def positive_weights(value, name, *, count=None, maximum=2048):
    if (
        not isinstance(value, list)
        or len(value) > maximum
        or any(
            type(v) not in (int, float) or not np.isfinite(v) or not 1e-6 <= v <= 1e6 for v in value
        )
        or (value and count is not None and len(value) != count)
    ):
        raise ValidationError(
            f"{name} needs positive finite weights in [1e-6,1e6], one per entry", name
        )
    return value


def validate_training(p):
    if set(p) != set(TRAINING):
        raise ValidationError("Forest training settings have unexpected fields")
    for key, limits in RANGES.items():
        limits.check(p[key], key, key)
    if p["termination"] not in TERMINATION or type(p["calculate_importance"]) is not bool:
        raise ValidationError("Invalid forest termination or importance setting")
    if p["termination"] == "tree_limit_or_oob_error" and p["oob_error"] <= 0:
        raise ValidationError("OOB termination needs a positive target", "oob_error")
    positive_weights(p["class_priors"], "class_priors", maximum=64)


def number(node, name):
    value = node.getNode(name)
    if not (value.isInt() or value.isReal()):
        raise ValidationError("Missing numeric forest field: " + name)
    result = value.real()
    if not np.isfinite(result):
        raise ValidationError("Nonfinite forest field: " + name)
    return result


def sequence(node, name, count):
    value = node.getNode(name)
    if not value.isSeq() or value.size() != count:
        raise ValidationError("Invalid forest sequence: " + name)
    if any(not (value.at(i).isInt() or value.at(i).isReal()) for i in range(count)):
        raise ValidationError("Nonnumeric forest sequence: " + name)
    values = [value.at(i).real() for i in range(count)]
    if not np.isfinite(values).all():
        raise ValidationError("Nonfinite forest sequence: " + name)
    return values


def inspect_forest(node, model):
    """Check the generated ordered-feature/native-YAML subset before native model loading."""
    dimensions, classes = model["dimensions"], model["class_labels"]
    p = model["training_parameters"]
    allowed = {
        "format",
        "is_classifier",
        "var_all",
        "var_count",
        "ord_var_count",
        "cat_var_count",
        "training_params",
        "global_var_idx",
        "var_idx",
        "var_type",
        "cat_ofs",
        "class_labels",
        "missing_subst",
        "oob_error",
        "var_importance",
        "ntrees",
        "trees",
    }
    if set(node.keys()) - allowed:
        raise ValidationError("Unsupported native forest fields")
    if (
        number(node, "format") != 3
        or number(node, "is_classifier") != 1
        or number(node, "var_count") != dimensions
        or number(node, "var_all") != dimensions + 1
        or number(node, "ord_var_count") != dimensions
        or number(node, "cat_var_count") != 1
        or number(node, "global_var_idx") != 1
    ):
        raise ValidationError("Native forest type or feature dimensions differ")
    if (
        sequence(node, "class_labels", len(classes)) != list(range(len(classes)))
        or sequence(node, "var_idx", dimensions) != list(range(dimensions))
        or sequence(node, "var_type", dimensions + 1) != [0] * dimensions + [1]
        or sequence(node, "cat_ofs", 2 * dimensions) != [0] * (2 * dimensions)
    ):
        raise ValidationError("Native forest class/feature mapping differs")
    if np.any(np.abs(sequence(node, "missing_subst", dimensions + 1)) > 1e15):
        raise ValidationError("Invalid native missing-value substitutions")
    params = node.getNode("training_params")
    if set(params.keys()) - {
        "use_surrogates",
        "max_categories",
        "regression_accuracy",
        "max_depth",
        "min_sample_count",
        "cross_validation_folds",
        "priors",
        "nactive_vars",
    }:
        raise ValidationError("Unsupported native forest training fields")
    for key, expected in {
        "max_depth": p["max_depth"],
        "min_sample_count": p["min_sample_count"],
        "nactive_vars": p["active_variables"],
        "cross_validation_folds": 0,
        "use_surrogates": 0,
    }.items():
        if number(params, key) != expected:
            raise ValidationError("Native forest training settings differ: " + key)
    priors = params.getNode("priors")
    if p["class_priors"]:
        if (
            not priors.isMap()
            or sequence(priors, "sizes", 1) != [len(classes)]
            or priors.getNode("dt").string() != "f"
            or not np.allclose(
                sequence(priors, "data", len(classes)),
                np.float32(p["class_priors"]),
                rtol=1e-7,
                atol=0,
            )
        ):
            raise ValidationError("Native forest priors differ")
    elif not priors.empty():
        raise ValidationError("Unexpected native forest priors")
    count = number(node, "ntrees")
    if int(count) != count or not 1 <= count <= p["max_trees"]:
        raise ValidationError("Native forest tree count exceeds the declared limit")
    trees = node.getNode("trees")
    if not trees.isSeq() or trees.size() != count:
        raise ValidationError("Native forest tree list differs from its count")
    total = 0
    for index in range(int(count)):
        tree = trees.at(index)
        if set(tree.keys()) != {"nodes"}:
            raise ValidationError("Unsupported native forest tree fields")
        nodes = tree.getNode("nodes")
        if not nodes.isSeq() or nodes.size() < 1:
            raise ValidationError("Forest tree needs a root node")
        total += nodes.size()
        if total > MAX_NODES:
            raise ValidationError("Forest exceeds 50,000 nodes")
        pending = [0]
        for j in range(nodes.size()):
            item = nodes.at(j)
            if set(item.keys()) - {"depth", "value", "norm_class_idx", "splits"}:
                raise ValidationError("Unsupported forest node fields")
            if not pending or number(item, "depth") != pending.pop():
                raise ValidationError("Forest preorder tree topology is invalid")
            depth = number(item, "depth")
            label = number(item, "value")
            if (
                not 0 <= depth <= p["max_depth"]
                or label != int(label)
                or not 0 <= label < len(classes)
                or number(item, "norm_class_idx") != label
            ):
                raise ValidationError("Forest node depth or class is invalid")
            splits = item.getNode("splits")
            if not splits.empty():
                if not splits.isSeq() or splits.size() != 1 or depth >= p["max_depth"]:
                    raise ValidationError("Unsupported forest surrogate/depth split")
                split = splits.at(0)
                variable = number(split, "var")
                if int(variable) != variable or not 0 <= variable < dimensions:
                    raise ValidationError("Forest split feature index is invalid")
                thresholds = [key for key in ("le", "gt") if not split.getNode(key).empty()]
                if len(thresholds) != 1 or set(split.keys()) - {"var", "quality", "le", "gt"}:
                    raise ValidationError("Only ordered numeric forest splits are supported")
                if abs(number(split, thresholds[0])) > 1e15 or number(split, "quality") < 0:
                    raise ValidationError("Invalid forest split threshold/quality")
                pending.extend([depth + 1, depth + 1])
        if pending:
            raise ValidationError("Forest tree has missing child nodes")
    importance = []
    if p["calculate_importance"]:
        importance = sequence(node, "var_importance", dimensions)
        if np.any(np.asarray(importance) < 0) or not (
            np.isclose(sum(importance), 1, atol=2e-6) or sum(importance) == 0
        ):
            raise ValidationError("Invalid native forest variable importance")
    elif not node.getNode("var_importance").empty():
        raise ValidationError("Unexpected native variable importance")
    oob = number(node, "oob_error")
    if oob < 0:
        raise ValidationError("Invalid native OOB error")
    calculated = p["calculate_importance"] or p["termination"] == "tree_limit_or_oob_error"
    return {
        "trees": int(count),
        "nodes": total,
        "importance": importance,
        "native_oob_statistic": oob if calculated else None,
        "oob_computation_requested": bool(calculated),
    }


def read_forest(model, *, load=True):
    if (
        not isinstance(model, dict)
        or set(model) != MODEL_KEYS
        or model.get("model") != "opencv_rtrees_classifier_v1"
    ):
        raise ValidationError("Connect a portable random forest classifier model")
    if len(canonical_json(model).encode()) > MAX_MODEL_BYTES or model["revision"] != model_revision(
        model
    ):
        raise ValidationError("Forest model exceeds 8 MiB or its revision differs")
    if (
        type(model["dimensions"]) is not int
        or not 1 <= model["dimensions"] <= 65_536
        or model["opencv_major"] != int(cv2.__version__.split(".")[0])
        or model["normalization"] not in ("none", "l2")
    ):
        raise ValidationError("Forest dimensions, normalization or OpenCV major version differs")
    classes = label_strings(model["class_labels"], "Class mapping", maximum=64)
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValidationError("Forest needs 2..64 distinct classes")
    feature_contract(model["feature_contract"])
    p = model["training_parameters"]
    if not isinstance(p, dict):
        raise ValidationError("Forest training parameters must be an object")
    validate_training(p)
    positive_weights(p["class_priors"], "class_priors", count=len(classes), maximum=64)
    if p["active_variables"] > model["dimensions"]:
        raise ValidationError("Active variables exceed model dimensions")
    summary = model["training_summary"]
    if not isinstance(summary, dict) or set(summary) != {
        "rows",
        "class_counts",
        "sample_weights_sha256",
    }:
        raise ValidationError("Forest training summary must be an object")
    rows, counts, digest = (
        summary["rows"],
        summary["class_counts"],
        summary["sample_weights_sha256"],
    )
    if (
        type(rows) is not int
        or not 2 <= rows <= 2048
        or not isinstance(counts, list)
        or len(counts) != len(classes)
        or any(type(n) is not int or n < 1 for n in counts)
        or sum(counts) != rows
        or (
            digest is not None
            and (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            )
        )
    ):
        raise ValidationError("Invalid forest training row/class/weight summary")
    text = model["forest_yaml"]
    if not isinstance(text, str) or not 1 <= len(text.encode()) <= MAX_MODEL_BYTES:
        raise ValidationError("Native forest YAML is missing or exceeds 8 MiB")
    storage = cv2.FileStorage(text, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
    try:
        node = storage.getNode("forest")
        report = inspect_forest(node, model)
        native = None
        if load:
            native = cv2.ml.RTrees_create()
            native.read(node)
            if (
                not native.isTrained()
                or not native.isClassifier()
                or native.getVarCount() != model["dimensions"]
            ):
                raise ValidationError("Loaded native forest is not a matching classifier")
    finally:
        storage.release()
    return native, report


def importance_plot(values):
    canvas = np.full((400, 680, 3), 249, np.uint8)
    order = np.argsort(-np.asarray(values), kind="stable")[:16]
    maximum = max(max(values), 1e-12)
    for row, index in enumerate(order):
        y = 36 + row * 21
        cv2.putText(
            canvas,
            f"{index}: {values[index]:.3g}",
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (35, 35, 35),
            1,
        )
        cv2.rectangle(
            canvas, (140, y - 11), (140 + int(505 * values[index] / maximum), y), (190, 130, 60), -1
        )
    cv2.putText(
        canvas,
        "Native permutation importance; feature indices, not image pixels",
        (12, 385),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (35, 35, 35),
        1,
    )
    return canvas


class _ForestBlock(Block):
    category, block_type, subtype = "ML", "ML", "Classical classification"
    process_inputs_readonly = True
    preview_safe = True
    parallel_safe = True


class TrainForest(_ForestBlock):
    type_id, title = "ml.rtrees_train", "Train OpenCV random forest classifier"
    inputs = (
        Port("samples", "any"),
        Port("labels", "data", required=False, required_when_connected=True),
        Port("sample_weights", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("model", "data"),
        Port("importance", "data", required=False),
        Port("report", "data"),
    )
    defaults = {**TRAINING, "normalization": "none", "labels": [], "sample_weights": []}
    parameter_choices = {"termination": TERMINATION, "normalization": ("none", "l2")}
    parameter_ranges = RANGES
    parallel_safe = False  # Native stochastic training; do not alter the caller's global RNG seed.
    payload_reference = "samples"
    parameter_help = {
        "max_depth": "Maximum tree depth (root is zero). Bounds model complexity.",
        "min_sample_count": "Native minimum samples used to split a node.",
        "active_variables": "Features considered per split; 0 uses the native rounded square root. "
        "Positive values must not exceed the descriptor length.",
        "max_trees": "Hard training tree limit; retained even with optional OOB stopping.",
        "termination": "Always keep the tree limit; optionally stop when native OOB error "
        "is below the target.",
        "oob_error": "OOB error target used only by OOB termination; needs a positive value. "
        "The weighted native statistic is not held-out accuracy.",
        "calculate_importance": "Compute native permutation importance and OOB statistics. "
        "Adds substantial feature-dependent training cost; the work budget is checked first.",
        "class_priors": "Optional positive relative class weights in first-appearance label order.",
        "normalization": "None or per-row L2, stored with the model and applied to queries.",
        "labels": "One class string per training row; a wired Labels input overrides this list.",
        "sample_weights": "[] gives equal weights. Otherwise one positive value per row; "
        "the wired Sample weights input overrides this list.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        validate_training({key: p[key] for key in TRAINING})
        label_strings(p["labels"], "Training labels")
        positive_weights(p["sample_weights"], "sample_weights")

    def process(self, inputs):
        p = self.parameters
        values, contract = descriptor_matrix(inputs["samples"], max_abs=1e15)
        if not isinstance(inputs["samples"], FlowGroup):
            contract = self.input_payloads["samples"].get("feature_contract")
        labels = label_strings(inputs.get("labels", p["labels"]), "Training labels")
        if len(labels) != len(values):
            raise ValidationError("Supply one class label per training row", "labels")
        classes = list(dict.fromkeys(labels))
        if not 2 <= len(classes) <= 64:
            raise ValidationError("Forest training needs 2..64 classes", "labels")
        weights = positive_weights(
            inputs.get("sample_weights", p["sample_weights"]), "sample_weights", count=len(values)
        )
        priors = positive_weights(p["class_priors"], "class_priors", count=len(classes), maximum=64)
        if p["active_variables"] > values.shape[1]:
            raise ValidationError(
                "Active variables exceed the descriptor length", "active_variables"
            )
        work = (
            values.size
            * int(p["max_trees"])
            * (values.shape[1] if p["calculate_importance"] else 1)
        )
        if work > MAX_WORK:
            raise ValidationError("Forest training work exceeds 64 million feature operations")
        values = normalize_features(values, p["normalization"])
        native = cv2.ml.RTrees_create()
        native.setMaxDepth(int(p["max_depth"]))
        native.setMinSampleCount(int(p["min_sample_count"]))
        native.setActiveVarCount(int(p["active_variables"]))
        native.setCalculateVarImportance(p["calculate_importance"])
        native.setCVFolds(0)
        native.setUseSurrogates(False)
        criteria = cv2.TERM_CRITERIA_MAX_ITER
        if p["termination"] == "tree_limit_or_oob_error":
            criteria |= cv2.TERM_CRITERIA_EPS
        native.setTermCriteria((criteria, int(p["max_trees"]), float(p["oob_error"])))
        if priors:
            native.setPriors(np.float32(priors))
        ids = np.int32([classes.index(label) for label in labels])
        data = cv2.ml.TrainData_create(
            values, cv2.ml.ROW_SAMPLE, ids, sampleWeights=np.float32(weights) if weights else None
        )
        if not native.train(data) or not native.isClassifier():
            raise ValidationError("OpenCV could not train the forest classifier")
        storage = cv2.FileStorage(
            "", cv2.FILE_STORAGE_WRITE | cv2.FILE_STORAGE_MEMORY | cv2.FILE_STORAGE_FORMAT_YAML
        )
        try:
            native.write(storage, "forest")
            text = storage.releaseAndGetString()
        finally:
            storage.release()
        summary = {
            "rows": len(values),
            "class_counts": [labels.count(c) for c in classes],
            "sample_weights_sha256": hashlib.sha256(np.float32(weights).tobytes()).hexdigest()
            if weights
            else None,
        }
        model = {
            "model": "opencv_rtrees_classifier_v1",
            "opencv_major": int(cv2.__version__.split(".")[0]),
            "dimensions": values.shape[1],
            "class_labels": classes,
            "normalization": p["normalization"],
            "feature_contract": feature_contract(contract),
            "training_parameters": {key: copy.deepcopy(p[key]) for key in TRAINING},
            "training_summary": summary,
            "forest_yaml": text,
        }
        model["revision"] = model_revision(model)
        _, report = read_forest(model, load=False)
        importance = report.pop("importance")
        report.update(
            summary, dimensions=values.shape[1], revision=model["revision"], classes=classes
        )
        graphics = (
            {"Feature importance": importance_plot(importance)}
            if importance and self.generate_graphics
            else {}
        )
        return Result(
            {
                "model": model,
                "importance": importance if importance else NO_VALUE,
                "report": report,
            },
            graphics=graphics,
            message=f"Trained {report['trees']} trees for {len(classes)} classes",
        )


class PredictForest(_ForestBlock):
    type_id, title = "ml.rtrees_predict", "Predict OpenCV random forest classes"
    inputs = (
        Port("model", "data"),
        Port("samples", "data"),
        Port("image", required=False, required_when_connected=True),
    )
    outputs = (
        Port("labels", "data"),
        Port("label", "data", required=False),
        Port("candidate_labels", "data"),
        Port("votes", "data"),
        Port("vote_fractions", "data"),
        Port("accepted", "data"),
        Port("image", required=False),
        Port("report", "data"),
    )
    defaults = {
        "contract_policy": "require_match",
        "min_vote_fraction": 0.0,
        "min_vote_margin": 0.0,
        "reject_vote_ties": True,
        "accepted_labels": [],
    }
    parameter_choices = {"contract_policy": ("require_match", "dimensions_only")}
    parameter_ranges = {
        "min_vote_fraction": NumericRange(0, 1),
        "min_vote_margin": NumericRange(0, 1),
    }
    payload_reference = "samples"
    parameter_help = {
        "contract_policy": "Match saved preprocessing, or explicitly check only dimensions. "
        "Stored normalization always applies.",
        "min_vote_fraction": "Inclusive minimum winning vote fraction; agreement is not "
        "a calibrated probability.",
        "min_vote_margin": "Inclusive minimum (winner minus runner-up votes) / actual tree count.",
        "reject_vote_ties": "Withhold recognized labels on tied maximum votes; candidate "
        "and votes remain visible.",
        "accepted_labels": "Optional allowed trained classes. Known disallowed classes still "
        "emit their labels with an inspection failure.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        label_strings(p["accepted_labels"], "Accepted labels", maximum=64)

    def process(self, inputs):
        model = inputs["model"]
        _, info = read_forest(model, load=False)
        values, _ = descriptor_matrix(inputs["samples"], max_abs=1e15)
        p = self.parameters
        if values.shape[1] != model["dimensions"]:
            raise ValidationError("Prediction dimensions differ from the trained forest")
        contract = feature_contract(self.input_payloads["samples"].get("feature_contract"))
        if p["contract_policy"] == "require_match" and contract != model["feature_contract"]:
            raise ValidationError(
                "Prediction preprocessing differs from the trained feature contract"
            )
        if len(values) * info["nodes"] > MAX_WORK:
            raise ValidationError("Forest prediction work exceeds 64 million node comparisons")
        if not set(p["accepted_labels"]) <= set(model["class_labels"]):
            raise ValidationError(
                "Accepted labels must be in the trained class mapping", "accepted_labels"
            )
        image = validate_prediction_image(self, inputs, len(values))
        values = normalize_features(values, model["normalization"])
        native, _ = read_forest(model)
        predicted = native.predict(values)[1]
        table = native.getVotes(values, 0)
        classes = model["class_labels"]
        if (
            table is None
            or table.shape != (len(values) + 1, len(classes))
            or table.dtype.kind not in "iu"
            or table[0].tolist() != list(range(len(classes)))
            or np.any(table[1:] < 0)
            or np.any(table[1:].sum(axis=1) != info["trees"])
        ):
            raise ValidationError("Native forest vote table is invalid")
        if (
            predicted is None
            or predicted.shape != (len(values), 1)
            or not np.isfinite(predicted).all()
            or np.any(predicted != np.rint(predicted))
            or np.any(predicted < 0)
            or np.any(predicted >= len(classes))
        ):
            raise ValidationError("Native forest returned invalid class IDs")
        ids = predicted.ravel().astype(int)
        votes = table[1:]
        counts = votes[np.arange(len(values)), ids]
        if np.any(counts != votes.max(axis=1)):
            raise ValidationError("Native forest candidate disagrees with its vote table")
        fractions = votes / info["trees"]
        winners = counts / info["trees"]
        margins = (counts - np.sort(votes, axis=1)[:, -2]) / info["trees"]
        ties = np.count_nonzero(votes == counts[:, None], axis=1) > 1
        reasons = [[] for _ in values]
        for i in range(len(values)):
            if winners[i] < p["min_vote_fraction"]:
                reasons[i].append("rtrees_vote_fraction")
            if margins[i] < p["min_vote_margin"]:
                reasons[i].append("rtrees_vote_margin")
            if p["reject_vote_ties"] and ties[i]:
                reasons[i].append("rtrees_vote_tie")
        known = np.array([not reason for reason in reasons])
        candidates = [classes[i] for i in ids]
        labels = [label if known[i] else None for i, label in enumerate(candidates)]
        accepted = known & np.array(
            [not p["accepted_labels"] or c in p["accepted_labels"] for c in candidates]
        )
        passed = bool(accepted.all())
        text = f"{labels[0] or 'Unknown'} · votes={winners[0]:.0%} · margin={margins[0]:.0%}"
        drawings = prediction_drawing(self, image, text, passed, "Random forest class")
        report = {
            "count": len(values),
            "revision": model["revision"],
            "trees": info["trees"],
            "class_mapping": classes,
            "winning_vote_fractions": winners.tolist(),
            "vote_margins": margins.tolist(),
            "vote_ties": ties.tolist(),
            "unknown_reasons": reasons,
            "unknown_count": int(np.count_nonzero(~known)),
            "probabilities_available": False,
            "judgment_settings": copy.deepcopy(p),
        }
        failures = (["rtrees_unknown_class"] if not known.all() else []) + (
            ["rtrees_unaccepted_label"] if np.any(known & ~accepted) else []
        )
        return Result(
            {
                "labels": labels,
                "label": labels[0] if len(labels) == 1 and labels[0] is not None else NO_VALUE,
                "candidate_labels": candidates,
                "votes": votes.tolist(),
                "vote_fractions": fractions.tolist(),
                "accepted": accepted.tolist(),
                "image": image,
                "report": report,
            },
            drawings=drawings,
            passed=passed,
            fail_reasons=failures,
            message=text if len(values) == 1 else f"{len(values)} forest predictions",
        )


class SaveForest(_ForestBlock):
    type_id, title = "ml.rtrees_save", "Save random forest classifier"
    category, block_type, subtype = "Output", "Output", "ML models"
    inputs, outputs = (Port("model", "data"),), (Port("path", "data"),)
    defaults = {"path": "forest-classifier.json"}
    preview_safe = False

    def process(self, inputs):
        model = inputs["model"]
        read_forest(model, load=False)
        path = local_path(self, self.parameters["path"])
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical_json(model) + "\n")
        return Result({"path": str(path)}, message="Saved forest; existing files preserved")


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValidationError("Duplicate forest model JSON key: " + key)
        result[key] = value
    return result


def _constant(token):
    raise ValidationError("Nonfinite forest model JSON number: " + token)


class LoadForest(_ForestBlock):
    type_id, title = "ml.rtrees_load", "Load random forest classifier"
    outputs = (Port("model", "data"),)
    defaults = {"path": ""}
    project_asset_parameters = (*Block.project_asset_parameters, "path")

    def process(self, inputs):
        path = local_path(self, self.parameters["path"])
        with path.open("rb") as stream:
            raw = stream.read(MAX_MODEL_BYTES + 1)
        if len(raw) > MAX_MODEL_BYTES:
            raise ValidationError("Forest model exceeds 8 MiB")
        model = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant)
        _, info = read_forest(model, load=False)
        return Result({"model": model}, message=f"Loaded {info['trees']} forest trees")


RTREES_BLOCKS = (TrainForest, PredictForest, SaveForest, LoadForest)
