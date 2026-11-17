"""Image-level classification evaluation with explicit abstentions and capture groups."""

import copy
import hashlib
from collections import Counter

import cv2
import numpy as np

from .classifier_common import canonical_json, label_strings
from .core import Block, Port, Result
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, FlowGroup, frame
from .tickets import ticket_key

MAX_SAMPLES = 10_000
STATUSES = ("predicted", "rejected", "missing", "error")
METRIC_DEFAULTS = {
    "classes": [],
    "min_accuracy": 1.0,
    "min_coverage": 1.0,
    "thresholds": [],
    "score_direction": "higher",
}
CLASSIFICATION_DEFAULTS = {
    **METRIC_DEFAULTS,
    "block": "",
    "output": "labels",
    "value_path": "0",
    "score_output": "",
    "score_path": "",
    "score_kind": "score",
    "training_groups": [],
}


def text(value, name, *, maximum=128, empty=False):
    if not isinstance(value, str) or not (0 if empty else 1) <= len(value) <= maximum:
        raise ValidationError(f"{name} must be text of up to {maximum} characters")
    return value


def error_text(value):
    message = str(value)[:4096]
    while len(canonical_json(message).encode()) > 4096:
        message = message[: len(message) // 2]
    return message


def case_metadata(case):
    label = case.get("class_label")
    if label is not None:
        text(label, "Class label")
    text(case.get("capture_group", case["id"]), "Capture group")
    digest = case.get("image_sha256")
    if digest is not None and (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValidationError("Image SHA-256 must have 64 lowercase hexadecimal characters")


def metric_settings(parameters):
    if set(parameters) - set(METRIC_DEFAULTS):
        raise ValidationError("Unexpected classifier metric settings")
    p = {**METRIC_DEFAULTS, **parameters}
    labels = label_strings(p["classes"], "Classes", maximum=128)
    if len(set(labels)) != len(labels):
        raise ValidationError("Classes must be distinct")
    for key in ("min_accuracy", "min_coverage"):
        if type(p[key]) not in (int, float):
            raise ValidationError(key + " must be a number", key)
        p[key] = float(NumericRange(0, 1).check(p[key], key, key))
    thresholds = p["thresholds"]
    if (
        not isinstance(thresholds, list)
        or len(thresholds) > 128
        or any(type(v) not in (int, float) or not np.isfinite(v) for v in thresholds)
        or len(set(thresholds)) != len(thresholds)
    ):
        raise ValidationError("Thresholds must be up to 128 distinct finite numbers", "thresholds")
    if p["score_direction"] not in ("higher", "lower"):
        raise ValidationError("Score direction must be higher or lower")
    return p


def classification_config(value, cases=()):
    if not isinstance(value, dict) or set(value) - set(CLASSIFICATION_DEFAULTS):
        raise ValidationError("Invalid classifier evaluation configuration")
    p = {**copy.deepcopy(CLASSIFICATION_DEFAULTS), **copy.deepcopy(value)}
    metric_settings({k: p[k] for k in METRIC_DEFAULTS})
    for key in ("block", "output"):
        text(p[key], key, maximum=200)
    for key in ("value_path", "score_output", "score_path", "score_kind"):
        text(p[key], key, maximum=256, empty=key != "score_kind")
    for key in ("value_path", "score_path"):
        if p[key] and (len(p[key].split(".")) > 16 or any(not part for part in p[key].split("."))):
            raise ValidationError("Output paths need up to 16 dotted fields/indices")
    groups = label_strings(p["training_groups"], "Training groups", maximum=MAX_SAMPLES)
    if len(set(groups)) != len(groups):
        raise ValidationError("Training groups must be distinct")
    overlap = set(groups) & {c.get("capture_group", c["id"]) for c in cases}
    if overlap:
        raise ValidationError(
            "Training/evaluation capture groups overlap: " + ", ".join(sorted(overlap)[:8])
        )
    if p["thresholds"] and not p["score_output"]:
        raise ValidationError("Threshold sweeps need a score output", "score_output")
    return p


def split_image_groups(
    samples, *, test_fraction=0.25, validation_fraction=0.0, seed=0, require_training_classes=True
):
    for name, value in (
        ("test_fraction", test_fraction),
        ("validation_fraction", validation_fraction),
    ):
        NumericRange(0, 1).check(value, name, name)
        if type(value) not in (int, float):
            raise ValidationError(name + " must be a number", name)
    if type(require_training_classes) is not bool:
        raise ValidationError("Require training classes must be a boolean")
    NumericRange(-2147483648, 2147483647, whole=True).check(seed, "Seed", "seed")
    if test_fraction + validation_fraction >= 1:
        raise ValidationError("Test and validation fractions must leave training groups")
    if not isinstance(samples, list) or not 1 <= len(samples) <= MAX_SAMPLES:
        raise ValidationError("Use 1..10,000 labeled image records")
    ids, groups, hashes = set(), {}, {}
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValidationError("Each image record must be an object")
        identity = text(sample.get("id"), "Image ID")
        text(sample.get("path"), "Image path", maximum=4096)
        label = text(sample.get("class_label"), "Class label")
        group = text(sample.get("capture_group"), "Explicit capture group")
        case_metadata(sample)
        if identity in ids:
            raise ValidationError("Duplicate image ID: " + identity)
        ids.add(identity)
        groups.setdefault(group, []).append(copy.deepcopy(sample))
        keys = [("path", sample["path"].replace("\\", "/"))]
        if sample.get("image_sha256"):
            keys.append(("hash", sample["image_sha256"]))
        for key in keys:
            if key in hashes and hashes[key] != (group, label):
                raise ValidationError("Repeated image path/hash crosses capture groups or labels")
            hashes[key] = group, label
    order = sorted(
        groups, key=lambda g: (hashlib.sha256(f"{int(seed)}\0{g}".encode()).hexdigest(), g)
    )

    def count(fraction):
        return max(1, int(len(order) * fraction + 0.5)) if fraction > 0 else 0

    test_count, validation_count = count(test_fraction), count(validation_fraction)
    if test_count + validation_count >= len(order):
        raise ValidationError("Not enough groups for nonempty requested splits and training")
    names = (
        ["test"] * test_count
        + ["validation"] * validation_count
        + ["train"] * (len(order) - test_count - validation_count)
    )
    assignments = dict(zip(order, names, strict=True))
    result = {
        name: sorted(
            [s for group in order if assignments[group] == name for s in groups[group]],
            key=lambda s: s["id"],
        )
        for name in ("train", "validation", "test")
    }
    all_classes = {s["class_label"] for s in samples}
    absent = sorted(all_classes - {s["class_label"] for s in result["train"]})
    if require_training_classes and absent:
        raise ValidationError(
            "Training split lacks classes; change groups/seed/fractions: " + ", ".join(absent)
        )
    report = {
        "method": "sha256_group_shuffle_v1",
        "seed": int(seed),
        "group_assignments": assignments,
        "group_counts": dict(Counter(names)),
        "class_counts": {
            name: dict(Counter(s["class_label"] for s in rows)) for name, rows in result.items()
        },
        "missing_training_classes": absent,
        "fractions_apply_to": "groups",
        "stratified": False,
        "image_hashes_verified": False,
    }
    return result, report


def normalize_records(records, expected_cases=None):
    if isinstance(records, FlowGroup):
        records = [item.value for item in records.items]
    if isinstance(records, dict) and records.get("model") == "validation_report_v1":
        records = [row["classification"] for row in records["cases"] if "classification" in row]
    if not isinstance(records, list) or len(records) > MAX_SAMPLES:
        raise ValidationError("Use up to 10,000 classification records")
    try:
        encoded = canonical_json(records).encode()
    except (ValueError, TypeError) as exc:
        raise ValidationError("Classification records require finite JSON values") from exc
    if len(encoded) > 16 * 1024**2:
        raise ValidationError("Classification records exceed 16 MiB")
    ids, result = set(), []
    for value in records:
        if not isinstance(value, dict) or len(canonical_json(value).encode()) > 16384:
            raise ValidationError("Classification records must be JSON objects up to 16 KiB")
        record = copy.deepcopy(value)
        identity = text(record.get("id"), "Record ID")
        if identity in ids:
            raise ValidationError("Duplicate prediction record ID: " + identity)
        ids.add(identity)
        for field in ("group", "score_kind"):
            if field in record:
                text(record[field], field, maximum=256)
        for field in ("expected", "prediction"):
            if record.get(field) is not None:
                text(record[field], field)
        if record.get("status") not in STATUSES or (record["status"] == "predicted") != (
            record.get("prediction") is not None
        ):
            raise ValidationError("Prediction status and recognized label disagree")
        score = record.get("score")
        if score is not None and (type(score) not in (int, float) or not np.isfinite(score)):
            raise ValidationError("Scores must be finite numbers or null")
        result.append(record)
    if expected_cases is not None:
        if not isinstance(expected_cases, list) or len(expected_cases) > MAX_SAMPLES:
            raise ValidationError("Expected cases exceed 10,000 images")
        expected = {}
        for case in expected_cases:
            if not isinstance(case, dict):
                raise ValidationError("Expected images must be objects")
            identity = text(case.get("id"), "Expected image ID")
            if identity in expected:
                raise ValidationError("Duplicate expected image ID")
            case_metadata(case)
            expected[identity] = case
        if ids - expected.keys():
            raise ValidationError("Predictions contain unexpected image IDs")
        for record in result:
            case = expected[record["id"]]
            if record.get("expected") != case.get("class_label"):
                raise ValidationError("Prediction record truth differs from the expected image")
            if record.get("group", case.get("capture_group", case["id"])) != case.get(
                "capture_group", case["id"]
            ):
                raise ValidationError("Prediction record capture group differs")
        for identity in sorted(expected.keys() - ids):
            case = expected[identity]
            result.append(
                {
                    "id": identity,
                    "expected": case.get("class_label"),
                    "prediction": None,
                    "group": case.get("capture_group", identity),
                    "status": "missing",
                    "score": None,
                }
            )
    return result


def _metrics(records, classes):
    observed = {
        r[key] for r in records for key in ("expected", "prediction") if r.get(key) is not None
    }
    labels = list(classes) + sorted(observed - set(classes))
    if len(labels) > 128:
        raise ValidationError("Evaluation exceeds 128 observed classes")
    indices = {label: i for i, label in enumerate(labels)}
    counts = np.zeros((len(labels), len(labels) + 3), np.int64)
    status_counts = Counter(r["status"] for r in records)
    for row in records:
        if row.get("expected") is None:
            continue
        column = (
            indices[row["prediction"]]
            if row["status"] == "predicted"
            else len(labels) + ("rejected", "missing", "error").index(row["status"])
        )
        counts[indices[row["expected"]], column] += 1
    support = counts.sum(axis=1)
    recognized = int(counts[:, : len(labels)].sum())
    correct = int(np.trace(counts[:, : len(labels)]))
    labeled = int(support.sum())

    def ratio(a, b):
        return a / b if b else None

    per_class = []
    for i, label in enumerate(labels):
        tp = int(counts[i, i])
        fp = int(counts[:, i].sum()) - tp
        fn = int(support[i]) - tp
        per_class.append(
            {
                "label": label,
                "support": int(support[i]),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": ratio(tp, tp + fp),
                "recall": ratio(tp, tp + fn),
                "f1": ratio(2 * tp, 2 * tp + fp + fn),
            }
        )
    macro = {}
    for name in ("precision", "recall", "f1"):
        values = [row[name] for row in per_class if row[name] is not None]
        macro[name] = float(np.mean(values)) if values else None
        macro[name + "_classes"] = len(values)
    return {
        "classes": labels,
        "columns": [{"kind": "class", "label": label} for label in labels]
        + [{"kind": status} for status in ("rejected", "missing", "error")],
        "confusion": counts.tolist(),
        "per_class": per_class,
        "macro": macro,
        "summary": {
            "total": len(records),
            "labeled": labeled,
            "unlabeled": len(records) - labeled,
            "correct": correct,
            "recognized_labeled": recognized,
            "status_counts": dict(status_counts),
            "accuracy": ratio(correct, labeled),
            "coverage": ratio(recognized, labeled),
            "selective_accuracy": ratio(correct, recognized),
            "balanced_accuracy": macro["recall"],
            "weighted_f1": sum(r["support"] * (r["f1"] or 0) for r in per_class) / labeled
            if labeled
            else None,
        },
    }


def evaluate_classification(records, *, expected_cases=None, **parameters):
    p = metric_settings(parameters)
    rows = normalize_records(records, expected_cases)
    result = _metrics(rows, p["classes"])
    result.update(
        model="image_classification_evaluation_v1",
        records=rows,
        expected_ids_verified=expected_cases is not None,
        thresholds=[],
    )
    for threshold in p["thresholds"]:
        sweep = copy.deepcopy(rows)
        kinds = set()
        for row in sweep:
            if row["status"] != "predicted":
                continue
            if row.get("score") is None:
                raise ValidationError(
                    "Every recognized prediction needs a score for threshold sweeps"
                )
            kinds.add(row.get("score_kind", "score"))
            if (
                row["score"] < threshold
                if p["score_direction"] == "higher"
                else row["score"] > threshold
            ):
                row.update(status="rejected", prediction=None)
        if len(kinds) > 1:
            raise ValidationError("Cannot mix score meanings in one threshold sweep")
        result["thresholds"].append(
            {
                "threshold": threshold,
                "direction": p["score_direction"],
                "summary": _metrics(sweep, result["classes"])["summary"],
            }
        )
    summary = result["summary"]
    summary["passed"] = (
        bool(summary["labeled"])
        and summary["accuracy"] >= p["min_accuracy"]
        and summary["coverage"] >= p["min_coverage"]
        and not summary["status_counts"].get("missing", 0)
        and not summary["status_counts"].get("error", 0)
    )
    result["settings"] = {key: copy.deepcopy(p[key]) for key in METRIC_DEFAULTS}
    return result


def output_path(value, path):
    for part in path.split(".") if path else ():
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, (list, tuple)) and part.isdecimal() and int(part) < len(value):
            value = value[int(part)]
        else:
            raise ValidationError("Output path is unavailable: " + path)
    return value


def capture_classification(graph, config, case, ticket, errors):
    row = {
        "id": case["id"],
        "group": case.get("capture_group", case["id"]),
        "expected": case.get("class_label"),
        "prediction": None,
        "status": "missing",
        "score": None,
        "score_kind": config["score_kind"],
    }
    try:
        block = graph.blocks[config["block"]]
        if block.state == "error":
            row.update(status="error", error=error_text(block.error))
            return row
        packets = block.current_emissions.get(config["output"], [])
        if not packets:
            if errors:
                row.update(status="error", error="Program errors prevented a current prediction")
            return row
        if len(packets) != 1:
            raise ValidationError("Image evaluation needs exactly one prediction event per image")
        packet = packets[0]
        if (
            ticket is None
            or packet.payload.get("ticket") is None
            or ticket_key(packet.payload["ticket"]) != ticket_key(ticket)
        ):
            raise ValidationError("Prediction ticket does not match the current validation image")
        if (
            config["value_path"] == "0"
            and isinstance(packet.value, (list, tuple))
            and len(packet.value) != 1
        ):
            raise ValidationError("Image evaluation requires one recognized-label entry")
        label = output_path(packet.value, config["value_path"])
        if label is not None:
            text(label, "Predicted class label")
        row.update(prediction=label, status="predicted" if label is not None else "rejected")
        if config["score_output"]:
            scores = block.current_emissions.get(config["score_output"], [])
            if len(scores) != 1 or ticket_key(scores[0].payload.get("ticket", {})) != ticket_key(
                ticket
            ):
                raise ValidationError("A current matching score event is required")
            score = output_path(scores[0].value, config["score_path"])
            if type(score) not in (int, float) or not np.isfinite(score):
                raise ValidationError("Configured classifier score must be a finite scalar")
            row["score"] = float(score)
        report = block.last_result.outputs.get("report") if block.last_result else None
        if isinstance(report, dict) and isinstance(report.get("revision"), str):
            row["model_revision"] = report["revision"][:128]
    except Exception as exc:
        row.update(
            prediction=None, status="error", error=error_text(f"{type(exc).__name__}: {exc}")
        )
    return row


def confusion_plot(report):
    size = min(16, len(report["classes"]))
    columns = list(range(size)) + list(range(len(report["classes"]), len(report["classes"]) + 3))
    canvas = np.full((160 + 28 * size, 190 + 42 * len(columns), 3), 248, np.uint8)
    matrix = (
        np.asarray(report["confusion"], np.int64).reshape(len(report["classes"]), -1)
        if report["classes"]
        else np.zeros((0, 3), np.int64)
    )
    peak = max(1, int(matrix.max()) if matrix.size else 0)
    for i in range(size):
        cv2.putText(
            canvas, str(i), (30, 74 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1
        )
        for j, column in enumerate(columns):
            value = int(matrix[i, column])
            shade = 245 - int(170 * value / peak)
            color = (shade, 240, shade) if i == column else (shade, shade, 245)
            x, y = 70 + 42 * j, 56 + 28 * i
            cv2.rectangle(canvas, (x, y), (x + 39, y + 25), color, -1)
            cv2.putText(
                canvas, str(value), (x + 3, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1
            )
    for j, column in enumerate(columns):
        label = (
            str(column)
            if column < len(report["classes"])
            else ("R", "M", "E")[column - len(report["classes"])]
        )
        cv2.putText(
            canvas, label, (74 + j * 42, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1
        )
    cv2.putText(
        canvas,
        "True class rows; predicted columns. R=reject M=missing E=error",
        (12, canvas.shape[0] - 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (30, 30, 30),
        1,
    )
    cv2.putText(
        canvas,
        "Class indices map to Report. First 16 classes shown; full counts retained.",
        (12, canvas.shape[0] - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (30, 30, 30),
        1,
    )
    return canvas


class SplitImageGroups(Block):
    type_id, title = "ml.split_image_groups", "Split labeled images by capture group"
    category, block_type, subtype = "ML", "ML", "Evaluation"
    inputs = (Port("samples", "data", required=False, required_when_connected=True),)
    outputs = (
        Port("train", "data"),
        Port("validation", "data"),
        Port("test", "data"),
        Port("report", "data"),
    )
    defaults = {
        "samples": [],
        "test_fraction": 0.25,
        "validation_fraction": 0.0,
        "seed": 0,
        "require_training_classes": True,
    }
    parameter_ranges = {
        "test_fraction": NumericRange(0, 1),
        "validation_fraction": NumericRange(0, 1),
        "seed": NumericRange(-2147483648, 2147483647, whole=True),
    }
    parameter_help = {
        "samples": "Image records: id, path, class_label, capture_group, optional image_sha256. "
        "Groups represent correlated captures. Paths/hash claims cannot cross groups or labels; "
        "files are not read here.",
        "test_fraction": "Fraction of whole groups, rounded with at least one if positive.",
        "validation_fraction": "Additional validation groups; leave at least one training group.",
        "seed": "Deterministic group hash order; independent of input order or global RNG state.",
        "require_training_classes": "Reject missing training classes. Split is not stratified.",
    }
    process_inputs_readonly = True
    parallel_safe = preview_safe = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["test_fraction"] + p["validation_fraction"] >= 1:
            raise ValidationError("Fractions must leave training groups", "test_fraction")

    def process(self, inputs):
        p = self.parameters
        result, report = split_image_groups(
            inputs.get("samples", p["samples"]), **{k: p[k] for k in p if k != "samples"}
        )
        return Result(
            {**result, "report": report},
            message=f"{len(result['train'])} train / {len(result['test'])} test images",
        )


class EvaluateClassification(Block):
    type_id, title = "ml.evaluate_classification", "Evaluate image class predictions"
    category, block_type, subtype = "ML", "ML", "Evaluation"
    inputs = (Port("records", "any", required=False, required_when_connected=True),)
    outputs = (Port("report", "data"), Port("plot", required=False))
    defaults = {**METRIC_DEFAULTS, "records": [], "expected_cases": []}
    parameter_choices = {"score_direction": ("higher", "lower")}
    parameter_ranges = {"min_accuracy": NumericRange(0, 1), "min_coverage": NumericRange(0, 1)}
    parameter_help = {
        "records": "Per-image id, expected, prediction, status, optional score/score_kind. "
        "Status is predicted/rejected/missing/error. A validation report may be wired directly.",
        "expected_cases": "Full set: id, class_label, capture_group. Missing IDs remain scored.",
        "classes": "Optional class order; observed extra labels are appended.",
        "min_accuracy": "Correct / all labeled images, including rejected/missing/errors.",
        "min_coverage": "Recognized / all labeled images, including wrong recognized labels.",
        "thresholds": "Distinct cutoffs; recognized rows need comparable finite scores. "
        "Sweeps can further reject recognized rows, not recover upstream rejected candidates.",
        "score_direction": "Keep >= cutoff for higher, <= for lower; scores are not probabilities.",
    }
    process_inputs_readonly = True
    parallel_safe = preview_safe = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        metric_settings({k: p[k] for k in METRIC_DEFAULTS})

    def process(self, inputs):
        p = self.parameters
        value = inputs.get("records", p["records"])
        expected = p["expected_cases"] or None
        if isinstance(value, dict) and value.get("model") == "validation_report_v1":
            expected = value["set"]["cases"]
        report = evaluate_classification(
            value, expected_cases=expected, **{k: p[k] for k in METRIC_DEFAULTS}
        )
        plot = confusion_plot(report) if self.generate_graphics else NO_VALUE
        payloads = (
            {"plot": {"spatial": frame([plot.shape[1], plot.shape[0]], self.id + ":confusion")}}
            if plot is not NO_VALUE
            else {}
        )
        passed = report["summary"]["passed"]
        return Result(
            {"report": report, "plot": plot},
            passed=passed,
            payloads=payloads,
            graphics={"Confusion matrix": plot} if plot is not NO_VALUE else {},
            fail_reasons=[] if passed else ["classification_evaluation_limits"],
            message=f"{report['summary']['correct']}/{report['summary']['labeled']} correct labels",
        )


CLASSIFICATION_EVALUATION_BLOCKS = (SplitImageGroups, EvaluateClassification)
