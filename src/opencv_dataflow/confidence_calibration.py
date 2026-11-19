"""Temperature scaling for ordered image class scores and selective predictions."""

import hashlib
import re

import cv2
import numpy as np

from .classification_evaluation import text
from .classifier_common import canonical_json, label_strings, model_revision
from .core import Block, Port, Result
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, frame
from .tickets import ticket_key

CALIBRATION = "class_score_calibration_v1"
SCORE_KINDS = ("logits", "probabilities")
MAX_CLASSES = 128
MAX_SAMPLES = 4096
MAX_ELEMENTS = 262144
PARAMETERS = {
    "classes": [],
    "score_kind": "logits",
    "score_source": "",
}
PARAMETER_HELP = {
    "classes": "Ordered labels, one per score column. Order must match the classifier.",
    "score_kind": "Raw logits or positive probabilities summing to one. Do not use top-K scores.",
    "score_source": "SHA-256 identity of the scorer and preprocessing revision. "
    "With a wired DNN receipt, leave blank to derive it.",
}
THRESHOLDS = {"minimum_confidence": 0.65, "minimum_margin": 0.05}
THRESHOLD_HELP = {
    "minimum_confidence": "Abstain below this calibrated top-class probability.",
    "minimum_margin": "Abstain below this gap between the top two probabilities.",
}
UNIT_RANGE = NumericRange(0, 1)
TEMP_RANGE = NumericRange(0.01, 100)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def revision_text(value, name):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValidationError(f"{name} requires a lowercase SHA-256 identity")
    return value


def ordered_classes(value):
    classes = label_strings(value, "Classes", maximum=MAX_CLASSES)
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValidationError("Declare at least two distinct classes in tensor column order")
    return classes


def group_list(value, name):
    groups = label_strings(value, name, maximum=MAX_SAMPLES)
    if len(set(groups)) != len(groups):
        raise ValidationError(f"{name} must be distinct")
    return groups


def real(value, name, allowed):
    if type(value) not in (float, int):
        raise ValidationError(f"{name} must be a number")
    return float(allowed.check(value, name, name))


def centered_scores(values, count, score_kind):
    if score_kind not in SCORE_KINDS:
        raise ValidationError("Choose logits or probabilities")
    if not isinstance(values, list) or len(values) != count:
        raise ValidationError("Provide the complete score vector in class order")
    if any(type(v) not in (float, int) or not -1e6 <= v <= 1e6 for v in values):
        raise ValidationError("Scores must be finite numbers within +/- 1,000,000")
    scores = np.asarray(values, dtype=np.float64)
    if score_kind == "probabilities":
        if np.any(scores <= 0) or not np.isclose(scores.sum(), 1.0, rtol=0, atol=1e-6):
            raise ValidationError("Probabilities must be positive and sum to one")
        scores = np.log(scores)
    return scores - scores.max()


def score_table(records, classes, kind, *, allow_unknown=False):
    if not isinstance(records, list) or not 1 <= len(records) <= MAX_SAMPLES:
        raise ValidationError("Provide 1..4,096 labeled image score records")
    if len(records) * len(classes) > MAX_ELEMENTS:
        raise ValidationError("Score table exceeds 262,144 values")
    index = dict(zip(classes, range(len(classes)), strict=True))
    rows, groups, ids, targets = [], set(), [], []
    seen = set()
    for entry in records:
        if not isinstance(entry, dict) or set(entry) != {"id", "capture_group", "label", "scores"}:
            raise ValidationError("Each score row needs id, capture_group, label and scores")
        identity = text(entry["id"], "Image ID")
        if identity in seen:
            raise ValidationError("Duplicate image ID: " + identity)
        seen.add(identity)
        ids.append(identity)
        groups.add(text(entry["capture_group"], "Capture group"))
        label = entry["label"]
        if label is None and allow_unknown:
            targets.append(-1)
        elif isinstance(label, str) and label in index:
            targets.append(index[label])
        else:
            raise ValidationError("Known image labels must appear in the ordered classes")
        rows.append(centered_scores(entry["scores"], len(classes), kind))
    order = sorted(range(len(ids)), key=ids.__getitem__)
    return np.stack(rows)[order], np.asarray(targets)[order], groups


def probabilities(centered, temperature):
    weights = np.exp(centered / temperature)
    return weights / weights.sum(axis=-1, keepdims=True)


def log_loss(centered, targets, temperature):
    scaled = centered / temperature
    return float(
        np.mean(np.log(np.exp(scaled).sum(axis=1)) - scaled[np.arange(len(targets)), targets])
    )


def fit_temperature(centered, targets, low, high):
    """NLL is convex in inverse temperature. A bracketed derivative finds its minimum."""
    true_values = centered[np.arange(len(targets)), targets]

    def derivative(inverse):
        weighted = probabilities(centered, 1 / inverse) * centered
        return float(np.mean(weighted.sum(axis=1) - true_values))

    a, b = 1 / high, 1 / low
    da, db = derivative(a), derivative(b)
    if da == db == 0:
        return 1.0
    if da >= 0:
        return high
    if db <= 0:
        return low
    from scipy.optimize import brentq

    return float(1 / brentq(derivative, a, b, xtol=1e-12))


def fit_calibration(
    records,
    *,
    classes,
    score_kind,
    score_source,
    training_groups,
    minimum_temperature,
    maximum_temperature,
):
    classes = ordered_classes(classes)
    revision_text(score_source, "Score source")
    training = set(group_list(training_groups, "Training groups"))
    low = real(minimum_temperature, "Minimum temperature", TEMP_RANGE)
    high = real(maximum_temperature, "Maximum temperature", TEMP_RANGE)
    if not low < 1 < high:
        raise ValidationError("Temperature bounds must bracket 1")
    scores, targets, groups = score_table(records, classes, score_kind)
    if groups & training:
        raise ValidationError("Calibration capture groups overlap classifier training")
    if len(groups) < 2 or len(set(targets)) != len(classes):
        raise ValidationError("Calibration needs all classes and at least two capture groups")
    temperature = fit_temperature(scores, targets, low, high)
    model = {
        "model": CALIBRATION,
        "temperature": temperature,
        "classes": list(classes),
        "score_kind": score_kind,
        "score_source": score_source,
        "fit_groups": sorted(groups),
        "training_groups": sorted(training),
        "fit_sha256": digest(sorted(records, key=lambda row: row["id"])),
    }
    model["revision"] = model_revision(model)
    return model, {
        "samples": len(records),
        "groups": len(groups),
        "training_groups_declared": bool(training),
        "raw_nll": log_loss(scores, targets, 1.0),
        "calibrated_nll": log_loss(scores, targets, temperature),
        "at_temperature_bound": temperature in (low, high),
    }


def checked_model(model, *, classes, score_kind, score_source):
    if (
        not isinstance(model, dict)
        or set(model)
        != {
            "model",
            "temperature",
            "classes",
            "score_kind",
            "score_source",
            "fit_groups",
            "training_groups",
            "fit_sha256",
            "revision",
        }
        or model["model"] != CALIBRATION
    ):
        raise ValidationError("Connect or enter a temperature calibration model")
    if ordered_classes(model["classes"]) != ordered_classes(classes):
        raise ValidationError("Classifier class order differs from the fitted calibration")
    if model["score_kind"] != score_kind or score_kind not in SCORE_KINDS:
        raise ValidationError("Score kind differs from the fitted calibration")
    if model["score_source"] != revision_text(score_source, "Score source"):
        raise ValidationError("Classifier or preprocessing differs from the fitted calibration")
    real(model["temperature"], "Temperature", TEMP_RANGE)
    fit = group_list(model["fit_groups"], "Calibration groups")
    training = group_list(model["training_groups"], "Training groups")
    if len(fit) < 2 or set(fit) & set(training):
        raise ValidationError("Calibration model has invalid capture groups")
    revision_text(model["fit_sha256"], "Calibration data SHA-256")
    if model["revision"] != model_revision(model):
        raise ValidationError("Calibration model revision mismatch")
    return model


def dnn_score_source(receipt, name, tensor=None):
    """Fingerprint model bytes, settings, actual backend, preprocessing and output name."""
    if not isinstance(receipt, dict) or receipt.get("model") != "onnx_inference_receipt_v1":
        raise ValidationError("Connect an ONNX inference receipt")
    calls = receipt.get("calls")
    if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict):
        raise ValidationError("Use the receipt for a single native inference call")
    call = calls[0]
    outputs = call.get("outputs", [])
    matches = (
        [output for output in outputs if isinstance(output, dict) and output.get("name") == name]
        if isinstance(outputs, list)
        else []
    )
    if len(matches) != 1:
        raise ValidationError("Receipt does not uniquely describe the selected tensor")
    if tensor is not None:
        output = matches[0]
        if output.get("shape") != tensor["shape"] or output.get("dtype") != tensor["dtype"]:
            raise ValidationError("DNN tensor shape or dtype differs from the receipt")
        if "sha256" in output:
            raw = np.asarray(tensor["values"], dtype=np.float32).tobytes()
            if hashlib.sha256(raw).hexdigest() != output["sha256"]:
                raise ValidationError("DNN tensor values differ from the receipt hash")
    artifact = call.get("artifact")
    model_sha = revision_text(
        artifact.get("sha256") if isinstance(artifact, dict) else None, "ONNX model"
    )
    settings = revision_text(receipt.get("configuration_revision"), "DNN configuration")
    return digest(
        {
            "model": model_sha,
            "configuration": settings,
            "preprocessing": call.get("preprocessing"),
            "execution": call.get("selected_execution"),
            "tensor": name,
        }
    )


def reliability(prob, targets, bins):
    top = np.argmax(prob, axis=1)
    confidence = prob[np.arange(len(top)), top]
    correct = targets == top
    index = np.minimum((confidence * bins).astype(int), bins - 1)
    counts = np.bincount(index, minlength=bins)
    accuracy = np.bincount(index, weights=correct, minlength=bins)
    means = np.bincount(index, weights=confidence, minlength=bins)
    brier = float(np.mean(1 - 2 * prob[np.arange(len(top)), targets] + np.sum(prob**2, axis=1)))
    return {
        "accuracy": float(correct.mean()),
        "brier": max(0.0, brier),
        "ece": float(np.sum(np.abs(accuracy - means)) / len(top)),
        "bins": [
            {
                "low": i / bins,
                "high": (i + 1) / bins,
                "count": int(count),
                "accuracy": float(accuracy[i] / count) if count else None,
                "confidence": float(means[i] / count) if count else None,
            }
            for i, count in enumerate(counts)
        ],
    }


def selection(prob, minimum_confidence, minimum_margin):
    top = np.argmax(prob, axis=1)
    confidence = prob[np.arange(len(top)), top]
    second = np.partition(prob, -2, axis=1)[:, -2]
    margin = confidence - second
    return top, confidence, margin, (confidence >= minimum_confidence) & (margin >= minimum_margin)


def audit_calibration(
    records, model, *, classes, score_kind, score_source, bins, minimum_confidence, minimum_margin
):
    checked_model(model, classes=classes, score_kind=score_kind, score_source=score_source)
    bins = NumericRange(2, 50, whole=True).check(bins, "Bins", "bins")
    real(minimum_confidence, "Minimum confidence", UNIT_RANGE)
    real(minimum_margin, "Minimum margin", UNIT_RANGE)
    scores, targets, groups = score_table(records, classes, score_kind, allow_unknown=True)
    if groups & set(model["training_groups"] + model["fit_groups"]):
        raise ValidationError("Evaluation capture groups overlap training or calibration")
    known = targets >= 0
    if not np.any(known):
        raise ValidationError("Evaluation needs at least one held-out known class image")
    calibrated = probabilities(scores, model["temperature"])
    raw = reliability(probabilities(scores[known], 1.0), targets[known], bins)
    adjusted = reliability(calibrated[known], targets[known], bins)
    raw["nll"] = log_loss(scores[known], targets[known], 1.0)
    adjusted["nll"] = log_loss(scores[known], targets[known], model["temperature"])
    top, _, _, accepted = selection(calibrated, minimum_confidence, minimum_margin)
    recognized = accepted & known
    unknown = ~known
    return {
        "model": "class_calibration_evaluation_v1",
        "calibration_revision": model["revision"],
        "evaluation_groups": sorted(groups),
        "raw": raw,
        "calibrated": adjusted,
        "known": int(known.sum()),
        "known_accepted": int(recognized.sum()),
        "known_coverage": float(accepted[known].mean()),
        "accepted_accuracy": float((targets[recognized] == top[recognized]).mean())
        if recognized.any()
        else None,
        "ambiguous": int(unknown.sum()),
        "ambiguous_accepted": int((unknown & accepted).sum()),
        "minimum_confidence": minimum_confidence,
        "minimum_margin": minimum_margin,
    }


def reliability_plot(report):
    image = np.full((410, 480, 3), 28, np.uint8)
    x0, y0, width, height = 56, 358, 360, 300

    def point(confidence, accuracy):
        return x0 + round(width * confidence), y0 - round(height * accuracy)

    def label(message, xy, color=(222, 222, 222)):
        cv2.putText(image, message, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    cv2.rectangle(image, point(0, 1), point(1, 0), (120, 120, 120), 1)
    cv2.line(image, point(0, 0), point(1, 1), (105, 105, 105), 1, cv2.LINE_AA)
    for key, name, color, offset in (
        ("raw", "Raw", (90, 180, 255), 0),
        ("calibrated", "Calibrated", (170, 255, 120), 192),
    ):
        for item in report[key]["bins"]:
            if item["count"]:
                xy = point(item["confidence"], item["accuracy"])
                cv2.circle(image, xy, 5, color, 2, cv2.LINE_AA)
        label(f"{name} ECE {report[key]['ece']:.4f}", (60 + offset, 36), color)
    label("Top-class accuracy", (58, 53))
    label("0", (40, 364))
    label("1", (40, 67))
    label("1", (414, 379))
    label("Mean confidence", (165, 391))
    return image


class FitConfidenceCalibration(Block):
    type_id, title = "ml.fit_confidence_calibration", "Fit classifier confidence calibration"
    category, block_type, subtype = "ML", "ML", "Training"
    inputs = (Port("records", "data", required=False, required_when_connected=True),)
    outputs = (Port("calibration", "data"), Port("report", "data"))
    defaults = {
        **PARAMETERS,
        "records": [],
        "training_groups": [],
        "minimum_temperature": 0.05,
        "maximum_temperature": 20.0,
    }
    parameter_choices = {"score_kind": SCORE_KINDS}
    parameter_ranges = {"minimum_temperature": TEMP_RANGE, "maximum_temperature": TEMP_RANGE}
    parameter_help = {
        **PARAMETER_HELP,
        "records": "Separate calibration images: id, capture_group, label, full scores.",
        "training_groups": "Known training capture groups; any overlap is rejected.",
        "minimum_temperature": "Lower fit bound, less than one.",
        "maximum_temperature": "Upper fit bound, greater than one.",
    }
    process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        ordered_classes(p["classes"])
        revision_text(p["score_source"], "Score source")
        group_list(p["training_groups"], "Training groups")
        low = real(p["minimum_temperature"], "Minimum temperature", TEMP_RANGE)
        high = real(p["maximum_temperature"], "Maximum temperature", TEMP_RANGE)
        if not low < 1 < high:
            raise ValidationError(
                "Temperature bounds must bracket 1", "minimum_temperature", "maximum_temperature"
            )

    def process(self, inputs):
        params = dict(self.parameters)
        records = inputs.get("records", params.pop("records"))
        model, report = fit_calibration(records, **params)
        return Result(
            {"calibration": model, "report": report},
            message=f"Temperature {model['temperature']:.4g} from {report['samples']} images",
        )


class ApplyConfidenceCalibration(Block):
    type_id, title = "ml.apply_confidence_calibration", "Calibrate confidence and abstain"
    category, block_type, subtype = "ML", "ML", "Classification"
    inputs = (
        Port("tensors", "data"),
        Port("receipt", "data", required=False, required_when_connected=True),
        Port("calibration", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("label", "text", required=False),
        Port("confidence", "number"),
        Port("accepted", "boolean"),
        Port("decision", "data"),
    )
    defaults = {**PARAMETERS, **THRESHOLDS, "tensor_name": "", "calibration": {}}
    parameter_choices = {"score_kind": SCORE_KINDS}
    parameter_ranges = dict.fromkeys(THRESHOLDS, UNIT_RANGE)
    parameter_help = {
        **PARAMETER_HELP,
        **THRESHOLD_HELP,
        "tensor_name": "Select a single [C] or [1,C] output by name. Blank requires one output.",
        "calibration": "Saved fit model; the wired model takes precedence.",
    }
    process_inputs_readonly = preview_safe = parallel_safe = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        ordered_classes(p["classes"])
        text(p["tensor_name"], "Tensor name", maximum=256, empty=True)
        for key in THRESHOLDS:
            real(p[key], key, UNIT_RANGE)
        if p["score_source"] != "":
            revision_text(p["score_source"], "Score source")

    def process(self, inputs):
        p = self.parameters
        tensors = inputs["tensors"]
        name = p["tensor_name"]
        if not name and isinstance(tensors, dict) and len(tensors) == 1:
            name = next(iter(tensors))
        tensor = tensors.get(name) if isinstance(tensors, dict) and name else None
        count = len(p["classes"])
        shape = tensor.get("shape") if isinstance(tensor, dict) else None
        if (
            not isinstance(tensor, dict)
            or tensor.get("model") != "tensor_v1"
            or tensor.get("dtype") != "float32"
            or not isinstance(shape, list)
            or shape not in ([count], [1, count])
            or any(type(n) is not int for n in shape)
        ):
            raise ValidationError("Select a single-image [C] or [1,C] float32 class tensor")
        centered = centered_scores(tensor.get("values"), count, p["score_kind"])
        source = p["score_source"]
        if "receipt" in inputs:
            wired = dnn_score_source(inputs["receipt"], name, tensor)
            if source and source != wired:
                raise ValidationError(
                    "Wired DNN receipt disagrees with the configured score source"
                )
            a = self.input_payloads.get("tensors", {}).get("ticket")
            b = self.input_payloads.get("receipt", {}).get("ticket")
            if (a is None) != (b is None) or (a is not None and ticket_key(a) != ticket_key(b)):
                raise ValidationError("DNN tensor and receipt tickets differ")
            source = wired
        model = checked_model(
            inputs.get("calibration", p["calibration"]),
            classes=p["classes"],
            score_kind=p["score_kind"],
            score_source=source,
        )
        prob = probabilities(centered, model["temperature"])
        top, confidence, margin, accepted = selection(
            prob[None, :], p["minimum_confidence"], p["minimum_margin"]
        )
        index, certainty, gap, passed = (
            int(top[0]),
            float(confidence[0]),
            float(margin[0]),
            bool(accepted[0]),
        )
        label = model["classes"][index]
        reasons = []
        if certainty < p["minimum_confidence"]:
            reasons.append(
                {
                    "code": "classification.low_confidence",
                    "message": "Calibrated confidence below minimum",
                }
            )
        if gap < p["minimum_margin"]:
            reasons.append(
                {"code": "classification.small_margin", "message": "Top two classes are too close"}
            )
        return Result(
            {
                "label": label if passed else NO_VALUE,
                "confidence": certainty,
                "accepted": passed,
                "decision": {
                    "model": "calibrated_class_decision_v1",
                    "candidate": label,
                    "accepted": passed,
                    "confidence": certainty,
                    "margin": gap,
                    "raw_probabilities": probabilities(centered, 1.0).tolist(),
                    "classes": list(model["classes"]),
                    "probabilities": prob.tolist(),
                    "calibration_revision": model["revision"],
                    "score_source": source,
                    "minimum_confidence": p["minimum_confidence"],
                    "minimum_margin": p["minimum_margin"],
                },
            },
            passed=passed,
            score=certainty,
            message=f"{'' if passed else 'Abstained: '}{label} at {certainty:.1%}",
            fail_reasons=reasons,
            event_data={
                "confidence": certainty,
                "margin": gap,
                "calibration_revision": model["revision"],
            },
        )


class EvaluateConfidenceCalibration(Block):
    type_id, title = "ml.evaluate_confidence_calibration", "Evaluate classifier reliability"
    category, block_type, subtype = "ML", "ML", "Evaluation"
    inputs = (
        Port("records", "data", required=False, required_when_connected=True),
        Port("calibration", "data", required=False, required_when_connected=True),
    )
    outputs = (Port("report", "data"), Port("plot", required=False))
    defaults = {**PARAMETERS, **THRESHOLDS, "bins": 10, "records": [], "calibration": {}}
    parameter_choices = {"score_kind": SCORE_KINDS}
    parameter_ranges = {
        **dict.fromkeys(THRESHOLDS, UNIT_RANGE),
        "bins": NumericRange(2, 50, whole=True),
    }
    parameter_help = {
        **PARAMETER_HELP,
        **THRESHOLD_HELP,
        "records": "Separate test images with id, capture_group, label and complete scores. "
        "Use null label for deliberately ambiguous or unknown controls.",
        "calibration": "Saved fit model; the wired model takes precedence.",
        "bins": "Equal-width top-confidence bins for expected calibration error and plots.",
    }
    process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        ordered_classes(p["classes"])
        revision_text(p["score_source"], "Score source")

    def process(self, inputs):
        p = dict(self.parameters)
        records = inputs.get("records", p.pop("records"))
        model = inputs.get("calibration", p.pop("calibration"))
        p["bins"] = int(p["bins"])
        report = audit_calibration(records, model, **p)
        plot = reliability_plot(report) if self.generate_graphics else NO_VALUE
        return Result(
            {"report": report, "plot": plot},
            message=f"Known coverage {report['known_coverage']:.1%}; ECE "
            f"{report['raw']['ece']:.4f} to {report['calibrated']['ece']:.4f}",
            graphics={"Reliability": plot} if plot is not NO_VALUE else {},
            payloads={"plot": {"spatial": frame([480, 410], self.id + ":reliability")}}
            if plot is not NO_VALUE
            else {},
        )


CONFIDENCE_BLOCKS = (
    FitConfidenceCalibration,
    ApplyConfidenceCalibration,
    EvaluateConfidenceCalibration,
)
