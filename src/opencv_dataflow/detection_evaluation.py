"""Bounded, class-aware axis-aligned detection evaluation; independent of model engines."""

import copy
from collections import Counter

import cv2
import numpy as np

from .classification_evaluation import case_metadata, error_text, text
from .classifier_common import canonical_json, label_strings
from .core import Block, Port, Result
from .objects import validate_collection
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, FlowGroup, ValuePacket, frame, require_same_frame
from .tickets import ticket_key

MAX_IMAGES, MAX_OBJECTS, MAX_PER_IMAGE, MAX_PAIRS = 10000, 20000, 256, 2_000_000
METRICS = {
    "classes": [],
    "iou_threshold": 0.5,
    "score_threshold": 0.0,
    "iou_thresholds": [0.5, 0.75],
    "score_thresholds": [],
    "ap_interpolation": "all_points",
    "min_precision": 1.0,
    "min_recall": 1.0,
}
DEFAULTS = {**METRICS, "block": "", "output": "collection", "training_groups": []}


def number(value, name, low=-1e12, high=1e12):
    if type(value) not in (int, float) or not np.isfinite(value) or not low <= value <= high:
        raise ValidationError(f"{name} needs a finite number in [{low}, {high}]", name)
    return float(value)


def settings(value):
    if not isinstance(value, dict) or set(value) - set(METRICS):
        raise ValidationError("Invalid detector metric settings")
    p = {**copy.deepcopy(METRICS), **copy.deepcopy(value)}
    labels = label_strings(p["classes"], "Classes", maximum=128)
    if len(set(labels)) != len(labels):
        raise ValidationError("Class labels must be distinct")
    for key in ("iou_threshold", "min_precision", "min_recall"):
        number(p[key], key, 0.000001 if key == "iou_threshold" else 0, 1)
    number(p["score_threshold"], "score_threshold")
    for key, limit, low, high in (
        ("iou_thresholds", 10, 0.000001, 1),
        ("score_thresholds", 32, -1e12, 1e12),
    ):
        values = p[key]
        if not isinstance(values, list) or len(values) > limit:
            raise ValidationError(f"{key} supports up to {limit} cutoffs", key)
        for item in values:
            number(item, key, low, high)
        if len(set(values)) != len(values):
            raise ValidationError("Cutoffs must be distinct", key)
    if p["ap_interpolation"] not in ("all_points", "101_points"):
        raise ValidationError("Choose all_points or 101_points AP interpolation")
    return p


def image_size(value):
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(n) is not int or not 1 <= n <= 16384 for n in value)
        or value[0] * value[1] > 64_000_000
    ):
        raise ValidationError("Image size needs [width, height], at most 16384/axis and 64M pixels")
    return list(value)


def objects(value, size, *, predicted=False):
    if not isinstance(value, list) or len(value) > MAX_PER_IMAGE:
        raise ValidationError("Use up to 256 objects per image and side")
    result, ids = [], set()
    for item in value:
        allowed = {"id", "label", "bbox", "score"} if predicted else {"id", "label", "bbox"}
        if not isinstance(item, dict) or set(item) != allowed:
            raise ValidationError("Objects need id, label, XYWH bbox, plus score for predictions")
        identity = text(item["id"], "Object ID")
        text(item["label"], "Object label")
        if identity in ids:
            raise ValidationError("Duplicate object ID within one image/side")
        ids.add(identity)
        box = item["bbox"]
        if not isinstance(box, list) or len(box) != 4:
            raise ValidationError("Bounding boxes need [left, top, width, height]")
        x, y, w, h = [number(v, "bbox", -1e6, 1e6) for v in box]
        if w <= 0 or h <= 0 or w * h == 0:
            raise ValidationError(
                "Bounding box width/height and representable area must be positive"
            )
        if not predicted and (x < 0 or y < 0 or x + w > size[0] or y + h > size[1]):
            raise ValidationError("Ground-truth boxes must be inside their declared source image")
        if predicted:
            number(item["score"], "score")
        result.append(copy.deepcopy(item))
    return sorted(result, key=lambda item: item["id"])


def truth(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"size", "objects"}:
        raise ValidationError("Detection truth needs size and objects, or null for unlabeled")
    size = image_size(value["size"])
    return {"size": size, "objects": objects(value["objects"], size)}


def detection_config(value, cases=()):
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValidationError("Invalid detector evaluation configuration")
    p = {**copy.deepcopy(DEFAULTS), **copy.deepcopy(value)}
    settings({k: p[k] for k in METRICS})
    text(p["block"], "Detector block", maximum=200)
    text(p["output"], "Detector output", maximum=200)
    groups = label_strings(p["training_groups"], "Training groups", maximum=MAX_IMAGES)
    if len(set(groups)) != len(groups):
        raise ValidationError("Training groups must be distinct")
    if set(groups) & {case.get("capture_group", case["id"]) for case in cases}:
        raise ValidationError("Training/evaluation capture groups overlap")
    return p


def records(value, expected_cases=None):
    if isinstance(value, FlowGroup):
        value = [item.value for item in value.items]
    if isinstance(value, dict) and value.get("model") == "validation_report_v1":
        value = [row["detection"] for row in value["cases"] if "detection" in row]
    if not isinstance(value, list) or len(value) > MAX_IMAGES:
        raise ValidationError("Use up to 10,000 detection image records")
    try:
        if len(canonical_json(value).encode()) > 16 * 1024**2:
            raise ValidationError("Detection records exceed 16 MiB")
    except (ValueError, TypeError) as exc:
        raise ValidationError("Detection records need finite bounded JSON values") from exc
    result, by_id = [], {}
    for record in value:
        if not isinstance(record, dict):
            raise ValidationError("Each detection image record must be an object")
        item = copy.deepcopy(record)
        identity = text(item.get("id"), "Image ID")
        if identity in by_id:
            raise ValidationError("Duplicate detection image ID")
        if "group" in item:
            text(item["group"], "Capture group")
        actual = image_size(item["size"]) if item.get("size") is not None else None
        item["truth"] = truth(item.get("truth"))
        if item.get("status") == "predicted" and actual is None:
            raise ValidationError("Observed predictions require the current source image size")
        if (
            item.get("status") == "predicted"
            and item["truth"] is not None
            and item["truth"]["size"] != actual
        ):
            raise ValidationError("Detection truth and source image sizes differ")
        item["predictions"] = objects(item.get("predictions"), actual, predicted=True)
        if item.get("status") not in ("predicted", "missing", "error"):
            raise ValidationError("Detection status must be predicted, missing or error")
        if item["status"] != "predicted" and item["predictions"]:
            raise ValidationError("Missing/error images cannot have current predictions")
        by_id[identity] = item
        result.append(item)
    if expected_cases is not None:
        if not isinstance(expected_cases, list) or len(expected_cases) > MAX_IMAGES:
            raise ValidationError("Expected cases exceed 10,000 images")
        expected = {}
        for case in expected_cases:
            if not isinstance(case, dict):
                raise ValidationError("Expected cases must be objects")
            identity = text(case.get("id"), "Expected image ID")
            if identity in expected:
                raise ValidationError("Duplicate expected image ID")
            case_metadata(case)
            expected[identity] = case
            known = truth(case.get("detection_truth"))
            if identity in by_id:
                found = by_id[identity]
                if found["truth"] != known or found.get(
                    "group", case.get("capture_group", identity)
                ) != case.get("capture_group", identity):
                    raise ValidationError("Detection record truth/group differs from expected case")
            else:
                result.append(
                    {
                        "id": identity,
                        "truth": known,
                        "size": None,
                        "group": case.get("capture_group", identity),
                        "status": "missing",
                        "predictions": [],
                    }
                )
        if by_id.keys() - expected.keys():
            raise ValidationError("Unexpected detection image IDs")
    total = sum(
        len(row["predictions"]) + len((row["truth"] or {}).get("objects", [])) for row in result
    )
    pairs = sum(
        len(row["predictions"]) * len((row["truth"] or {}).get("objects", [])) for row in result
    )
    if total > MAX_OBJECTS or pairs > MAX_PAIRS:
        raise ValidationError("Evaluation exceeds 20,000 objects or 2,000,000 IoU pairs")
    return sorted(result, key=lambda item: item["id"])


def box_iou(predictions, expected):
    a = np.asarray([p["bbox"] for p in predictions], float).reshape(-1, 4)
    b = np.asarray([p["bbox"] for p in expected], float).reshape(-1, 4)
    # Work relative to the first box instead of adding tiny widths to large
    # absolute coordinates, which can round an otherwise identical box's IoU.
    offset = b[None, :, :2] - a[:, None, :2]
    low = np.maximum(0, offset)
    high = np.minimum(a[:, None, 2:], offset + b[None, :, 2:])
    intersection = np.prod(np.maximum(0, high - low), axis=2)
    return intersection / (
        np.prod(a[:, 2:], axis=1)[:, None] + np.prod(b[:, 2:], axis=1)[None, :] - intersection
    )


def ratio(a, b):
    return a / b if b else None


def counts(tp, fp, support):
    return {
        "tp": tp,
        "fp": fp,
        "fn": support - tp,
        "support": support,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, support),
        "f1": ratio(2 * tp, tp + fp + support),
    }


def average_precision(curve, support, method):
    if not support:
        return None
    if not curve:
        return 0.0
    precision = np.maximum.accumulate(np.asarray([p["precision"] for p in curve])[::-1])[::-1]
    recall = np.asarray([p["recall"] for p in curve])
    if method == "101_points":
        return float(
            np.mean(
                [precision[recall >= target].max(initial=0) for target in np.linspace(0, 1, 101)]
            )
        )
    return float(np.sum(np.diff(np.r_[0, recall]) * precision))


def evaluate_detections(value, *, expected_cases=None, **parameters):
    p = settings(parameters)
    rows = records(value, expected_cases)
    labeled = [r for r in rows if r["truth"] is not None]
    classes = list(p["classes"]) + sorted(
        {o["label"] for row in labeled for o in row["truth"]["objects"] + row["predictions"]}
        - set(p["classes"])
    )
    if len(classes) > 128:
        raise ValidationError("Evaluation exceeds 128 classes")
    support = Counter(o["label"] for r in labeled for o in r["truth"]["objects"])
    prepared = []
    for row in labeled:
        predictions = sorted(row["predictions"], key=lambda item: (-item["score"], item["id"]))
        expected = row["truth"]["objects"]
        prepared.append((row, predictions, expected, box_iou(predictions, expected)))
    thresholds = list(dict.fromkeys([p["iou_threshold"], *p["iou_thresholds"]]))
    reports = []
    for cutoff in thresholds:
        events, images = [], []
        for row, predictions, expected, matrix in prepared:
            used = set()
            current = []
            for i, prediction in enumerate(predictions):
                candidates = [
                    j
                    for j, target in enumerate(expected)
                    if j not in used
                    and target["label"] == prediction["label"]
                    and matrix[i, j] >= cutoff
                ]
                match = (
                    min(candidates, key=lambda j: (-matrix[i, j], expected[j]["id"]))
                    if candidates
                    else None
                )
                if match is not None:
                    used.add(match)
                event = {
                    "image": row["id"],
                    "prediction": prediction["id"],
                    "label": prediction["label"],
                    "score": prediction["score"],
                    "truth": expected[match]["id"] if match is not None else None,
                    "iou": float(matrix[i, match]) if match is not None else None,
                    "tp": match is not None,
                }
                events.append(event)
                current.append(event)
            selected = [e for e in current if e["score"] >= p["score_threshold"]]
            selected_truth = {e["truth"] for e in selected if e["tp"]}
            images.append(
                {
                    "id": row["id"],
                    **counts(
                        len(selected_truth), sum(not e["tp"] for e in selected), len(expected)
                    ),
                    "matches": selected,
                    "missed_truth": [o["id"] for o in expected if o["id"] not in selected_truth],
                }
            )
        events.sort(key=lambda e: (-e["score"], e["image"], e["prediction"]))
        if cutoff == p["iou_threshold"]:
            base_events = events
        per_class = []
        for label in classes:
            found = [e for e in events if e["label"] == label]
            curve, tp, fp = [], 0, 0
            for i, event in enumerate(found):
                tp += event["tp"]
                fp += not event["tp"]
                if i + 1 == len(found) or event["score"] != found[i + 1]["score"]:
                    curve.append({"score": event["score"], **counts(tp, fp, support[label])})
            selected = [e for e in found if e["score"] >= p["score_threshold"]]
            per_class.append(
                {
                    "label": label,
                    **counts(
                        sum(e["tp"] for e in selected),
                        sum(not e["tp"] for e in selected),
                        support[label],
                    ),
                    "curve": curve,
                    "ap": average_precision(curve, support[label], p["ap_interpolation"]),
                }
            )
        selected = [e for e in events if e["score"] >= p["score_threshold"]]
        ap = [r["ap"] for r in per_class if r["ap"] is not None]
        summary = {
            **counts(
                sum(e["tp"] for e in selected),
                sum(not e["tp"] for e in selected),
                sum(support.values()),
            ),
            "map": float(np.mean(ap)) if ap else None,
            "ap_classes": len(ap),
        }
        reports.append(
            {"iou_threshold": cutoff, "summary": summary, "per_class": per_class, "images": images}
        )
    base = reports[0]
    summary = {
        **base["summary"],
        "total_images": len(rows),
        "labeled_images": len(labeled),
        "unlabeled_images": len(rows) - len(labeled),
        "status_counts": dict(Counter(r["status"] for r in rows)),
    }
    maps = [report["summary"]["map"] for report in reports if report["summary"]["map"] is not None]
    summary["mean_ap_over_iou"] = float(np.mean(maps)) if maps else None
    summary["passed"] = (
        bool(labeled)
        and not any(r["status"] != "predicted" for r in rows)
        and (
            (summary["precision"] is not None and summary["precision"] >= p["min_precision"])
            or (summary["tp"] + summary["fp"] == 0 and summary["support"] == 0)
        )
        and (summary["recall"] is None or summary["recall"] >= p["min_recall"])
    )
    sweeps = []
    for score in p["score_thresholds"]:
        selected = [e for e in base_events if e["score"] >= score]
        sweeps.append(
            {
                "score_threshold": score,
                **counts(
                    sum(e["tp"] for e in selected),
                    sum(not e["tp"] for e in selected),
                    summary["support"],
                ),
            }
        )
    return {
        "model": "image_detection_evaluation_v1",
        "settings": p,
        "classes": classes,
        "records": rows,
        "expected_ids_verified": expected_cases is not None,
        "summary": summary,
        "per_class": base["per_class"],
        "images": base["images"],
        "iou_reports": reports,
        "score_sweeps": sweeps,
    }


def capture_detections(graph, config, case, ticket, errors, source_id):
    known = truth(case.get("detection_truth"))
    row = {
        "id": case["id"],
        "group": case.get("capture_group", case["id"]),
        "truth": known,
        "size": None,
        "status": "missing",
        "predictions": [],
    }
    try:
        block = graph.blocks[config["block"]]
        if block.state == "error":
            raise ValidationError(block.error)
        packets = block.current_emissions.get(config["output"], [])
        if not packets:
            if errors:
                raise ValidationError("Program errors prevented a current detection")
            return row
        sources = graph.blocks[source_id].current_emissions.get("image", [])
        if len(packets) != 1 or len(sources) != 1:
            raise ValidationError("Evaluation needs one current source image and collection event")
        packet, source = packets[0], sources[0]
        if ticket is None or any(
            ticket_key(item.payload.get("ticket", {})) != ticket_key(ticket)
            for item in (packet, source)
        ):
            raise ValidationError(
                "Detection/source ticket does not match the current validation image"
            )
        value = packet.value
        validate_collection(value)
        if value["kind"] != "boxes" or value.get("origin", [0, 0]) != [0, 0]:
            raise ValidationError("Use source-grid boxes with zero collection origin")
        contexts = [packet, source, ValuePacket(None, value["context"])]
        if not all(item.payload.get("spatial", {}).get("root_id") is not None for item in contexts):
            raise ValidationError("Detection evaluation requires explicit source-grid metadata")
        require_same_frame(contexts, message="Detection boxes use a different source image grid")
        size = list(source.value.shape[1::-1])
        row["size"] = size
        if value["image_size"] != size or (known is not None and known["size"] != size):
            raise ValidationError(
                "Detection/ground-truth sizes differ from the current source image"
            )
        if any(item.get("orientation_measured", False) for item in value["items"]):
            raise ValidationError("Rotated geometry needs a separate oriented evaluation adapter")
        predictions = [
            {
                "id": str(item["id"]),
                "label": item["label"],
                "score": item["score"],
                "bbox": item["geometry"]["bbox"],
            }
            for item in value["items"]
        ]
        row.update(
            size=size, predictions=objects(predictions, size, predicted=True), status="predicted"
        )
    except Exception as exc:
        row.update(status="error", predictions=[], error=error_text(f"{type(exc).__name__}: {exc}"))
    return row


def precision_recall_plot(report):
    plot = np.full((400, 760, 3), 250, np.uint8)
    cv2.rectangle(plot, (45, 30), (485, 310), (80, 80, 80), 1)
    for index, row in enumerate(report["per_class"][:16]):
        points = [
            (45 + round(440 * p["recall"]), 310 - round(280 * p["precision"]))
            for p in row["curve"]
            if p["recall"] is not None
        ]
        color = ((47 + index * 83) % 200, (121 + index * 57) % 200, (17 + index * 43) % 200)
        cv2.rectangle(plot, (510, 20 + 21 * index), (530, 32 + 21 * index), color, -1)
        label = f"{index}: {row['label'][:20]}".encode("ascii", "replace").decode()
        cv2.putText(
            plot, label, (540, 31 + 21 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1
        )
        if points:
            cv2.polylines(plot, [np.asarray(points, np.int32)], False, color, 2)
            for point in points:
                cv2.circle(plot, point, 3, color, -1)
    cv2.putText(
        plot,
        "Recall 0..1; Precision 0..1. First 16 classes; full names/counts in Report.",
        (25, 380),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (40, 40, 40),
        1,
    )
    return plot


class EvaluateDetections(Block):
    type_id, title = "ml.evaluate_detections", "Evaluate image detections"
    category, block_type, subtype = "ML", "ML", "Evaluation"
    inputs = (Port("records", "any", required=False, required_when_connected=True),)
    outputs = (Port("report", "data"), Port("plot", required=False))
    defaults = {**METRICS, "records": [], "expected_cases": []}
    parameter_choices = {"ap_interpolation": ("all_points", "101_points")}
    parameter_ranges = {
        "iou_threshold": NumericRange(0.000001, 1),
        "score_threshold": NumericRange(-1e12, 1e12),
        "min_precision": NumericRange(0, 1),
        "min_recall": NumericRange(0, 1),
    }
    parameter_help = {
        "records": "Records: id, size, truth, predictions, status. Accepts reports/groups too.",
        "expected_cases": "Full expected set with detection_truth; absent IDs count as missing.",
        "classes": "Optional display order. Observed extra labels stay in the report.",
        "iou_threshold": "Class-aware score-greedy matching; continuous source XYWH edges.",
        "score_threshold": "Keep >= cutoff for precision/recall; AP uses all supplied scores.",
        "iou_thresholds": "Up to ten extra IoU cutoffs for separate AP/precision/recall reports.",
        "score_thresholds": "Up to 32 cutoffs; cannot recover detections removed upstream.",
        "ap_interpolation": "Envelope area or 101 recall samples. Equal-score points grouped.",
        "min_precision": "TP/(TP+FP). Clean negative-only sets pass without numeric precision.",
        "min_recall": "Operating TP/ground-truth objects. Missing/error images always fail.",
    }
    parallel_safe = preview_safe = process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        settings({k: p[k] for k in METRICS})

    def process(self, inputs):
        p = self.parameters
        value = inputs.get("records", p["records"])
        expected = p["expected_cases"] or None
        if isinstance(value, dict) and value.get("model") == "validation_report_v1":
            expected = value["set"]["cases"]
        report = evaluate_detections(value, expected_cases=expected, **{k: p[k] for k in METRICS})
        plot = precision_recall_plot(report) if self.generate_graphics else NO_VALUE
        payloads = (
            {"plot": {"spatial": frame(list(plot.shape[1::-1]), self.id + ":pr")}}
            if plot is not NO_VALUE
            else {}
        )
        passed, summary = report["summary"]["passed"], report["summary"]
        return Result(
            {"report": report, "plot": plot},
            passed=passed,
            score=summary["f1"] if summary["f1"] is not None else float(passed),
            payloads=payloads,
            graphics={"Precision-recall": plot} if plot is not NO_VALUE else {},
            fail_reasons=[] if passed else ["detection_evaluation_limits"],
            message=f"{summary['tp']} TP / {summary['fp']} FP / {summary['fn']} missed objects",
        )
