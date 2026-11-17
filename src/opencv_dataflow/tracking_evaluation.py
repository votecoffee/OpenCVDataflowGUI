"""Offline bounded tracking metrics with explicit identity and fragmentation semantics."""

import numpy as np

from .detection_evaluation import box_iou, number, ratio
from .parameter_validation import ValidationError
from .tracking_assignment import associate_costs


def _records(value):
    if not isinstance(value, list) or len(value) > 128:
        raise ValidationError("Use at most 128 tracking records on each side per frame")
    found = set()
    for row in value:
        if not isinstance(row, dict) or set(row) - {"id", "bbox", "label"}:
            raise ValidationError("Tracking records need id, XYWH bbox and optional label")
        identity = row.get("id")
        if (
            type(identity) not in (int, str)
            or len(str(identity)) > 256
            or (type(identity), identity) in found
        ):
            raise ValidationError("Tracking IDs must be unique bounded integers or text per frame")
        found.add((type(identity), identity))
        bounds = row.get("bbox")
        if not isinstance(bounds, list) or len(bounds) != 4:
            raise ValidationError("Tracking boxes need four XYWH values")
        for i, v in enumerate(bounds):
            number(v, "bbox", 1e-6 if i > 1 else -1e6, 1e6)
        if row.get("label") is not None and (
            type(row["label"]) not in (int, str) or len(str(row["label"])) > 256
        ):
            raise ValidationError("Tracking labels need bounded text or integers")
    return value


def evaluate_tracking(frames, *, iou_threshold=0.5):
    """Score exactly supplied predictions; callers explicitly choose observed-only or all tracks.

    Truth absence is unannotated for that identity, not a missed observation. An
    ID switch compares its last match even across absence. Fragmentation requires
    a labeled unmatched observation between two matches. No HOTA/IDF1 claim.
    """
    number(iou_threshold, "iou_threshold", 1e-6, 1)
    if not isinstance(frames, list) or not 1 <= len(frames) <= 10000:
        raise ValidationError("Use 1..10000 fully annotated tracking frames")
    work = total = 0
    for frame in frames:
        if not isinstance(frame, dict) or set(frame) != {"truth", "predictions"}:
            raise ValidationError("Every tracking frame needs explicit truth and predictions")
        expected, predicted = _records(frame["truth"]), _records(frame["predictions"])
        work += len(expected) * len(predicted)
        total += len(expected) + len(predicted)
        if work > 2_000_000 or total > 50_000:
            raise ValidationError("Tracking evaluation exceeds record/comparison budget")
    tp = fp = fn = switches = fragments = 0
    histories, per_frame, matched_ious = {}, [], []
    for index, frame in enumerate(frames):
        expected, predicted = frame["truth"], frame["predictions"]
        overlaps = box_iou(expected, predicted)
        allowed = overlaps >= iou_threshold
        for r, row in enumerate(expected):
            for c, other in enumerate(predicted):
                if row.get("label") is not None and other.get("label") is not None:
                    allowed[r, c] &= row["label"] == other["label"]
        matches = associate_costs(1 - overlaps, allowed)
        tp += len(matches)
        fp += len(predicted) - len(matches)
        fn += len(expected) - len(matches)
        events = []
        for r, row in enumerate(expected):
            key = (type(row["id"]), row["id"])
            history = histories.setdefault(key, {"last": None, "gap": False})
            if r not in matches:
                if history["last"] is not None:
                    history["gap"] = True
                continue
            c = matches[r]
            identity = (type(predicted[c]["id"]), predicted[c]["id"])
            switched = history["last"] is not None and history["last"] != identity
            fragmented = history["gap"]
            switches += int(switched)
            fragments += int(fragmented)
            matched_ious.append(float(overlaps[r, c]))
            events.append(
                {
                    "truth_id": row["id"],
                    "track_id": predicted[c]["id"],
                    "iou": float(overlaps[r, c]),
                    "id_switch": switched,
                    "fragment": fragmented,
                }
            )
            history.update(last=identity, gap=False)
        per_frame.append(
            {
                "frame": index,
                "tp": len(matches),
                "fp": len(predicted) - len(matches),
                "fn": len(expected) - len(matches),
                "matches": events,
            }
        )
    return {
        "model": "tracking_evaluation_v1",
        "frames": len(frames),
        "summary": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "id_switches": switches,
            "fragmentations": fragments,
            "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn),
            "mota": 1 - (fn + fp + switches) / (tp + fn) if tp + fn else None,
            "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else None,
        },
        "iou_threshold": iou_threshold,
        "per_frame": per_frame,
        "semantics": (
            "Supplied predictions only; ID switches retain last match across absence; "
            "fragments require labeled misses. Not HOTA or IDF1."
        ),
    }
