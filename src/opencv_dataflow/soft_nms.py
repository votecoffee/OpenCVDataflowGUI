"""Class-aware score-decay suppression with explicit native integer-box geometry."""

import copy

import cv2
import numpy as np

from .core import Block, Port, Result
from .detection_fusion import rectangle
from .graphics_layers import GraphicDrawing, graphic_object
from .objects import validate_collection
from .parameter_validation import NumericRange, ValidationError
from .payloads import ValuePacket, check_spatial, require_same_frame, validate_payload
from .processing import image_value
from .recipes import unique_ref
from .selection import SELECTION_INPUTS, contains, selection_mask
from .tickets import ticket_key

METHODS = {
    "linear": cv2.dnn.SoftNMSMethod_SOFTNMS_LINEAR,
    "gaussian": cv2.dnn.SoftNMSMethod_SOFTNMS_GAUSSIAN,
}


def native_box(bounds, policy):
    edges = (
        np.r_[np.floor(bounds[:2]), np.ceil(bounds[2:])] if policy == "cover" else np.rint(bounds)
    ).astype(np.int64)
    size = edges[2:] - edges[:2]
    # Rect<int>::area and the pairwise area sum must stay representable in int32.
    if np.any(size <= 0) or np.any(size > 32767):
        raise ValidationError("Rounded native boxes need positive dimensions at most 32767 pixels")
    return [int(n) for n in (*edges[:2], *size)]


def source_context(block, data, size):
    contexts = [
        block.input_payloads.get("image", {}),
        data.get("context", {}),
        block.input_payloads.get("collection", {}),
    ]
    for context in contexts:
        validate_payload(context)
        check_spatial(context, size)
    require_same_frame(
        [ValuePacket(None, p) for p in contexts],
        message="Suppression inputs belong to different image frames",
    )
    frames = [p["spatial"] for p in contexts if "spatial" in p]
    if frames and any(f["root_size"] != frames[0]["root_size"] for f in frames[1:]):
        raise ValidationError("Suppression input root dimensions disagree")
    for key in ("pyramid", "recipe_ref"):
        values = [p[key] for p in contexts if key in p]
        if values and any(v != values[0] for v in values[1:]):
            raise ValidationError(f"Suppression input {key} contexts disagree")
    tickets = [p["ticket"] for p in contexts if "ticket" in p]
    if tickets and any(ticket_key(t) != ticket_key(tickets[0]) for t in tickets[1:]):
        raise ValidationError("Suppression inputs belong to different acquisition tickets")
    context = copy.deepcopy(contexts[0])
    for p in contexts[1:]:
        for key in ("spatial", "pyramid", "ticket", "recipe_ref"):
            if key in p and key not in context:
                context[key] = copy.deepcopy(p[key])
    recipe = unique_ref([ValuePacket(None, p) for p in contexts])
    if recipe is not None:
        context["recipe_ref"] = copy.deepcopy(recipe)
    return context, {
        "ticket_checked": "ticket" in contexts[0] and "ticket" in contexts[2],
        "frame_checked": bool(contexts[0].get("spatial", {}).get("root_id"))
        and any(p.get("spatial", {}).get("root_id") for p in contexts[1:]),
    }


class SuppressDetections(Block):
    type_id, title = "ml.soft_nms", "Suppress overlapping detections (Soft-NMS)"
    category = block_type = "ML"
    subtype = "Object detection"
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = (*SELECTION_INPUTS, Port("collection", "data", visual_type="collection"))
    outputs = (
        Port("image"),
        Port("collection", "data", visual_type="collection"),
        Port("count", "integer"),
        Port("indices", "data"),
        Port("scores", "data"),
        Port("report", "data"),
    )
    defaults = {
        "method": "linear",
        "score_threshold": 0.25,
        "iou_threshold": 0.5,
        "sigma": 0.5,
        "top_k": 0,
        "class_agnostic": False,
        "box_source": "unclipped",
        "rounding": "cover",
        "accepted_class_ids": [],
        "minimum_count": 1,
        "maximum_count": 0,
        "include_candidate_details": False,
        "maximum_raw_detections": 10_000,
        "maximum_candidates": 2048,
        "maximum_results": 2048,
        "maximum_image_pixels": 16_777_216,
    }
    parameter_choices = {
        "method": ("linear", "gaussian", "hard", "none"),
        "box_source": ("unclipped", "bbox"),
        "rounding": ("cover", "nearest"),
    }
    parameter_ranges = {
        "score_threshold": NumericRange(0, 1),
        "iou_threshold": NumericRange(0, 1),
        "sigma": NumericRange(0.000001, 100),
        "top_k": NumericRange(0, 4096, whole=True),
        "minimum_count": NumericRange(0, 4096, whole=True),
        "maximum_count": NumericRange(0, 4096, whole=True),
        "maximum_raw_detections": NumericRange(1, 50_000, whole=True),
        "maximum_candidates": NumericRange(1, 4096, whole=True),
        "maximum_results": NumericRange(1, 4096, whole=True),
        "maximum_image_pixels": NumericRange(1, 64_000_000, whole=True),
    }
    parameter_help = {
        "method": "Native linear/Gaussian Soft-NMS, hard NMS comparison, or no suppression.",
        "score_threshold": (
            "Float32 scores: Soft/None keep equality, native Hard requires strictly greater. "
            "Soft modes apply the threshold again after decay."
        ),
        "iou_threshold": (
            "Linear decays above this IoU; Hard suppresses above it. Gaussian ignores it."
        ),
        "sigma": "Positive float32 Gaussian decay denominator: exp(-IoU squared / sigma).",
        "top_k": (
            "Maximum retained per class, or globally when class-agnostic; zero unlimited. "
            "Hard applies this after native suppression, not as a candidate prefilter."
        ),
        "class_agnostic": "Allow overlapping classes to suppress/decay each other.",
        "box_source": (
            "Require unclipped_xyxy, or use bbox XYWH explicitly. Output geometry stays unchanged."
        ),
        "rounding": (
            "Native Soft-NMS requires integer boxes. Cover floors left/top and ceils right/bottom. "
            "Nearest rounds edges, ties to even; collapsed boxes reject. Geometry is not rounded."
        ),
        "accepted_class_ids": (
            "Empty accepts all; otherwise distinct nonnegative integer class IDs."
        ),
        "minimum_count": "Fail below this retained count; valid empty collections still emit.",
        "maximum_count": "Zero disables the upper inspection-count criterion.",
        "include_candidate_details": (
            "Optional bounded trace with source indices/IDs, native boxes and retained scores. "
            "Rejected final scores are unknown because native Soft-NMS does not return them."
        ),
        "maximum_raw_detections": "Reject excess input records before validation/native work.",
        "maximum_candidates": "Bound quadratic native work; reject excess without truncating.",
        "maximum_results": "Reject excess output records instead of publishing partial results.",
        "maximum_image_pixels": "Source image budget; maximum axis 16384 pixels.",
    }
    input_help = {
        "image": "Unchanged review image in the collection's source grid.",
        "collection": (
            "Zero-origin axis-aligned boxes with score/class_id/label, e.g. Decode detections "
            "with NMS off. Already removed proposals cannot be recovered."
        ),
        "mask": "Optional center selection in the source image; intersects Region.",
        "region": "Selects the pixel cell containing the chosen source box center; no cropping.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for key, limits in self.parameter_ranges.items():
            if type(p[key]) not in (int, float) or (limits.whole and type(p[key]) is not int):
                raise ValidationError("Use typed numeric suppression settings", key)
        for key in ("class_agnostic", "include_candidate_details"):
            if type(p[key]) is not bool:
                raise ValidationError("Use a checkbox boolean", key)
        ids = p["accepted_class_ids"]
        if (
            not isinstance(ids, list)
            or len(ids) > 4096
            or any(type(v) is not int or not 0 <= v <= 1_000_000 for v in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValidationError(
                "Class IDs must be distinct nonnegative integers", "accepted_class_ids"
            )
        if p["maximum_count"] and p["minimum_count"] > p["maximum_count"]:
            raise ValidationError(
                "Minimum count exceeds maximum count", "minimum_count", "maximum_count"
            )

    def process(self, inputs):
        p = self.parameters
        image = image_value(inputs["image"])
        h, w = image.shape[:2]
        if max(h, w) > 16384 or h * w > p["maximum_image_pixels"]:
            raise ValidationError(
                "Suppression image exceeds pixel/axis budget", "maximum_image_pixels"
            )
        data = inputs["collection"]
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("items"), list)
            or len(data["items"]) > p["maximum_raw_detections"]
        ):
            raise ValidationError(
                "Suppression input exceeds raw detection budget", "maximum_raw_detections"
            )
        validate_collection(data)
        if (
            data["kind"] != "boxes"
            or data.get("origin", [0, 0]) != [0, 0]
            or data["image_size"] != [w, h]
        ):
            raise ValidationError(
                "Suppression requires zero-origin boxes matching the source image size"
            )
        context, checked = source_context(self, data, [w, h])
        selected = selection_mask(inputs, image)
        threshold = float(np.float32(p["score_threshold"]))
        records, candidates, labels = [], [], {}
        for index, item in enumerate(data["items"]):
            score, cid, label = item.get("score"), item.get("class_id"), item.get("label")
            if (
                type(score) not in (int, float)
                or not np.isfinite(score)
                or not 0 <= score <= 1
                or type(cid) is not int
                or not 0 <= cid <= 1_000_000
                or not isinstance(label, str)
                or not label
                or len(label) > 256
                or item.get("orientation_measured", False)
            ):
                raise ValidationError(
                    "Boxes need probability scores, class IDs/labels and unmeasured orientation"
                )
            if cid in labels and labels[cid] != label:
                raise ValidationError("Class IDs have inconsistent labels")
            labels[cid] = label
            rectangle(item["geometry"].get("bbox"), "Displayed bbox")
            bounds = rectangle(
                item["geometry"].get(
                    "unclipped_xyxy" if p["box_source"] == "unclipped" else "bbox"
                ),
                "Suppression box",
                xyxy=p["box_source"] == "unclipped",
            )
            quantized = native_box(bounds, p["rounding"])
            score32 = float(np.float32(score))
            status = "candidate"
            if score32 < threshold or (p["method"] == "hard" and score32 == threshold):
                status = "confidence"
            elif p["accepted_class_ids"] and cid not in p["accepted_class_ids"]:
                status = "class_filter"
            elif not contains(selected, np.floor((bounds[:2] + bounds[2:]) / 2)):
                status = "selection"
            records.append(
                {
                    "source_index": index,
                    "id": item["id"],
                    "class_id": cid,
                    "original_score": score,
                    "input_score_float32": score32,
                    "native_bbox": quantized,
                    "status": status,
                    "updated_score": None,
                }
            )
            if status == "candidate":
                candidates.append(index)
        if len(candidates) > p["maximum_candidates"]:
            raise ValidationError(
                "Suppression exceeds quadratic candidate budget", "maximum_candidates"
            )
        groups = {}
        for i in candidates:
            groups.setdefault(0 if p["class_agnostic"] else records[i]["class_id"], []).append(i)
        retained = []
        for indices in groups.values():
            boxes = [records[i]["native_bbox"] for i in indices]
            scores = [records[i]["input_score_float32"] for i in indices]
            if p["method"] in METHODS:
                updated, keep = cv2.dnn.softNMSBoxes(
                    boxes,
                    scores,
                    threshold,
                    p["iou_threshold"],
                    top_k=p["top_k"],
                    sigma=p["sigma"],
                    method=METHODS[p["method"]],
                )
            else:
                keep = (
                    np.asarray(cv2.dnn.NMSBoxes(boxes, scores, threshold, p["iou_threshold"]))
                    .reshape(-1)
                    .tolist()
                    if p["method"] == "hard"
                    else sorted(range(len(indices)), key=lambda k: (-scores[k], k))
                )
                keep = keep[: p["top_k"]] if p["top_k"] else keep
                updated = [scores[int(k)] for k in keep]
            keep, updated = np.asarray(keep), np.asarray(updated)
            if (
                keep.ndim != 1
                or updated.ndim != 1
                or len(keep) != len(updated)
                or len(set(keep.tolist())) != len(keep)
                or not np.isfinite(updated).all()
                or any(
                    isinstance(k, (bool, np.bool_))
                    or not isinstance(k, (int, np.integer))
                    or not 0 <= k < len(indices)
                    for k in keep
                )
                or np.any(updated < threshold)
                or np.any(updated > 1)
            ):
                raise ValueError("Native suppression returned invalid or misaligned scores/indices")
            for k, score in zip(keep, updated, strict=True):
                source = indices[int(k)]
                if score > records[source]["input_score_float32"]:
                    raise ValueError("Native suppression unexpectedly increased a score")
                retained.append((source, float(score)))
        retained.sort(key=lambda row: (-row[1], row[0]))
        if len(retained) > p["maximum_results"]:
            raise ValidationError("Suppression exceeds result budget", "maximum_results")
        for i in candidates:
            records[i]["status"] = "suppressed_or_top_k"
        items, drawings = [], []
        for index, score in retained:
            record = records[index]
            record.update(status="retained", updated_score=score)
            item = copy.deepcopy(data["items"][index])
            item["score"] = item["metrics"]["score"] = score
            item["soft_nms"] = {"method": p["method"], **copy.deepcopy(record)}
            items.append(item)
        if self.generate_graphics:
            candidate_set = set(candidates)
            for record in records:
                if record["source_index"] not in candidate_set:
                    continue
                x, y, width, height = data["items"][record["source_index"]]["geometry"]["bbox"]
                color = (60, 230, 120) if record["status"] == "retained" else (70, 90, 230)
                value = record["updated_score"]
                text = (
                    f"{record['id']}: {value:.3f}"
                    if value is not None
                    else f"{record['id']}: removed"
                )
                drawings.extend(
                    (
                        graphic_object(
                            "polyline",
                            [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
                            color,
                            closed=True,
                        ),
                        graphic_object("label", [[x, y]], color, text=text),
                    )
                )
        output = copy.deepcopy(data)
        output.update(
            items=items,
            count=len(items),
            context={
                **copy.deepcopy(data.get("context", {})),
                **{k: copy.deepcopy(context[k]) for k in ("spatial", "pyramid") if k in context},
            },
        )
        report = {
            "model": "soft_nms_v1",
            "method": p["method"],
            "class_aware": not p["class_agnostic"],
            "box_source": p["box_source"],
            "rounding": p["rounding"],
            "native_coordinate_convention": "integer_source_pixel_edges",
            "output_geometry_unchanged": True,
            "score_threshold_float32": threshold,
            "threshold_relation": ">" if p["method"] == "hard" else ">=",
            "iou_threshold_used": p["method"] in ("linear", "hard"),
            "top_k_scope": "retained_per_group",
            "context_checks": checked,
            "raw_count": len(records),
            "selected_candidates": len(candidates),
            "retained": len(items),
            "filtered_counts": {
                s: sum(r["status"] == s for r in records)
                for s in ("confidence", "class_filter", "selection", "suppressed_or_top_k")
            },
        }
        if p["include_candidate_details"]:
            report["candidates"] = records
        count = len(items)
        passed = count >= p["minimum_count"] and (
            not p["maximum_count"] or count <= p["maximum_count"]
        )
        return Result(
            {
                "image": inputs["image"],
                "collection": output,
                "count": count,
                "indices": [i for i, _ in retained],
                "scores": [s for _, s in retained],
                "report": report,
            },
            payloads={port.name: copy.deepcopy(context) for port in self.outputs},
            passed=passed,
            score=max((s for _, s in retained), default=0.0),
            fail_reasons=[]
            if passed
            else [
                {
                    "code": "suppression.count",
                    "message": f"Retained {count}; count criterion failed",
                }
            ],
            message=f"{count}/{len(records)} detections retained by {p['method']}",
            drawings={"Detection suppression": GraphicDrawing(tuple(drawings))} if drawings else {},
        )
