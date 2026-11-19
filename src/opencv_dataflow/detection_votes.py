"""Native class-aware rectangle voting with verified source memberships."""

import copy
from collections import Counter, defaultdict

import cv2
import numpy as np

from .core import Block, Port, Result
from .detection_fusion import rectangle
from .graphics_layers import GraphicDrawing, graphic_object
from .objects import validate_collection
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .processing import image_value
from .soft_nms import native_box, source_context


def components(boxes, eps):
    """Find transitive connected components under the four-edge SimilarRects rule."""
    family = list(range(len(boxes)))
    edges = np.column_stack((boxes[:, :2], boxes[:, :2] + boxes[:, 2:]))

    def root(i):
        while family[i] != i:
            family[i] = family[family[i]]
            i = family[i]
        return i

    for i in range(len(family) - 1):
        delta = eps * np.minimum(boxes[i, 2:], boxes[i + 1 :, 2:]).sum(axis=1) * 0.5
        close = np.max(np.abs(edges[i] - edges[i + 1 :]), axis=1) <= delta
        leader = root(i)
        for j in np.flatnonzero(close).tolist():
            family[root(i + 1 + j)] = leader
    result = defaultdict(list)
    for i in range(len(family)):
        result[root(i)].append(i)
    return list(result.values())


def native_key(boxes, members):
    summed = boxes[members].sum(axis=0, dtype=np.int64)
    # OpenCV averages integer rectangle coordinates with a float32 reciprocal.
    mean = np.rint(summed.astype(np.float32) * (np.float32(1) / np.float32(len(members))))
    return tuple(mean.astype(np.int64).tolist()), len(members)


def voted_members(boxes, eps, minimum_votes, singleton_policy):
    original = np.asarray(boxes, dtype=np.int32).reshape(-1, 4)
    grouped, weights = cv2.groupRectangles(boxes, minimum_votes - 1, eps)
    grouped = np.asarray(grouped).reshape(-1, 4)
    weights = np.asarray(weights).reshape(-1)
    if (
        len(grouped) != len(weights)
        or (len(weights) and (grouped.dtype.kind not in "iu" or weights.dtype.kind not in "iu"))
        or np.any(weights < minimum_votes)
    ):
        raise ValidationError("Unexpected native rectangle grouping results")
    native = Counter(
        (tuple(box), votes) for box, votes in zip(grouped.tolist(), weights.tolist(), strict=True)
    )
    sets = components(original, eps)
    keys = [native_key(original, row) for row in sets]
    expected = Counter(key for key in keys if key[1] >= minimum_votes)
    for key, count in native.items():
        if count != expected[key]:
            raise ValidationError("Native rectangles cannot be assigned unambiguous vote sources")
    for key, members in zip(keys, sets, strict=True):
        box, votes = key
        if key in native:
            status = "grouped"
        elif votes == 1 and singleton_policy == "keep":
            status = "singleton_kept"
        elif votes < minimum_votes:
            status = "below_minimum_votes"
        else:
            status = "native_suppressed"
        yield list(box), members, status


def id_order(value):
    return isinstance(value, str), value


def representative(entries, native, size):
    x, y, native_w, native_h = native
    w, h = min(native_w, size[0] - x), min(native_h, size[1] - y)
    box = [x, y, w, h]
    members = sorted(entries, key=lambda item: id_order(item["id"]))
    source_ids = [item["id"] for item in members]
    cid = members[0]["class_id"]
    kind = members[0].get("score_kind")
    score = members[0].get("score")
    if len(members) > 1:
        comparable = kind is not None and all(
            item.get("score_kind") == kind and item.get("score") is not None for item in members
        )
        score = max(item["score"] for item in members) if comparable else None
    result = {
        "id": source_ids[0],
        "source_ids": source_ids,
        "class_id": cid,
        "label": members[0]["label"],
        "vote_count": len(members),
        "orientation_measured": False,
        "metrics": {
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "area": w * h,
            "center_x": x + w / 2,
            "center_y": y + h / 2,
            "aspect_ratio": w / h,
            "score": score,
            "angle": None,
            "strength": len(members),
            "vote_count": len(members),
            "class_id": cid,
        },
        "geometry": {
            "bbox": box,
            "unclipped_xyxy": [x, y, x + native_w, y + native_h],
            "clipped": box != native,
            "points": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
        },
    }
    if score is not None:
        result["score"] = score
        if kind is not None:
            result["score_kind"] = kind
    return result


def class_inputs(items, size, rounding):
    labels, classes = {}, defaultdict(list)
    for entry in items:
        class_id, label = entry.get("class_id"), entry.get("label")
        score, kind = entry.get("score"), entry.get("score_kind")
        identity = entry["id"]
        if (
            type(class_id) is not int
            or not 0 <= class_id <= 1_000_000
            or not isinstance(label, str)
            or not 1 <= len(label) <= 256
            or entry.get("orientation_measured", False)
            or (
                score is not None
                and (type(score) not in (int, float) or not -1e12 <= score <= 1e12)
            )
            or (kind is not None and (not isinstance(kind, str) or not 1 <= len(kind) <= 128))
            or (isinstance(identity, str) and not 1 <= len(identity) <= 256)
            or (type(identity) is int and not -(2**63) <= identity < 2**63)
        ):
            raise ValidationError("Voting needs axis-aligned boxes, class labels and bounded IDs")
        if class_id in labels and label != labels[class_id]:
            raise ValidationError("Class IDs have inconsistent labels")
        labels[class_id] = label
        bounds = rectangle(entry["geometry"].get("bbox"), "Visible voting box")
        if np.any(bounds[:2] < 0) or np.any(bounds[2:] - size > 1e-7):
            raise ValidationError("Visible voting box extends outside the source image")
        bounds[2:] = np.minimum(bounds[2:], size)
        boxes = native_box(bounds, rounding)
        classes[class_id].append((entry, boxes))
    return classes


class GroupDetectionVotes(Block):
    type_id, title = "ml.group_detection_votes", "Group detections by spatial votes"
    category, block_type, subtype = "ML", "ML", "Object detection"
    inputs = (
        Port("collection", "data", visual_type="collection"),
        Port("image", required=False, required_when_connected=True),
    )
    outputs = (
        Port("collection", "data", visual_type="collection"),
        Port("count", "integer"),
        Port("report", "data"),
        Port("image", required=False),
    )
    payload_reference = "collection"
    defaults = {
        "epsilon": 0.2,
        "minimum_votes": 2,
        "singleton_policy": "drop",
        "rounding": "cover",
        "minimum_count": 1,
        "maximum_count": 0,
        "maximum_candidates": 2048,
    }
    parameter_choices = {
        "singleton_policy": ("drop", "keep"),
        "rounding": ("cover", "nearest"),
    }
    parameter_ranges = {
        "epsilon": NumericRange(0, 1),
        "minimum_votes": NumericRange(2, 4096, whole=True),
        "minimum_count": NumericRange(0, 4096, whole=True),
        "maximum_count": NumericRange(0, 4096, whole=True),
        "maximum_candidates": NumericRange(1, 4096, whole=True),
    }
    parameter_help = {
        "epsilon": "Relative tolerance on all four rectangle edges. Neighbor chains join "
        "transitively within each class. Zero joins exact integer duplicates.",
        "minimum_votes": "Minimum group size. OpenCV can suppress a small contained group.",
        "singleton_policy": "Keep or drop isolated single votes. Groups of two or more must "
        "still meet Minimum votes.",
        "rounding": "Use visible bbox. Cover rounds outward; Nearest rounds edges to nearest. "
        "The grouped result is an integer rectangle.",
        "minimum_count": "Minimum number of retained representative boxes for inspection pass.",
        "maximum_count": "Maximum retained boxes; zero disables the upper count gate.",
        "maximum_candidates": "Bound the input count before pairwise grouping.",
    }
    input_help = {
        "collection": "Axis-aligned boxes in one zero-origin image, with class IDs.",
        "image": "Optional unchanged source image for overlays. Must match the box context.",
    }
    process_inputs_readonly = parallel_safe = preview_safe = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for key, limits in self.parameter_ranges.items():
            if type(p[key]) not in (int, float) or (limits.whole and type(p[key]) is not int):
                raise ValidationError("Use a typed numeric value", key)
        if p["maximum_count"] and p["maximum_count"] < p["minimum_count"]:
            raise ValidationError(
                "Maximum count cannot be below Minimum count", "maximum_count", "minimum_count"
            )

    def process(self, inputs):
        p = self.parameters
        data = inputs["collection"]
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("items"), list)
            or len(data["items"]) > p["maximum_candidates"]
        ):
            raise ValidationError("Vote grouping exceeds candidate allowance", "maximum_candidates")
        validate_collection(data)
        if data["kind"] != "boxes" or data.get("origin", [0, 0]) != [0, 0]:
            raise ValidationError("Vote grouping requires zero-origin detection boxes")
        size = data["image_size"]
        image = inputs.get("image", NO_VALUE)
        if image is not NO_VALUE and list(image_value(image).shape[1::-1]) != size:
            raise ValidationError("Voting boxes differ from the review image dimensions")
        context, checked = source_context(self, data, size, operation="Vote grouping")
        classes = class_inputs(data["items"], size, p["rounding"])
        items, records, graphics = [], [], []
        totals = Counter()
        for cid, pairs in sorted(classes.items()):
            entries, boxes = zip(*pairs, strict=True)
            for box, members, status in voted_members(
                boxes, p["epsilon"], p["minimum_votes"], p["singleton_policy"]
            ):
                group = representative([entries[i] for i in members], box, size)
                retain = status in ("grouped", "singleton_kept")
                if retain:
                    items.append(group)
                totals[status] += 1
                records.append(
                    {
                        "id": group["id"],
                        "class_id": cid,
                        "source_ids": group["source_ids"],
                        "vote_count": group["vote_count"],
                        "bbox": group["geometry"]["bbox"],
                        "native_bbox": box,
                        "status": status,
                    }
                )
                if retain and image is not NO_VALUE and self.generate_graphics:
                    outline = group["geometry"]["points"]
                    color = (60, 230, 120) if status == "grouped" else (40, 195, 240)
                    graphics.extend(
                        (
                            graphic_object("polyline", outline, color, closed=True),
                            graphic_object(
                                "label",
                                [outline[0]],
                                color,
                                text=f"{group['label']}: {group['vote_count']} votes",
                            ),
                        )
                    )
        items.sort(key=lambda item: (item["class_id"], id_order(item["id"])))
        records.sort(key=lambda item: (item["class_id"], id_order(item["id"])))
        count = len(items)
        passed = count >= p["minimum_count"] and (
            not p["maximum_count"] or count <= p["maximum_count"]
        )
        output = {
            "model": "vision_collection_v1",
            "kind": "boxes",
            "count": count,
            "items": items,
            "image_size": size.copy(),
            "origin": [0, 0],
            "context": {
                key: copy.deepcopy(context[key]) for key in ("spatial", "pyramid") if key in context
            },
        }
        return Result(
            {
                "image": image,
                "collection": output,
                "count": count,
                "report": {
                    "model": "detection_votes_v1",
                    "input_count": len(data["items"]),
                    "retained": count,
                    "group_totals": dict(totals),
                    "epsilon": p["epsilon"],
                    "minimum_votes": p["minimum_votes"],
                    "rounding": p["rounding"],
                    "singleton_policy": p["singleton_policy"],
                    "context_checks": checked,
                    "groups": records,
                },
            },
            payloads={port.name: copy.deepcopy(context) for port in self.outputs},
            passed=passed,
            fail_reasons=[]
            if passed
            else [
                {
                    "code": "detection_votes.count",
                    "message": "Grouped count outside configured limits",
                }
            ],
            message=f"{count} retained boxes from {len(data['items'])} spatial votes",
            drawings={"Spatial votes": GraphicDrawing(tuple(graphics))} if graphics else {},
        )
