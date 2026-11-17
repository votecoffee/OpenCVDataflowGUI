"""Explicit single-image detector tensor adapters, source geometry and native NMS."""

import copy

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .ml_common import selected_tensor
from .objects import collection
from .parameter_validation import NumericRange, ValidationError
from .payloads import ValuePacket, require_same_frame
from .processing import image_value
from .selection import SELECTION_INPUTS, contains, selection_mask
from .tickets import ticket_key


def preprocessing_geometry(value, image):
    if not isinstance(value, dict):
        raise ValidationError("Connect the inference preprocessing record")
    size = value.get("network_size")
    if (
        value.get("input_size") != list(image.shape[1::-1])
        or not isinstance(size, list)
        or len(size) != 2
        or any(type(n) is not int or not 1 <= n <= 4096 for n in size)
    ):
        raise ValidationError("Preprocessing sizes do not match the source image/network")
    try:
        forward = np.asarray(value.get("input_to_network"), float)
        reverse = np.asarray(value.get("network_to_input"), float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("Preprocessing maps must be finite numeric 3x3 matrices") from exc
    if (
        forward.shape != (3, 3)
        or reverse.shape != (3, 3)
        or not np.isfinite(forward).all()
        or not np.isfinite(reverse).all()
        or np.any(np.abs(forward) > 1e7)
        or np.any(np.abs(reverse) > 1e7)
        or not np.allclose(forward @ reverse, np.eye(3), atol=1e-7, rtol=0)
        or forward[0, 0] <= 0
        or forward[1, 1] <= 0
        or not np.allclose(
            forward[[0, 1, 2, 2, 2], [1, 0, 0, 1, 2]], [0, 0, 0, 0, 1], atol=1e-9, rtol=0
        )
    ):
        raise ValidationError("Preprocessing must supply inverse positive axis-scale/offset maps")
    return size, reverse


def check_context(block, *, kind="Detection"):
    contexts = [
        block.input_payloads.get(name, {}) for name in ("image", "tensors", "preprocessing")
    ]
    require_same_frame(
        [ValuePacket(None, p) for p in contexts],
        message=f"{kind} tensors/preprocessing use another image frame",
    )
    tickets = [p["ticket"] for p in contexts if "ticket" in p]
    if tickets and any(ticket_key(t) != ticket_key(tickets[0]) for t in tickets[1:]):
        raise ValidationError(f"{kind} inputs belong to different acquisition tickets")
    recipes = [p["recipe_ref"] for p in contexts if "recipe_ref" in p]
    if recipes and any(r != recipes[0] for r in recipes[1:]):
        raise ValidationError(f"{kind} inputs belong to different recipe revisions")
    return {
        "ticket_checked": len(tickets) == 3,
        "frame_checked": all(p.get("spatial", {}).get("root_id") is not None for p in contexts),
    }


def probability(values, mode):
    if mode == "sigmoid":
        z = np.exp(-np.abs(values))
        return np.where(values >= 0, 1 / (1 + z), z / (1 + z))
    if np.any(values < 0) or np.any(values > 1):
        raise ValidationError("Probability scores must be in [0,1]; choose sigmoid for logits")
    return values


def ssd_head_rows(tensors, p):
    """Normalize explicitly named single-image SSD heads into DetectionOutput rows."""
    heads = {}
    for key in (
        "boxes_output_name",
        "classes_output_name",
        "scores_output_name",
        "count_output_name",
    ):
        name = p[key]
        if not name:
            continue
        record = tensors.get(name) if isinstance(tensors, dict) else None
        if not isinstance(record, dict) or record.get("dtype") != "float32":
            raise ValidationError("Select existing dequantized float32 SSD heads", key)
        try:
            heads[key] = selected_tensor(tensors, name, allow_empty=True).astype(np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError(str(exc), key) from exc
    boxes = heads["boxes_output_name"]
    if boxes.ndim == 3 and boxes.shape[0] == 1:
        boxes = boxes[0]
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValidationError("SSD boxes need [N,4] or [1,N,4]", "boxes_output_name")
    length = len(boxes)
    if length > p["maximum_raw_rows"]:
        raise ValidationError("SSD heads exceed the raw-row budget", "maximum_raw_rows")
    vectors = []
    for key in ("classes_output_name", "scores_output_name"):
        values = heads[key]
        if values.ndim == 2 and values.shape[0] == 1:
            values = values[0]
        if values.shape != (length,):
            raise ValidationError("SSD classes/scores need matching [N] or [1,N]", key)
        vectors.append(values)
    count = length
    if p["count_output_name"]:
        values = heads["count_output_name"]
        if (
            values.shape not in ((1,), (1, 1))
            or values.item() != np.floor(values.item())
            or not 0 <= values.item() <= length
        ):
            raise ValidationError("SSD count needs one integer in 0..N", "count_output_name")
        count = int(values.item())
    boxes = boxes[:count]
    if p["box_coordinate_order"] == "yxyx":
        boxes = boxes[:, [1, 0, 3, 2]]
    return np.column_stack((np.zeros(count), vectors[0][:count], vectors[1][:count], boxes))


def tensor_rows(inputs, p):
    if p["format"] == "ssd_heads":
        return ssd_head_rows(inputs["tensors"], p)
    try:
        tensors = inputs["tensors"]
        chosen = tensors.get(p["output_name"]) if p["output_name"] else next(iter(tensors.values()))
        if not isinstance(chosen, dict) or chosen.get("dtype") != "float32":
            raise ValidationError("Use dequantized float32 tensor_v1 detector outputs")
        values = selected_tensor(inputs["tensors"], p["output_name"], allow_empty=True).astype(
            np.float64
        )
    except (AttributeError, StopIteration, TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(str(exc)) from exc
    if p["format"] == "ssd":
        if values.ndim > 2:
            if any(n != 1 for n in values.shape[:-2]):
                raise ValidationError("SSD decoding requires a single-image batch")
            values = values.reshape(values.shape[-2:])
        if values.ndim != 2 or values.shape[1] != 7:
            raise ValidationError("SSD rows must contain image ID, class, score and XYXY")
    else:
        if values.ndim == 3 and values.shape[0] == 1:
            values = values[0]
        if values.ndim != 2:
            raise ValidationError("Detection decoding requires [N,C], [C,N], or one leading batch")
        if p["layout"] == "channels_first":
            values = values.T
        width = values.shape[1]
        minimum = 6 if p["format"] in ("xywh_objectness", "xyxy_score_class") else 5
        if width < minimum or width > 4096 or (p["format"] == "xyxy_score_class" and width != 6):
            raise ValidationError("Detection row width does not match the explicit output format")
    if len(values) > p["maximum_raw_rows"]:
        raise ValidationError("Detector tensor exceeds the raw-row budget", "maximum_raw_rows")
    return values


def candidates(values, p, size):
    kind = p["format"]
    if kind in ("xywh_objectness", "xywh_scores"):
        start = 5 if kind == "xywh_objectness" else 4
        scores = probability(values[:, start:], p["score_activation"])
        if p["labels"] and len(p["labels"]) != scores.shape[1]:
            raise ValidationError("Provide exactly one label per score channel", "labels")
        class_ids = scores.argmax(axis=1)
        confidence = scores[np.arange(len(scores)), class_ids]
        objectness = (
            probability(values[:, 4], p["score_activation"]) if start == 5 else np.ones(len(values))
        )
        confidence = (confidence * objectness).astype(np.float32).astype(np.float64)
        valid = objectness > p["objectness_threshold"] if start == 5 else np.ones(len(values), bool)
        boxes = values[:, :4].copy()
    else:
        ssd = kind in ("ssd", "ssd_heads")
        class_ids = values[:, 1] if ssd else values[:, 5]
        raw_scores = np.where(values[:, 0] >= 0, values[:, 2], 0) if ssd else values[:, 4]
        confidence = probability(raw_scores, p["score_activation"])
        boxes = values[:, 3:7].copy() if ssd else values[:, :4].copy()
        valid = np.ones(len(values), bool)
        if kind == "ssd":
            valid &= values[:, 0] >= 0  # Negative image IDs mark unused SSD rows.
            if np.any(values[valid, 0] != 0):
                raise ValidationError("SSD contains a different image/batch ID")
    valid &= confidence > p["confidence_threshold"]
    indices = np.flatnonzero(valid)
    boxes, confidence, class_ids = boxes[indices], confidence[indices], class_ids[indices]
    if (
        np.any(class_ids != np.floor(class_ids))
        or np.any(class_ids < 0)
        or np.any(class_ids > 1_001_000)
    ):
        raise ValidationError("Detection class IDs must be nonnegative integers")
    class_ids = class_ids.astype(np.int64) + p["class_id_offset"]
    if np.any(class_ids < 0) or np.any(class_ids > 1_000_000):
        raise ValidationError("Adjusted class IDs are outside 0..1000000", "class_id_offset")
    if p["labels"] and np.any(class_ids >= len(p["labels"])):
        raise ValidationError("Class ID is outside the supplied label table", "labels")
    if p["grid_decode"] == "yolox":
        if any(size[0] % s or size[1] % s for s in p["strides"]):
            raise ValidationError("YOLOX axes must be divisible by each stride", "strides")
        expected = sum((size[0] // s) * (size[1] // s) for s in p["strides"])
        if expected != len(values):
            raise ValidationError("YOLOX row count does not match the declared network/strides")
        grids, strides = [], []
        for stride in p["strides"]:
            y, x = np.indices((size[1] // stride, size[0] // stride))
            grids.append(np.column_stack((x.ravel(), y.ravel())))
            strides.extend([stride] * x.size)
        grid, strides = np.concatenate(grids), np.asarray(strides)[:, None]
        if np.any(np.abs(boxes[:, 2:4]) > 20):
            raise ValidationError("YOLOX log-size exceeds the bounded exponential range")
        boxes[:, :2] = (boxes[:, :2] + grid[indices]) * strides[indices]
        boxes[:, 2:4] = np.exp(boxes[:, 2:4]) * strides[indices]
    if p["coordinate_units"] == "normalized":
        boxes *= np.array([*size, *size])
    if kind.startswith("xywh"):
        boxes[:, :2] -= boxes[:, 2:4] / 2
        boxes[:, 2:4] += boxes[:, :2]
    if (
        not np.isfinite(boxes).all()
        or np.any(np.abs(boxes) > 1e7)
        or np.any(boxes[:, 2:4] <= boxes[:, :2])
    ):
        raise ValidationError(
            "Scored detections need finite, bounded boxes with positive dimensions"
        )
    return boxes, confidence, class_ids, indices


class DecodeDetections(Block):
    type_id, title = "ml.decode_detections", "Decode object detections"
    category = block_type = "ML"
    subtype = "Object detection"
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = (*SELECTION_INPUTS, Port("tensors", "data"), Port("preprocessing", "data"))
    outputs = (
        Port("image"),
        Port("collection", "data", visual_type="collection"),
        Port("count", "integer"),
        Port("report", "data"),
    )
    defaults = {
        "output_name": "",
        "format": "xywh_scores",
        "boxes_output_name": "bboxes",
        "classes_output_name": "labels",
        "scores_output_name": "scores",
        "count_output_name": "",
        "box_coordinate_order": "xyxy",
        "layout": "rows",
        "coordinate_units": "network_pixels",
        "grid_decode": "none",
        "strides": [8, 16, 32],
        "score_activation": "probability",
        "labels": [],
        "accepted_class_ids": [],
        "class_id_offset": 0,
        "confidence_threshold": 0.25,
        "objectness_threshold": 0.0,
        "nms_enabled": True,
        "class_agnostic_nms": False,
        "nms_iou": 0.45,
        "nms_eta": 1.0,
        "nms_top_k": 0,
        "border_policy": "clip",
        "minimum_count": 1,
        "maximum_count": 0,
        "maximum_raw_rows": 100_000,
        "maximum_candidates": 10_000,
        "maximum_results": 1000,
        "maximum_image_pixels": 16_777_216,
    }
    parameter_choices = {
        "format": ("xywh_scores", "xywh_objectness", "xyxy_score_class", "ssd", "ssd_heads"),
        "box_coordinate_order": ("xyxy", "yxyx"),
        "layout": ("rows", "channels_first"),
        "coordinate_units": ("network_pixels", "normalized"),
        "grid_decode": ("none", "yolox"),
        "score_activation": ("probability", "sigmoid"),
        "border_policy": ("clip", "discard", "keep"),
    }
    parameter_ranges = {
        **{
            k: NumericRange(0, 1)
            for k in ("confidence_threshold", "objectness_threshold", "nms_iou")
        },
        "nms_eta": NumericRange(0.001, 1),
        "nms_top_k": NumericRange(0, 100_000, whole=True),
        "class_id_offset": NumericRange(-1000, 1000, whole=True),
        **{k: NumericRange(0, 100_000, whole=True) for k in ("minimum_count", "maximum_count")},
        "maximum_raw_rows": NumericRange(1, 100_000, whole=True),
        "maximum_candidates": NumericRange(1, 10_000, whole=True),
        "maximum_results": NumericRange(1, 10_000, whole=True),
        "maximum_image_pixels": NumericRange(1, 64_000_000, whole=True),
    }
    parameter_help = {
        "output_name": "Exact packed tensor name; empty selects the first. Ignored for ssd_heads.",
        "boxes_output_name": "SSD heads only: exact boxes tensor, [N,4] or [1,N,4].",
        "classes_output_name": "SSD heads only: exact explicit class-ID tensor, [N] or [1,N].",
        "scores_output_name": "SSD heads only: exact confidence tensor, [N] or [1,N].",
        "count_output_name": (
            "SSD heads only: optional [1] or [1,1] integer count; only the first count rows "
            "are active. Empty uses all rows. Storage shapes/row budget are checked first."
        ),
        "box_coordinate_order": "SSD heads only: XYXY or top/left/bottom/right (YXYX).",
        "format": (
            "Explicit row schema. XYWH scores suits decoded YOLOv8-style heads; objectness "
            "suits YOLOv5/YOLOX. SSD supports packed rows or explicitly named separate heads."
        ),
        "layout": (
            "Rows [N,C] or channels-first [C,N], optionally with one leading batch of size "
            "one. SSD is fixed."
        ),
        "coordinate_units": (
            "Box coordinates relative to the network input; normalized values scale by its "
            "width/height."
        ),
        "grid_decode": (
            "YOLOX expands XY offsets and exponential sizes using the exact network stride grids."
        ),
        "strides": (
            "Ascending distinct YOLOX head strides, normally [8,16,32]; all must divide "
            "network axes."
        ),
        "score_activation": (
            "Probability validates [0,1]; sigmoid explicitly converts class/objectness logits."
        ),
        "labels": (
            "Optional zero-based string table. Score-head formats require one entry per "
            "class channel."
        ),
        "accepted_class_ids": (
            "Empty keeps all classes. Otherwise keep these adjusted, zero-based IDs."
        ),
        "class_id_offset": (
            "Adjustment for explicit-ID SSD/XYXY rows, e.g. -1 for one-based classes."
        ),
        "confidence_threshold": (
            "Keep scores strictly above this value. Objectness heads multiply class and objectness."
        ),
        "objectness_threshold": (
            "Objectness-head gate before combined-score filtering; other formats ignore it."
        ),
        "nms_enabled": (
            "Disable when the model already performs suppression, or to inspect duplicate "
            "proposals."
        ),
        "class_agnostic_nms": (
            "Otherwise suppress separately per class, including boxes with negative coordinates."
        ),
        "nms_iou": (
            "Native hard-NMS overlap threshold, applied to unclipped boxes in the source grid."
        ),
        "nms_eta": "Native adaptive IoU factor in (0,1]; one keeps a fixed threshold.",
        "nms_top_k": (
            "Native pre-NMS score cap per class, or globally in class-agnostic mode; zero "
            "is unlimited."
        ),
        "border_policy": (
            "Clip partly visible boxes, discard them, or keep their unclipped coordinates. "
            "Fully outside boxes drop."
        ),
        "minimum_count": "Fail the inspection below this many retained detections.",
        "maximum_count": (
            "Zero disables the upper count judgment; positive values bound retained detections."
        ),
        "maximum_raw_rows": (
            "Reject larger tensors before row processing; tensor_v1 also caps total values."
        ),
        "maximum_candidates": (
            "Reject excess valid candidates before quadratic NMS; never silently truncate."
        ),
        "maximum_results": (
            "Reject excess retained results instead of emitting an incomplete collection."
        ),
        "maximum_image_pixels": (
            "Reject larger source images before selection; collection axes are limited to 16384."
        ),
    }
    input_help = {
        "image": "Original image whose source coordinates and ticket own the detections.",
        "tensors": "Single-image tensor_v1 outputs from the matching inference activation.",
        "preprocessing": (
            "Matching inference resize/letterbox maps. Known frame/ticket/recipe mismatches reject."
        ),
        "mask": (
            "Optional image-grid selection; only detections whose unclipped center is "
            "selected survive."
        ),
        "region": (
            "Optional region intersected with the mask. Does not crop or alter network inference."
        ),
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for name, limits in self.parameter_ranges.items():
            if type(p[name]) not in (int, float) or (limits.whole and type(p[name]) is not int):
                raise ValidationError("Use typed numeric decoder settings", name)
        if not isinstance(p["output_name"], str) or len(p["output_name"]) > 256:
            raise ValidationError("Output name must be text up to 256 characters", "output_name")
        head_keys = (
            "boxes_output_name",
            "classes_output_name",
            "scores_output_name",
            "count_output_name",
        )
        for name in head_keys:
            if not isinstance(p[name], str) or len(p[name]) > 256:
                raise ValidationError("Head name must be text up to 256 characters", name)
        if p["format"] == "ssd_heads":
            names = [p[name] for name in head_keys if p[name]]
            if any(not p[name].strip() for name in head_keys[:3]) or len(set(names)) != len(names):
                raise ValidationError("SSD head names must be nonempty and distinct", *head_keys)
        for name in ("nms_enabled", "class_agnostic_nms"):
            if type(p[name]) is not bool:
                raise ValidationError("Use a checkbox boolean", name)
        if (
            not isinstance(p["labels"], list)
            or len(p["labels"]) > 4096
            or any(not isinstance(s, str) or not s or len(s) > 256 for s in p["labels"])
        ):
            raise ValidationError("Labels must be a bounded list of nonempty strings", "labels")
        classes = p["accepted_class_ids"]
        if (
            not isinstance(classes, list)
            or len(classes) > 4096
            or any(type(n) is not int or not 0 <= n <= 1_000_000 for n in classes)
            or len(set(classes)) != len(classes)
        ):
            raise ValidationError(
                "Class filters need distinct nonnegative integer IDs", "accepted_class_ids"
            )
        strides = p["strides"]
        if (
            not isinstance(strides, list)
            or not 1 <= len(strides) <= 6
            or any(type(n) is not int or not 1 <= n <= 1024 for n in strides)
            or strides != sorted(set(strides))
        ):
            raise ValidationError(
                "Strides must be 1..6 increasing distinct integers in 1..1024", "strides"
            )
        if p["grid_decode"] == "yolox" and (
            p["format"] != "xywh_objectness" or p["coordinate_units"] != "network_pixels"
        ):
            raise ValidationError(
                "YOLOX grids require network-pixel XYWH/objectness rows",
                "grid_decode",
                "format",
                "coordinate_units",
            )
        if p["format"].startswith("xywh") and p["class_id_offset"] != 0:
            raise ValidationError(
                "Class offset only applies to explicit-ID row formats", "class_id_offset"
            )
        if p["maximum_count"] and p["minimum_count"] > p["maximum_count"]:
            raise ValidationError(
                "Minimum count exceeds maximum count", "minimum_count", "maximum_count"
            )

    def process(self, inputs):
        p = self.parameters
        source = inputs["image"]
        if (
            min(source.shape[:2]) < 1
            or max(source.shape[:2]) > 16384
            or source.shape[0] * source.shape[1] > p["maximum_image_pixels"]
        ):
            raise ValidationError(
                "Source exceeds the detection pixel/axis budget", "maximum_image_pixels"
            )
        image = image_value(source)
        context = self.input_payloads.get("image", {})
        checked = check_context(self)
        size, matrix = preprocessing_geometry(inputs["preprocessing"], image)
        rows = tensor_rows(inputs, p)
        boxes, scores, classes, source_indices = candidates(rows, p, size)
        # Positive independent XY scales preserve rectangles and IoU. Keep float edges.
        network_boxes = boxes.copy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * matrix[0, 0] + matrix[0, 2]
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * matrix[1, 1] + matrix[1, 2]
        if not np.isfinite(boxes).all() or np.any(np.abs(boxes) > 1e7):
            raise ValidationError("Mapped boxes exceed the finite source-coordinate budget")
        h, w = image.shape[:2]
        clipped = boxes.clip([0, 0, 0, 0], [w, h, w, h])
        visible = np.all(clipped[:, 2:] > clipped[:, :2], axis=1)
        selected = selection_mask(inputs, image)
        valid = visible.copy()
        if p["border_policy"] == "discard":
            valid &= np.all(boxes == clipped, axis=1)
        if p["accepted_class_ids"]:
            valid &= np.isin(classes, p["accepted_class_ids"])
        for i in np.flatnonzero(valid):
            valid[i] = contains(selected, np.floor((boxes[i, :2] + boxes[i, 2:]) / 2))
        indices = np.flatnonzero(valid).tolist()
        if len(indices) > p["maximum_candidates"]:
            raise ValidationError(
                "Scored/selected boxes exceed the NMS candidate budget", "maximum_candidates"
            )
        indices.sort(key=lambda i: (-float(scores[i]), int(source_indices[i])))
        retained = []
        if p["nms_enabled"]:
            grouped = {}
            for i in indices:
                grouped.setdefault(int(classes[i]), []).append(i)
            groups = [indices] if p["class_agnostic_nms"] else list(grouped.values())
            for group in groups:
                if not group:
                    continue
                xywh = boxes[group].copy()
                xywh[:, 2:] -= xywh[:, :2]
                keep = cv2.dnn.NMSBoxes(
                    xywh.tolist(),
                    scores[group].tolist(),
                    p["confidence_threshold"],
                    p["nms_iou"],
                    eta=p["nms_eta"],
                    top_k=p["nms_top_k"],
                )
                retained.extend(group[int(i)] for i in np.asarray(keep).reshape(-1))
            retained.sort(key=lambda i: (-float(scores[i]), int(source_indices[i])))
        else:
            retained = indices
        if len(retained) > p["maximum_results"]:
            raise ValidationError("Retained detections exceed the result budget", "maximum_results")
        items, graphics = [], []
        for i in retained:
            bounds = clipped[i] if p["border_policy"] == "clip" else boxes[i]
            x, y, right, bottom = map(float, bounds)
            width, height = right - x, bottom - y
            class_id, score = int(classes[i]), float(scores[i])
            label = p["labels"][class_id] if p["labels"] else str(class_id)
            corners = [[x, y], [right, y], [right, bottom], [x, bottom]]
            items.append(
                {
                    "id": int(source_indices[i]),
                    "class_id": class_id,
                    "label": label,
                    "score": score,
                    "orientation_measured": False,
                    "metrics": {
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height,
                        "area": width * height,
                        "center_x": (x + right) / 2,
                        "center_y": (y + bottom) / 2,
                        "aspect_ratio": width / height,
                        "score": score,
                        "class_id": class_id,
                        "angle": None,
                    },
                    "geometry": {
                        "points": corners,
                        "bbox": [x, y, width, height],
                        "unclipped_xyxy": boxes[i].tolist(),
                        "network_xyxy": network_boxes[i].tolist(),
                        "crosses_image_border": bool(np.any(boxes[i] != clipped[i])),
                        "clipped": p["border_policy"] == "clip"
                        and bool(np.any(boxes[i] != clipped[i])),
                    },
                }
            )
            if self.generate_graphics:
                graphics.extend(
                    (
                        graphic_object("polyline", corners, (60, 230, 120), closed=True),
                        graphic_object(
                            "label", [[x, y]], (60, 230, 120), text=f"{label}: {score:.3f}"
                        ),
                    )
                )
        count = len(items)
        passed = count >= p["minimum_count"] and (
            not p["maximum_count"] or count <= p["maximum_count"]
        )
        report = {
            "model": "object_detection_decode_v1",
            "format": p["format"],
            "ssd_heads": {
                key: p[key]
                for key in (
                    "boxes_output_name",
                    "classes_output_name",
                    "scores_output_name",
                    "count_output_name",
                    "box_coordinate_order",
                )
            }
            if p["format"] == "ssd_heads"
            else None,
            "layout": p["layout"],
            "grid_decode": p["grid_decode"],
            "raw_rows": len(rows),
            "scored_rows": len(scores),
            "selected_candidates": len(indices),
            "retained": count,
            "nms_enabled": p["nms_enabled"],
            "nms_class_aware": not p["class_agnostic_nms"],
            "context_checks": checked,
            "coordinate_convention": "continuous_source_pixel_edges",
            "preprocessing": copy.deepcopy(inputs["preprocessing"]),
        }
        return Result(
            {
                "image": inputs["image"],
                "collection": collection("boxes", items, image, context),
                "count": count,
                "report": report,
            },
            payloads={
                name: copy.deepcopy(context) for name in ("image", "collection", "count", "report")
            },
            passed=passed,
            score=max((item["score"] for item in items), default=0.0),
            message=f"{count} detections after selection/NMS",
            fail_reasons=[]
            if passed
            else [
                {
                    "code": "detection.count",
                    "message": (
                        f"Retained {count}; required minimum {p['minimum_count']}, "
                        f"maximum {p['maximum_count'] or 'unbounded'}"
                    ),
                }
            ],
            drawings={"Object detections": GraphicDrawing(tuple(graphics))} if graphics else {},
        )
