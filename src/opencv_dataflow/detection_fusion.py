"""Bounded cross-tile detection mapping and class-aware native suppression."""

import copy

import cv2
import numpy as np

from .core import Block, Port, Result
from .detection_decode import DecodeDetections
from .graphics_layers import GraphicDrawing, graphic_object
from .objects import collection, validate_collection
from .outcomes import failed_input
from .parameter_validation import NumericRange, ValidationError
from .payloads import FlowGroup, ValuePacket, check_spatial, require_same_frame, validate_payload
from .processing import image_value
from .recipes import unique_ref
from .selection import SELECTION_INPUTS, contains, selection_mask
from .tickets import ticket_key


def spatial_context(payload, size):
    validate_payload(payload)
    check_spatial(payload, size)
    spatial = payload.get("spatial", {})
    if not spatial.get("root_id"):
        raise ValidationError("Tile fusion requires a known shared spatial root")
    return spatial


def edge_mapping(source, reference):
    if source["root_id"] != reference["root_id"] or source["root_size"] != reference["root_size"]:
        raise ValidationError("Tile detections belong to another image root")
    matrix = np.linalg.solve(
        np.asarray(reference["to_root"], float), np.asarray(source["to_root"], float)
    )
    if not np.isfinite(matrix).all() or not np.allclose(matrix[2], [0, 0, 1], rtol=0, atol=1e-9):
        raise ValidationError("Tile fusion requires an affine relative mapping")
    axes = np.abs(matrix[:2, :2]) > 1e-9
    if not (np.all(axes.sum(axis=0) == 1) and np.all(axes.sum(axis=1) == 1)):
        raise ValidationError(
            "Tile fusion supports crop/scale/flip/right-angle maps, not shear or arbitrary rotation"
        )
    matrix[:2, :2][~axes] = 0
    # Spatial metadata maps pixel centers. Detector boxes use continuous pixel edges.
    edges = matrix.copy()
    edges[:2, 2] += 0.5 - matrix[:2, :2] @ np.array([0.5, 0.5])
    if np.any(np.abs(edges) > 1e7):
        raise ValidationError("Tile mapping exceeds the coordinate budget")
    return edges


def rectangle(value, name, *, xyxy=False):
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or any(type(v) not in (int, float) for v in value)
    ):
        raise ValidationError(f"{name} needs four numeric pixel-edge coordinates")
    box = np.asarray(value, float)
    if not xyxy:
        box[2:] += box[:2]
    if not np.isfinite(box).all() or np.any(np.abs(box) > 1e7) or np.any(box[2:] <= box[:2]):
        raise ValidationError(f"{name} requires a finite, bounded positive rectangle")
    return box


def mapped_box(box, matrix):
    x, y, right, bottom = box
    corners = np.array([[x, y, 1], [right, y, 1], [right, bottom, 1], [x, bottom, 1]]) @ matrix.T
    return np.r_[corners[:, :2].min(axis=0), corners[:, :2].max(axis=0)]


def touches_internal_edge(box, size, matrix, reference_size, margin):
    w, h = size
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]) @ matrix.T
    touching = (box[1] <= margin, box[2] >= w - margin, box[3] >= h - margin, box[0] <= margin)
    for index, touch in enumerate(touching):
        if not touch:
            continue
        a, b = corners[index], corners[(index + 1) % 4]
        axis = 0 if abs(a[0] - b[0]) < 1e-7 else 1
        if 1e-7 < a[axis] < reference_size[axis] - 1e-7:
            return True
    return False


class FuseTileDetections(Block):
    type_id, title = "ml.fuse_tile_detections", "Fuse tile detections"
    category = block_type = "ML"
    subtype = "Object detection"
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = (*SELECTION_INPUTS, Port("detections", "any", visual_type="collection"))
    outputs = (
        Port("image"),
        Port("collection", "data", visual_type="collection"),
        Port("count", "integer"),
        Port("report", "data"),
    )
    defaults = {
        **{
            key: DecodeDetections.defaults[key]
            for key in (
                "confidence_threshold",
                "accepted_class_ids",
                "nms_enabled",
                "class_agnostic_nms",
                "nms_eta",
                "nms_top_k",
                "border_policy",
                "minimum_count",
                "maximum_count",
                "maximum_results",
                "maximum_image_pixels",
            )
        },
        "expected_tiles": 0,
        "box_source": "bbox",
        "nms_iou": 0.5,
        "edge_policy": "keep",
        "edge_margin": 0.0,
        "include_candidate_details": False,
        "maximum_raw_detections": 10_000,
        "maximum_candidates": 4096,
    }
    parameter_choices = {
        "box_source": ("bbox", "unclipped"),
        "edge_policy": ("keep", "drop_internal"),
        "border_policy": ("clip", "discard", "keep"),
    }
    parameter_ranges = {
        **{
            key: DecodeDetections.parameter_ranges[key]
            for key in (
                "confidence_threshold",
                "nms_iou",
                "nms_eta",
                "nms_top_k",
                "minimum_count",
                "maximum_count",
                "maximum_results",
                "maximum_image_pixels",
            )
        },
        "expected_tiles": NumericRange(0, 64, whole=True),
        "edge_margin": NumericRange(0, 16384),
        "maximum_raw_detections": NumericRange(1, 50_000, whole=True),
        "maximum_candidates": NumericRange(1, 10_000, whole=True),
    }
    parameter_help = {
        **{
            key: DecodeDetections.parameter_help[key]
            for key in (
                "confidence_threshold",
                "accepted_class_ids",
                "nms_enabled",
                "class_agnostic_nms",
                "nms_iou",
                "nms_eta",
                "nms_top_k",
                "border_policy",
                "minimum_count",
                "maximum_count",
                "maximum_results",
                "maximum_image_pixels",
            )
        },
        "confidence_threshold": (
            "Scores and threshold are compared as float32, strictly greater than, matching "
            "native NMS. Original tile scores remain in source_detection provenance."
        ),
        "expected_tiles": (
            "Zero accepts any complete supplied group. Positive values require this many "
            "distinct tile labels; this does not prove full image coverage."
        ),
        "box_source": (
            "Use each local bbox, or explicitly require geometry.unclipped_xyxy from every "
            "detection. No partial-object reconstruction or weighted averaging."
        ),
        "edge_policy": (
            "Keep all boxes, or discard those touching internal tile edges before NMS. "
            "Reference outer edges are exempt; sufficient tile overlap remains necessary."
        ),
        "edge_margin": (
            "Local tile pixels around an internal edge. Only used by drop_internal; must "
            "leave a nonempty tile interior."
        ),
        "include_candidate_details": (
            "Include a bounded per-detection mapping/filter/NMS trace. Off avoids optional "
            "report overhead."
        ),
        "maximum_raw_detections": (
            "Reject excess total detections across tiles before mapping and suppression."
        ),
        "maximum_candidates": (
            "Reject excess score/class/region/border-selected candidates before quadratic "
            "native NMS; never silently truncate."
        ),
    }
    input_help = {
        "image": (
            "Reference image owning the output grid/ticket; known spatial root and "
            "acquisition ticket are required."
        ),
        "detections": (
            "Flat FlowGroup of tile vision_collection_v1 boxes, e.g. Join parallel "
            "branches. Each packet must retain its tile spatial frame and the reference "
            "ticket/recipe."
        ),
        "mask": (
            "Optional reference-grid center selection, intersected with region; no image cropping."
        ),
        "region": "Optional reference-grid selection using each unclipped mapped box center.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for key, limits in self.parameter_ranges.items():
            if type(p[key]) not in (int, float) or (limits.whole and type(p[key]) is not int):
                raise ValidationError("Use typed numeric fusion settings", key)
        for key in ("nms_enabled", "class_agnostic_nms", "include_candidate_details"):
            if type(p[key]) is not bool:
                raise ValidationError("Use a checkbox boolean", key)
        classes = p["accepted_class_ids"]
        if (
            not isinstance(classes, list)
            or len(classes) > 4096
            or any(type(v) is not int or not 0 <= v <= 1_000_000 for v in classes)
            or len(set(classes)) != len(classes)
        ):
            raise ValidationError(
                "Class filters need distinct nonnegative integer IDs", "accepted_class_ids"
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
                "Reference image exceeds the fusion pixel/axis budget", "maximum_image_pixels"
            )
        context = self.input_payloads.get("image", {})
        reference = spatial_context(context, [w, h])
        if "ticket" not in context:
            raise ValidationError(
                "Tile fusion requires an acquisition ticket on the reference image"
            )
        identity = ticket_key(context["ticket"])
        group = inputs["detections"]
        if not isinstance(group, FlowGroup):
            raise ValidationError("Connect a flat FlowGroup of tile detection collections")
        if (
            not 1 <= len(group.items) <= 64
            or len(group.labels) != len(group.items)
            or any(not isinstance(packet, ValuePacket) for packet in group.items)
        ):
            raise ValidationError("Tile groups require 1..64 labeled value packets")
        if any(
            not isinstance(label, str) or not label or len(label) > 256 for label in group.labels
        ) or len(set(group.labels)) != len(group.labels):
            raise ValidationError("Tile labels must be distinct nonempty strings")
        if p["expected_tiles"] and len(group.items) != p["expected_tiles"]:
            raise ValidationError("Detection group has an unexpected tile count", "expected_tiles")
        aggregate = self.input_payloads.get("detections", {})
        if "ticket" in aggregate and ticket_key(aggregate["ticket"]) != identity:
            raise ValidationError("Detection group belongs to another acquisition ticket")
        recipe = unique_ref(
            [ValuePacket(None, context), ValuePacket(None, aggregate), *group.items]
        )
        selected = selection_mask(inputs, image)
        threshold = float(np.float32(p["confidence_threshold"]))
        tiles, records, candidates, labels = [], [], [], {}
        raw_count = 0
        for tile_index, (label, packet) in enumerate(zip(group.labels, group.items, strict=True)):
            data = packet.value
            if not isinstance(data, dict) or not isinstance(data.get("items"), list):
                raise ValidationError("Each tile must supply a detection collection")
            raw_count += len(data["items"])
            if raw_count > p["maximum_raw_detections"]:
                raise ValidationError(
                    "Tile detections exceed the raw detection budget", "maximum_raw_detections"
                )
            validate_collection(data)
            if data["kind"] != "boxes" or data.get("origin", [0, 0]) != [0, 0]:
                raise ValidationError(
                    "Tile detections require boxes with zero local collection origin"
                )
            if "spatial" in data.get("context", {}) and "spatial" in packet.payload:
                require_same_frame(
                    [ValuePacket(None, data["context"]), packet],
                    message="Collection and wire tile frames disagree",
                )
                if data["context"]["spatial"].get("root_size") != packet.payload["spatial"].get(
                    "root_size"
                ):
                    raise ValidationError("Collection and wire tile root dimensions disagree")
            payload = {**data.get("context", {}), **packet.payload}
            if (
                "pyramid" in data.get("context", {})
                and "pyramid" in packet.payload
                and data["context"]["pyramid"] != packet.payload["pyramid"]
            ):
                raise ValidationError("Collection and wire pyramid contexts disagree")
            recipe = unique_ref(
                [
                    ValuePacket(None, {"recipe_ref": recipe} if recipe else {}),
                    ValuePacket(None, payload),
                ]
            )
            tile_frame = spatial_context(payload, data["image_size"])
            if "ticket" not in payload or ticket_key(payload["ticket"]) != identity:
                raise ValidationError("Tile detections belong to another acquisition ticket")
            matrix = edge_mapping(tile_frame, reference)
            size = data["image_size"]
            if p["edge_policy"] == "drop_internal" and 2 * p["edge_margin"] >= min(size):
                raise ValidationError(
                    "Edge margin must leave a nonempty tile interior", "edge_margin"
                )
            failed = failed_input(packet)
            tiles.append(
                {
                    "index": tile_index,
                    "label": label,
                    "image_size": size,
                    "to_reference_edges": matrix.tolist(),
                    "failed": failed,
                    "source_spatial": copy.deepcopy(tile_frame),
                    "source_pyramid": copy.deepcopy(payload.get("pyramid")),
                }
            )
            for item in data["items"]:
                class_id, score, name = item.get("class_id"), item.get("score"), item.get("label")
                if (
                    type(class_id) is not int
                    or not 0 <= class_id <= 1_000_000
                    or type(score) not in (int, float)
                    or not np.isfinite(score)
                    or not 0 <= score <= 1
                    or not isinstance(name, str)
                    or not name
                    or len(name) > 256
                    or item.get("orientation_measured", False)
                ):
                    raise ValidationError(
                        "Tile boxes need nonnegative class IDs, labels, probability scores "
                        "and unmeasured orientation"
                    )
                score = float(np.float32(score))
                if class_id in labels and labels[class_id] != name:
                    raise ValidationError("Tile class IDs have inconsistent label mappings")
                labels[class_id] = name
                local = rectangle(item["geometry"].get("bbox"), "Local bbox")
                geometry = (
                    local
                    if p["box_source"] == "bbox"
                    else rectangle(
                        item["geometry"].get("unclipped_xyxy"), "Local unclipped box", xyxy=True
                    )
                )
                box = mapped_box(geometry, matrix)
                if not np.isfinite(box).all() or np.any(np.abs(box) > 1e7):
                    raise ValidationError("Mapped detections exceed the coordinate budget")
                clipped = box.clip([0, 0, 0, 0], [w, h, w, h])
                reason = "candidate"
                if score <= threshold:
                    reason = "confidence"
                elif p["accepted_class_ids"] and class_id not in p["accepted_class_ids"]:
                    reason = "class_filter"
                elif p["edge_policy"] == "drop_internal" and touches_internal_edge(
                    local, size, matrix, [w, h], p["edge_margin"]
                ):
                    reason = "internal_edge"
                elif np.any(clipped[2:] <= clipped[:2]):
                    reason = "outside_reference"
                elif p["border_policy"] == "discard" and np.any(box != clipped):
                    reason = "reference_border"
                elif not contains(selected, np.floor((box[:2] + box[2:]) / 2)):
                    reason = "selection"
                record = {
                    "index": len(records),
                    "tile_index": tile_index,
                    "detection_id": item["id"],
                    "score": float(score),
                    "class_id": class_id,
                    "label": name,
                    "status": reason,
                    "mapped_xyxy": box.tolist(),
                }
                records.append(record)
                if reason == "candidate":
                    candidates.append((record, box, clipped, item))
        if len(candidates) > p["maximum_candidates"]:
            raise ValidationError(
                "Selected detections exceed the NMS candidate budget", "maximum_candidates"
            )
        candidates.sort(
            key=lambda value: (-value[0]["score"], value[0]["tile_index"], value[0]["index"])
        )
        retained = []
        if p["nms_enabled"]:
            groups = {}
            for i, (record, *_rest) in enumerate(candidates):
                key = 0 if p["class_agnostic_nms"] else record["class_id"]
                groups.setdefault(key, []).append(i)
            for indices in groups.values():
                boxes = [
                    np.r_[
                        candidates[i][1][:2], candidates[i][1][2:] - candidates[i][1][:2]
                    ].tolist()
                    for i in indices
                ]
                keep = cv2.dnn.NMSBoxes(
                    boxes,
                    [candidates[i][0]["score"] for i in indices],
                    threshold,
                    p["nms_iou"],
                    eta=p["nms_eta"],
                    top_k=p["nms_top_k"],
                )
                retained.extend(indices[int(i)] for i in np.asarray(keep).reshape(-1))
            retained.sort()
        else:
            retained = list(range(len(candidates)))
        if len(retained) > p["maximum_results"]:
            raise ValidationError("Fused detections exceed the result budget", "maximum_results")
        for record, *_rest in candidates:
            record["status"] = "nms_or_top_k"
        items, drawings = [], []
        for i in retained:
            record, box, clipped, original = candidates[i]
            record["status"] = "retained"
            bounds = clipped if p["border_policy"] == "clip" else box
            x, y, right, bottom = map(float, bounds)
            width, height = right - x, bottom - y
            points = [[x, y], [right, y], [right, bottom], [x, bottom]]
            items.append(
                {
                    "id": len(items),
                    "class_id": record["class_id"],
                    "label": record["label"],
                    "score": record["score"],
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
                        "score": record["score"],
                        "class_id": record["class_id"],
                        "angle": None,
                    },
                    "geometry": {
                        "bbox": [x, y, width, height],
                        "points": points,
                        "unclipped_xyxy": box.tolist(),
                        "crosses_image_border": bool(np.any(box != clipped)),
                        "clipped": p["border_policy"] == "clip" and bool(np.any(box != clipped)),
                    },
                    "provenance": {
                        "tile_index": record["tile_index"],
                        "tile_label": tiles[record["tile_index"]]["label"],
                        "box_source": p["box_source"],
                        "source_detection": copy.deepcopy(original),
                    },
                }
            )
            if self.generate_graphics:
                drawings.extend(
                    (
                        graphic_object("polyline", points, (60, 230, 120), closed=True),
                        graphic_object(
                            "label",
                            [[x, y]],
                            (60, 230, 120),
                            text=f"{record['label']}: {record['score']:.3f}",
                        ),
                    )
                )
        count = len(items)
        reasons = []
        if count < p["minimum_count"] or (p["maximum_count"] and count > p["maximum_count"]):
            reasons.append(
                {
                    "code": "tile_detections.count",
                    "message": (
                        f"Retained {count}; required {p['minimum_count']}.."
                        f"{p['maximum_count'] or 'unbounded'}"
                    ),
                }
            )
        if any(tile["failed"] for tile in tiles):
            reasons.append(
                {
                    "code": "tile_detections.failed_tile",
                    "message": "A supplied tile carries a failed inspection outcome",
                }
            )
        report = {
            "model": "tile_detection_fusion_v1",
            "tiles": tiles,
            "raw_detections": raw_count,
            "selected_candidates": len(candidates),
            "retained": count,
            "nms_enabled": p["nms_enabled"],
            "class_aware_nms": not p["class_agnostic_nms"],
            "box_source": p["box_source"],
            "edge_policy": p["edge_policy"],
            "expected_tile_count_checked": bool(p["expected_tiles"]),
            "coordinate_convention": "continuous_reference_pixel_edges",
            "partial_object_reconstruction": False,
            "filtered_counts": {
                status: sum(r["status"] == status for r in records)
                for status in (
                    "confidence",
                    "class_filter",
                    "internal_edge",
                    "outside_reference",
                    "reference_border",
                    "selection",
                    "nms_or_top_k",
                )
            },
        }
        if p["include_candidate_details"]:
            report["candidates"] = records
        context = copy.deepcopy(context)
        if recipe is not None:
            context["recipe_ref"] = copy.deepcopy(recipe)
        return Result(
            {
                "image": inputs["image"],
                "collection": collection("boxes", items, image, context),
                "count": count,
                "report": report,
            },
            payloads={
                key: copy.deepcopy(context) for key in ("image", "collection", "count", "report")
            },
            passed=not reasons,
            score=max((item["score"] for item in items), default=0.0),
            fail_reasons=reasons,
            message=f"{count} detections from {len(tiles)} tiles after mapping/NMS",
            drawings={"Fused tile detections": GraphicDrawing(tuple(drawings))} if drawings else {},
        )
