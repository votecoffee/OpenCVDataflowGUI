"""Explicit oriented detector heads, affine source footprints and rotated suppression."""

import copy
import math

import cv2
import numpy as np

from .core import Block, Port, Result
from .detection_decode import (
    DecodeDetections,
    candidates,
    check_context,
    preprocessing_geometry,
    tensor_rows,
)
from .graphics_layers import GraphicDrawing, graphic_object
from .objects import collection
from .parameter_validation import NumericRange, ValidationError
from .poses import make_pose
from .processing import image_value
from .selection import contains, selection_mask


def polygon_area(points):
    if len(points) < 3:
        return 0.0
    local = np.asarray(points, np.float64) - points[0]
    return (
        abs(
            float(
                np.sum(
                    local[:, 0] * np.roll(local[:, 1], -1) - local[:, 1] * np.roll(local[:, 0], -1)
                )
            )
        )
        / 2
    )


def clip_image(points, width, height):
    result = list(np.asarray(points, np.float64))
    for axis, boundary, sign in ((0, 0, 1), (0, width, -1), (1, 0, 1), (1, height, -1)):
        incoming, result = result, []
        if not incoming:
            break
        previous = incoming[-1]
        before = sign * (previous[axis] - boundary) >= 0
        for current in incoming:
            inside = sign * (current[axis] - boundary) >= 0
            if inside != before:
                t = (boundary - previous[axis]) / (current[axis] - previous[axis])
                crossed = previous + t * (current - previous)
                crossed[axis] = boundary
                result.append(crossed)
            if inside:
                result.append(current)
            previous, before = current, inside
    return np.asarray(result, np.float64).reshape(-1, 2)


def rotated_iou(a, b):
    area_a, area_b = a[1][0] * a[1][1], b[1][0] * b[1][1]
    flag, points = cv2.rotatedRectangleIntersection(a, b)
    if flag == cv2.INTERSECT_NONE:
        return 0.0
    if flag == cv2.INTERSECT_FULL:
        # This build's native NMS treats full containment as IoU=1. Correctly
        # distinguish a small inner detection from an identical rectangle here.
        intersection = min(area_a, area_b)
    elif points is None:
        raise ValidationError("Native rotated intersection returned no partial boundary")
    else:
        intersection = float(cv2.contourArea(cv2.convexHull(points)))
    if not math.isfinite(intersection) or intersection < 0:
        raise ValidationError("Native rotated intersection returned an invalid area")
    intersection = min(intersection, area_a, area_b)
    return intersection / (area_a + area_b - intersection)


def suppress(group, rectangles, scores, p, bounds):
    if not group:
        return []
    if p["nms_method"] == "opencv":
        kept = cv2.dnn.NMSBoxesRotated(
            [rectangles[i] for i in group],
            scores[group].tolist(),
            p["confidence_threshold"],
            p["nms_iou"],
            eta=p["nms_eta"],
            top_k=p["nms_top_k"],
        )
        return [group[int(i)] for i in np.asarray(kept).reshape(-1)]
    retained, threshold = [], p["nms_iou"]
    for index in group[: p["nms_top_k"] or None]:
        box = bounds[index]
        if all(
            box[0] >= bounds[old][2]
            or bounds[old][0] >= box[2]
            or box[1] >= bounds[old][3]
            or bounds[old][1] >= box[3]
            or rotated_iou(rectangles[index], rectangles[old]) <= threshold
            for old in retained
        ):
            retained.append(index)
            if p["nms_eta"] < 1 and threshold > 0.5:
                threshold *= p["nms_eta"]
    return retained


def oriented_candidates(rows, p, size):
    explicit = p["format"] == "xywha_score_class"
    minimum = 7 if explicit or p["format"] == "xywha_objectness" else 6
    if rows.shape[1] < minimum or explicit and rows.shape[1] != 7:
        raise ValidationError("Rotated row width disagrees with the selected format")
    angle_index = rows.shape[1] - 1 if p["format"] == "xywh_scores_angle" else 4
    angles = rows[:, angle_index]
    values = np.delete(rows, angle_index, axis=1)
    if explicit:
        values[:, :2] -= values[:, 2:4] / 2
        values[:, 2:4] += values[:, :2]
    adapter = {
        **p,
        "grid_decode": "none",
        "format": "xyxy_score_class"
        if explicit
        else "xywh_objectness"
        if p["format"] == "xywha_objectness"
        else "xywh_scores",
    }
    boxes, scores, classes, source_indices = candidates(values, adapter, size)
    theta = angles[source_indices] * (180 / np.pi if p["angle_unit"] == "radians" else 1)
    if np.any(np.abs(theta) > 36000):
        raise ValidationError("Scored angles must remain within 100 turns")
    if p["angle_direction"] == "counterclockwise":
        theta = -theta
    theta %= int(p["angle_period"])
    xywh = np.column_stack(((boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]))
    return xywh, theta, scores, classes, source_indices


class DecodeRotatedDetections(DecodeDetections):
    type_id, title = "ml.decode_rotated_detections", "Decode rotated detections"
    preview_safe = parallel_safe = process_inputs_readonly = True
    outputs = (*DecodeDetections.outputs, Port("matches", "data"))
    defaults = {
        **{
            k: v
            for k, v in DecodeDetections.defaults.items()
            if k not in ("grid_decode", "strides")
        },
        "format": "xywh_scores_angle",
        "angle_unit": "degrees",
        "angle_direction": "image_clockwise",
        "angle_period": "180",
        "nms_method": "polygon_iou",
        "maximum_candidates": 2000,
        "maximum_pair_comparisons": 2_000_000,
    }
    parameter_choices = {
        **{k: v for k, v in DecodeDetections.parameter_choices.items() if k != "grid_decode"},
        "format": ("xywh_scores_angle", "xywha_scores", "xywha_objectness", "xywha_score_class"),
        "angle_unit": ("degrees", "radians"),
        "angle_direction": ("image_clockwise", "counterclockwise"),
        "angle_period": ("180", "360"),
        "nms_method": ("polygon_iou", "opencv"),
    }
    parameter_ranges = {
        **DecodeDetections.parameter_ranges,
        "maximum_pair_comparisons": NumericRange(1, 20_000_000, whole=True),
    }
    parameter_help = {
        **{
            k: v
            for k, v in DecodeDetections.parameter_help.items()
            if k not in ("grid_decode", "strides")
        },
        "format": "Explicit center XYWH, angle and scores; scores-angle places angle last.",
        "angle_unit": "Units of the model's raw angle column, before direction conversion.",
        "angle_direction": "Positive image-clockwise angles, or reverse counterclockwise angles.",
        "angle_period": "180: unoriented width axis; 360: caller asserts a directed model angle.",
        "nms_method": "Polygon IoU corrects nested-box overlap; OpenCV preserves native behavior.",
        "nms_iou": "Unclipped network-rectangle IoU; shared affine maps preserve polygon IoU.",
        "maximum_pair_comparisons": "Reject quadratic work above the sum of group n*(n-1)/2.",
        "border_policy": "Clip polygons, discard crossings or keep them; invisible boxes drop.",
        "class_id_offset": "Offset explicit-ID rows only; score-head channels use zero-based IDs.",
    }

    def validate_parameters(self, parameters=None):
        Block.validate_parameters(self, parameters)
        p = self.parameters if parameters is None else parameters
        # Reuse the existing pure shared detector constraints, without exposing
        # axis-only grid/stride controls or pretending the rotated schema is XYXY.
        shared = {
            **DecodeDetections.defaults,
            **{k: v for k, v in p.items() if k in DecodeDetections.defaults},
        }
        shared["format"] = (
            "xyxy_score_class" if p["format"] == "xywha_score_class" else "xywh_scores"
        )
        DecodeDetections().validate_parameters(shared)
        if type(p["maximum_pair_comparisons"]) is not int:
            raise ValidationError(
                "Pair comparison budget must be an integer", "maximum_pair_comparisons"
            )

    def process(self, inputs):
        p, source = self.parameters, inputs["image"]
        h, w = source.shape[:2]
        if max(h, w) > 16384 or h * w > p["maximum_image_pixels"]:
            raise ValidationError("Source exceeds rotated detection image limits")
        image = image_value(source)
        context = self.input_payloads.get("image", {})
        checked = check_context(self)
        size, inverse = preprocessing_geometry(inputs["preprocessing"], image)
        rows = tensor_rows(inputs, {**p, "format": "xywh_scores"})
        xywh, angles, scores, classes, source_indices = oriented_candidates(rows, p, size)
        centers = xywh[:, :2] * np.diag(inverse)[:2] + inverse[:2, 2]
        selected = selection_mask(inputs, image)
        class_selected = (
            np.isin(classes, p["accepted_class_ids"])
            if p["accepted_class_ids"]
            else np.ones(len(scores), bool)
        )
        indices = [
            i for i in np.flatnonzero(class_selected) if contains(selected, np.floor(centers[i]))
        ]
        if len(indices) > p["maximum_candidates"]:
            raise ValidationError("Scored/selected detections exceed the candidate budget")
        rectangles, geometries, native_bounds = {}, {}, {}
        for i in indices:
            x, y, bw, bh = map(float, xywh[i])
            if max(abs(x), abs(y), bw, bh) > 1e6:
                raise ValidationError("Rotated model coordinates must stay within one million")
            rectangle = ((x, y), (bw, bh), float(angles[i]))
            native_points = cv2.boxPoints(rectangle).astype(np.float64)
            native_area = polygon_area(native_points)
            if native_area <= 0 or not math.isclose(native_area, bw * bh, rel_tol=0.001, abs_tol=0):
                raise ValidationError("Rotated rectangle loses area at native float32 precision")
            angle = math.radians(float(angles[i]))
            local = np.array(
                [
                    [math.cos(angle), -math.sin(angle), x],
                    [math.sin(angle), math.cos(angle), y],
                    [0, 0, 1],
                ]
            )
            transform = inverse @ local
            corners = np.array(
                [[-bw / 2, -bh / 2], [bw / 2, -bh / 2], [bw / 2, bh / 2], [-bw / 2, bh / 2]]
            )
            points = (np.c_[corners, np.ones(4)] @ transform.T)[:, :2]
            if not np.isfinite(points).all() or np.max(np.abs(points)) > 1e6:
                raise ValidationError("Mapped rotated footprint exceeds source-coordinate limits")
            clipped = clip_image(points, w, h)
            if polygon_area(clipped) <= 0:
                continue
            crosses = bool(np.any(points < [0, 0]) or np.any(points > [w, h]))
            if p["border_policy"] == "discard" and crosses:
                continue
            rectangles[i] = rectangle
            # Conservative bounds avoid native intersection calls for disjoint
            # parts while allowing float32 corner-rounding uncertainty.
            margin = max(1.0, float(np.max(np.abs(native_points)))) * np.finfo(np.float32).eps * 8
            native_bounds[i] = [
                *(native_points.min(axis=0) - margin),
                *(native_points.max(axis=0) + margin),
            ]
            geometries[i] = (points, clipped, transform, crosses, corners)
        indices = sorted(rectangles, key=lambda i: (-float(scores[i]), int(source_indices[i])))
        groups = {}
        for i in indices:
            groups.setdefault(0 if p["class_agnostic_nms"] else int(classes[i]), []).append(i)
        comparisons = sum(
            (min(len(g), p["nms_top_k"] or len(g)) * (min(len(g), p["nms_top_k"] or len(g)) - 1))
            // 2
            for g in groups.values()
        )
        if p["nms_enabled"] and comparisons > p["maximum_pair_comparisons"]:
            raise ValidationError("Rotated NMS exceeds the pair comparison budget")
        retained = (
            [i for g in groups.values() for i in suppress(g, rectangles, scores, p, native_bounds)]
            if p["nms_enabled"]
            else indices
        )
        retained.sort(key=lambda i: (-float(scores[i]), int(source_indices[i])))
        if len(retained) > p["maximum_results"]:
            raise ValidationError("Retained rotated detections exceed the result budget")
        items, graphics = [], []
        for i in retained:
            points, clipped, transform, crosses, local_corners = geometries[i]
            final = clipped if p["border_policy"] == "clip" else points
            low, high = final.min(axis=0), final.max(axis=0)
            center, ax, ay = transform[:2, 2], transform[:2, 0], transform[:2, 1]
            source_angle = math.degrees(math.atan2(ax[1], ax[0])) % int(p["angle_period"])
            width_axis = np.linalg.norm(points[1] - points[0])
            height_axis = np.linalg.norm(points[3] - points[0])
            orthogonal = abs(float(ax @ ay)) <= 1e-9 * np.linalg.norm(ax) * np.linalg.norm(ay)
            pose = make_pose(
                transform,
                [w, h],
                context,
                local_units="network_input_pixels",
                orientation_source="model_width_axis",
                orientation_period=int(p["angle_period"]),
            )
            class_id, score = int(classes[i]), float(scores[i])
            label = p["labels"][class_id] if p["labels"] else str(class_id)
            item = {
                "id": int(source_indices[i]),
                "class_id": class_id,
                "label": label,
                "score": score,
                "orientation_measured": False,
                "orientation_predicted": True,
                "pose": pose,
                "metrics": {
                    "x": float(low[0]),
                    "y": float(low[1]),
                    "width": float(high[0] - low[0]),
                    "height": float(high[1] - low[1]),
                    "center_x": float(center[0]),
                    "center_y": float(center[1]),
                    "area": polygon_area(final),
                    "full_area": polygon_area(points),
                    "aspect_ratio": float((high[0] - low[0]) / (high[1] - low[1])),
                    "angle": source_angle,
                    "score": score,
                    "class_id": class_id,
                    "width_axis_length": float(width_axis),
                    "height_axis_length": float(height_axis),
                    "major_size": float(max(width_axis, height_axis)) if orthogonal else None,
                    "minor_size": float(min(width_axis, height_axis)) if orthogonal else None,
                    "visible_fraction": min(1.0, polygon_area(clipped) / polygon_area(points)),
                },
                "geometry": {
                    "points": final.tolist(),
                    "unclipped_points": points.tolist(),
                    "bbox": [*low.tolist(), *(high - low).tolist()],
                    "network_xywha": [*xywh[i].tolist(), float(angles[i])],
                    "local_corners": local_corners.tolist(),
                    "orthogonal": bool(orthogonal),
                    "crosses_image_border": crosses,
                    "clipped": p["border_policy"] == "clip" and crosses,
                },
            }
            items.append(item)
            if self.generate_graphics:
                end = center + ax * min(float(xywh[i, 2]) / 2, 20)
                graphics.extend(
                    (
                        graphic_object("polyline", final, (60, 230, 120), closed=True),
                        graphic_object("polyline", [center, end], (80, 180, 255)),
                        graphic_object(
                            "label",
                            [low],
                            (60, 230, 120),
                            text=f"{label}: {score:.3f}, {source_angle:.1f}deg",
                        ),
                    )
                )
        count = len(items)
        passed = count >= p["minimum_count"] and (
            not p["maximum_count"] or count <= p["maximum_count"]
        )
        return Result(
            {
                "image": source,
                "collection": collection("boxes", items, image, context),
                "count": count,
                "matches": {
                    "model": "pattern_matches_v1",
                    "count": count,
                    "items": copy.deepcopy(items),
                },
                "report": {
                    "model": "rotated_detection_decode_v1",
                    "format": p["format"],
                    "layout": p["layout"],
                    "angle_unit": p["angle_unit"],
                    "angle_direction": p["angle_direction"],
                    "opencv_version": cv2.__version__,
                    "raw_rows": len(rows),
                    "scored_rows": len(scores),
                    "selected_candidates": len(indices),
                    "retained": count,
                    "nms_method": p["nms_method"],
                    "nms_enabled": p["nms_enabled"],
                    "nms_class_aware": not p["class_agnostic_nms"],
                    "pair_comparison_bound": comparisons,
                    "nms_grid": "unclipped_network_rectangles",
                    "context_checks": checked,
                    "coordinate_convention": "continuous_source_pixel_edges",
                    "pose_local_units": "network_input_pixels",
                    "angle_period": int(p["angle_period"]),
                    "preprocessing": copy.deepcopy(inputs["preprocessing"]),
                },
            },
            payloads={port.name: copy.deepcopy(context) for port in self.outputs},
            passed=passed,
            score=max((i["score"] for i in items), default=0.0),
            message=f"{count} rotated detections after selection/NMS",
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
            drawings={"Rotated detections": GraphicDrawing(tuple(graphics))} if graphics else {},
        )
