"""Feature detector/matcher benchmarking using caller-supplied image registration truth."""

import copy
import math

import cv2
import numpy as np

from .core import Block, Port, Result
from .feature_benchmark import (
    geometric_edges,
    in_support,
    maximum_matching,
    precision_recall_curve,
    project,
    reference_matrix,
    threshold_metrics,
)
from .features import overlay_image
from .graphics_layers import GraphicDrawing, draw_object, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, ValuePacket, frame, require_same_frame
from .processing import image_value
from .registration import radius_data
from .selection import selection_mask
from .tickets import ticket_key


def feature_data(value, image, maximum):
    if not isinstance(value, dict) or value.get("image_size") != list(image.shape[1::-1]):
        raise ValidationError("Feature image_size differs from its evaluation image")
    keys, descriptors = value.get("keypoints"), value.get("descriptors")
    if not isinstance(keys, list) or len(keys) > maximum:
        raise ValidationError(
            "Feature count exceeds limit or keypoints are invalid", "maximum_features"
        )
    if (
        not isinstance(descriptors, list)
        or len(descriptors) != len(keys)
        or any(not isinstance(row, list) or len(row) > 512 for row in descriptors)
    ):
        raise ValidationError("Invalid or oversized descriptor table")
    try:
        descriptors, points, layout, norm = radius_data(value)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise ValidationError(f"Invalid feature data: {exc}") from exc
    if np.any(points < 0) or np.any(points >= image.shape[1::-1]):
        raise ValidationError("Feature centers must lie inside their image")
    return descriptors, points, layout, norm


def candidate_rows(inputs, query, train, query_points, train_points, qkeep, tkeep, norm, p):
    if "matches" in inputs:
        data = inputs["matches"]
        if not isinstance(data, dict) or not isinstance(data.get("matches"), list):
            raise ValidationError("Matches must be a paired feature-match record")
        if len(data["matches"]) > p["maximum_matches"]:
            raise ValidationError("Match table exceeds limit", "maximum_matches")
        for role in ("query", "train"):
            if data.get(role + "_size") != inputs[role]["image_size"]:
                raise ValidationError("Match record image sizes differ from feature sets")
        rows = []
        for i, match in enumerate(data["matches"]):
            if not isinstance(match, dict):
                raise ValidationError("Each match must contain feature indices and distance")
            q, t, distance = (
                match.get("query_index"),
                match.get("train_index"),
                match.get("distance"),
            )
            if (
                type(q) is not int
                or type(t) is not int
                or not 0 <= q < len(query)
                or not 0 <= t < len(train)
            ):
                raise ValidationError("Match feature indices are out of range")
            if (
                isinstance(distance, bool)
                or not isinstance(distance, (int, float))
                or not math.isfinite(distance)
                or not 0 <= distance <= 1e9
            ):
                raise ValidationError("Match distances must be finite nonnegative values up to 1e9")
            for role, index, points in (("query", q, query_points), ("train", t, train_points)):
                supplied = data.get(role + "_points")
                if not isinstance(supplied, list) or len(supplied) != len(data["matches"]):
                    raise ValidationError("Match coordinates disagree with indexed feature centers")
                try:
                    point = np.asarray(supplied[i], np.float64)
                except (ValueError, TypeError, OverflowError) as exc:
                    raise ValidationError("Match coordinates must be finite pixel pairs") from exc
                if point.shape != (2,) or not np.allclose(point, points[index], rtol=0, atol=1e-6):
                    raise ValidationError("Match coordinates disagree with indexed feature centers")
            rows.append(
                {
                    "query_index": q,
                    "train_index": t,
                    "distance": float(distance),
                    "rank": float(distance),
                }
            )
        return rows, "external_distance"
    qindices, tindices = np.flatnonzero(qkeep), np.flatnonzero(tkeep)
    if not len(qindices) or not len(tindices):
        return [], p["ranking"]
    neighbors = 2 if p["ranking"] == "ratio" else 1
    matched = cv2.BFMatcher(norm).knnMatch(
        query[qindices], train[tindices], k=min(neighbors, len(tindices))
    )
    rows = []
    for source, matches in zip(qindices, matched, strict=True):
        if not matches:
            continue
        first = matches[0]
        rank, why = float(first.distance), ""
        if neighbors == 2:
            if len(matches) < 2 or matches[1].distance == 0:
                rank, why = None, "insufficient_or_zero_second_distance"
            else:
                rank = float(first.distance / matches[1].distance)
        rows.append(
            {
                "query_index": int(source),
                "train_index": int(tindices[first.trainIdx]),
                "distance": float(first.distance),
                "rank": rank,
                "unranked_reason": why,
            }
        )
    if len(rows) > p["maximum_matches"]:
        raise ValidationError("Generated matches exceed limit", "maximum_matches")
    return rows, p["ranking"]


class EvaluateImageFeatures(Block):
    type_id, title = "inspect.evaluate_features", "Evaluate image features and matches"
    category, block_type, subtype = "Inspection", "Inspection", "Feature evaluation"
    preview_safe = parallel_safe = process_inputs_readonly = True
    composes_result_image = True
    inputs = (
        Port("query_image"),
        Port("train_image"),
        Port("query", "data"),
        Port("train", "data"),
        Port("transform", "data", required=False, required_when_connected=True),
        Port("matches", "data", required=False, required_when_connected=True),
        Port("query_mask", required=False),
        Port("train_mask", required=False),
        Port("query_region", "data", required=False),
        Port("train_region", "data", required=False),
    )
    outputs = (
        Port("metrics", "data"),
        Port("repeatability", "float", required=False),
        Port("precision", "float", required=False),
        Port("recall", "float", required=False),
        Port("correspondences", "data"),
        Port("matches", "data"),
        Port("curve", "data"),
        Port("canvas", required=False),
        Port("overlay", required=False),
    )
    defaults = {
        "reference_transform": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "reprojection_tolerance": 3.0,
        "border_margin": 0.0,
        "ranking": "distance",
        "match_threshold": 64.0,
        "minimum_correspondences": 1,
        "minimum_repeatability": 0.0,
        "minimum_precision": 0.0,
        "minimum_recall": 0.0,
        "maximum_features": 2000,
        "maximum_pairs": 4000000,
        "maximum_geometry_edges": 100000,
        "maximum_matches": 10000,
        "maximum_curve_points": 64,
        "maximum_image_pixels": 8388608,
        "preview_width": 1200,
        "preview_height": 600,
        "maximum_graphic_matches": 200,
    }
    parameter_choices = {"ranking": ("distance", "ratio")}
    parameter_ranges = {
        "reprojection_tolerance": NumericRange(0.000001, 1024),
        "border_margin": NumericRange(0, 4096),
        "match_threshold": NumericRange(0, 1e9),
        "minimum_correspondences": NumericRange(0, 4096, whole=True),
        "minimum_repeatability": NumericRange(0, 1),
        "minimum_precision": NumericRange(0, 1),
        "minimum_recall": NumericRange(0, 1),
        "maximum_features": NumericRange(1, 4096, whole=True),
        "maximum_pairs": NumericRange(1, 8000000, whole=True),
        "maximum_geometry_edges": NumericRange(1, 200000, whole=True),
        "maximum_matches": NumericRange(1, 20000, whole=True),
        "maximum_curve_points": NumericRange(2, 256, whole=True),
        "maximum_image_pixels": NumericRange(1, 16777216, whole=True),
        "preview_width": NumericRange(128, 2048, whole=True),
        "preview_height": NumericRange(128, 1024, whole=True),
        "maximum_graphic_matches": NumericRange(0, 2000, whole=True),
    }
    parameter_help = {
        "reference_transform": "Known query-to-train homography; wired Transform overrides.",
        "reprojection_tolerance": "Inclusive target-pixel center error; geometry-only correctness.",
        "border_margin": "Boundary margin in each image's pixels; selections use keypoint centers.",
        "ranking": "Automatic NN distance or ratio; external Matches always ranks by distance.",
        "match_threshold": "Inclusive operating threshold; ratio normally uses 0..1.",
        "minimum_correspondences": "Minimum one-to-one geometric pairs; no-support never passes.",
        "minimum_repeatability": "Correspondence count divided by smaller eligible feature count.",
        "minimum_precision": "Unique correct pairs / accepted rows. Duplicates reduce precision.",
        "minimum_recall": "Unique correct pairs / geometric correspondences.",
        "maximum_pairs": "Full input query×train pair guard, before descriptor/geometry matching.",
        "maximum_geometry_edges": "Valid geometry edge cap before one-to-one assignment.",
        "maximum_curve_points": "Maximum sampled thresholds, including operating and highest rank.",
        "maximum_image_pixels": "Per-image cap before grayscale/preview or numeric evaluation.",
        "maximum_features": "Per-input feature count cap before descriptor conversion.",
        "maximum_matches": "Bound diagnostic rows and external candidate lists.",
        "preview_width": "Maximum side-by-side diagnostic width; output has its own grid.",
        "preview_height": "Maximum diagnostic height; images scale together.",
        "maximum_graphic_matches": "Draw at most this many accepted candidates, best rank first.",
    }
    input_help = {
        "query": "ORB/BRISK/AKAZE/SIFT/KAZE/BRIEF/FREAK/DAISY feature data for Query image.",
        "train": "Same descriptor family/settings for Train image.",
        "transform": "Independent query-to-train truth. Fitting these matches biases evaluation.",
        "matches": "Indexed paired match record; descriptor distance is used as rank.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = ["metrics", "repeatability", "precision", "recall", "overlay"]

    def process(self, inputs):
        p, values, selections, sizes = self.parameters, {}, {}, {}
        for role in ("query", "train"):
            image = inputs[role + "_image"]
            if image.shape[0] * image.shape[1] > p["maximum_image_pixels"]:
                raise ValidationError("Evaluation image exceeds pixel cap", "maximum_image_pixels")
            image_value(image)
            sizes[role] = list(image.shape[1::-1])
            values[role] = feature_data(inputs[role], image, p["maximum_features"])
            image_payload = self.input_payloads.get(role + "_image", {})
            feature_payload = self.input_payloads.get(role, {})
            try:
                require_same_frame(
                    [ValuePacket(None, image_payload), ValuePacket(None, feature_payload)]
                )
            except ValueError as exc:
                raise ValidationError("Features and image use different coordinate frames") from exc
            if (
                "ticket" in image_payload
                and "ticket" in feature_payload
                and ticket_key(image_payload["ticket"]) != ticket_key(feature_payload["ticket"])
            ):
                raise ValidationError("Features and image have different acquisition tickets")
            selected = {"image": image}
            for key in ("mask", "region"):
                if role + "_" + key in inputs:
                    selected[key] = inputs[role + "_" + key]
            selections[role] = selection_mask(selected, image)
        query, qpoints, layout, norm = values["query"]
        train, tpoints, other_layout, _ = values["train"]
        if layout != other_layout:
            raise ValidationError("Descriptor families/settings differ between query and train")
        if len(query) * len(train) > p["maximum_pairs"]:
            raise ValidationError("Feature pairs exceed configured limit", "maximum_pairs")
        matrix, inverse = reference_matrix(
            inputs.get("transform", p["reference_transform"]), sizes["query"], sizes["train"]
        )
        projected, back = project(qpoints, matrix), project(tpoints, inverse)
        qkeep = in_support(
            qpoints, sizes["query"], selections["query"], p["border_margin"]
        ) & in_support(projected, sizes["train"], selections["train"], p["border_margin"])
        tkeep = in_support(
            tpoints, sizes["train"], selections["train"], p["border_margin"]
        ) & in_support(back, sizes["query"], selections["query"], p["border_margin"])
        adjacency = geometric_edges(
            projected,
            tpoints,
            qkeep,
            tkeep,
            p["reprojection_tolerance"],
            p["maximum_geometry_edges"],
        )
        truth = maximum_matching(adjacency, len(train))
        rows, ranking = candidate_rows(
            inputs, query, train, qpoints, tpoints, qkeep, tkeep, norm, p
        )
        for i, row in enumerate(rows):
            q, t = row["query_index"], row["train_index"]
            row.update(
                index=i,
                eligible=bool(qkeep[q] and tkeep[t]),
                geometrically_correct=t in adjacency[q],
                error_pixels=float(np.linalg.norm(projected[q] - tpoints[t])),
            )
        operating, pairing = threshold_metrics(
            rows, p["match_threshold"], len(truth), len(query), len(train)
        )
        curve, sampled = precision_recall_curve(
            rows,
            p["match_threshold"],
            len(truth),
            len(query),
            len(train),
            p["maximum_curve_points"],
        )
        count = min(int(qkeep.sum()), int(tkeep.sum()))
        repeatability = len(truth) / count if count else None
        used = set()
        for row in sorted(
            rows, key=lambda r: (float("inf") if r["rank"] is None else r["rank"], r["index"])
        ):
            pair = row["query_index"], row["train_index"]
            accepted = (
                row["eligible"] and row["rank"] is not None and row["rank"] <= p["match_threshold"]
            )
            row["accepted"] = accepted
            row["unique_true_positive"] = bool(
                accepted and pairing.get(pair[0]) == pair[1] and pair not in used
            )
            if row["unique_true_positive"]:
                used.add(pair)
        reasons = []
        if not count or not truth:
            reasons.append("feature_evaluation_no_correspondences")
        if len(truth) < p["minimum_correspondences"]:
            reasons.append("feature_evaluation_insufficient_correspondences")
        for metric, value in (
            ("repeatability", repeatability),
            ("precision", operating["precision"]),
            ("recall", operating["recall"]),
        ):
            if p["minimum_" + metric] > 0 and (value is None or value < p["minimum_" + metric]):
                reasons.append("feature_evaluation_" + metric + "_below_minimum")
        metrics = {
            "model": "feature_evaluation_v1",
            "metric": "center_reprojection_maximum_cardinality",
            "transform_independence": "caller_supplied_not_verified",
            "query_to_train": matrix.tolist(),
            "query_count": len(query),
            "train_count": len(train),
            "eligible_query": int(qkeep.sum()),
            "eligible_train": int(tkeep.sum()),
            "geometric_edges": sum(map(len, adjacency)),
            "correspondence_count": len(truth),
            "repeatability": repeatability,
            "valid": {
                "repeatability": repeatability is not None,
                "precision": operating["precision"] is not None,
                "recall": operating["recall"] is not None,
            },
            "ranking": ranking,
            "operating": operating,
            "curve_sampled": sampled,
            "candidate_rows": len(rows),
            "outside_common_support": sum(not r["eligible"] for r in rows),
            "unranked_rows": sum(r["rank"] is None for r in rows),
            "reasons": reasons,
            "query_context": copy.deepcopy(self.input_payloads.get("query_image", {})),
            "train_context": copy.deepcopy(self.input_payloads.get("train_image", {})),
        }
        correspondences = [
            {
                "query_index": q,
                "train_index": t,
                "error_pixels": float(np.linalg.norm(projected[q] - tpoints[t])),
            }
            for q, t in sorted(truth.items())
        ]
        canvas, overlay, drawings, graphics, contexts = self.render(
            inputs, rows, qpoints, tpoints, projected, curve, metrics
        )
        outputs = {
            "metrics": metrics,
            "repeatability": repeatability if repeatability is not None else NO_VALUE,
            "precision": operating["precision"] if operating["precision"] is not None else NO_VALUE,
            "recall": operating["recall"] if operating["recall"] is not None else NO_VALUE,
            "correspondences": correspondences,
            "matches": rows,
            "curve": curve,
            "canvas": canvas,
            "overlay": overlay,
        }
        contexts = {**{name: {} for name in outputs}, **contexts}
        return Result(
            outputs,
            passed=not reasons,
            fail_reasons=reasons,
            payloads=contexts,
            drawings=drawings,
            graphics=graphics,
            score=repeatability,
            message=(
                f"{len(truth)} correspondences; "
                f"{operating['true_positives']}/{operating['accepted']} correct predictions"
            ),
        )

    def render(self, inputs, rows, query, train, projected, curve, metrics):
        if not self.generate_graphics:
            return NO_VALUE, NO_VALUE, {}, {}, {}
        p = self.parameters
        left, right = overlay_image(inputs["query_image"]), overlay_image(inputs["train_image"])
        scale = min(
            1.0,
            p["preview_width"] / (left.shape[1] + right.shape[1]),
            p["preview_height"] / max(left.shape[0], right.shape[0]),
        )
        left_width = max(1, min(p["preview_width"] - 1, round(left.shape[1] * scale)))
        right_width = max(1, min(p["preview_width"] - left_width, round(right.shape[1] * scale)))
        a, b = [
            cv2.resize(im, (width, max(1, round(im.shape[0] * scale))))
            for im, width in ((left, left_width), (right, right_width))
        ]
        canvas = np.full((max(a.shape[0], b.shape[0]), a.shape[1] + b.shape[1], 3), 25, np.uint8)
        canvas[: a.shape[0], : a.shape[1]], canvas[: b.shape[0], a.shape[1] :] = a, b
        qm = np.diag([a.shape[1] / left.shape[1], a.shape[0] / left.shape[0], 1.0])
        tm = np.diag([b.shape[1] / right.shape[1], b.shape[0] / right.shape[0], 1.0])
        for transform in (qm, tm):
            transform[0, 2] = (transform[0, 0] - 1) / 2
            transform[1, 2] = (transform[1, 1] - 1) / 2
        tm[0, 2] += a.shape[1]
        q, t, expected = project(query, qm), project(train, tm), project(projected, tm)
        objects = []
        for row in sorted(
            (r for r in rows if r["accepted"]), key=lambda r: (r["rank"], r["index"])
        )[: int(p["maximum_graphic_matches"])]:
            qi, ti = row["query_index"], row["train_index"]
            color = (60, 230, 120) if row["unique_true_positive"] else (40, 60, 245)
            objects.append(graphic_object("line", [q[qi], t[ti]], color, thickness=1))
            objects.append(
                graphic_object(
                    "line", [expected[qi] - [2, 0], expected[qi] + [2, 0]], (230, 200, 40)
                )
            )
        overlay = canvas.copy()
        for obj in objects:
            draw_object(overlay, obj, np.eye(3))
        context = {"spatial": frame([canvas.shape[1], canvas.shape[0]], self.id + ":review")}
        metrics["preview"] = {"query_to_canvas": qm.tolist(), "train_to_canvas": tm.tolist()}
        plot = np.full((360, 480, 3), 25, np.uint8)
        cv2.rectangle(plot, (45, 30), (450, 320), (130, 130, 130), 1)
        valid = [
            point
            for point in curve
            if point["precision"] is not None and point["recall"] is not None
        ]
        if valid:
            line = np.asarray(
                [[45 + round(405 * v["recall"]), 320 - round(290 * v["precision"])] for v in valid],
                np.int32,
            )
            cv2.polylines(plot, [line], False, (60, 230, 120), 2)
            for point in line:
                cv2.circle(plot, tuple(point), 2, (60, 230, 120), -1)
        cv2.putText(
            plot,
            "Precision vs recall (threshold samples)",
            (40, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
        )
        cv2.putText(
            plot, "Recall: 0 to 1", (160, 346), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1
        )
        return (
            canvas,
            overlay,
            {"matched pairs": GraphicDrawing(tuple(objects), source="canvas", from_output=True)}
            if objects
            else {},
            {"precision recall": plot},
            {"canvas": context, "overlay": context},
        )
