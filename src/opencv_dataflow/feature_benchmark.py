"""Bounded center-reprojection and one-to-one matching metrics for known image pairs."""

from collections import deque

import cv2
import numpy as np

from .parameter_validation import ValidationError
from .selection import contains


def reference_matrix(value, query_size, train_size):
    if isinstance(value, dict):
        for key, size in (("source_image_size", query_size), ("target_image_size", train_size)):
            if value.get(key) is not None and value[key] != size:
                raise ValidationError(
                    "Reference transform image sizes differ from evaluation images"
                )
        value = value.get("source_to_target")
    try:
        matrix = np.asarray(value, np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValidationError("Reference transform must be a finite 3x3 matrix") from exc
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not np.any(matrix):
        raise ValidationError("Reference transform must be a finite nonzero 3x3 matrix")
    matrix = matrix / np.max(np.abs(matrix))
    if np.linalg.cond(matrix) > 1e12:
        raise ValidationError("Reference transform is singular or ill-conditioned")
    inverse = np.linalg.inv(matrix)
    for transform, size in ((matrix, query_size), (inverse, train_size)):
        width, height = size
        corners = np.array(
            [[0, 0, 1], [width - 1, 0, 1], [width - 1, height - 1, 1], [0, height - 1, 1]]
        )
        denominator = corners @ transform[2]
        relative = denominator / np.max(np.abs(denominator))
        if not (np.all(relative > 1e-12) or np.all(relative < -1e-12)):
            raise ValidationError("Reference transform has a projective horizon in an image")
    return matrix, inverse


def project(points, matrix):
    if not len(points):
        return np.empty((0, 2), np.float64)
    mapped = cv2.perspectiveTransform(points[:, None, :], matrix).reshape(-1, 2)
    if not np.isfinite(mapped).all():
        raise ValidationError("Reference projection is nonfinite")
    return mapped


def in_support(points, size, selection, margin):
    return np.asarray(
        [
            margin <= x < size[0] - margin
            and margin <= y < size[1] - margin
            and contains(selection, (x, y))
            for x, y in points
        ],
        dtype=bool,
    )


def geometric_edges(query, train, selected_query, selected_train, tolerance, maximum_edges):
    """Forward error in target pixels; rows retain original feature indices."""
    adjacency = [[] for _ in query]
    query_indices, train_indices = np.flatnonzero(selected_query), np.flatnonzero(selected_train)
    count = 0
    target = train[train_indices]
    for start in range(0, len(query_indices), 64):
        indices = query_indices[start : start + 64]
        delta = query[indices, None, :] - target[None, :, :]
        distances = np.hypot(delta[..., 0], delta[..., 1])
        for row, index in enumerate(indices):
            valid = np.flatnonzero(distances[row] <= tolerance)
            count += len(valid)
            if count > maximum_edges:
                raise ValidationError(
                    "Geometric candidate edges exceed limit", "maximum_geometry_edges"
                )
            # Stable preference affects the diagnostic assignment, not maximum cardinality.
            valid = sorted(valid, key=lambda j: (distances[row, j], int(train_indices[j])))
            adjacency[int(index)] = [int(train_indices[j]) for j in valid]
    return adjacency


def maximum_matching(adjacency, train_count):
    """Iterative augmenting paths: maximum cardinality, no recursion or greedy undercount."""
    query_match = [-1] * len(adjacency)
    train_match = [-1] * train_count
    for root in range(len(adjacency)):
        if query_match[root] >= 0:
            continue
        parents, seen_query, queue, free = {}, {root}, deque([root]), -1
        while queue and free < 0:
            query = queue.popleft()
            for train in adjacency[query]:
                if train in parents:
                    continue
                parents[train] = query
                owner = train_match[train]
                if owner < 0:
                    free = train
                    break
                if owner not in seen_query:
                    seen_query.add(owner)
                    queue.append(owner)
        while free >= 0:
            query = parents[free]
            previous = query_match[query]
            query_match[query], train_match[free] = free, query
            free = previous
    return {q: t for q, t in enumerate(query_match) if t >= 0}


def threshold_metrics(rows, threshold, truth_count, query_count, train_count):
    accepted = [
        row
        for row in rows
        if row["rank"] is not None and row["eligible"] and row["rank"] <= threshold
    ]
    adjacency = [[] for _ in range(query_count)]
    for row in accepted:
        if row["geometrically_correct"] and row["train_index"] not in adjacency[row["query_index"]]:
            adjacency[row["query_index"]].append(row["train_index"])
    pairing = maximum_matching(adjacency, train_count)
    true_count = len(pairing)
    return {
        "threshold": float(threshold),
        "accepted": len(accepted),
        "true_positives": true_count,
        "false_positives": len(accepted) - true_count,
        "false_negatives": truth_count - true_count,
        "precision": true_count / len(accepted) if accepted else None,
        "recall": true_count / truth_count if truth_count else None,
    }, pairing


def precision_recall_curve(
    rows, operating_threshold, truth_count, query_count, train_count, maximum
):
    values = sorted({row["rank"] for row in rows if row["eligible"] and row["rank"] is not None})
    sampled = len(values) > maximum - 1
    if sampled:
        indices = (
            [len(values) - 1]
            if maximum == 2
            else np.linspace(0, len(values) - 1, maximum - 1, dtype=int)
        )
        values = [values[i] for i in indices]
    thresholds = sorted(set(values) | {float(operating_threshold)})
    curve = [
        threshold_metrics(rows, threshold, truth_count, query_count, train_count)[0]
        for threshold in thresholds
    ]
    return curve, sampled
