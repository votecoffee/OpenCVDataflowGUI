"""JSON object collections and measured native shape descriptors."""

import copy
import math

import cv2
import numpy as np

METRICS = (
    "area",
    "width",
    "height",
    "x",
    "y",
    "center_x",
    "center_y",
    "aspect_ratio",
    "angle",
    "length",
    "perimeter",
    "major_size",
    "minor_size",
    "rotated_aspect_ratio",
    "extent",
    "solidity",
    "circularity",
    "distance",
    "line_width",
    "precision",
    "nfa",
    *(f"hu_{i}" for i in range(1, 8)),
    *(f"hu_log_{i}" for i in range(1, 8)),
)


def shape_metrics(points, moments, area, perimeter, bounds, center, hull_area):
    x, y, width, height = bounds
    (_, _), (a, b), _ = cv2.minAreaRect(np.asarray(points, np.float32).reshape(-1, 1, 2))
    major, minor = max(a, b), min(a, b)
    metrics = dict.fromkeys(METRICS)
    anisotropy = math.hypot(moments["mu20"] - moments["mu02"], 2 * moments["mu11"])
    variance = abs(moments["mu20"]) + abs(moments["mu02"])
    angle = (
        math.degrees(0.5 * math.atan2(2 * moments["mu11"], moments["mu20"] - moments["mu02"])) % 180
        if anisotropy > max(1, variance) * 1e-12
        else None
    )
    metrics.update(
        area=float(area),
        perimeter=float(perimeter),
        x=float(x),
        y=float(y),
        width=float(width),
        height=float(height),
        center_x=float(center[0]),
        center_y=float(center[1]),
        aspect_ratio=width / height,
        angle=angle,
        major_size=float(major),
        minor_size=float(minor),
        rotated_aspect_ratio=major / minor if minor > 0 else None,
        extent=area / (width * height),
        solidity=area / hull_area if hull_area > 0 else None,
        circularity=4 * math.pi * area / perimeter**2 if perimeter > 0 else None,
    )
    hu = cv2.HuMoments(moments).reshape(-1) if moments["m00"] > 0 else [None] * 7
    for index, value in enumerate(hu, 1):
        metrics[f"hu_{index}"] = float(value) if value is not None else None
        metrics[f"hu_log_{index}"] = (
            -math.copysign(1, value) * math.log10(abs(value))
            if value
            else 0.0
            if value == 0
            else None
        )
    return metrics


def contour_item(contour, index, hierarchy=None):
    points = np.asarray(contour).reshape(-1, 2)
    moments = cv2.moments(contour)
    center = (
        [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]
        if moments["m00"]
        else points.mean(axis=0).tolist()
    )
    area = cv2.contourArea(contour)
    metrics = shape_metrics(
        points,
        moments,
        area,
        cv2.arcLength(contour, True),
        cv2.boundingRect(contour),
        center,
        cv2.contourArea(cv2.convexHull(contour)),
    )
    geometry = {"points": points.tolist()}
    if hierarchy is not None:
        geometry["hierarchy"] = list(map(int, hierarchy))
    return {"id": int(index), "metrics": metrics, "geometry": geometry}


def blob_item(binary_roi, x, y, index):
    contours, hierarchy = cv2.findContours(binary_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    moments = cv2.moments(binary_roi, binaryImage=True)
    center = [x + moments["m10"] / moments["m00"], y + moments["m01"] / moments["m00"]]
    points = np.vstack([c.reshape(-1, 2) for c in contours])
    hull = cv2.convexHull(points)
    hull_mask = np.zeros_like(binary_roi)
    cv2.drawContours(hull_mask, [hull], -1, 1, -1)
    metrics = shape_metrics(
        points,
        moments,
        cv2.countNonZero(binary_roi),
        sum(cv2.arcLength(c, True) for c in contours),
        (x, y, binary_roi.shape[1], binary_roi.shape[0]),
        center,
        cv2.countNonZero(hull_mask),
    )
    offset = np.array([x, y])
    return {
        "id": int(index),
        "metrics": metrics,
        "geometry": {
            "contours": [(c.reshape(-1, 2) + offset).tolist() for c in contours],
            "hierarchy": hierarchy[0].tolist(),
        },
    }


def line_item(segment, index, width=None, precision=None, nfa=None):
    points = np.asarray(segment, float).reshape(2, 2)
    delta = points[1] - points[0]
    bounds = cv2.boundingRect(points.astype(np.float32))
    x, y, w, h = bounds
    metrics = dict.fromkeys(METRICS)
    metrics.update(
        x=float(x),
        y=float(y),
        width=float(w),
        height=float(h),
        aspect_ratio=w / h,
        center_x=float(points[:, 0].mean()),
        center_y=float(points[:, 1].mean()),
        length=float(np.linalg.norm(delta)),
        angle=math.degrees(math.atan2(delta[1], delta[0])) % 180 if np.any(delta) else None,
        line_width=float(width) if width is not None else None,
        precision=float(precision) if precision is not None else None,
        nfa=float(nfa) if nfa is not None else None,
    )
    return {"id": int(index), "metrics": metrics, "geometry": {"points": points.tolist()}}


def collection(kind, items, image, payload, origin=(0, 0)):
    return {
        "model": "vision_collection_v1",
        "kind": kind,
        "count": len(items),
        "items": items,
        "image_size": [image.shape[1], image.shape[0]],
        "origin": list(map(int, origin)),
        "context": {
            key: copy.deepcopy(payload[key]) for key in ("spatial", "pyramid") if key in payload
        },
    }


def validate_collection(value):
    if (
        not isinstance(value, dict)
        or value.get("model") != "vision_collection_v1"
        or value.get("kind") not in ("blobs", "lines", "contours")
    ):
        raise ValueError("Expected a vision_collection_v1 blob, line or contour collection")
    items = value.get("items")
    size, origin = value.get("image_size"), value.get("origin", [0, 0])
    if (
        not isinstance(items, list)
        or len(items) > 100000
        or value.get("count") != len(items)
        or not isinstance(size, list)
        or len(size) != 2
        or any(type(v) is not int or v <= 0 for v in size)
        or max(size) > 16384
        or size[0] * size[1] > 64_000_000
        or len(origin) != 2
        or any(type(v) is not int for v in origin)
    ):
        raise ValueError(
            "Invalid collection count, dimensions or origin "
            "(100000 items / 64M pixels / 16384 axis limits)"
        )
    ids = set()
    for item in items:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("id"), (int, str))
            or isinstance(item["id"], bool)
            or item["id"] in ids
            or not isinstance(item.get("geometry"), dict)
            or not isinstance(item.get("metrics"), dict)
        ):
            raise ValueError("Collection items need unique IDs, geometry and numeric metrics")
        ids.add(item["id"])
        for number in item["metrics"].values():
            if number is not None and (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(number)
            ):
                raise ValueError("Collection metrics must be finite numbers or null")
    return value
