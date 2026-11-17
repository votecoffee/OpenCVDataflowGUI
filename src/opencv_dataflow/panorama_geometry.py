"""Bounded rotating-camera geometry and independent forward panorama projection."""

import copy
import hashlib
import json

import cv2
import numpy as np

from .parameter_validation import ValidationError
from .payloads import check_spatial, frame
from .tickets import validate_ticket

MODEL = "rotation_panorama_v1"
PROJECTIONS = ("spherical", "cylindrical", "plane")


def rotation(value):
    matrix = np.asarray(value, np.float64)
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-3, rtol=0)
        or not 0.999 <= np.linalg.det(matrix) <= 1.001
    ):
        raise ValidationError("Panorama camera rotation must be finite and proper orthonormal")
    return matrix


def intrinsics(value):
    matrix = np.asarray(value, np.float64)
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or matrix[0, 0] < 1e-4
        or matrix[1, 1] < 1e-4
        or not np.allclose(matrix[2], [0, 0, 1])
        or abs(matrix[0, 1]) + abs(matrix[1, 0]) > 1e-7
        or np.max(abs(matrix)) > 1e7
    ):
        raise ValidationError("Use finite non-skewed pinhole panorama intrinsics")
    return matrix


def project_points(points, camera, projection, scale, origin=(0, 0)):
    """Forward equations independent of native warp maps; camera rays -> panorama pixels."""
    points = np.asarray(points, np.float64)
    if points.size == 0:
        points = points.reshape(0, 2)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or len(points) > 100000
        or not np.isfinite(points).all()
        or (points.size and np.max(abs(points)) > 1_000_000)
    ):
        raise ValidationError("Supply up to 100000 finite [x,y] source points")
    K = intrinsics(camera["K"])
    R = rotation(camera["camera_to_panorama"])
    rays = np.c_[points, np.ones(len(points))] @ np.linalg.inv(K).T @ R.T
    x, y, z = rays.T
    if projection == "plane":
        valid = z > 1e-8
        denominator = np.where(valid, z, 1)
        result = np.c_[scale * x / denominator, scale * y / denominator]
    elif projection in ("spherical", "cylindrical"):
        radius = np.hypot(x, z)
        valid = radius > 1e-8
        v = (
            scale * y / np.where(valid, radius, 1)
            if projection == "cylindrical"
            else scale * (np.pi / 2 + np.arctan2(y, radius))
        )
        result = np.c_[scale * np.arctan2(x, z), v]
    else:
        raise ValidationError("Unknown panorama projection")
    return result - np.asarray(origin), valid


def source_boundary(size, samples=24):
    w, h = size
    x = np.linspace(0, w - 1, samples)
    y = np.linspace(0, h - 1, samples)
    return np.concatenate(
        (
            np.c_[x, np.zeros(samples)],
            np.c_[np.full(samples, w - 1), y],
            np.c_[x[::-1], np.full(samples, h - 1)],
            np.c_[np.zeros(samples), y[::-1]],
        )
    )


def layout_frame(cameras, projection, scale, origin, size, anchor_label, block_id):
    document = {
        "model": MODEL,
        "projection": projection,
        "scale": float(scale),
        "origin": list(origin),
        "image_size": list(size),
        "anchor_label": anchor_label,
        "cameras": copy.deepcopy(cameras),
    }

    encoded = json.dumps(document, sort_keys=True, allow_nan=False)
    identity = hashlib.sha256(encoded.encode()).hexdigest()
    document = json.loads(encoded)
    document["spatial"] = frame(size, f"panorama:{block_id}:{identity}")
    return document


def validate_layout(layout):
    if (
        not isinstance(layout, dict)
        or layout.get("model") != MODEL
        or layout.get("projection") not in PROJECTIONS
        or not isinstance(layout.get("cameras"), list)
        or not 2 <= len(layout["cameras"]) <= 8
    ):
        raise ValidationError("Use a rotating panorama layout with 2–8 retained cameras")
    size, origin = layout.get("image_size"), layout.get("origin")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(type(v) is not int or not 1 <= v <= 16384 for v in size)
        or np.prod(size) > 64_000_000
        or not isinstance(origin, list)
        or len(origin) != 2
        or any(type(v) is not int or abs(v) > 1_000_000 for v in origin)
        or type(layout.get("scale")) not in (int, float)
        or not np.isfinite(layout["scale"])
        or not 1 <= layout["scale"] <= 16384
    ):
        raise ValidationError("Invalid bounded panorama canvas or projection scale")
    labels, indices = [], []
    for camera in layout["cameras"]:
        if (
            not isinstance(camera, dict)
            or not isinstance(camera.get("label"), str)
            or not 1 <= len(camera["label"]) <= 1024
            or type(camera.get("index")) is not int
            or not 0 <= camera["index"] <= 7
        ):
            raise ValidationError("Camera records require source labels")
        labels.append(camera["label"])
        indices.append(camera["index"])
        intrinsics(camera.get("K"))
        rotation(camera.get("camera_to_panorama"))
        source = camera.get("source_size")
        if (
            not isinstance(source, list)
            or len(source) != 2
            or any(type(v) is not int or not 8 <= v <= 16384 for v in source)
        ):
            raise ValidationError("Invalid panorama source dimensions")
        if camera.get("source_spatial") is not None:
            check_spatial({"spatial": camera["source_spatial"]}, source)
        if camera.get("source_ticket") is not None:
            validate_ticket(camera["source_ticket"])
    if (
        len(set(labels)) != len(labels)
        or len(set(indices)) != len(indices)
        or layout.get("anchor_label") not in labels
    ):
        raise ValidationError("Camera labels must be unique and include the anchor")
    expected = layout_frame(
        layout["cameras"],
        layout["projection"],
        layout["scale"],
        origin,
        size,
        layout["anchor_label"],
        "",
    )
    spatial = layout.get("spatial")
    if (
        not isinstance(spatial, dict)
        or spatial.get("image_size") != size
        or spatial.get("root_size") != size
        or not isinstance(spatial.get("root_id"), str)
        or not spatial["root_id"].endswith(expected["spatial"]["root_id"].split(":")[-1])
        or spatial.get("to_root") != np.eye(3).tolist()
    ):
        raise ValidationError("Panorama layout geometry/spatial identity changed")
    return layout


def camera_homography(source, target):
    return (
        intrinsics(target["K"])
        @ rotation(target["camera_to_panorama"]).T
        @ rotation(source["camera_to_panorama"])
        @ np.linalg.inv(intrinsics(source["K"]))
    )


def mapped_error(matrix, a, b):
    projected = cv2.perspectiveTransform(a.reshape(-1, 1, 2).astype(np.float64), matrix).reshape(
        -1, 2
    )
    if not np.isfinite(projected).all():
        return float("inf")
    return float(np.sqrt(np.mean(np.sum((projected - b) ** 2, axis=1))))
