"""Matched RGB/absolute-coordinate crops for the published single-object configuration.

Adapted from GDRNPP 990003d core/utils/data_utils.py and
core/gdrn_modeling/datasets/data_loader.py (Apache-2.0). Modifications: bounded
standalone NumPy/OpenCV interface, no dataset registry or Detectron2 runtime.
"""

import hashlib
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Crop:
    image: np.ndarray
    coordinates: np.ndarray
    camera: np.ndarray
    center: np.ndarray
    box_size: np.ndarray
    scale: float
    source_from_input: np.ndarray
    source_from_output: np.ndarray
    source_box: tuple
    source_size: tuple
    working_size: tuple


def affine(center, scale, output):
    # Retain the reference implementation's float32 control-point rounding.
    center = np.asarray(center, np.float32)
    source = np.array(
        [center, center + [0, -scale / 2], center + [-scale / 2, -scale / 2]], np.float32
    )
    target = np.array([[output / 2, output / 2], [output / 2, 0], [0, 0]], np.float32)
    return cv2.getAffineTransform(source, target)


def validate_frame(rgb, camera):
    rgb = np.asarray(rgb)
    k = np.asarray(camera, np.float64)
    if (
        rgb.dtype != np.uint8
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        or min(rgb.shape[:2]) < 2
        or rgb.shape[0] * rgb.shape[1] > 16_000_000
    ):
        raise ValueError("Provide a bounded uint8 RGB image")
    if (
        k.shape != (3, 3)
        or not np.isfinite(k).all()
        or min(k[0, 0], k[1, 1]) <= 0
        or not np.allclose(k[2], [0, 0, 1])
        or abs(k[0, 1]) > 1e-10
        or abs(k[1, 0]) > 1e-10
    ):
        raise ValueError("Provide rectified pinhole intrinsics with positive focal lengths")
    return rgb, k


def frame_signature(rgb, camera):
    rgb, camera = validate_frame(rgb, camera)
    return dict(
        rgb_sha256=hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest(),
        camera_matrix=camera.tolist(),
        source_size=[rgb.shape[1], rgb.shape[0]],
    )


def validate_box(box, *, width, height):
    bounds = np.asarray(box, np.float64)
    if (
        bounds.shape != (4,)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:2] < 0)
        or np.any(bounds[2:] > [width, height])
        or np.any(bounds[2:] - bounds[:2] < 1)
    ):
        raise ValueError(
            "Use a nonempty clipped source-image XYXY box, with exclusive upper bounds"
        )
    return bounds


def prepare_crop(rgb, camera, box):
    rgb, k = validate_frame(rgb, camera)
    height, width = rgb.shape[:2]
    bounds = validate_box(box, width=width, height=height)
    ratio = min(540 / min(height, width), 720 / max(height, width))
    working = (int(width * ratio + 0.5), int(height * ratio + 0.5))
    sx, sy = working[0] / width, working[1] / height
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    if working != (width, height):
        bgr = cv2.resize(bgr, working, interpolation=cv2.INTER_LINEAR)
    k = k.copy()
    k[0] *= sx
    k[1] *= sy
    adjusted = bounds * [sx, sy, sx, sy]
    center = ((adjusted[:2] + adjusted[2:]) / 2).astype(np.float32)
    size = (adjusted[2:] - adjusted[:2]).astype(np.float32)
    scale = float(min(max(size) * 1.5, max(working)))
    input_affine, output_affine = affine(center, scale, 256), affine(center, scale, 64)
    image = cv2.warpAffine(bgr, input_affine, (256, 256), flags=cv2.INTER_LINEAR)
    x = np.linspace(0, 1, working[0], endpoint=False, dtype=np.float32)
    y = np.linspace(0, 1, working[1], endpoint=False, dtype=np.float32)
    coordinates = np.stack(np.meshgrid(x, y), axis=-1)
    coordinates = cv2.warpAffine(coordinates, output_affine, (64, 64), flags=cv2.INTER_LINEAR)
    source_from_working = np.diag([1 / sx, 1 / sy, 1])

    def inverse(matrix):
        return source_from_working @ np.linalg.inv(np.vstack([matrix, [0, 0, 1]]))

    return Crop(
        np.ascontiguousarray(image.transpose(2, 0, 1), np.float32) / 255,
        np.ascontiguousarray(coordinates.transpose(2, 0, 1)),
        k,
        center,
        size,
        scale,
        inverse(input_affine),
        inverse(output_affine),
        tuple(bounds.tolist()),
        (width, height),
        working,
    )
