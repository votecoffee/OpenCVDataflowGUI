"""Camera and transform contracts shared by OpenCV 3D wrappers."""

import copy

import cv2
import numpy as np

from .core import Block
from .processing import finite, integer


class Tool3D(Block):
    category, block_type, subtype = "3D vision", "3D vision", "Geometry"


def matrix(value, shape, name="Matrix"):
    value = np.asarray(value, np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite {shape} matrix")
    return np.ascontiguousarray(value)


def points(value, dimensions=3, minimum=1):
    if hasattr(value, "points"):
        value = value.points
    value = np.asarray(value, np.float64)
    if (
        value.ndim != 2
        or value.shape[1] != dimensions
        or not minimum <= len(value) <= 1_000_000
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"Expected {minimum}..1,000,000 finite {dimensions}D points")
    return np.ascontiguousarray(value)


def image_size(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Image size must be [width,height]")
    return tuple(integer(v, "Image dimension", 1, 16384) for v in value)


def camera(value):
    if not isinstance(value, dict):
        raise ValueError("Connect a camera model")
    k = matrix(value.get("camera_matrix", value.get("matrix")), (3, 3), "Camera matrix")
    if (
        k[0, 0] <= 0
        or k[1, 1] <= 0
        or not np.allclose(k[2], [0, 0, 1])
        or abs(k[0, 1]) > 1e-9
        or abs(k[1, 0]) > 1e-9
    ):
        raise ValueError("Camera requires positive focal lengths, zero skew and [0,0,1] last row")
    lens = value.get("lens", "pinhole")
    d = np.asarray(value.get("distortion", []), np.float64).reshape(-1)
    if (
        lens not in ("pinhole", "fisheye", "omnidir")
        or not np.isfinite(d).all()
        or (lens == "pinhole" and len(d) not in (0, 4, 5, 8, 12, 14))
        or (lens != "pinhole" and len(d) != 4)
    ):
        raise ValueError("Invalid camera lens/distortion coefficients")
    size = image_size(value.get("image_size"))
    units = value.get("units", "m")
    if units not in ("m", "mm", "unitless"):
        raise ValueError("Camera units must be m/mm/unitless")
    frame = value.get("frame", "camera")
    if not isinstance(frame, str) or not frame:
        raise ValueError("Camera frame must be named")
    if lens == "omnidir":
        finite(value.get("xi"), "Omnidirectional xi")
    return k, d, size


def camera_record(k, d, size, units="m", frame="camera", lens="pinhole", **extra):
    value = {
        "model": "camera_model_v1",
        "camera_matrix": np.asarray(k).tolist(),
        "distortion": np.asarray(d).reshape(-1).tolist(),
        "image_size": list(size),
        "units": units,
        "frame": frame,
        "lens": lens,
        **extra,
    }
    camera(value)
    return value


def rigid(value):
    value = value.get("source_to_target") if isinstance(value, dict) else value
    transform = matrix(value, (4, 4), "3D transform")
    if (
        not np.allclose(transform[3], [0, 0, 0, 1])
        or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-5)
    ):
        raise ValueError("Rigid 3D transforms require proper orthonormal rotation")
    return transform


def pose_record(transform, source="object", target="camera", units="m", **extra):
    transform = rigid(transform)
    if units not in ("m", "mm", "unitless") or not all(
        isinstance(v, str) and v for v in (source, target)
    ):
        raise ValueError("A 3D pose requires explicit units and named source/target frames")
    return {
        "model": "pose_3d_v1",
        "source_to_target": transform.tolist(),
        "source_frame": source,
        "target_frame": target,
        "units": units,
        "rvec": cv2.Rodrigues(transform[:3, :3])[0].reshape(-1).tolist(),
        "translation": transform[:3, 3].tolist(),
        **extra,
    }


def pose_matrix(rotation, translation):
    result = np.eye(4)
    result[:3, :3] = matrix(rotation, (3, 3), "Rotation")
    result[:3, 3] = np.asarray(translation).reshape(3)
    return rigid(result)


def require_pinhole(model):
    k, d, size = camera(model)
    if model.get("lens", "pinhole") != "pinhole":
        raise ValueError("Use a pinhole model or rectify fisheye/omnidirectional inputs first")
    return k, d, size


def cloud_graphics(block, cloud):
    if not block.run_profile.graphics:
        return {}
    from .pointclouds import render_cloud

    return {"point cloud": render_cloud(cloud)}


def camera_for_image(model, image):
    k, d, size = camera(model)
    if size != (image.shape[1], image.shape[0]):
        raise ValueError("Camera image size differs from input; supply scaled/rectified intrinsics")
    return k, d, size


def transformed_cloud(cloud, transform, target_frame):
    from .pointclouds import PointCloud

    xyz = (
        cv2.perspectiveTransform(cloud.points.reshape(-1, 1, 3), rigid(transform)).reshape(-1, 3)
        if len(cloud.points)
        else cloud.points.copy()
    )
    normals = cloud.normals @ rigid(transform)[:3, :3].T if cloud.normals is not None else None
    return PointCloud(
        xyz,
        cloud.units,
        target_frame,
        copy.deepcopy(cloud.colors),
        normals,
        copy.deepcopy(cloud.pixels),
        cloud.source_size,
        copy.deepcopy(cloud.metadata),
    )
