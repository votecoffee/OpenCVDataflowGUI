"""Camera and transform contracts shared by OpenCV 3D wrappers."""

import copy

import cv2
import numpy as np

from .calibration_revisions import REVISION, camera_binding, check_capture
from .core import Block
from .payloads import metadata
from .processing import finite, integer


class Tool3D(Block):
    category, block_type, subtype = "3D vision", "3D vision", "Geometry"


def plane_sample_normal(a, b, c):
    """Return the unit normal of a noncollinear three-point sample."""
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    length = np.linalg.norm(normal)
    if length <= 1e-8 * np.linalg.norm(ab) * np.linalg.norm(ac):
        return None
    return normal / length


def fit_plane_normal(points):
    """Return the orthogonal least-squares normal and a point on the plane."""
    center = points.mean(axis=0)
    _, singular, directions = np.linalg.svd(points - center, full_matrices=False)
    if singular[1] <= 1e-8 * singular[0]:
        return None
    return directions[-1], center


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
    camera_binding(value)
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


def project_camera_pixels(xyz, model, k, d, rvec, translation):
    """Project using the declared lens family and return an Nx2 pixel array."""
    lens = model.get("lens", "pinhole")
    if lens == "fisheye":
        pixels, _ = cv2.fisheye.projectPoints(xyz[:, None], rvec, translation, k, d)
    elif lens == "omnidir":
        pixels, _ = cv2.omnidir.projectPoints(xyz[:, None], rvec, translation, k, model["xi"], d)
    else:
        pixels, _ = cv2.projectPoints(xyz, rvec, translation, k, d)
    return pixels.reshape(-1, 2)


def undistort_camera_pixels(xy, model, k, d, projection=None):
    """Return normalized or pinhole-projected rays in an Nx2 array."""
    xy = xy.reshape(-1, 1, 2)
    lens = model.get("lens", "pinhole")
    if lens == "fisheye":
        result = cv2.fisheye.undistortPoints(xy, k, d, P=projection)
    elif lens == "omnidir":
        result = cv2.omnidir.undistortPoints(xy, k, d, np.array([model["xi"]]), np.eye(3))
        if projection is not None:
            result = cv2.perspectiveTransform(result, projection)
    else:
        result = cv2.undistortPoints(xy, k, d, P=projection)
    return result.reshape(-1, 2)


def stereo_cameras(left, right):
    a, b = camera(left), camera(right)
    lens = left.get("lens", "pinhole")
    if (
        a[2] != b[2]
        or left["units"] != right["units"]
        or right.get("lens", "pinhole") != lens
    ):
        raise ValueError("Stereo cameras must share lens family, image size and length units")
    if lens == "omnidir" and min(left["xi"], right["xi"]) < 0:
        raise ValueError("Omnidirectional stereo requires nonnegative xi")
    return a, b, lens


def require_pinhole(model):
    k, d, size = camera(model)
    if model.get("lens", "pinhole") != "pinhole":
        raise ValueError("Use a pinhole model or rectify fisheye/omnidirectional inputs first")
    return k, d, size


def cloud_graphics(block, cloud):
    if not block.capture_graphics:
        return {}
    from .pointclouds import render_cloud

    return {"point cloud": render_cloud(cloud)}


def camera_for_image(model, image):
    k, d, size = camera(model)
    if size != (image.shape[1], image.shape[0]):
        raise ValueError("Camera image size differs from input; supply scaled/rectified intrinsics")
    check_capture(model.get(REVISION), metadata(image), size)
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
