"""Detect and qualify multiple native OpenCV planes on one calibrated depth grid."""

import math

import cv2
import numpy as np

from .core import Port, Result
from .geometry3d_common import Tool3D, fit_plane_normal
from .images import DepthArray, ImageArray
from .parameter_validation import NumericRange, ValidationError
from .ptcloud3d import NORMAL_METHODS, depth_intrinsics, organized_normals
from .selection import selection_mask
from .stereo3d import metric_depth, rgbd_function

_INVALID_LABEL = 65535
_MAX_PIXELS = 1_048_576
_MAX_TILES = 4096
_MAX_PLANES = 254
_MAX_XYZ_M = 1e6


def _qualified_plane(rows, xyz, tolerance, minimum):
    """Refit native members in float64 and check the final metric residuals."""
    points = xyz[rows].astype(np.float64)
    for _ in range(3):
        if len(rows) < minimum:
            return None
        fit = fit_plane_normal(points)
        if fit is None:
            return None
        normal, center = fit
        offset = float(-normal @ center)
        if offset < 0 or (offset == 0 and normal[2] > 0):
            normal, offset = -normal, -offset
        residual = np.abs(points @ normal + offset)
        keep = residual <= tolerance
        if keep.all():
            break
        rows, points, residual = rows[keep], points[keep], residual[keep]
    if len(rows) < minimum:
        return None
    return rows, [*normal.tolist(), offset], residual


def _plane_regions(xyz, native_labels, coefficients, valid, tolerance, minimum, cancelled):
    native = native_labels.ravel()
    count = len(coefficients)
    selected = np.flatnonzero(valid.ravel() & (native < count))
    labels = native[selected]
    groups = selected[np.argsort(labels, kind="stable")]
    sizes = np.bincount(labels, minlength=count)
    planes, start = [], 0
    flat_xyz = xyz[..., :3].reshape(-1, 3)
    for native_id, size in enumerate(sizes):
        if cancelled():
            raise InterruptedError("Organized plane detection cancelled")
        end = start + size
        if size >= minimum and np.isfinite(coefficients[native_id]).all():
            result = _qualified_plane(groups[start:end], flat_xyz, tolerance, minimum)
            if result is not None:
                planes.append(result)
        start = end
    # Larger regions get earlier IDs. Pixel order breaks ties independently of native IDs.
    return sorted(planes, key=lambda plane: (-len(plane[0]), plane[0][0]))


def _preview(labels):
    palette = np.zeros((256, 3), np.uint8)
    hsv = np.zeros((1, 256, 3), np.uint8)
    hsv[0, :, 0] = (np.arange(256) * 47) % 180
    hsv[0, :, 1:] = [170, 235]
    palette[:] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0]
    palette[0] = (72, 72, 82)
    palette[255] = (18, 18, 18)
    return palette[np.where(labels == _INVALID_LABEL, 255, labels)]


class FindOrganizedPlanes(Tool3D):
    type_id = "3d.organized_planes"
    title, subtype = "Find planes on a depth grid", "RGB-D"
    description = "Detect multiple metric planes and label their verified members."
    preview_safe = parallel_safe = process_inputs_readonly = True
    payload_reference = "depth"
    inputs = (
        Port("depth", "depth"),
        Port("camera", "data"),
        Port("mask", required=False, visual_type="region"),
        Port("region", "data", required=False, visual_type="region"),
    )
    outputs = (
        Port("labels"),
        Port("planes", "data"),
        Port("valid_mask", visual_type="region"),
        Port("unmatched_mask", visual_type="region"),
        Port("unmatched_depth", "depth"),
        Port("report", "data"),
    )
    defaults = {
        "block_size": 40,
        "minimum_plane_pixels": 1600,
        "threshold_m": 0.01,
        "sensor_error_a": 0.0,
        "sensor_error_b": 0.0,
        "sensor_error_c": 0.0,
        "normal_method": "fals",
        "normal_window": "5",
        "minimum_planes": 1,
        "input_units": "m",
        "units_per_sample": 0.001,
    }
    parameter_choices = {
        "normal_method": ("none", *NORMAL_METHODS),
        "normal_window": ("1", "3", "5", "7"),
        "input_units": ("m", "mm"),
    }
    parameter_ranges = {
        "block_size": NumericRange(4, 256, whole=True),
        "minimum_plane_pixels": NumericRange(3, _MAX_PIXELS, whole=True),
        "threshold_m": NumericRange(1e-6, 1000),
        "minimum_planes": NumericRange(1, _MAX_PLANES, whole=True),
        "units_per_sample": NumericRange(1e-9, 1000),
        **{f"sensor_error_{k}": NumericRange(-1000, 1000) for k in "abc"},
    }
    parameter_help = {
        "block_size": "OpenCV seed tile side length in pixels.",
        "minimum_plane_pixels": "Minimum native and verified support for one plane.",
        "threshold_m": "Native detection distance and final orthogonal residual limit in meters.",
        "sensor_error_a": "OpenCV native quadratic sensor error coefficient for metric Z.",
        "sensor_error_b": "OpenCV native linear sensor error coefficient for metric Z.",
        "sensor_error_c": "OpenCV native constant sensor error term in meters.",
        "normal_method": "Organized OpenCV normal estimator, or geometry-only detection.",
        "normal_window": "Native organized normal window in pixels.",
        "minimum_planes": "Minimum verified planes required to pass.",
        "input_units": "Length units of untagged depth.",
        "units_per_sample": "Length in input units represented by one uint16 count.",
    }
    input_help = {
        "depth": "Positive axial camera Z. Zero, nonfinite and uint16 65535 are invalid.",
        "camera": "Matching undistorted pinhole intrinsics and optical frame.",
        "mask": "Optional raster selection on the source pixel grid.",
        "region": "Optional region intersected with the raster selection.",
    }
    output_help = {
        "labels": "uint16 per-pixel labels. 0 unmatched, 1…254 planes, 65535 invalid or excluded.",
        "planes": "Metric unit-normal equations ax+by+cz+d=0, support and residuals.",
        "valid_mask": "Selected usable depth pixels.",
        "unmatched_mask": "Selected valid pixels without a verified plane.",
        "unmatched_depth": "Metric Z at unmatched pixels, NaN elsewhere.",
        "report": "Optical frame, label meanings, native candidate count and coverage.",
    }

    def process(self, inputs):
        p, depth, camera_model = self.parameters, inputs["depth"], inputs["camera"]
        raw = np.asarray(depth)
        if raw.ndim != 2 or raw.dtype not in (np.uint16, np.float32, np.float64):
            raise ValidationError("Depth must be a uint16, float32 or float64 plane", "depth")
        h, w = raw.shape
        if min(w, h) < 2 or w * h > _MAX_PIXELS:
            raise ValidationError("Depth grid requires 2×2 or more and at most 1,048,576 pixels")
        tile = int(p["block_size"])
        if math.ceil(h / tile) * math.ceil(w / tile) > _MAX_TILES:
            raise ValidationError(
                "Use larger tiles to stay within the native tile budget", "block_size"
            )
        k, size = depth_intrinsics(camera_model, depth)
        camera_frame = camera_model.get("frame", "camera")
        if self.input_payloads["depth"].get("optical_frame", camera_frame) != camera_frame:
            raise ValidationError("Depth optical frame differs from the camera model", "camera")
        with np.errstate(over="ignore", invalid="ignore"):
            values = metric_depth(depth, p["units_per_sample"], p["input_units"])
        if raw.dtype == np.uint16:
            values[raw == 65535] = np.nan
        xyz = rgbd_function("depthTo3d")(values, k)
        # OpenCV 5.0.0 uses Vec4f for both grids in the native plane kernel.
        if xyz.shape != (h, w, 4) or xyz.dtype != np.float32:
            raise RuntimeError("OpenCV depthTo3d must return a four-channel float32 point grid")
        valid = np.all(np.isfinite(xyz[..., :3]) & (np.abs(xyz[..., :3]) <= _MAX_XYZ_M), axis=2)
        valid &= np.isfinite(values)
        valid_depth = int(valid.sum())
        mask = selection_mask(inputs, depth)
        if mask is not None:
            valid &= mask != 0
        xyz[~valid] = np.nan
        valid_count = int(valid.sum())
        minimum = int(p["minimum_plane_pixels"])
        native_count, unreported = 0, 0
        planes = []
        if valid_count >= minimum:
            normals = None
            if p["normal_method"] != "none":
                window = int(p["normal_window"])
                if window > min(w, h):
                    raise ValidationError("Normal window exceeds the depth grid", "normal_window")
                normals = organized_normals(xyz, k, p["normal_method"], window)
                if normals.shape != xyz.shape or normals.dtype != np.float32:
                    raise RuntimeError("OpenCV normals must match the four-channel point grid")
            if self._cancelled():
                raise InterruptedError("Organized plane detection cancelled")
            native_labels, coefficients = cv2.findPlanes(
                xyz,
                normals,
                block_size=tile,
                min_size=minimum,
                threshold=float(p["threshold_m"]),
                sensor_error_a=float(p["sensor_error_a"]),
                sensor_error_b=float(p["sensor_error_b"]),
                sensor_error_c=float(p["sensor_error_c"]),
                method=cv2.RGBD_PLANE_METHOD_DEFAULT,
            )
            if native_labels.shape != (h, w) or native_labels.dtype != np.uint8:
                raise RuntimeError("OpenCV returned an invalid organized plane label map")
            coefficients = np.empty((0, 4)) if coefficients is None else coefficients.reshape(-1, 4)
            native_count = len(coefficients)
            if native_count > _MAX_PLANES:
                raise RuntimeError("OpenCV returned too many organized plane coefficients")
            unreported = int(
                (valid & (native_labels != 255) & (native_labels >= native_count)).sum()
            )
            planes = _plane_regions(
                xyz,
                native_labels,
                coefficients,
                valid,
                float(p["threshold_m"]),
                minimum,
                self._cancelled,
            )
        labels = np.where(valid, 0, _INVALID_LABEL).astype(np.uint16)
        models = []
        for label, (rows, coefficients, residual) in enumerate(planes, start=1):
            labels.ravel()[rows] = label
            models.append(
                {
                    "label": label,
                    "coefficients": coefficients,
                    "support": len(rows),
                    "rmse_m": float(np.sqrt(np.mean(residual**2))),
                    "max_residual_m": float(residual.max()),
                }
            )
        unmatched = labels == 0
        z = DepthArray(np.where(unmatched, values, np.nan), "m")
        report = {
            "frame": camera_frame,
            "units": "m",
            "image_size": list(size),
            "native_planes": native_count,
            "plane_count": len(models),
            "invalid_depth_pixels": h * w - valid_depth,
            "excluded_valid_pixels": valid_depth - valid_count,
            "valid_pixels": valid_count,
            "unmatched_pixels": int(unmatched.sum()),
            "native_labels_without_coefficients": unreported,
            "invalid_label": _INVALID_LABEL,
            "unmatched_label": 0,
        }
        passed = len(models) >= p["minimum_planes"]
        return Result(
            {
                "labels": ImageArray(labels, "gray"),
                "planes": {"frame": camera_frame, "units": "m", "items": models},
                "valid_mask": valid.astype(np.uint8) * 255,
                "unmatched_mask": unmatched.astype(np.uint8) * 255,
                "unmatched_depth": z,
                "report": report,
            },
            passed=passed,
            fail_reasons=[] if passed else ["organized_planes_insufficient_support"],
            graphics={"Plane labels": _preview(labels)} if self.capture_graphics else {},
            message=f"{len(models)} planes · {report['unmatched_pixels']:,} unmatched pixels",
        )
