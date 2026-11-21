"""Calibrated metric reconstruction from native omnidirectional stereo matches."""

import cv2
import numpy as np

from .calibration_revisions import REVISION, applied_payload, digest, is_full_grid
from .core import Port, Result
from .disparity_search import supported_disparity_pixels
from .geometry3d_common import Tool3D, camera_for_image, cloud_graphics
from .images import DepthArray, ImageArray, opencv_image
from .parameter_validation import NumericRange, ValidationError
from .payloads import frame
from .selection import selection_mask
from .stereo3d import cloud_from_grid, validate_rig
from .tickets import ticket_key
from .wide_stereo import geometry as stereo_geometry

_OMNIDIR_GRID = "omnidir_stereo_grid"
_MAX_PIXELS = 4_194_304
_MAX_SEARCH_WORK = 134_217_728
_METERS = {"m": 1.0, "mm": 0.001}
_MODES = {"perspective": "RECTIFY_PERSPECTIVE", "longitude_latitude": "RECTIFY_LONGLATI"}


def _virtual_intrinsics(size, horizontal, vertical):
    width, height = size
    fx = (width - 1) / np.deg2rad(horizontal)
    fy = (height - 1) / np.deg2rad(vertical)
    center = (np.asarray(size) - 1) / 2 - np.pi / 2 * np.asarray([fx, fy])
    return np.array([[fx, 0, center[0]], [0, fy, center[1]], [0, 0, 1]], dtype=np.float64)


def _optical_domain(size, virtual, rotation, xi, angular):
    width, height = size
    x = (np.arange(width, dtype=np.float32) - virtual[0, 2]) / virtual[0, 0]
    y = ((np.arange(height, dtype=np.float32) - virtual[1, 2]) / virtual[1, 1])[:, None]
    a, b, c = rotation[:, 2]
    if angular:
        cosine = -a * np.cos(x) + np.sin(x) * (-b * np.cos(y) + c * np.sin(y))
    else:
        cosine = (a * x + b * y + c) / np.sqrt(1 + x * x + y * y)
    min_cosine = -min(xi, 1 / xi) if xi > 0 else 0.0
    return cosine > min_cosine + 1e-8


def _rectified_support(image, mask, calibrated, model, rotation, virtual, size, mode, block):
    k, d, _ = calibrated
    flag = getattr(cv2.omnidir, _MODES[mode])
    mx, my = cv2.omnidir.initUndistortRectifyMap(
        k, d, np.array([model["xi"]]), rotation, virtual, size, cv2.CV_32FC1, flag
    )
    usable = selection_mask({"mask": mask}, image)
    if usable is None:
        usable = np.full(image.shape[:2], 255, np.uint8)
    warped = cv2.remap(usable, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    domain = _optical_domain(size, virtual, rotation, model["xi"], mode == "longitude_latitude")
    supported = np.uint8((warped == 255) & domain)
    if block > 1:
        supported = cv2.erode(
            supported,
            np.ones((block, block), np.uint8),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return supported != 0


def _triangulate(x, y, disparity, virtual, baseline, mode):
    fx, fy = virtual[0, 0], virtual[1, 1]
    u, v = (x - virtual[0, 2]) / fx, (y - virtual[1, 2]) / fy
    if mode == "perspective":
        z = baseline * fx / disparity
        return np.column_stack((u * z, v * z, z))
    # At a shared latitude, horizontal disparity is the angle between two rays.
    # The sine rule on the baseline triangle gives radial distance along the left ray.
    delta = disparity / fx
    right_angle = u - delta
    radius = baseline * np.sin(right_angle) / np.sin(delta)
    return (
        np.column_stack((-np.cos(u), -np.sin(u) * np.cos(v), np.sin(u) * np.sin(v)))
        * radius[:, None]
    )


def _native_image(source):
    result = opencv_image(source)
    if result.dtype != np.uint8:
        raise ValidationError("Native omnidirectional stereo requires 8-bit input images")
    if result.ndim == 3 and result.shape[2] == 1:
        return result[..., 0]
    if result.ndim == 3 and result.shape[2] == 4:
        return cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)
    return result


class ReconstructOmnidirStereo(Tool3D):
    type_id, title, subtype = (
        "3d.omnidir_stereo_reconstruct",
        "Reconstruct omnidirectional stereo",
        "Stereo",
    )
    inputs = (
        Port("left"),
        Port("right"),
        Port("rig", "data"),
        Port("left_mask", required=False, visual_type="region", required_when_connected=True),
        Port("right_mask", required=False, visual_type="region", required_when_connected=True),
    )
    outputs = (
        Port("left"),
        Port("right"),
        Port("disparity"),
        Port("valid_mask", visual_type="region"),
        Port("depth", "depth"),
        Port("cloud", "cloud"),
        Port("report", "data"),
    )
    defaults = {
        "projection": "perspective",
        "width": 0,
        "height": 0,
        "omnidir_focal_length": 0.0,
        "horizontal_span_degrees": 100.0,
        "vertical_span_degrees": 75.0,
        "num_disparities": 64,
        "block_size": 5,
        "point_type": "xyzrgb",
        "minimum_depth_m": 0.001,
        "maximum_depth_m": 100.0,
        "sample_step": 2,
        "max_points": 200000,
    }
    parameter_choices = {"projection": tuple(_MODES), "point_type": ("xyzrgb", "xyz")}
    parameter_ranges = {
        "width": NumericRange(0, 4096, whole=True),
        "height": NumericRange(0, 4096, whole=True),
        "omnidir_focal_length": NumericRange(0, 32768),
        "horizontal_span_degrees": NumericRange(1, 175),
        "vertical_span_degrees": NumericRange(1, 175),
        "num_disparities": NumericRange(16, 512, whole=True),
        "block_size": NumericRange(1, 31, whole=True, odd=True),
        "minimum_depth_m": NumericRange(0.000001, 1000),
        "maximum_depth_m": NumericRange(0.000001, 1000),
        "sample_step": NumericRange(1, 256, whole=True),
        "max_points": NumericRange(1, 2_000_000, whole=True),
    }
    parameter_help = {
        "projection": "Perspective pixels or a longitude/latitude angular grid.",
        "width": "Rectified width. Zero uses the input width.",
        "height": "Rectified height. Zero uses the input height.",
        "omnidir_focal_length": "Perspective focal length in pixels. Zero derives it from the rig.",
        "horizontal_span_degrees": "Longitude span for angular rectification.",
        "vertical_span_degrees": "Latitude span for angular rectification.",
        "num_disparities": "Native SGBM search width, in positive multiples of 16 pixels.",
        "block_size": "Odd native SGBM matching window and required support footprint.",
        "point_type": "Emit XYZ, or XYZ with color sampled from the rectified left view.",
        "minimum_depth_m": "Nearest accepted rectified camera-Z in meters.",
        "maximum_depth_m": "Farthest accepted rectified camera-Z in meters.",
        "sample_step": "Pixel pitch for cloud samples. Depth and disparity use the full grid.",
        "max_points": "Maximum cloud size. Increase Sample Step for bigger outputs.",
    }
    input_help = {
        "left": "Raw 8-bit capture on the calibrated left omnidirectional camera grid.",
        "right": "Matching 8-bit capture on the calibrated right camera grid.",
        "rig": "Metric stereo rig with two calibrated Mei omnidirectional cameras.",
        "left_mask": "Optional uint8 mask on the raw left camera grid.",
        "right_mask": "Optional uint8 mask on the raw right camera grid.",
    }
    payload_reference = "left"
    process_inputs_readonly = parallel_safe = preview_safe = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if int(p["num_disparities"]) % 16:
            raise ValidationError("Search width must be a multiple of 16", "num_disparities")
        if float(p["minimum_depth_m"]) >= float(p["maximum_depth_m"]):
            raise ValidationError(
                "Set increasing depth limits", "minimum_depth_m", "maximum_depth_m"
            )

    def process(self, inputs):
        p, rig = self.parameters, inputs["rig"]
        cam_left, cam_right, pose = validate_rig(rig)
        if rig["left"].get("lens") != "omnidir" or rig["units"] not in _METERS:
            raise ValidationError("Use two Mei omnidirectional cameras in a metric rig")
        if rig["left"]["frame"] == rig["right"]["frame"]:
            raise ValidationError("Left and right camera frames must have distinct names")
        raw_size = cam_left[2]
        if np.prod(raw_size) > _MAX_PIXELS:
            raise ValidationError("Raw stereo images are limited to 4,194,304 pixels per view")
        for side in ("left", "right"):
            camera_for_image(rig[side], inputs[side])
            context = self.input_payloads[side]
            if context.get("optical_frame", rig[side]["frame"]) != rig[side]["frame"] or (
                "spatial" in context and not is_full_grid(context, raw_size)
            ):
                raise ValidationError(
                    f"{side.title()} image must use the raw calibrated camera grid"
                )
        tickets = [self.input_payloads[name].get("ticket") for name in self.input_payloads]
        if len({ticket_key(ticket) for ticket in tickets if ticket is not None}) > 1:
            raise ValidationError("Stereo captures and masks must share the same ticket")

        size = int(p["width"]) or raw_size[0], int(p["height"]) or raw_size[1]
        disparities, block = int(p["num_disparities"]), int(p["block_size"])
        if size[0] <= disparities + block or size[1] < max(2, block):
            raise ValidationError("Rectified image is too small for the matching window and search")
        if np.prod(size) > _MAX_PIXELS or np.prod(size) * disparities > _MAX_SEARCH_WORK:
            raise ValidationError(
                "Stereo search exceeds the work limit. Reduce output size or search."
            )
        pose = pose.copy()
        pose[:3, 3] *= _METERS[rig["units"]]
        baseline = float(np.linalg.norm(pose[:3, 3]))
        r1, r2, perspective, _, _, _, _ = stereo_geometry(
            rig,
            (cam_left, cam_right),
            pose,
            size,
            {"zero_disparity": True, "omnidir_focal_length": float(p["omnidir_focal_length"])},
        )
        mode = p["projection"]
        virtual = perspective[:, :3].copy()
        if mode == "longitude_latitude":
            virtual = _virtual_intrinsics(
                size,
                float(p["horizontal_span_degrees"]),
                float(p["vertical_span_degrees"]),
            )
        a, b = (_native_image(inputs[side]) for side in ("left", "right"))
        if a.ndim != b.ndim:
            a, b = (
                cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
                for image in (a, b)
            )
        k1, d1, _ = cam_left
        k2, d2, _ = cam_right
        native_d, left, right, compact = cv2.omnidir.stereoReconstruct(
            a,
            b,
            k1,
            d1,
            np.array([rig["left"]["xi"]]),
            k2,
            d2,
            np.array([rig["right"]["xi"]]),
            pose[:3, :3],
            pose[:3, 3],
            getattr(cv2.omnidir, _MODES[mode]),
            disparities,
            block,
            newSize=size,
            Knew=virtual,
            pointType=cv2.omnidir.XYZ,
        )
        if native_d is None or native_d.dtype != np.float32 or native_d.shape != size[::-1]:
            raise ValidationError("Native stereo did not return its expected float pixel disparity")
        if (
            left is None
            or right is None
            or left.shape[:2] != size[::-1]
            or right.shape[:2] != size[::-1]
        ):
            raise ValidationError("Native stereo returned an inconsistent rectified image grid")
        allowed_left, allowed_right = (
            _rectified_support(
                inputs[side],
                inputs.get(f"{side}_mask"),
                calibrated,
                rig[side],
                rotation,
                virtual,
                size,
                mode,
                block,
            )
            for side, calibrated, rotation in (("left", cam_left, r1), ("right", cam_right, r2))
        )
        positive = np.isfinite(native_d) & (native_d > 0)
        ids = supported_disparity_pixels(native_d, positive & allowed_left, allowed_right)
        y, x = np.divmod(ids, size[0])
        points = _triangulate(x, y, native_d.flat[ids].astype(np.float64), virtual, baseline, mode)
        low, high = float(p["minimum_depth_m"]), float(p["maximum_depth_m"])
        keep = np.isfinite(points).all(axis=1) & (points[:, 2] >= low) & (points[:, 2] <= high)
        xyz = np.full((*size[::-1], 3), np.nan, np.float32)
        xyz.reshape(-1, 3)[ids[keep]] = points[keep]
        valid = np.isfinite(xyz[..., 2])
        depth = DepthArray(xyz[..., 2].copy(), "m", low, high)
        left, right = (
            ImageArray(image, "gray" if image.ndim == 2 else "bgr") for image in (left, right)
        )

        rectification = {
            "model": "omnidir_stereo_rectification_v1",
            "projection": mode,
            "image_size": list(size),
            "virtual_matrix": virtual.tolist(),
            "left_rotation": r1.tolist(),
            "right_rotation": r2.tolist(),
            "left_frame": f"{rig['left']['frame']}:rectified",
            "right_frame": f"{rig['right']['frame']}:rectified",
            "baseline_m": baseline,
            "rig_sha256": digest(rig),
        }
        grid = digest(rectification)
        payloads = {}
        for side in ("left", "right"):
            context = dict(self.input_payloads[side])
            original = context.get("spatial")
            context["spatial"] = frame(size, f"{self.id}:omnidir:{side}:{grid[:12]}")
            context["optical_frame"] = rectification[f"{side}_frame"]
            context.pop("pyramid", None)
            applied_payload(context, rig[side].get(REVISION))
            context[_OMNIDIR_GRID] = {
                "sha256": grid,
                "side": side,
                "projection": mode,
                "source": original,
            }
            payloads[side] = context
        cloud = cloud_from_grid(
            xyz,
            "m",
            rectification["left_frame"],
            image=left if p["point_type"] == "xyzrgb" else None,
            step=int(p["sample_step"]),
            maximum=int(p["max_points"]),
            payload=payloads["left"],
        )
        count = int(np.count_nonzero(valid))
        report = {
            **rectification,
            "model": "omnidir_stereo_reconstruction_v1",
            "input_rig_units": rig["units"],
            "units": "m",
            "native_disparity_units": "px",
            "native_disparity_scale": 1.0,
            "native_positive_pixels": int(np.count_nonzero(positive)),
            "geometrically_supported_pixels": len(ids),
            "valid_pixels": count,
            "valid_at_or_below_15_px": int(np.count_nonzero(valid & (native_d <= 15))),
            "native_cloud_points": 0 if compact is None else len(compact.reshape(-1, 3)),
            "cloud_points": len(cloud.points),
        }
        payloads.update(
            {
                name: payloads["left"]
                for name in ("depth", "cloud", "disparity", "valid_mask", "report")
            }
        )
        return Result(
            {
                "left": left,
                "right": right,
                "disparity": ImageArray(np.where(valid, native_d, np.nan), "gray"),
                "valid_mask": ImageArray(np.uint8(valid) * 255, "gray"),
                "depth": depth,
                "cloud": cloud,
                "report": report,
            },
            payloads=payloads,
            graphics=cloud_graphics(self, cloud) if len(cloud.points) else {},
            passed=bool(len(cloud.points)),
            score=count / native_d.size,
            fail_reasons=[] if len(cloud.points) else ["omnidir_stereo_no_supported_points"],
            message=f"{len(cloud.points)} cloud samples from {count} valid disparity pixels",
        )
