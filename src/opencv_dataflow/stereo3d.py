"""Stereo calibration/rectification/disparity and physical RGB-D reconstruction."""

import copy

import cv2
import numpy as np

from .core import Port, Result
from .geometry3d_common import (
    Tool3D,
    camera_for_image,
    camera_record,
    cloud_graphics,
    image_size,
    matrix,
    points,
    pose_matrix,
    pose_record,
    require_pinhole,
    rigid,
)
from .images import DepthArray, ImageArray, image_format, preview_rgb
from .parameter_validation import NumericRange
from .payloads import NO_VALUE, frame, metadata
from .pointclouds import PointCloud
from .processing import finite, gray_image, integer
from .selection import selection_mask


def rgbd_function(name):
    function = getattr(cv2, name, None) or getattr(getattr(cv2, "rgbd", None), name, None)
    if function is None:
        raise RuntimeError(f"This OpenCV build does not expose RGB-D {name}")
    return function


def metric_depth(depth, scale, units):
    values = np.asarray(depth, np.float32).copy()
    source_units = getattr(depth, "units", units)
    if source_units not in ("m", "mm"):
        raise ValueError("Depth units must be m or mm")
    if np.issubdtype(depth.dtype, np.integer):
        values *= finite(scale, "Depth units per sample", 0.000000001)
    if source_units == "mm":
        values *= 0.001
    values[(values <= 0) | ~np.isfinite(values)] = np.nan
    return values


def cloud_from_grid(
    xyz, units, target_frame, mask=None, image=None, step=1, maximum=200000, payload=None
):
    h, w = xyz.shape[:2]
    yy, xx = np.mgrid[:h:step, :w:step]
    sampled = xyz[::step, ::step, :3]
    valid = np.isfinite(sampled).all(axis=2)
    if mask is not None:
        valid &= mask[::step, ::step] != 0
    pixels = np.c_[xx[valid], yy[valid]].astype(np.int32)
    values = sampled[valid]
    if len(values) > maximum:
        raise ValueError(
            "Cloud exceeds max_points; increase sample_step or reduce the selected region"
        )
    colors = preview_rgb(image)[pixels[:, 1], pixels[:, 0]] if image is not None else None
    return PointCloud(
        values,
        units,
        target_frame,
        colors=colors,
        pixels=pixels,
        source_size=(w, h),
        metadata=copy.deepcopy(payload or {}),
    )


class StereoRig(Tool3D):
    type_id, title, subtype = "3d.stereo_rig", "Define stereo rig", "Stereo"
    inputs = (Port("left_camera", "data"), Port("right_camera", "data"), Port("transform", "data"))
    outputs = (Port("rig", "data"),)

    def process(self, inputs):
        left, right, pose = (inputs[k] for k in ("left_camera", "right_camera", "transform"))
        _, _, size = require_pinhole(left)
        if require_pinhole(right)[2] != size or left["units"] != right["units"]:
            raise ValueError("Stereo cameras must have matching image size and length units")
        if (
            pose["source_frame"] != left["frame"]
            or pose["target_frame"] != right["frame"]
            or pose["units"] != left["units"]
        ):
            raise ValueError(
                "Stereo transform must map left camera to right camera in matching units"
            )
        if np.linalg.norm(rigid(pose)[:3, 3]) <= 1e-10:
            raise ValueError("Stereo baseline must be nonzero")
        return Result(
            {
                "rig": {
                    "model": "stereo_rig_v1",
                    "left": left,
                    "right": right,
                    "transform": pose,
                    "image_size": list(size),
                    "units": left["units"],
                }
            }
        )


def validate_rig(rig):
    if not isinstance(rig, dict) or rig.get("model") != "stereo_rig_v1":
        raise ValueError("Connect a calibrated/defined stereo rig")
    tool = StereoRig()
    tool.process(
        {"left_camera": rig["left"], "right_camera": rig["right"], "transform": rig["transform"]}
    )
    return require_pinhole(rig["left"]), require_pinhole(rig["right"]), rigid(rig["transform"])


class CalibrateStereo(Tool3D):
    type_id, title, subtype = "3d.stereo_calibrate", "Calibrate stereo cameras", "Calibration"
    inputs = (
        Port("object_views", "data"),
        Port("left_views", "data"),
        Port("right_views", "data"),
        Port("left_camera", "data"),
        Port("right_camera", "data"),
    )
    outputs = (
        Port("rig", "data"),
        Port("rms", "number"),
        Port("fundamental", "data"),
        Port("essential", "data"),
    )
    defaults = {
        "fix_intrinsics": True,
        "same_focal_length": False,
        "zero_tangent": False,
        "fix_principal_point": False,
        "fix_aspect_ratio": False,
        "rational_model": False,
        "fix_k1": False,
        "fix_k2": False,
        "fix_k3": False,
        "fix_k4": False,
        "fix_k5": False,
        "fix_k6": False,
        "iterations": 100,
        "epsilon": 1e-6,
        "max_rms": 2.0,
    }

    def process(self, inputs):
        left, right = inputs["left_camera"], inputs["right_camera"]
        k1, d1, size = require_pinhole(left)
        k2, d2, size2 = require_pinhole(right)
        if size != size2 or left["units"] != right["units"]:
            raise ValueError("Stereo camera sizes/units must match")
        groups = [inputs[k] for k in ("object_views", "left_views", "right_views")]
        if (
            any(not isinstance(g, list) for g in groups)
            or not len(groups[0]) >= 3
            or len({len(g) for g in groups}) != 1
        ):
            raise ValueError(
                "Stereo calibration needs at least three matched object/left/right views"
            )
        object_views, a, b = [], [], []
        for obj, xy_left, xy_right in zip(*groups):
            obj, xy_left, xy_right = (
                points(obj, 3, 4),
                points(xy_left, 2, 4),
                points(xy_right, 2, 4),
            )
            if not len(obj) == len(xy_left) == len(xy_right):
                raise ValueError("Stereo calibration point counts differ within a view")
            object_views.append(obj.astype(np.float32))
            a.append(xy_left.astype(np.float32))
            b.append(xy_right.astype(np.float32))
        p = self.parameters
        flags = cv2.CALIB_FIX_INTRINSIC if p["fix_intrinsics"] else cv2.CALIB_USE_INTRINSIC_GUESS
        for key, flag in {
            "same_focal_length": cv2.CALIB_SAME_FOCAL_LENGTH,
            "zero_tangent": cv2.CALIB_ZERO_TANGENT_DIST,
            "fix_principal_point": cv2.CALIB_FIX_PRINCIPAL_POINT,
            "fix_aspect_ratio": cv2.CALIB_FIX_ASPECT_RATIO,
            "rational_model": cv2.CALIB_RATIONAL_MODEL,
            **{f"fix_k{i}": getattr(cv2, f"CALIB_FIX_K{i}") for i in range(1, 7)},
        }.items():
            if p[key]:
                flags |= flag
        values = cv2.stereoCalibrate(
            object_views,
            a,
            b,
            k1,
            d1,
            k2,
            d2,
            size,
            flags=flags,
            criteria=(
                cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
                integer(p["iterations"], "Iterations", 1, 10000),
                finite(p["epsilon"], "Epsilon", 0),
            ),
        )
        rms, k1, d1, k2, d2, r, t, e, f = values
        left = camera_record(k1, d1, size, left["units"], left["frame"])
        right = camera_record(k2, d2, size, right["units"], right["frame"])
        pose = pose_record(pose_matrix(r, t), left["frame"], right["frame"], left["units"])
        rig = (
            StereoRig()
            .process({"left_camera": left, "right_camera": right, "transform": pose})
            .outputs["rig"]
        )
        passed = rms <= finite(p["max_rms"], "Maximum RMS", 0)
        return Result(
            {"rig": rig, "rms": float(rms), "essential": e.tolist(), "fundamental": f.tolist()},
            passed=passed,
            fail_reasons=[] if passed else ["stereo_calibration_rms"],
        )


class RectifyStereo(Tool3D):
    type_id, title, subtype = "3d.stereo_rectify", "Rectify stereo images", "Stereo"
    inputs = (Port("left"), Port("right"), Port("rig", "data"))
    outputs = (Port("left"), Port("right"), Port("rectification", "data"))
    defaults = {
        "alpha": 0.0,
        "zero_disparity": True,
        "width": 0,
        "height": 0,
        "interpolation": "linear",
        "border_value": 0.0,
    }
    parameter_choices = {"interpolation": ("nearest", "linear", "cubic", "lanczos4")}

    def process(self, inputs):
        rig, p = inputs["rig"], self.parameters
        (k1, d1, size), (k2, d2, _), pose = validate_rig(rig)
        for image in (inputs["left"], inputs["right"]):
            if image.shape[:2] != (size[1], size[0]):
                raise ValueError("Stereo image sizes must match calibrated sizes")
        alpha = finite(p["alpha"], "Alpha")
        if not -1 <= alpha <= 1:
            raise ValueError("Rectification alpha must be in [-1,1]")
        output_size = (
            integer(p["width"], "Width", 0, 4096) or size[0],
            integer(p["height"], "Height", 0, 4096) or size[1],
        )
        r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(
            k1,
            d1,
            k2,
            d2,
            size,
            pose[:3, :3].copy(),
            pose[:3, 3].copy().reshape(3, 1),
            flags=cv2.CALIB_ZERO_DISPARITY if p["zero_disparity"] else 0,
            alpha=alpha,
            newImageSize=output_size,
        )
        result, payloads = {}, {}
        for name, k, d, r, projection in (("left", k1, d1, r1, p1), ("right", k2, d2, r2, p2)):
            mx, my = cv2.initUndistortRectifyMap(k, d, r, projection, output_size, cv2.CV_32FC1)
            result[name] = cv2.remap(
                np.asarray(inputs[name]),
                mx,
                my,
                {
                    "nearest": cv2.INTER_NEAREST,
                    "linear": cv2.INTER_LINEAR,
                    "cubic": cv2.INTER_CUBIC,
                    "lanczos4": cv2.INTER_LANCZOS4,
                }[p["interpolation"]],
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=finite(p["border_value"], "Border value"),
            )
            result[name] = ImageArray(result[name], image_format(inputs[name]))
            payloads[name] = {
                **metadata(inputs[name]),
                "spatial": frame(output_size, f"{self.id}:rectified:{name}"),
            }
            payloads[name].pop("pyramid", None)
        result["rectification"] = {
            "model": "stereo_rectification_v1",
            "Q": q.tolist(),
            "P1": p1.tolist(),
            "P2": p2.tolist(),
            "R1": r1.tolist(),
            "R2": r2.tolist(),
            "roi_left": list(roi1),
            "roi_right": list(roi2),
            "image_size": list(output_size),
            "units": rig["units"],
            "frame": f"{rig['left']['frame']}:rectified",
        }
        return Result(result, payloads=payloads)


class StereoDisparity(Tool3D):
    type_id, title, subtype = "3d.disparity", "Stereo BM / SGBM disparity", "Stereo"
    inputs = (Port("left"), Port("right"))
    outputs = (
        Port("disparity"),
        Port("valid_mask"),
        Port("preview"),
        Port("confidence", required=False),
    )
    defaults = {
        "algorithm": "sgbm",
        "min_disparity": 0,
        "num_disparities": 64,
        "block_size": 5,
        "p1": 0,
        "p2": 0,
        "mode": "sgbm_3way",
        "uniqueness_ratio": 10,
        "speckle_window": 0,
        "speckle_range": 2,
        "disp12_max_diff": 1,
        "prefilter_cap": 31,
        "prefilter_size": 9,
        "prefilter_type": "xsobel",
        "texture_threshold": 10,
        "wls": False,
        "wls_lambda": 8000.0,
        "wls_sigma": 1.5,
        "wls_lr_threshold": 24,
        "wls_depth_radius": -1,
        "minimum_wls_confidence": 0.0,
    }
    parameter_ranges = {
        "wls_lr_threshold": NumericRange(0, 4096, whole=True),
        "wls_depth_radius": NumericRange(-1, 255, whole=True),
        "minimum_wls_confidence": NumericRange(0, 1),
    }
    parameter_help = {
        "wls_lr_threshold": "WLS left-right consistency tolerance in sixteenths of a pixel.",
        "wls_depth_radius": "Radius near depth jumps. -1 uses the native default.",
        "minimum_wls_confidence": "Reject WLS estimates below this value. Zero disables gating.",
    }
    parameter_choices = {
        "algorithm": ("bm", "sgbm"),
        "mode": ("sgbm", "sgbm_3way", "hh", "hh4"),
        "prefilter_type": ("xsobel", "normalized"),
    }

    def process(self, inputs):
        a, b = (gray_image(inputs[n], (np.uint8,)) for n in ("left", "right"))
        if a.shape != b.shape:
            raise ValueError("Rectified stereo image dimensions must match")
        p = self.parameters
        disparities = integer(p["num_disparities"], "Number disparities", 16, 4096)
        if disparities % 16 or disparities >= a.shape[1]:
            raise ValueError("num_disparities must be a multiple of 16 smaller than image width")
        block_size = integer(
            p["block_size"], "Block size", 5 if p["algorithm"] == "bm" else 1, 255, odd=True
        )
        if block_size >= min(a.shape):
            raise ValueError("Stereo block must fit inside the image")
        minimum = integer(p["min_disparity"], "Minimum disparity", -4096, 4096)
        if p["algorithm"] == "bm":
            matcher = cv2.StereoBM_create(disparities, block_size)
            matcher.setTextureThreshold(
                integer(p["texture_threshold"], "Texture threshold", 0, 10000)
            )
            matcher.setPreFilterSize(
                integer(p["prefilter_size"], "Prefilter size", 5, 255, odd=True)
            )
            matcher.setPreFilterType(
                cv2.StereoBM_PREFILTER_XSOBEL
                if p["prefilter_type"] == "xsobel"
                else cv2.StereoBM_PREFILTER_NORMALIZED_RESPONSE
            )
        else:
            p1 = integer(p["p1"], "P1", 0, 1000000) or 8 * block_size**2
            p2 = integer(p["p2"], "P2", 0, 1000000) or 32 * block_size**2
            if p2 <= p1:
                raise ValueError("SGBM P2 must exceed P1")
            matcher = cv2.StereoSGBM_create(
                numDisparities=disparities,
                blockSize=block_size,
                P1=p1,
                P2=p2,
                mode={
                    "sgbm": cv2.StereoSGBM_MODE_SGBM,
                    "sgbm_3way": cv2.StereoSGBM_MODE_SGBM_3WAY,
                    "hh": cv2.StereoSGBM_MODE_HH,
                    "hh4": cv2.StereoSGBM_MODE_HH4,
                }[p["mode"]],
            )
        matcher.setMinDisparity(minimum)
        matcher.setPreFilterCap(integer(p["prefilter_cap"], "Prefilter cap", 1, 63))
        matcher.setUniquenessRatio(integer(p["uniqueness_ratio"], "Uniqueness ratio", 0, 100))
        matcher.setSpeckleWindowSize(integer(p["speckle_window"], "Speckle window", 0, 10000))
        matcher.setSpeckleRange(integer(p["speckle_range"], "Speckle range", 0, 256))
        matcher.setDisp12MaxDiff(integer(p["disp12_max_diff"], "Left-right difference", -1, 256))
        raw = matcher.compute(a, b)
        confidence = NO_VALUE
        if p["wls"]:
            right_matcher = cv2.ximgproc.createRightMatcher(matcher)
            right_raw = right_matcher.compute(b, a)
            wls = cv2.ximgproc.createDisparityWLSFilter(matcher)
            wls.setLambda(finite(p["wls_lambda"], "WLS lambda", 0.0001))
            wls.setSigmaColor(finite(p["wls_sigma"], "WLS sigma", 0.0001))
            wls.setLRCthresh(int(p["wls_lr_threshold"]))
            radius = int(p["wls_depth_radius"])
            if radius >= 0:
                wls.setDepthDiscontinuityRadius(radius)
            raw = wls.filter(raw, a, disparity_map_right=right_raw)
            confidence = wls.getConfidenceMap() / 255.0
            confidence[~np.isfinite(confidence)] = 0
            np.clip(confidence, 0, 1, out=confidence)
        valid = raw > (minimum - 1) * 16
        if confidence is not NO_VALUE:
            confidence[~valid] = 0
            valid &= confidence >= float(p["minimum_wls_confidence"])
        disparity = raw.astype(np.float32) / 16.0
        disparity[~valid] = np.nan
        preview = cv2.applyColorMap(
            np.uint8(
                np.clip((np.nan_to_num(disparity, nan=minimum) - minimum) / disparities, 0, 1) * 255
            ),
            cv2.COLORMAP_TURBO,
        )
        preview[~valid] = 0
        return Result(
            {
                "disparity": disparity,
                "valid_mask": valid.astype(np.uint8) * 255,
                "preview": preview,
                "confidence": confidence,
            },
            passed=bool(valid.any()),
            score=float(valid.mean()),
            message=f"Disparity in pixels · {int(np.count_nonzero(valid))} valid estimates",
        )


class ReprojectDisparity(Tool3D):
    type_id, title, subtype = "3d.reproject", "Disparity to 3D cloud", "Reconstruction"
    inputs = (
        Port("disparity"),
        Port("rectification", "data"),
        Port("image", required=False),
        Port("mask", required=False),
        Port("region", "data", required=False),
    )
    outputs = (Port("cloud", "cloud"), Port("depth", "depth"), Port("count", "number"))
    defaults = {
        "input_scale": 1.0,
        "sample_step": 2,
        "max_points": 200000,
        "min_depth": 0.001,
        "max_depth": 100.0,
    }

    def process(self, inputs):
        p, rec = self.parameters, inputs["rectification"]
        disparity = np.asarray(inputs["disparity"], np.float32)
        if disparity.ndim != 2 or image_size(rec["image_size"]) != (
            disparity.shape[1],
            disparity.shape[0],
        ):
            raise ValueError("Disparity must match rectification image size")
        xyz = cv2.reprojectImageTo3D(
            disparity * finite(p["input_scale"], "Input scale", 0.00000001),
            matrix(rec["Q"], (4, 4), "Q"),
        )
        low, high = (
            finite(p["min_depth"], "Minimum depth", 0),
            finite(p["max_depth"], "Maximum depth", 0),
        )
        if high <= low:
            raise ValueError("Maximum depth must exceed minimum depth (in rig units)")
        valid = np.isfinite(xyz).all(axis=2) & (xyz[..., 2] >= low) & (xyz[..., 2] <= high)
        xyz[~valid] = np.nan
        color = inputs.get("image")
        if color is not None and color.shape[:2] != disparity.shape:
            raise ValueError("Cloud color image must match disparity dimensions")
        cloud = cloud_from_grid(
            xyz,
            rec["units"],
            rec["frame"],
            selection_mask(inputs, disparity),
            color,
            integer(p["sample_step"], "Sample step", 1, 256),
            integer(p["max_points"], "Max points", 1, 2000000),
            metadata(inputs["disparity"]),
        )
        depth = DepthArray(xyz[..., 2], rec["units"], low, high)
        return Result(
            {"cloud": cloud, "depth": depth, "count": len(cloud.points)},
            graphics=cloud_graphics(self, cloud),
            passed=bool(len(cloud.points)),
        )


class DepthToCloud(Tool3D):
    type_id, title, subtype = "3d.depth_cloud", "Depth to point cloud", "RGB-D"
    inputs = (
        Port("depth", "depth"),
        Port("camera", "data"),
        Port("image", required=False),
        Port("mask", required=False),
        Port("region", "data", required=False),
    )
    outputs = (Port("cloud", "cloud"), Port("count", "number"))
    defaults = {
        "input_units": "m",
        "units_per_sample": 0.001,
        "output_units": "m",
        "sample_step": 2,
        "max_points": 200000,
    }
    parameter_choices = {"input_units": ("m", "mm"), "output_units": ("m", "mm")}

    def process(self, inputs):
        model, depth, p = inputs["camera"], inputs["depth"], self.parameters
        k, d, _ = camera_for_image(model, depth)
        if model.get("lens", "pinhole") != "pinhole" or np.any(d):
            raise ValueError("Depth projection requires undistorted pinhole intrinsics")
        values = metric_depth(depth, p["units_per_sample"], p["input_units"])
        xyz = rgbd_function("depthTo3d")(values, k)
        if p["output_units"] == "mm":
            xyz *= 1000
        image = inputs.get("image")
        if image is not None and image.shape[:2] != depth.shape:
            raise ValueError("Color/depth must already be registered and have matching dimensions")
        cloud = cloud_from_grid(
            xyz,
            p["output_units"],
            model["frame"],
            selection_mask(inputs, depth),
            image,
            integer(p["sample_step"], "Sample step", 1, 256),
            integer(p["max_points"], "Max points", 1, 2000000),
            metadata(depth),
        )
        return Result(
            {"cloud": cloud, "count": len(cloud.points)},
            graphics=cloud_graphics(self, cloud),
            passed=bool(len(cloud.points)),
            message=f"{len(cloud.points)} XYZ points in {cloud.units}",
        )


class RegisterDepth(Tool3D):
    type_id, title, subtype = "3d.register_depth", "Register depth to camera", "RGB-D"
    inputs = (
        Port("depth", "depth"),
        Port("depth_camera", "data"),
        Port("target_camera", "data"),
        Port("transform", "data"),
    )
    outputs = (Port("depth", "depth"), Port("preview"))
    defaults = {
        "input_units": "m",
        "units_per_sample": 0.001,
        "dilate": False,
        "display_min": 0.0,
        "display_max": 5.0,
    }
    parameter_choices = {"input_units": ("m", "mm")}

    def process(self, inputs):
        a, b, t, p = (
            inputs["depth_camera"],
            inputs["target_camera"],
            inputs["transform"],
            self.parameters,
        )
        ka, da, _ = require_pinhole(a)
        kb, db, size = require_pinhole(b)
        camera_for_image(a, inputs["depth"])
        if np.any(da):
            raise ValueError("Source depth must be undistorted before registration")
        if (
            t["source_frame"] != a["frame"]
            or t["target_frame"] != b["frame"]
            or t["units"] not in ("m", "mm")
        ):
            raise ValueError(
                "Depth transform must map the source camera to target camera with metric units"
            )
        transform = rigid(t).copy()
        transform[:3, 3] *= 0.001 if t["units"] == "mm" else 1
        depth = metric_depth(inputs["depth"], p["units_per_sample"], p["input_units"])
        values = rgbd_function("registerDepth")(
            ka, kb, db, transform, depth, size, depthDilation=p["dilate"]
        )
        values[(values <= 0) | ~np.isfinite(values)] = np.nan
        low, high = (
            finite(p["display_min"], "Display min", 0),
            finite(p["display_max"], "Display max", 0),
        )
        if high <= low:
            raise ValueError("Display max must exceed display min")
        output = DepthArray(values, "m", low, high)
        payload = {**metadata(inputs["depth"]), "spatial": frame(size, f"{self.id}:registered")}
        payload.pop("pyramid", None)
        return Result(
            {"depth": output, "preview": cv2.cvtColor(preview_rgb(output), cv2.COLOR_RGB2BGR)},
            payloads={"depth": payload},
            passed=bool(np.isfinite(output).any()),
        )


STEREO3D_BLOCKS = (
    StereoRig,
    CalibrateStereo,
    RectifyStereo,
    StereoDisparity,
    ReprojectDisparity,
    DepthToCloud,
    RegisterDepth,
)
