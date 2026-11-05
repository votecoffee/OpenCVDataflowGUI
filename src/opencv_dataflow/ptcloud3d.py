"""OpenCV 5 ptcloud kernels: organized normals, two-frame odometry and TSDF fusion."""

import cv2
import numpy as np

from .core import Port, Result
from .geometry3d_common import Tool3D, camera_for_image, cloud_graphics, pose_record, rigid
from .images import DepthArray
from .payloads import NO_VALUE, FlowGroup, frame, metadata
from .pointclouds import PointCloud
from .processing import finite, image_value, integer
from .selection import selection_mask
from .stereo3d import cloud_from_grid, metric_depth, rgbd_function

NORMAL_METHODS = {
    "fals": "FALS",
    "sri": "SRI",
    "cross_product": "CROSS_PRODUCT",
}


def depth_intrinsics(model, depth):
    k, d, size = camera_for_image(model, depth)
    if model.get("lens", "pinhole") != "pinhole" or np.any(d):
        raise ValueError("Supply depth registered to undistorted pinhole intrinsics")
    return k, size


class DepthNormals(Tool3D):
    type_id, title, subtype = "3d.depth_normals", "Organized RGB-D normals", "RGB-D"
    inputs = (
        Port("depth", "depth"),
        Port("camera", "data"),
        Port("image", required=False),
        Port("mask", required=False),
        Port("region", "data", required=False),
    )
    outputs = (Port("cloud", "cloud"), Port("preview"), Port("count", "number"))
    defaults = {
        "input_units": "m",
        "units_per_sample": 0.001,
        "method": "fals",
        "window_size": "5",
        "sample_step": 2,
        "max_points": 200000,
    }
    parameter_choices = {
        "input_units": ("m", "mm"),
        "method": tuple(NORMAL_METHODS),
        "window_size": ("1", "3", "5", "7"),
    }

    def process(self, inputs):
        p, model, depth = self.parameters, inputs["camera"], inputs["depth"]
        k, (w, h) = depth_intrinsics(model, depth)
        values = metric_depth(depth, p["units_per_sample"], p["input_units"])
        xyz = rgbd_function("depthTo3d")(values, k)
        method = getattr(cv2, f"RGBD_NORMALS_RGBD_NORMALS_METHOD_{NORMAL_METHODS[p['method']]}")
        computer = rgbd_function("RgbdNormals_create")(
            rows=h, cols=w, depth=cv2.CV_32F, K=k, window_size=int(p["window_size"]), method=method
        )
        normal = computer.apply(xyz)[..., :3]
        length = np.linalg.norm(normal, axis=2)
        valid = np.isfinite(normal).all(axis=2) & np.isfinite(values) & (length > 1e-8)
        mask = selection_mask(inputs, depth)
        if mask is not None:
            valid &= mask != 0
        image = inputs.get("image")
        if image is not None and image.shape[:2] != depth.shape:
            raise ValueError("Color image must be registered to the depth grid")
        normalized = np.zeros_like(normal)
        normalized[valid] = normal[valid] / length[valid, None]
        cloud = cloud_from_grid(
            xyz,
            "m",
            model.get("frame", "camera"),
            valid,
            image,
            integer(p["sample_step"], "Sample step", 1, 256),
            integer(p["max_points"], "Max points", 1, 2000000),
            metadata(depth),
        )
        cloud.normals = normalized[cloud.pixels[:, 1], cloud.pixels[:, 0]]
        preview = np.uint8(np.clip((normalized[..., ::-1] + 1) * 127.5, 0, 255))
        preview[~valid] = 0
        return Result(
            {"cloud": cloud, "preview": preview, "count": len(cloud.points)},
            passed=bool(len(cloud.points)),
            graphics=cloud_graphics(self, cloud),
        )


class RGBDOdometry(Tool3D):
    type_id, title, subtype = "3d.odometry", "RGB-D frame odometry", "RGB-D motion"
    inputs = (
        Port("source_depth", "depth"),
        Port("target_depth", "depth"),
        Port("camera", "data"),
        Port("source_image", required=False),
        Port("target_image", required=False),
    )
    outputs = (
        Port("pose", "data"),
        Port("translation_m", "number"),
        Port("rotation_degrees", "number"),
    )
    defaults = {
        "mode": "depth",
        "algorithm": "common",
        "source_frame": "previous_camera",
        "target_frame": "current_camera",
        "input_units": "m",
        "units_per_sample": 0.001,
        "iteration_counts": [7, 7, 7, 10],
        "min_gradient_magnitudes": [10.0, 10.0, 10.0, 10.0],
        "min_depth": 0.1,
        "max_depth": 4.0,
        "max_depth_difference": 0.07,
        "max_points_fraction": 0.07,
        "max_translation_m": 0.15,
        "max_rotation_degrees": 15.0,
        "normal_window": "5",
    }
    parameter_choices = {
        "mode": ("depth", "rgb", "rgbd"),
        "algorithm": ("common", "fast"),
        "input_units": ("m", "mm"),
        "normal_window": ("1", "3", "5", "7"),
    }

    def process(self, inputs):
        if not hasattr(cv2, "OdometrySettings") or not hasattr(cv2, "Odometry"):
            raise RuntimeError("OpenCV 5 ptcloud Odometry bindings are unavailable")
        p, model = self.parameters, inputs["camera"]
        a, b = inputs["source_depth"], inputs["target_depth"]
        k, size = depth_intrinsics(model, a)
        depth_intrinsics(model, b)
        a, b = (metric_depth(v, p["units_per_sample"], p["input_units"]) for v in (a, b))
        counts = p["iteration_counts"]
        gradients = p["min_gradient_magnitudes"]
        if (
            not isinstance(counts, list)
            or not 1 <= len(counts) <= 6
            or not isinstance(gradients, list)
            or len(counts) != len(gradients)
            or min(size) // 2 ** (len(counts) - 1) < 4
        ):
            raise ValueError(
                "Odometry pyramid lists must match and each level must be at least 4 pixels"
            )
        settings = cv2.OdometrySettings()
        settings.setCameraMatrix(k.astype(np.float32))
        # OpenCV 5 interprets the width as level count and reads one scalar per row.
        # A square matrix supplies both dimensions without reading past the rows.
        counts = np.asarray([integer(v, "Iterations", 1, 1000) for v in counts], np.int32)
        gradients = np.asarray([finite(v, "Minimum gradient", 0) for v in gradients], np.float32)
        settings.setIterCounts(np.repeat(counts[:, None], len(counts), axis=1))
        settings.setMinGradientMagnitudes(np.repeat(gradients[:, None], len(gradients), axis=1))
        if not np.array_equal(settings.getIterCounts().ravel(), counts) or not np.array_equal(
            settings.getMinGradientMagnitudes().ravel(), gradients
        ):
            raise RuntimeError("OpenCV rejected the requested odometry pyramid settings")
        low, high = (
            finite(p["min_depth"], "Minimum depth", 0),
            finite(p["max_depth"], "Maximum depth", 0),
        )
        fraction = finite(p["max_points_fraction"], "Max points fraction", 0.0001)
        if high <= low or fraction > 1:
            raise ValueError("Depth limits must be ordered and max_points_fraction <= 1")
        settings.setMinDepth(low)
        settings.setMaxDepth(high)
        settings.setMaxDepthDiff(
            finite(p["max_depth_difference"], "Maximum depth difference", 0.0001)
        )
        settings.setMaxPointsPart(fraction)
        settings.setMaxTranslation(finite(p["max_translation_m"], "Maximum translation", 0.0001))
        settings.setMaxRotation(finite(p["max_rotation_degrees"], "Maximum rotation", 0.0001))
        settings.setNormalWinSize(int(p["normal_window"]))
        algorithm = (
            cv2.ODOMETRY_ALGO_TYPE_COMMON
            if p["algorithm"] == "common"
            else cv2.ODOMETRY_ALGO_TYPE_FAST
        )
        mode = {
            "depth": cv2.ODOMETRY_TYPE_DEPTH,
            "rgb": cv2.ODOMETRY_TYPE_RGB,
            "rgbd": cv2.ODOMETRY_TYPE_RGB_DEPTH,
        }[p["mode"]]
        odometry = cv2.Odometry(mode, settings, algorithm)
        if p["mode"] == "depth":
            ok, transform = odometry.compute(a, b)
        else:
            if "source_image" not in inputs or "target_image" not in inputs:
                raise ValueError("RGB/RGB-D odometry requires two registered intensity images")
            images = [
                image_value(inputs[k], (np.uint8,), canonical=True)
                for k in ("source_image", "target_image")
            ]
            if any(im.shape[:2] != a.shape for im in images):
                raise ValueError("Odometry images must match the depth grid")
            images = [
                cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                if im.ndim == 2
                else cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
                if im.shape[2] == 4
                else im
                for im in images
            ]
            ok, transform = odometry.compute(a, images[0], b, images[1])
        if not ok:
            return Result(
                {"pose": NO_VALUE, "translation_m": 0.0, "rotation_degrees": 0.0},
                passed=False,
                fail_reasons=["odometry_tracking"],
                message="Odometry did not converge",
            )
        pose = pose_record(transform, p["source_frame"], p["target_frame"], "m")
        return Result(
            {
                "pose": pose,
                "translation_m": float(np.linalg.norm(pose["translation"])),
                "rotation_degrees": float(np.rad2deg(np.linalg.norm(pose["rvec"]))),
            }
        )


class FuseTSDF(Tool3D):
    type_id, title, subtype = (
        "3d.tsdf_fuse",
        "Fuse posed depth frames (TSDF)",
        "Volumetric reconstruction",
    )
    inputs = (
        Port("depth", "depth", required=False),
        Port("frames", "any", required=False),
        Port("poses", "data", visual_type="pose"),
        Port("camera", "data"),
    )
    outputs = (Port("cloud", "cloud"), Port("depth", "depth"), Port("count", "number"))
    defaults = {
        "resolution": [64, 64, 64],
        "voxel_size_m": 0.02,
        "volume_origin_m": [-0.64, -0.64, 0.5],
        "truncation_m": 0.06,
        "max_weight": 64,
        "max_depth_m": 5.0,
        "raycast_step_factor": 0.25,
        "input_units": "m",
        "units_per_sample": 0.001,
        "sample_step": 1,
        "max_points": 200000,
    }
    parameter_choices = {"input_units": ("m", "mm")}

    def process(self, inputs):
        if not hasattr(cv2, "Volume") or not hasattr(cv2, "VolumeSettings"):
            raise RuntimeError("OpenCV 5 ptcloud Volume bindings are unavailable")
        p, model = self.parameters, inputs["camera"]
        if ("depth" in inputs) == ("frames" in inputs):
            raise ValueError("Connect either one depth plane or a group of posed depth frames")
        group = inputs.get("frames")
        if group is not None and not isinstance(group, FlowGroup):
            raise ValueError("Frames must be a FlowGroup of depth planes")
        depths = [packet.value for packet in group.items] if group else [inputs["depth"]]
        poses = inputs["poses"] if isinstance(inputs["poses"], list) else [inputs["poses"]]
        if len(poses) != len(depths):
            raise ValueError("Supply one camera-to-volume/world pose per depth frame")
        if any(
            not isinstance(depth, np.ndarray)
            or depth.ndim != 2
            or depth.dtype not in (np.uint16, np.float32, np.float64)
            for depth in depths
        ):
            raise ValueError("TSDF frames must be depth arrays, not color images")
        k, (w, h) = depth_intrinsics(model, depths[0])
        target = poses[0]["target_frame"]
        transforms = []
        for depth, pose in zip(depths, poses):
            depth_intrinsics(model, depth)
            if (
                pose["source_frame"] != model.get("frame", "camera")
                or pose["target_frame"] != target
                or pose["units"] not in ("m", "mm")
            ):
                raise ValueError(
                    "TSDF poses must share one metric world frame and match camera source frame"
                )
            transform = rigid(pose).astype(np.float32)
            transform[:3, 3] *= 0.001 if pose["units"] == "mm" else 1.0
            transforms.append(transform)
        resolution = tuple(integer(v, "Volume dimension", 8, 256) for v in p["resolution"])
        if (
            len(resolution) != 3
            or any(v % 8 for v in resolution)
            or np.prod(resolution) > 8_000_000
        ):
            raise ValueError(
                "Volume requires three multiples of eight and at most 8 million voxels"
            )
        origin = np.asarray(p["volume_origin_m"], np.float32)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("Volume origin must be finite XYZ meters")
        volume_pose = np.eye(4, dtype=np.float32)
        volume_pose[:3, 3] = origin
        settings = cv2.VolumeSettings(cv2.VOLUME_TYPE_TSDF)
        settings.setVolumeResolution(resolution)
        settings.setVoxelSize(finite(p["voxel_size_m"], "Voxel size", 0.00001))
        settings.setVolumePose(volume_pose)
        settings.setTsdfTruncateDistance(finite(p["truncation_m"], "Truncation", 0.00001))
        settings.setMaxWeight(integer(p["max_weight"], "Max weight", 1, 255))
        maximum = finite(p["max_depth_m"], "Maximum depth", 0.001)
        settings.setMaxDepth(maximum)
        settings.setRaycastStepFactor(
            finite(p["raycast_step_factor"], "Raycast step factor", 0.001)
        )
        settings.setDepthFactor(1.0)
        settings.setIntegrateWidth(w)
        settings.setIntegrateHeight(h)
        settings.setRaycastWidth(w)
        settings.setRaycastHeight(h)
        settings.setCameraIntegrateIntrinsics(k.astype(np.float32))
        settings.setCameraRaycastIntrinsics(k.astype(np.float32))
        # Local OpenCV 5 OpenCL volume extraction is unreliable on this wheel/driver.
        # Use the CPU path and restore the current thread's OpenCL setting.
        opencl = cv2.ocl.useOpenCL()
        cv2.ocl.setUseOpenCL(False)
        try:
            volume = cv2.Volume(cv2.VOLUME_TYPE_TSDF, settings)
            for depth, pose in zip(depths, transforms):
                values = metric_depth(depth, p["units_per_sample"], p["input_units"])
                volume.integrate(np.nan_to_num(values, nan=0.0), pose)
            xyz, normals = volume.fetchPointsNormals()
            rendered, _ = volume.raycast(transforms[-1])
        finally:
            cv2.ocl.setUseOpenCL(opencl)
        xyz = (
            xyz.reshape(-1, xyz.shape[-1])[:, :3]
            if xyz is not None
            else np.empty((0, 3), np.float32)
        )
        normals = (
            normals.reshape(-1, normals.shape[-1])[:, :3]
            if normals is not None
            else np.empty((0, 3), np.float32)
        )
        lengths = np.linalg.norm(normals, axis=1)
        keep = np.isfinite(xyz).all(axis=1) & np.isfinite(normals).all(axis=1) & (lengths > 1e-8)
        idx = np.flatnonzero(keep)[:: integer(p["sample_step"], "Sample step", 1, 256)]
        if len(idx) > integer(p["max_points"], "Max points", 1, 2000000):
            raise ValueError("Volume cloud exceeds max_points; increase sample_step")
        cloud = PointCloud(xyz[idx], "m", target, normals=normals[idx] / lengths[idx, None])
        values = rendered[..., 2].copy()
        values[~np.isfinite(values) | (values <= 0)] = np.nan
        return Result(
            {"cloud": cloud, "depth": DepthArray(values, "m", 0.0, maximum), "count": len(idx)},
            payloads={"depth": {"spatial": frame((w, h), f"{self.id}:raycast")}},
            passed=bool(len(idx)),
            graphics=cloud_graphics(self, cloud),
            message=f"Fused {len(depths)} posed frames · {len(idx)} surface points",
        )


PTCLOUD3D_BLOCKS = (DepthNormals, RGBDOdometry, FuseTSDF)
