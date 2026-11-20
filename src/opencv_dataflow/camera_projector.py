"""Calibrate a pinhole camera/projector pair and reconstruct decoded correspondences."""

import cv2
import numpy as np

from .calibration_revisions import is_full_grid
from .core import Port, Result
from .geometry3d_common import (
    Tool3D,
    camera_for_image,
    camera_record,
    cloud_graphics,
    pose_matrix,
    pose_record,
    require_pinhole,
    rigid,
)
from .images import DepthArray
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, check_spatial, frame, image_with_payload, require_same_frame
from .pointclouds import PointCloud
from .selection import selection_mask
from .wide_stereo import read_views

RIG_MODEL = "camera_projector_rig_v1"
VIEW_NAMES = ("object_views", "camera_views", "projector_views")
TO_METERS = {"m": 1.0, "mm": 0.001}
MAX_GRID = 4_194_304


def optical_pair(camera, projector):
    a, b = require_pinhole(camera), require_pinhole(projector)
    if camera.get("units") not in TO_METERS or camera.get("units") != projector.get("units"):
        raise ValidationError("Camera and projector models require identical metric units")
    if camera["frame"] == projector["frame"]:
        raise ValidationError("Use distinct camera and projector frame names")
    return a, b


class CalibrateCameraProjector(Tool3D):
    type_id, title, subtype = (
        "3d.camera_projector_calibrate",
        "Calibrate camera and projector",
        "Structured light",
    )
    parallel_safe = True
    process_inputs_readonly = True
    inputs = (
        Port("camera", "data"),
        Port("projector", "data"),
        *(Port(key, "data", required=False, required_when_connected=True) for key in VIEW_NAMES),
    )
    outputs = (Port("rig", "data", required=False), Port("fit_report", "data"))
    defaults = {
        "path": "",
        "path_base": "project",
        "projector_intrinsics": "fixed",
        "iterations": 100,
        "epsilon": 1e-8,
        "max_rms_px": 1.0,
        "max_view_rms_px": 2.0,
    }
    parameter_choices = {
        "path_base": ("project", "working_directory"),
        "projector_intrinsics": ("fixed", "refine"),
    }
    parameter_ranges = {
        "iterations": NumericRange(1, 1000, whole=True),
        "epsilon": NumericRange(0, 1),
        "max_rms_px": NumericRange(1e-6, 1000),
        "max_view_rms_px": NumericRange(0, 1000),
    }
    parameter_help = {
        "path": "Optional JSON object with object_views, camera_views and projector_views.",
        "path_base": "Base for a relative observations file.",
        "projector_intrinsics": "Use fixed intrinsics or fit the projector on its own pixel grid.",
        "max_rms_px": "Maximum acceptable RMS across both devices and all views.",
        "max_view_rms_px": "Per-device RMS limit per view. Zero disables it.",
    }
    input_help = {
        "camera": "Calibrated pinhole camera. Camera intrinsics stay fixed.",
        "projector": "Projector pinhole model with fixed intrinsics or an initial guess.",
        "object_views": "Per-view target XYZ arrays in the declared metric units.",
        "camera_views": "Per-view camera XY arrays in target point order.",
        "projector_views": "Per-view decoded projector XY arrays in that same order.",
    }

    def process(self, inputs):
        p = self.parameters
        a, b = inputs["camera"], inputs["projector"]
        (k1, d1, size1), (k2, d2, size2) = optical_pair(a, b)
        raw = read_views(
            self,
            inputs,
            size1,
            "pinhole",
            names=VIEW_NAMES,
            second_size=size2,
            same_point_count=False,
        )
        obj, camera_xy, projector_xy = (
            [np.ascontiguousarray(view.reshape(-1, columns), np.float32) for view in group]
            for group, columns in zip(raw, (3, 2, 2), strict=True)
        )
        criteria = (
            cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
            int(p["iterations"]),
            float(p["epsilon"]),
        )
        projector_rms = None
        if p["projector_intrinsics"] == "refine":
            flags = cv2.CALIB_USE_INTRINSIC_GUESS
            if d2.size >= 8:
                flags |= cv2.CALIB_RATIONAL_MODEL
            if d2.size >= 12:
                flags |= cv2.CALIB_THIN_PRISM_MODEL
            if d2.size == 14:
                flags |= cv2.CALIB_TILTED_MODEL
            projector_rms, k2, d2, _, _ = cv2.calibrateCamera(
                obj, projector_xy, size2, k2, d2, flags=flags, criteria=criteria
            )
            if not np.isfinite(projector_rms):
                raise ValidationError("Native projector intrinsic fit returned a nonfinite RMS")
            b = camera_record(k2, d2, size2, b["units"], b["frame"])
        (
            rms,
            _camera_k,
            _camera_d,
            _projector_k,
            _projector_d,
            rotation,
            translation,
            _essential,
            _fundamental,
            rotations,
            offsets,
            per_view,
        ) = cv2.stereoCalibrateExtended(
            obj,
            camera_xy,
            projector_xy,
            k1,
            d1,
            k2,
            d2,
            size1,
            None,
            None,
            flags=cv2.CALIB_FIX_INTRINSIC,
            criteria=criteria,
        )
        errors = np.asarray(per_view).reshape(-1, 2)
        if len(errors) != len(obj) or not np.isfinite(rms) or not np.isfinite(errors).all():
            raise ValidationError("Native camera/projector fit returned invalid residuals")
        transform = pose_record(
            pose_matrix(rotation, translation), a["frame"], b["frame"], a["units"]
        )
        baseline = float(np.linalg.norm(translation)) * TO_METERS[a["units"]]
        if baseline <= 1e-9:
            raise ValidationError("Camera/projector fitting requires a nonzero baseline")
        for target, rv, tv in zip(obj, rotations, offsets, strict=True):
            camera_xyz = target @ cv2.Rodrigues(rv)[0].T + np.asarray(tv).ravel()
            if np.any(camera_xyz[:, 2] <= 0) or np.any(
                (camera_xyz @ rotation.T + np.asarray(translation).ravel())[:, 2] <= 0
            ):
                raise ValidationError("Calibrated target is behind the camera or projector")
        worst = float(errors.max())
        cutoff = float(p["max_view_rms_px"])
        reasons = []
        if rms > float(p["max_rms_px"]):
            reasons.append("camera_projector_rms")
        if cutoff and worst > cutoff:
            reasons.append("camera_projector_view_rms")
        report = {
            "rms_pixels": float(rms),
            "worst_view_rms_pixels": worst,
            "projector_intrinsic_rms_pixels": projector_rms,
            "baseline_m": baseline,
            "camera_size": list(size1),
            "projector_size": list(size2),
            "views": [
                {
                    "index": i,
                    "points": len(xyz),
                    "camera_rms_pixels": float(err[0]),
                    "projector_rms_pixels": float(err[1]),
                }
                for i, (xyz, err) in enumerate(zip(obj, errors, strict=True))
            ],
        }
        rig = {"model": RIG_MODEL, "camera": a, "projector": b, "transform": transform}
        return Result(
            {"rig": NO_VALUE if reasons else rig, "fit_report": report},
            passed=not reasons,
            fail_reasons=reasons,
            message=f"{len(obj)} views · {rms:.4g} px RMS · {baseline:.4g} m baseline",
        )


def reconstruct_pair(rig):
    if not isinstance(rig, dict) or rig.get("model") != RIG_MODEL:
        raise ValidationError("Connect a calibrated camera/projector rig")
    a, b, t = (rig[key] for key in ("camera", "projector", "transform"))
    optics = optical_pair(a, b)
    if (t.get("source_frame"), t.get("target_frame"), t.get("units")) != (
        a["frame"],
        b["frame"],
        a["units"],
    ):
        raise ValidationError("Rig pose must map the camera to the projector in model units")
    pose = rigid(t).copy()
    pose[:3, 3] *= TO_METERS[a["units"]]
    if np.linalg.norm(pose[:3, 3]) <= 1e-9:
        raise ValidationError("Camera/projector baseline must be nonzero")
    return optics, pose


def dense_input(block, inputs, camera, projector_size):
    first, second = inputs["projector_x"], inputs["projector_y"]
    arrays = (first, second)
    for axis, values in zip(("x", "y"), arrays, strict=True):
        if values.ndim != 2 or values.dtype not in (np.float32, np.float64):
            raise ValidationError("Projector-coordinate maps must be monochrome float images")
        camera_for_image(camera, values)
        decoded = block.input_payloads[f"projector_{axis}"].get("phase_shift")
        if decoded and (
            decoded.get("axis") != axis or decoded.get("projector_size") != list(projector_size)
        ):
            raise ValidationError("Decoded phase axis or projector dimensions do not match the rig")
    size = camera["image_size"]
    a, b = (block.input_payloads[f"projector_{axis}"] for axis in ("x", "y"))
    if a.get("pyramid") != b.get("pyramid"):
        raise ValidationError("Projector-coordinate maps have different pyramid levels")
    if not (is_full_grid(a, size) and is_full_grid(b, size)):
        if any(p.get("spatial", {}).get("root_id") is None for p in (a, b)):
            raise ValidationError("Transformed projector maps must use a shared known camera grid")
        require_same_frame(arrays, message="Projector-coordinate maps must share a camera grid")
    context = {key: a[key] for key in ("spatial", "pyramid", "ticket") if key in a}
    return arrays, context


def sparse_input(inputs, size):
    camera, projector = inputs["camera_points"], inputs["projector_points"]
    if (
        not isinstance(camera, list)
        or not isinstance(projector, list)
        or not 1 <= len(camera) <= 10000
    ):
        raise ValidationError("Supply 1 to 10,000 ordered camera/projector point pairs")
    flags = inputs.get("valid", [True] * len(camera))
    if (
        len(projector) != len(camera)
        or not isinstance(flags, list)
        or (len(flags) != len(camera) or any(type(value) is not bool for value in flags))
    ):
        raise ValidationError("Sparse pairs and the optional Boolean validity list must match")
    pixels = np.asarray(camera, dtype=float)
    if (
        pixels.shape != (len(camera), 2)
        or not np.isfinite(pixels).all()
        or np.any(pixels != np.rint(pixels))
        or np.any(pixels < 0)
        or np.any(pixels >= size)
        or len(np.unique(pixels, axis=0)) != len(pixels)
    ):
        raise ValidationError("Sparse camera points must be distinct whole pixels inside the grid")
    values = np.asarray(
        [
            v if flag and v is not None else [np.nan, np.nan]
            for flag, v in zip(flags, projector, strict=True)
        ],
        dtype=float,
    )
    if values.shape != pixels.shape:
        raise ValidationError("Sparse projector points must each be an XY pair or null")
    return pixels.astype(np.int32), values


def triangulate_band(pixels, projected, optics, pose, p):
    (k1, d1, _), (k2, d2, _) = optics
    uv = cv2.undistortPoints(pixels.astype(np.float64).reshape(-1, 1, 2), k1, d1).reshape(-1, 2)
    st = cv2.undistortPoints(projected.reshape(-1, 1, 2), k2, d2).reshape(-1, 2)
    if not np.isfinite(uv).all() or not np.isfinite(st).all():
        raise ValidationError("Calibrated undistortion returned nonfinite rays")
    r, t = pose[:3, :3], pose[:3, 3]
    left = np.column_stack((np.eye(3), np.zeros(3)))
    h = cv2.triangulatePoints(left, pose[:3], uv.T, st.T).T
    finite = np.isfinite(h).all(axis=1) & (np.abs(h[:, 3]) > 1e-12)
    xyz = h[:, :3] / np.where(finite, h[:, 3], 1)[:, None]
    depth = xyz[:, 2]
    in_front = finite & (depth >= float(p["minimum_z_m"])) & (depth <= float(p["maximum_z_m"]))
    in_front &= (xyz @ r.T + t)[:, 2] > 0
    a = np.column_stack((uv, np.ones(len(uv))))
    b = np.column_stack((st, np.ones(len(st)))) @ r
    dot = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    angle = np.rad2deg(np.arccos(np.clip(dot, -1, 1)))
    geometry = in_front & (angle >= float(p["minimum_ray_angle_degrees"]))
    errors = np.full(len(xyz), np.nan)
    if np.any(geometry):
        fit = np.ascontiguousarray(xyz[geometry])
        zero = np.zeros(3)
        cam_hat = cv2.projectPoints(fit, zero, zero, k1, d1)[0].reshape(-1, 2)
        proj_hat = cv2.projectPoints(fit, cv2.Rodrigues(r)[0], t, k2, d2)[0].reshape(-1, 2)
        errors[geometry] = np.maximum(
            np.linalg.norm(cam_hat - pixels[geometry], axis=1),
            np.linalg.norm(proj_hat - projected[geometry], axis=1),
        )
    accept = geometry & np.isfinite(errors) & (errors <= float(p["max_reprojection_error_px"]))
    errors[~np.isfinite(errors)] = np.nan
    return xyz, errors, accept


class ReconstructCameraProjector(Tool3D):
    type_id, title, subtype = (
        "3d.camera_projector_reconstruct",
        "Reconstruct structured-light depth",
        "Structured light",
    )
    parallel_safe = True
    process_inputs_readonly = True
    payload_reference = "projector_x"
    inputs = (
        Port("rig", "data"),
        Port("projector_x", required=False, required_when_connected=True),
        Port("projector_y", required=False, required_when_connected=True),
        *(
            Port(key, "data", required=False, required_when_connected=True)
            for key in ("camera_points", "projector_points", "valid")
        ),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("depth", "depth"),
        Port("cloud", "cloud"),
        Port("valid_mask", visual_type="region"),
        Port("reprojection_error"),
        Port("report", "data"),
    )
    defaults = {
        "sample_step": 1,
        "max_points": 1_000_000,
        "minimum_ray_angle_degrees": 0.25,
        "max_reprojection_error_px": 0.75,
        "minimum_z_m": 0.001,
        "maximum_z_m": 5.0,
    }
    parameter_ranges = {
        "sample_step": NumericRange(1, 512, whole=True),
        "max_points": NumericRange(1, 1_000_000, whole=True),
        "minimum_ray_angle_degrees": NumericRange(0.001, 90),
        "max_reprojection_error_px": NumericRange(0.00001, 1000),
        "minimum_z_m": NumericRange(1e-6, 100_000),
        "maximum_z_m": NumericRange(1e-6, 100_000),
    }
    parameter_help = {
        "sample_step": "Dense sampling stride in pixels. Sparse point order is retained.",
        "max_points": "Limit the number of submitted valid correspondences before triangulation.",
        "minimum_ray_angle_degrees": "Reject nearly parallel camera/projector ray pairs.",
        "max_reprojection_error_px": "Maximum allowed pixel error on either optical device.",
        "minimum_z_m": "Minimum accepted positive camera depth in meters.",
        "maximum_z_m": "Maximum accepted camera depth in meters.",
    }
    input_help = {
        "projector_x": "Dense X-coordinate map on the camera grid. Connect with projector_y.",
        "projector_y": "Dense Y-coordinate map on that same camera grid.",
        "camera_points": "Sparse integer camera pixels, such as the Gray-code decoder output.",
        "projector_points": "Sparse projector XY pixels matching camera_points in order.",
        "valid": "Optional sparse Boolean flag list in point order.",
        "mask": "Limit measurements to selected camera pixels.",
        "region": "Limit measurements to a camera region, intersecting the optional mask.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if float(p["maximum_z_m"]) <= float(p["minimum_z_m"]):
            raise ValidationError("Maximum Z must exceed minimum Z", "maximum_z_m", "minimum_z_m")

    def process(self, inputs):
        p = self.parameters
        optics, pose = reconstruct_pair(inputs["rig"])
        size, projector_size = optics[0][2], optics[1][2]
        if size[0] * size[1] > MAX_GRID:
            raise ValidationError("Camera grid exceeds the 4,194,304 pixel reconstruction limit")
        dense_keys = set(inputs) & {"projector_x", "projector_y"}
        sparse_keys = set(inputs) & {"camera_points", "projector_points", "valid"}
        if len(dense_keys) == 2 and not sparse_keys:
            maps, context = dense_input(self, inputs, inputs["rig"]["camera"], projector_size)
            stride = int(p["sample_step"])
            xmap, ymap = (m[::stride, ::stride] for m in maps)
            yy, xx = np.nonzero(np.isfinite(xmap) & np.isfinite(ymap))
            pixels = np.column_stack((xx, yy)).astype(np.int32) * stride
            projected = np.column_stack((xmap[yy, xx], ymap[yy, xx])).astype(np.float64)
        elif not dense_keys and {"camera_points", "projector_points"} <= sparse_keys:
            pixels, projected = sparse_input(inputs, size)
            context = {
                key: self.input_payloads["camera_points"][key]
                for key in ("spatial", "ticket")
                if key in self.input_payloads["camera_points"]
            }
        else:
            raise ValidationError("Supply both dense coordinate maps or both sparse point arrays")
        context.setdefault("spatial", frame(size, f"{self.id}:camera"))
        check_spatial(context, size)
        shape = size[::-1]
        chosen = selection_mask(inputs, image_with_payload(np.empty(shape, np.uint8), context))
        valid = np.isfinite(projected).all(axis=1)
        valid &= (projected >= -0.5).all(axis=1)
        valid &= (projected < np.asarray(projector_size) - 0.5).all(axis=1)
        if chosen is not None:
            valid &= chosen[pixels[:, 1], pixels[:, 0]] != 0
        pixels, projected = pixels[valid], projected[valid]
        if len(pixels) > int(p["max_points"]):
            raise ValidationError(
                "Too many correspondences. Increase sample step or limit the region."
            )

        depth = np.full(shape, np.nan, np.float32)
        errors = np.full(shape, np.nan, np.float32)
        observations, indices = [], []
        for start in range(0, len(pixels), 16384):
            band = slice(start, start + 16384)
            xy = pixels[band]
            xyz, error, accepted = triangulate_band(xy, projected[band], optics, pose, p)
            x, y = xy.T
            errors[y, x] = error
            output_xy = xy[accepted]
            output_xyz = xyz[accepted]
            observations.append(output_xyz)
            indices.append(output_xy)
            depth[output_xy[:, 1], output_xy[:, 0]] = output_xyz[:, 2]
        xyz = np.concatenate(observations) if observations else np.empty((0, 3), np.float32)
        xy = np.concatenate(indices) if indices else np.empty((0, 2), np.int32)
        model = inputs["rig"]["camera"]
        cloud = PointCloud(xyz, "m", model["frame"], pixels=xy, source_size=size, metadata=context)
        mask = np.isfinite(depth)
        output = {
            "depth": DepthArray(depth, "m", float(p["minimum_z_m"]), float(p["maximum_z_m"])),
            "cloud": cloud,
            "valid_mask": mask.astype(np.uint8) * 255,
            "reprojection_error": errors,
            "report": {
                "submitted_points": len(pixels),
                "accepted_points": len(xy),
                "rejected_points": len(pixels) - len(xy),
                "units": "m",
                "frame": model["frame"],
                "rms_pixels": float(np.sqrt(np.mean(errors[mask] ** 2))) if len(xy) else None,
            },
        }
        return Result(
            output,
            payloads={key: context for key in output},
            graphics=cloud_graphics(self, cloud),
            passed=bool(len(xy)),
            score=len(xy) / len(pixels) if len(pixels) else 0.0,
            fail_reasons=[] if len(xy) else ["no_valid_camera_projector_depth"],
            message=f"{len(xy)}/{len(pixels)} camera/projector depths reconstructed",
        )


CAMERA_PROJECTOR_BLOCKS = (CalibrateCameraProjector, ReconstructCameraProjector)
