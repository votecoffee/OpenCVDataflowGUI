"""Bounded native fusion of registered, posed RGB-D views."""

import math

import numpy as np

from .calibration_revisions import REVISION, check_capture, is_full_grid
from .cloud_projection import METERS
from .core import Port, Result, validate_value
from .depth_smoothing import normalize_depth_confidence
from .geometry3d_common import Tool3D, cloud_graphics, require_pinhole, rigid
from .images import image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, FlowGroup, ValuePacket, metadata, require_same_frame
from .pointclouds import PointCloud
from .ptcloud3d import FuseTSDF, cpu_tsdf_volume, fixed_volume_resolution
from .stereo3d import metric_depth
from .tickets import ticket_key

_REQUIRED = ("color", "depth", "pose")
_MIB = 1 << 20
_MAX_BATCH_PIXELS = 16 * _MIB


def _fields(group):
    if any(not isinstance(label, str) for label in group.labels):
        raise ValidationError("Packed RGB-D fields require text labels")
    fields = dict(zip(group.labels, group.items, strict=True))
    if (
        not set(_REQUIRED) <= fields.keys()
        or len(fields) != len(group.items)
        or not fields.keys() <= {*_REQUIRED, "confidence"}
    ):
        raise ValidationError("RGB-D views need color, depth and pose, with optional confidence")
    return fields


def _check_tickets(fields):
    tickets = {
        ticket_key(packet.payload["ticket"])
        for packet in fields.values()
        if "ticket" in packet.payload
    }
    if len(tickets) > 1:
        raise ValidationError("Color, depth, confidence and pose must refer to the same capture")


class PackPosedRGBD(Tool3D):
    type_id, title, subtype = "3d.pack_posed_rgbd", "Pack posed RGB-D view", "RGB-D"
    description = "Keep registered color, depth, a camera pose and optional confidence together."
    payload_reference = "depth"
    process_inputs_readonly = parallel_safe = preview_safe = True
    inputs = (
        Port("color"),
        Port("depth", "depth"),
        Port("pose", "data", visual_type="pose"),
        Port("confidence", required=False, required_when_connected=True),
    )
    outputs = (Port("view", "any"),)
    input_help = {
        "color": "Color aligned to the depth grid.",
        "depth": "Registered metric camera Z.",
        "pose": "Rigid pose mapping the camera to the world frame.",
        "confidence": "Optional confidence plane, float [0,1] or unsigned integer full scale.",
    }

    def process(self, inputs):
        fields = {
            name: ValuePacket(value, self.input_payloads[name]) for name, value in inputs.items()
        }
        _check_tickets(fields)
        if inputs["color"].shape[:2] != inputs["depth"].shape:
            raise ValidationError("Color and depth must share the same image dimensions")
        if "confidence" in inputs and inputs["confidence"].shape != inputs["depth"].shape:
            raise ValidationError("Confidence must match the scalar depth grid", "confidence")
        return Result({"view": FlowGroup(tuple(fields.values()), tuple(fields))})


def _views(group):
    if not isinstance(group, FlowGroup) or any(not isinstance(v, str) for v in group.labels):
        raise ValidationError("Connect one or more packed posed RGB-D views", "views")
    if all(isinstance(packet.value, FlowGroup) for packet in group.items):
        return [(label, _fields(packet.value)) for label, packet in zip(group.labels, group.items)]
    return [("view", _fields(group))]


def _view_transform(fields, model, size, grid_policy):
    _check_tickets(fields)
    shape = size[::-1]
    optical_frame = model.get("frame", "camera")
    raster_packets = [packet for name, packet in fields.items() if name != "pose"]
    for name, packet in fields.items():
        if name == "pose":
            continue
        value = packet.value
        validate_value(Port(name, "depth" if name == "depth" else "image"), value)
        if value.shape[:2] != shape or (name != "color" and value.ndim != 2):
            raise ValidationError(f"{name.capitalize()} must match the calibrated grid", name)
        payload = metadata(packet)
        if "spatial" in payload and not is_full_grid(payload, size):
            raise ValidationError(f"{name.capitalize()} must use the full calibrated grid", name)
        if payload.get("optical_frame", optical_frame) != optical_frame:
            raise ValidationError(f"{name.capitalize()} optical frame differs from camera", name)
        check_capture(model.get(REVISION), payload, size)
    if grid_policy == "shared_root":
        require_same_frame(raster_packets, message="RGB-D planes have different pixel grids")
    pose = fields["pose"].value
    if (
        not isinstance(pose, dict)
        or pose.get("source_frame") != optical_frame
        or not isinstance(pose.get("target_frame"), str)
        or not pose["target_frame"].strip()
        or pose.get("units") not in METERS
    ):
        raise ValidationError("Each metric pose must map the camera to a named world frame", "pose")
    matrix = rigid(pose).copy()
    matrix[:3, 3] *= METERS[pose["units"]]
    if np.max(np.abs(matrix[:3, 3])) > 100_000:
        raise ValidationError("Pose translations must be within 100 km of the world origin", "pose")
    return matrix.astype(np.float32), pose["target_frame"]


def _prepare_planes(fields, p):
    color = fields["color"].value
    encoding = image_format(color)
    if color.dtype != np.uint8 or color.shape[2:] != (3,) or encoding not in ("rgb", "bgr"):
        raise ValidationError("ColorTSDF requires three-channel uint8 RGB or BGR", "color")
    rgb = np.ascontiguousarray(color[..., ::-1] if encoding == "bgr" else color, dtype=np.float32)
    depth = metric_depth(fields["depth"].value, p["units_per_sample"], p["input_units"])
    valid = np.isfinite(depth) & (depth <= float(p["max_depth_m"]))
    valid_depth = int(valid.sum())
    confidence = fields.get("confidence")
    minimum = float(p["minimum_confidence"])
    if confidence is not None:
        scores = normalize_depth_confidence(confidence.value)
        valid &= (scores > 0) & (scores >= minimum)
    elif minimum:
        raise ValidationError(
            "Minimum confidence requires confidence in every view", "minimum_confidence"
        )
    depth[~valid] = 0
    used = int(valid.sum())
    return (
        depth,
        rgb,
        dict(
            valid_depth_pixels=valid_depth,
            integrated_pixels=used,
            confidence_rejected_pixels=valid_depth - used,
        ),
    )


def _extract(output, target, p):
    if all(array is None for array in output):
        return None
    xyz, normals, rgb = [array.reshape(-1, array.shape[-1])[:, :3] for array in output]
    lengths = np.linalg.norm(normals, axis=1)
    valid = np.isfinite(xyz).all(axis=1) & np.isfinite(normals).all(axis=1)
    valid &= np.isfinite(rgb).all(axis=1) & (lengths > 1e-8)
    rows = np.flatnonzero(valid)
    step = int(p["sample_step"])
    if (len(rows) + step - 1) // step > p["max_points"]:
        raise ValidationError("Surface exceeds maximum points. Increase sample step.", "max_points")
    # Native extraction ranges can finish in a different order on each thread.
    rows = rows[np.lexsort((xyz[rows, 2], xyz[rows, 1], xyz[rows, 0]))][::step]
    colors = np.rint(np.clip(rgb[rows], 0, 255)).astype(np.uint8)
    return PointCloud(
        xyz[rows], "m", target, normals=normals[rows] / lengths[rows, None], colors=colors
    )


class FuseColorTSDF(Tool3D):
    type_id, title, subtype = (
        "3d.tsdf_color",
        "Fuse posed RGB-D frames (ColorTSDF)",
        "Volumetric reconstruction",
    )
    description = "Reconstruct a fixed-volume colored surface from registered, posed depth and RGB."
    payload_reference = ""
    process_inputs_readonly = True
    inputs = (Port("views", "any"), Port("camera", "data"))
    outputs = (Port("cloud", "cloud", required=False), Port("report", "data"))
    defaults = {
        **{
            name: FuseTSDF.defaults[name]
            for name in (
                "resolution",
                "volume_origin_m",
                "voxel_size_m",
                "truncation_m",
                "max_weight",
                "max_depth_m",
                "input_units",
                "units_per_sample",
                "sample_step",
                "max_points",
            )
        },
        "grid_policy": "shared_root",
        "minimum_confidence": 0.0,
        "maximum_workspace_mib": 192,
    }
    parameter_choices = {
        "grid_policy": ("shared_root", "calibrated_grid"),
        "input_units": ("m", "mm"),
    }
    parameter_ranges = {
        "voxel_size_m": NumericRange(1e-5, 100),
        "truncation_m": NumericRange(1e-5, 100),
        "max_weight": NumericRange(1, 255, whole=True),
        "max_depth_m": NumericRange(0.001, 10_000),
        "units_per_sample": NumericRange(1e-9, 1000),
        "sample_step": NumericRange(1, 256, whole=True),
        "max_points": NumericRange(1, 2_000_000, whole=True),
        "minimum_confidence": NumericRange(0, 1),
        "maximum_workspace_mib": NumericRange(1, 512, whole=True),
    }
    parameter_help = {
        "resolution": "Fixed XYZ volume dimensions. Each dimension must be a multiple of eight.",
        "volume_origin_m": "Minimum corner of the volume in the world frame, in meters.",
        "voxel_size_m": "Metric voxel edge length.",
        "truncation_m": "Signed-distance band in meters.",
        "max_weight": "Cap on the native running TSDF observation weight.",
        "max_depth_m": "Discard observations past this camera Z in meters.",
        "input_units": "Physical units of untagged depth.",
        "units_per_sample": "Physical units per raw uint16 depth sample.",
        "sample_step": "Keep every nth extracted point in XYZ order.",
        "max_points": "Maximum colored points after sampling.",
        "grid_policy": "Require compatible provenance or explicitly assert calibrated alignment.",
        "minimum_confidence": "Discard zero confidence and values below this threshold.",
        "maximum_workspace_mib": "Budget for native volume, extraction and frame workspace.",
    }
    input_help = {
        "views": "One packed posed RGB-D view or a group of up to 64 packed views.",
        "camera": "Common undistorted pinhole intrinsics for the registered optical grid.",
    }
    output_help = {
        "cloud": "World-space points in meters with unit normals and fused RGB.",
        "report": "Native workspace estimate and per-view valid and confidence-rejected counts.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        try:
            fixed_volume_resolution(p)
        except (ValueError, TypeError) as exc:
            raise ValidationError(str(exc), "resolution") from exc
        origin = np.asarray(p["volume_origin_m"], float)
        if (
            origin.shape != (3,)
            or not np.isfinite(origin).all()
            or np.max(np.abs(origin)) > 100_000
        ):
            raise ValidationError(
                "Volume origin must be finite XYZ within 100 km", "volume_origin_m"
            )
        if float(p["truncation_m"]) < float(p["voxel_size_m"]):
            raise ValidationError("Truncation must span at least one voxel", "truncation_m")

    def process(self, inputs):
        p, model = self.parameters, inputs["camera"]
        k, distortion, size = require_pinhole(model)
        if np.any(distortion) or min(size) < 2:
            raise ValidationError(
                "ColorTSDF requires an undistorted pinhole grid of at least 2×2", "camera"
            )
        views = _views(inputs["views"])
        pixels = math.prod(size)
        if pixels * len(views) > _MAX_BATCH_PIXELS:
            raise ValidationError("RGB-D batch exceeds 16,777,216 calibrated pixels", "views")
        voxels = math.prod(fixed_volume_resolution(p))
        # Includes three crossings per voxel, native XYZ/normal/RGB vectors,
        # their temporary capacity and returned arrays, plus one prepared raster.
        workspace = 512 * voxels + 64 * pixels
        if workspace > p["maximum_workspace_mib"] * _MIB:
            raise ValidationError("ColorTSDF exceeds the workspace budget", "maximum_workspace_mib")
        transforms = [_view_transform(fields, model, size, p["grid_policy"]) for _, fields in views]
        target = transforms[0][1]
        if any(world != target for _, world in transforms):
            raise ValidationError("All RGB-D poses must share one world frame", "views")
        ticket = self.input_payloads["views"].get("ticket")
        payloads = {"cloud": {"ticket": ticket}, "report": {"ticket": ticket}} if ticket else {}
        report = dict(
            frame=target,
            units="m",
            voxels=voxels,
            workspace_estimate_bytes=workspace,
            integrated_frames=0,
            surface_points=0,
            views=[],
        )

        def failure(reason, message):
            return Result(
                {"cloud": NO_VALUE, "report": report},
                payloads=payloads,
                passed=False,
                fail_reasons=[reason],
                message=message,
            )

        if self._cancelled():
            return failure("color_tsdf_cancelled", "ColorTSDF cancelled")
        volume_settings = {**p, "raycast_step_factor": 0.25}
        cloud = None
        with cpu_tsdf_volume(
            volume_settings, k, size, float(p["max_depth_m"]), color=True
        ) as volume:
            for (label, fields), (transform, _) in zip(views, transforms, strict=True):
                if self._cancelled():
                    return failure("color_tsdf_cancelled", "ColorTSDF cancelled")
                depth, color, stats = _prepare_planes(fields, p)
                report["views"].append({"view": label, **stats})
                if stats["integrated_pixels"]:
                    volume.integrateColor(depth, color, transform)
                    report["integrated_frames"] += 1
            if report["integrated_frames"] and not self._cancelled():
                cloud = _extract(volume.fetchPointsNormalsColors(), target, p)
        if self._cancelled():
            return failure("color_tsdf_cancelled", "ColorTSDF cancelled")
        if cloud is None or not len(cloud.points):
            return failure("color_tsdf_empty", "No colored surface was recovered in the volume")
        report["surface_points"] = len(cloud.points)
        return Result(
            {"cloud": cloud, "report": report},
            payloads=payloads,
            graphics=cloud_graphics(self, cloud),
            message=f"Fused {report['integrated_frames']} views · {len(cloud.points)} points",
        )
