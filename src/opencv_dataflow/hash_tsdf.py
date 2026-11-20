"""Native HashTSDF with conservative admission before volume growth."""

import math
from itertools import product

import cv2
import numpy as np

from .core import Port, Result
from .geometry3d_common import Tool3D, cloud_graphics
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .pointclouds import PointCloud
from .ptcloud3d import cpu_tsdf_volume, tsdf_views
from .stereo3d import metric_depth

_MIB = 1 << 20
_NATIVE_POOL_BLOCKS = 8192
_MAX_BATCH_PIXELS = 16 * _MIB


def frame_block_box(depth, transform, k, origin, side, truncation):
    """Enclose all native allocation samples, including their truncation boxes."""
    y, x = np.nonzero(depth)
    if not len(y):
        return None
    used = depth[y, x]
    corners = np.asarray(
        list(product((x.min(), x.max()), (y.min(), y.max()), (used.min(), used.max()))), float
    )
    corners[:, :2] -= k[:2, 2]
    corners[:, :2] *= corners[:, 2, None] / np.diag(k)[:2]
    shift = transform[:3, 3].astype(float) - origin
    volume = corners @ transform[:3, :3].astype(float).T + shift
    # Allow for float32 reprojection, pose composition and block-index rounding.
    scale = max(
        1.0,
        abs(corners).max(),
        abs(shift).max(),
        abs(origin).max(),
        abs(transform[:3, 3]).max(),
        abs(volume).max(),
        truncation,
    )
    padding = truncation + 64 * np.finfo(np.float32).eps * scale
    lower = np.floor((volume.min(axis=0) - padding) / side)
    upper = np.floor((volume.max(axis=0) + padding) / side)
    if np.max(np.abs([lower, upper])) > 1_000_000:
        raise ValidationError("HashTSDF block coordinates exceed the safe native range")
    return tuple(lower.astype(int)), tuple(upper.astype(int))


def admit_blocks(reserved, box, limit):
    """Return the prospective union or reject without changing the reservation."""
    lower, upper = box
    axes = [range(a, b + 1) for a, b in zip(lower, upper, strict=True)]
    if math.prod(map(len, axes)) > limit:
        return None
    candidate = reserved | set(product(*axes))
    return candidate if len(candidate) <= limit else None


def _supported_points(xyz, frames, k, tolerance, cancelled):
    """Suppress crossings that are unsupported by any admitted camera depth."""
    supported = np.zeros(len(xyz), bool)
    width, height = frames[0][0].shape[::-1]
    for depth, pose in frames:
        for start in range(0, len(xyz), 65_536):
            if cancelled():
                return None
            band = slice(start, start + 65_536)
            rows = np.flatnonzero(~supported[band])
            camera = (xyz[start + rows] - pose[:3, 3]) @ pose[:3, :3]
            in_front = camera[:, 2] > 0
            rows, camera = rows[in_front], camera[in_front]
            uv = camera[:, :2] / camera[:, 2, None] * np.diag(k)[:2] + k[:2, 2]
            inside = (uv >= -0.5).all(axis=1) & (uv < [width - 0.5, height - 0.5]).all(axis=1)
            x, y = np.floor(uv[inside] + 0.5).astype(int).T
            observed = depth[y, x]
            good = (observed > 0) & (np.abs(observed - camera[inside, 2]) <= tolerance)
            supported[start + rows[inside][good]] = True
        if supported.all():
            break
    return supported


def _extract(volume, frames, target, k, p, cancelled):
    xyz, normals = volume.fetchPointsNormals()
    if xyz is None or normals is None:
        return NO_VALUE, 0, 0
    xyz, normals = (array.reshape(-1, array.shape[-1])[:, :3] for array in (xyz, normals))
    lengths = np.linalg.norm(normals, axis=1)
    valid = np.isfinite(xyz).all(axis=1) & np.isfinite(normals).all(axis=1) & (lengths > 1e-8)
    rows = np.flatnonzero(valid)
    selected = _supported_points(xyz[rows], frames, k, float(p["surface_tolerance_m"]), cancelled)
    if selected is None:
        return NO_VALUE, len(xyz), 0
    supported = rows[selected]
    step = int(p["sample_step"])
    if (len(supported) + step - 1) // step > p["max_points"]:
        return NO_VALUE, len(xyz), len(supported)
    ordered = np.lexsort((xyz[supported, 2], xyz[supported, 1], xyz[supported, 0]))[::step]
    keep = supported[ordered]
    return (
        PointCloud(xyz[keep], "m", target, normals=normals[keep] / lengths[keep, None]),
        len(xyz),
        len(supported),
    )


class FuseHashedTSDF(Tool3D):
    type_id, title, subtype = (
        "3d.tsdf_hash",
        "Fuse depth into a bounded HashTSDF",
        "Volumetric reconstruction",
    )
    description = "Grow a sparse depth volume within block and workspace limits."
    process_inputs_readonly = True
    payload_reference = ""
    inputs = (
        Port("depth", "depth", required=False, required_when_connected=True),
        Port("frames", "any", required=False, required_when_connected=True),
        Port("poses", "data", required=False, required_when_connected=True, visual_type="pose"),
        Port("camera", "data"),
    )
    outputs = (Port("cloud", "cloud", required=False), Port("report", "data"))
    defaults = {
        "camera_poses": [],
        "growth_mode": "bounded",
        "block_resolution": "8",
        "block_limit": 512,
        "maximum_workspace_mib": 192,
        "voxel_size_m": 0.02,
        "volume_origin_m": [-0.64, -0.64, 0.5],
        "truncation_m": 0.06,
        "max_weight": 32,
        "max_depth_m": 5.0,
        "surface_tolerance_m": 0.025,
        "input_units": "m",
        "units_per_sample": 0.001,
        "sample_step": 1,
        "max_points": 200_000,
    }
    parameter_choices = {
        "growth_mode": ("bounded", "seed_only"),
        "block_resolution": ("8", "16"),
        "input_units": ("m", "mm"),
    }
    parameter_ranges = {
        "block_limit": NumericRange(1, _NATIVE_POOL_BLOCKS, whole=True),
        "maximum_workspace_mib": NumericRange(1, 512, whole=True),
        "voxel_size_m": NumericRange(1e-5, 10),
        "truncation_m": NumericRange(1e-5, 10),
        "max_weight": NumericRange(1, 255, whole=True),
        "max_depth_m": NumericRange(0.001, 1000),
        "surface_tolerance_m": NumericRange(1e-6, 10),
        "units_per_sample": NumericRange(1e-9, 1000),
        "sample_step": NumericRange(1, 256, whole=True),
        "max_points": NumericRange(1, 2_000_000, whole=True),
    }
    input_help = {
        "depth": "Single depth frame, or connect a group to Frames.",
        "frames": "Ordered group of aligned metric depth frames with one pose per frame.",
        "poses": "One camera-to-world pose per depth frame. Overrides the saved poses.",
        "camera": "Common undistorted pinhole intrinsics.",
    }
    output_help = {
        "cloud": "World-space surface with normals and support from admitted depth measurements.",
        "report": "Per-frame reservations, growth decisions and native volume bounds.",
    }
    parameter_help = {
        "camera_poses": "Saved pose records in capture order if the Poses connector is unwired.",
        "growth_mode": "Allow bounded growth or freeze allocations after the first accepted frame.",
        "block_resolution": "Voxels on each edge of a native hash block.",
        "block_limit": "Maximum number of conservatively reserved blocks.",
        "maximum_workspace_mib": "Budget for native pool, extraction and depth workspace.",
        "voxel_size_m": "Metric voxel edge length.",
        "volume_origin_m": "World XYZ of the block-grid origin in meters.",
        "truncation_m": "Native integration band in meters.",
        "max_weight": "Native observation weight saturation.",
        "max_depth_m": "Exclude camera depths beyond this range before allocation.",
        "surface_tolerance_m": "Maximum camera Z difference for a supported surface point.",
        "input_units": "Units for untagged depth.",
        "units_per_sample": "Physical units per untagged uint16 depth count.",
        "sample_step": "Keep every nth supported point in sorted XYZ order.",
        "max_points": "Maximum cloud size after sampling.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not isinstance(p["camera_poses"], list) or len(p["camera_poses"]) > 64:
            raise ValidationError("Provide a list of at most 64 camera poses", "camera_poses")
        origin = np.asarray(p["volume_origin_m"], float)
        if origin.shape != (3,) or not np.isfinite(origin).all() or abs(origin).max() > 100_000:
            raise ValidationError("Volume origin needs finite XYZ within 100 km", "volume_origin_m")
        if float(p["truncation_m"]) < float(p["voxel_size_m"]):
            raise ValidationError("Truncation must span at least one voxel", "truncation_m")

    def process(self, inputs):
        p = self.parameters
        if "poses" not in inputs:
            inputs = {**inputs, "poses": p["camera_poses"]}
        depths, transforms, target, k, size = tsdf_views(inputs, inputs["camera"])
        pixels = math.prod(size) * len(depths)
        if min(size) < 2 or pixels > _MAX_BATCH_PIXELS:
            raise ValidationError(
                "HashTSDF needs a 2×2 or larger grid and at most 16,777,216 batch pixels"
            )
        if np.max(np.abs(k)) > 1e8 or np.min(k.diagonal()[:2]) < 1e-6:
            raise ValidationError("Intrinsics exceed the safe native HashTSDF range", "camera")
        if any(not np.isfinite(t).all() or abs(t[:3, 3]).max() > 100_000 for t in transforms):
            raise ValidationError("HashTSDF poses must be finite and within 100 km", "poses")
        resolution = int(p["block_resolution"])
        origin = np.asarray(p["volume_origin_m"], np.float32).astype(float)
        k = k.astype(np.float32).astype(float)
        side = float(np.float32(p["voxel_size_m"]) * np.float32(resolution))
        trunc = float(np.float32(p["truncation_m"]))
        maximum = float(np.float32(p["max_depth_m"]))
        workspace = int(p["maximum_workspace_mib"]) * _MIB
        # CPU OpenCV 5.0 allocates an 8,192-row native pool even for a small scan.
        baseline = _NATIVE_POOL_BLOCKS * resolution**3 * 2 + 4 * pixels + 64 * math.prod(size)
        per_block = 384 * resolution**3
        limit = min(int(p["block_limit"]), (workspace - baseline) // per_block)
        if limit < 1:
            raise ValidationError(
                "HashTSDF workspace cannot accommodate its native pool and one block",
                "maximum_workspace_mib",
            )
        settings = {**p, "resolution": [resolution] * 3, "raycast_step_factor": 0.25}
        reserved, admitted, rows = set(), [], []
        bounds = None
        rejected = 0
        cloud, extracted, supported = NO_VALUE, 0, 0
        with cpu_tsdf_volume(settings, k, size, maximum, hashed=True) as volume:
            for i, (depth, pose) in enumerate(zip(depths, transforms, strict=True)):
                if self._cancelled():
                    break
                depth = metric_depth(depth, p["units_per_sample"], p["input_units"])
                depth[~np.isfinite(depth) | (depth > maximum)] = 0
                box = frame_block_box(depth, pose, k, origin, side, trunc)
                row = dict(
                    frame=i,
                    reserved_blocks=len(reserved),
                    surface_block_overlap=0,
                    status="empty_depth",
                    growth_enabled=False,
                )
                rows.append(row)
                if box is None:
                    continue
                lower, upper = box
                row["surface_block_overlap"] = sum(
                    all(a <= v <= b for a, v, b in zip(lower, cell, upper, strict=True))
                    for cell in reserved
                )
                growth = p["growth_mode"] == "bounded" or not admitted
                if not growth and not row["surface_block_overlap"]:
                    row["status"] = "outside_seed_reservation"
                    continue
                if growth:
                    proposed = admit_blocks(reserved, box, limit)
                    if proposed is None:
                        rejected += 1
                        row["status"] = "rejected_before_growth"
                        continue
                    reserved = proposed
                volume.setEnableGrowth(growth)
                volume.integrate(depth, pose)
                admitted.append((depth, pose))
                row.update(
                    status="integrated",
                    reserved_blocks=len(reserved),
                    growth_enabled=volume.getEnableGrowth(),
                    native_visible_blocks=int(volume.getVisibleBlocks()),
                    native_volume_units=int(volume.getTotalVolumeUnits()),
                )
            if admitted and not self._cancelled():
                native_bounds = volume.getBoundingBox(cv2.VOLUME_VOLUME_UNIT)
                if native_bounds is not None:
                    bounds = (np.asarray(native_bounds).reshape(2, 3) + origin).tolist()
                cloud, extracted, supported = _extract(
                    volume, admitted, target, k, p, self._cancelled
                )
        report = dict(
            world_frame=target,
            units="m",
            block_edge_m=side,
            block_limit=limit,
            requested_block_limit=int(p["block_limit"]),
            reserved_blocks=len(reserved),
            frames=rows,
            integrated_frames=len(admitted),
            rejected_frames=rejected,
            bounds_world_m=bounds,
            workspace_estimate_bytes=baseline + per_block * len(reserved),
            allocation_counter="conservative reservation; native counters are diagnostic only",
            extracted_points=extracted,
            supported_points=supported,
            surface_points=0 if cloud is NO_VALUE else len(cloud.points),
        )
        reasons = []
        if self._cancelled():
            cloud = NO_VALUE
            report["surface_points"] = 0
            reasons.append("hash_tsdf_cancelled")
        if rejected:
            reasons.append("hash_tsdf_growth_limit")
        if cloud is NO_VALUE or not len(cloud.points):
            reasons.append("hash_tsdf_point_limit" if supported else "hash_tsdf_no_surface")
        source = "frames" if "frames" in inputs else "depth"
        ticket = self.input_payloads.get(source, {}).get("ticket")
        payloads = {key: {"ticket": ticket} for key in ("cloud", "report")} if ticket else {}
        return Result(
            {"cloud": cloud, "report": report},
            passed=not reasons,
            fail_reasons=reasons,
            payloads=payloads,
            graphics=cloud_graphics(self, cloud) if cloud is not NO_VALUE else {},
            message=(
                f"Growth limit rejected {rejected} views · {len(reserved)} reserved blocks"
                if rejected
                else f"{len(admitted)} integrated views · {len(reserved)} reserved blocks"
            ),
        )
