"""Forward-project selected positive camera-Z samples between calibrated views."""

import cv2
import numpy as np

from .calibration_revisions import REVISION, applied_payload
from .cloud_projection import MAX_PIXELS, METERS, as_mask, pixel_cells
from .core import Port, Result
from .geometry3d_common import (
    Tool3D,
    camera,
    camera_for_image,
    project_camera_pixels,
    rigid,
    undistort_camera_pixels,
)
from .graphics_layers import GraphicDrawing
from .images import DepthArray
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, frame, metadata, require_same_frame
from .raster_regions import MAX_REGION_RUNS, region_from_mask
from .selection import selection_mask
from .selection_remap import boundaries
from .stereo3d import metric_depth
from .tickets import ticket_key

_BATCH = 16_384
_ZERO = np.zeros(3, dtype=np.float64)


def _pose_between(source, target, pose):
    if (
        not isinstance(pose, dict)
        or pose.get("source_frame") != source["frame"]
        or pose.get("target_frame") != target["frame"]
        or pose.get("units") not in METERS
    ):
        raise ValidationError("Transform must map source to target optical frames in m or mm")
    transform = rigid(pose).copy()
    transform[:3, 3] *= METERS[pose["units"]]
    return transform


def _lift_samples(xy, z, model, optics, tolerance):
    k, d, _ = optics
    rays = undistort_camera_pixels(xy, model, k, d)
    usable = np.isfinite(rays).all(axis=1)
    indices = np.flatnonzero(usable)
    if len(indices):
        direction = np.column_stack((rays[indices], np.ones(len(indices))))
        # Reject inverse-model artifacts, including rear-facing rays folded onto positive Z.
        check = project_camera_pixels(direction, model, k, d, _ZERO, _ZERO)
        distance = np.hypot(*(check - xy[indices]).T)
        good = np.isfinite(distance) & (distance <= tolerance)
        usable[indices[~good]] = False
        return usable, direction[good] * z[usable, None]
    return usable, np.empty((0, 3), np.float64)


def _update_buffer(cells, z, source, depth, owner):
    if not len(cells):
        return
    order = np.lexsort((source, z, cells))
    sorted_cells = cells[order]
    first = np.r_[True, sorted_cells[1:] != sorted_cells[:-1]]
    best = order[first]
    best_cells = cells[best]
    # Chunks run in source scanline order. Keeping an equal old Z preserves the earlier sample.
    wins = z[best] < depth[best_cells]
    depth[best_cells[wins]] = z[best[wins]]
    owner[best_cells[wins]] = source[best[wins]]


def _project_region(z, samples, source, target, source_optics, target_optics, transform, p):
    k, d, size = target_optics
    depth = np.full(size[0] * size[1], np.inf, np.float64)
    owner = np.full(depth.shape, -1, np.int32)
    lowest, highest = float(p["minimum_target_z_m"]), float(p["maximum_target_z_m"])
    tolerance = float(p["inverse_tolerance_px"])
    invalid_ray = outside_depth = outside_frame = projected = 0
    for start in range(0, len(samples), _BATCH):
        ids = samples[start : start + _BATCH]
        y, x = np.divmod(ids, z.shape[1])
        xy = np.column_stack((x, y)).astype(np.float64)
        usable, xyz = _lift_samples(xy, z.flat[ids], source, source_optics, tolerance)
        invalid_ray += int(np.count_nonzero(~usable))
        xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
        keep = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] >= lowest) & (xyz[:, 2] <= highest)
        outside_depth += int(np.count_nonzero(~keep))
        if not keep.any():
            continue
        xyz = xyz[keep]
        uv = project_camera_pixels(xyz, target, k, d, _ZERO, _ZERO)
        inside, cells = pixel_cells(uv, size)
        outside_frame += int(np.count_nonzero(~inside))
        projected += len(cells)
        _update_buffer(cells, xyz[inside, 2], ids[usable][keep][inside], depth, owner)
    hits = np.flatnonzero(owner >= 0)
    return (
        hits,
        depth[hits],
        owner[hits],
        {
            "invalid_source_rays": invalid_ray,
            "outside_target_depth_samples": outside_depth,
            "outside_target_grid_samples": outside_frame,
            "projected_samples": projected,
            "colliding_samples": projected - len(hits),
        },
    )


class ReprojectDepthRegion(Tool3D):
    type_id, title, subtype = (
        "3d.reproject_depth_region",
        "Reproject depth-backed region",
        "Depth and RGB-D",
    )
    inputs = (
        Port("source_depth", "depth"),
        Port("source_camera", "data"),
        Port("target_camera", "data"),
        Port("transform", "data", visual_type="pose"),
        Port("mask", required=False, visual_type="region", required_when_connected=True),
        Port("region", "data", required=False, visual_type="region", required_when_connected=True),
        Port("target_image", required=False, required_when_connected=True),
        Port("target_depth", "depth", required=False, required_when_connected=True),
    )
    outputs = (
        Port("mask", visual_type="region"),
        Port("region", "data", required=False, visual_type="region"),
        Port("depth", "depth"),
        Port("source_support", visual_type="region"),
        Port("occluded_mask", visual_type="region"),
        Port("unverified_mask", visual_type="region"),
        Port("report", "data"),
    )
    defaults = {
        "source_units": "m",
        "source_units_per_sample": 0.001,
        "target_units": "m",
        "target_units_per_sample": 0.001,
        "visibility_policy": "reject_occluded",
        "reference_tolerance_m": 0.01,
        "minimum_target_z_m": 0.001,
        "maximum_target_z_m": 10.0,
        "inverse_tolerance_px": 0.25,
        "minimum_retained_fraction": 0.0,
        "emit_region": True,
        "maximum_region_runs": 100_000,
    }
    parameter_choices = {
        "source_units": ("m", "mm"),
        "target_units": ("m", "mm"),
        "visibility_policy": ("reject_occluded", "require_agreement"),
    }
    parameter_ranges = {
        "source_units_per_sample": NumericRange(1e-9, 1000),
        "target_units_per_sample": NumericRange(1e-9, 1000),
        "reference_tolerance_m": NumericRange(0),
        "minimum_target_z_m": NumericRange(1e-6, 100_000),
        "maximum_target_z_m": NumericRange(1e-6, 100_000),
        "inverse_tolerance_px": NumericRange(1e-6, 4.0),
        "minimum_retained_fraction": NumericRange(0, 1),
        "maximum_region_runs": NumericRange(1, MAX_REGION_RUNS, whole=True),
    }
    parameter_help = {
        "source_units": "Physical unit of an untagged source depth image.",
        "source_units_per_sample": "Physical units per raw uint16 source sample.",
        "target_units": "Physical unit of an untagged target depth image.",
        "target_units_per_sample": "Physical units per raw uint16 target sample.",
        "visibility_policy": "Keep unverified projections, or require matching target depth.",
        "reference_tolerance_m": "Allowed axial-Z difference for target depth agreement.",
        "minimum_target_z_m": "Clip target camera Z below this positive distance.",
        "maximum_target_z_m": "Clip target camera Z above this distance.",
        "inverse_tolerance_px": "Maximum source pixel round-trip error for a usable ray.",
        "minimum_retained_fraction": "Required fraction of selected source pixels retained.",
        "emit_region": "Encode the exact target mask as a reusable scanline region.",
        "maximum_region_runs": "Bound the exact region encoding.",
    }
    input_help = {
        "source_depth": "Positive axial Z on the supplied source camera's pixel grid.",
        "region": "Select a patch in source depth pixels. With Mask, use their intersection.",
        "mask": "Optional uint8 source selection.",
        "target_image": "Optional target raster that supplies output coordinates and an overlay.",
        "target_depth": "Optional observed axial Z used to test projection visibility.",
        "transform": "Rigid source-optical to target-optical pose, in meters or millimeters.",
    }
    payload_reference = "source_depth"
    parallel_safe = preview_safe = process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if float(p["minimum_target_z_m"]) >= float(p["maximum_target_z_m"]):
            raise ValidationError("Maximum target Z must exceed minimum target Z")
        if type(p["emit_region"]) is not bool:
            raise ValidationError("Emit region must be a boolean", "emit_region")

    def _target_payload(self, inputs, model, size):
        contexts = [inputs[name] for name in ("target_image", "target_depth") if name in inputs]
        target = {"spatial": frame(size, f"{self.id}:target")}
        if contexts:
            for value in contexts:
                camera_for_image(model, value)
            require_same_frame(
                contexts, message="Target context rasters use different pixel frames"
            )
            records = [metadata(value) for value in contexts]
            source = self.input_payloads["source_depth"]
            tickets = {
                ticket_key(item["ticket"]) for item in [source, *records] if "ticket" in item
            }
            if len(tickets) > 1:
                raise ValidationError("Source and target rasters must refer to the same ticket")
            primary = next(
                (item for item in records if item["spatial"].get("root_id") is not None), records[0]
            )
            target.update(
                {name: primary[name] for name in ("spatial", "pyramid") if name in primary}
            )
            target_ticket = next((item["ticket"] for item in records if "ticket" in item), None)
            if "ticket" not in source and target_ticket is not None:
                target["ticket"] = target_ticket
        return applied_payload(target, model.get(REVISION))

    def process(self, inputs):
        p = self.parameters
        source, target = inputs["source_camera"], inputs["target_camera"]
        source_depth = inputs["source_depth"]
        source_optics = camera_for_image(source, source_depth)
        target_optics = camera(target)
        size = target_optics[2]
        if source_depth.size > MAX_PIXELS or size[0] * size[1] > MAX_PIXELS:
            raise ValidationError("Source and target grids are limited to 8,388,608 pixels each")
        if source.get("units") not in METERS or target.get("units") not in METERS:
            raise ValidationError("Both camera models must declare metric units")
        transform = _pose_between(source, target, inputs["transform"])
        target_payload = self._target_payload(inputs, target, size)

        measured = inputs.get("target_depth")
        strict = p["visibility_policy"] == "require_agreement"
        if strict and measured is None:
            raise ValidationError(
                "Require agreement needs a target depth input", "visibility_policy"
            )
        selection = selection_mask(inputs, source_depth)
        valid = metric_depth(source_depth, p["source_units_per_sample"], p["source_units"])
        selected_count = (
            source_depth.size if selection is None else int(np.count_nonzero(selection))
        )
        candidates = np.isfinite(valid)
        if selection is not None:
            candidates &= selection != 0
        samples = np.flatnonzero(candidates)
        cells, z, owner, diagnostics = _project_region(
            valid, samples, source, target, source_optics, target_optics, transform, p
        )
        occluded = np.zeros(len(cells), bool)
        unverified = np.ones(len(cells), bool)
        if measured is not None:
            reference = metric_depth(measured, p["target_units_per_sample"], p["target_units"])
            delta = z - reference.flat[cells]
            tolerance = float(p["reference_tolerance_m"])
            occluded = delta > tolerance
            unverified = ~np.isfinite(delta) | (delta < -tolerance)
        allowed = ~(occluded | (unverified if strict else False))
        accepted = cells[allowed]
        mask = as_mask(size, accepted)
        depth = np.full(size[::-1], np.nan, np.float32)
        depth.flat[accepted] = z[allowed]
        retained = len(accepted) / selected_count if selected_count else 0.0
        reasons = []
        if not len(accepted):
            reasons.append("depth_region_no_coverage")
        elif retained < float(p["minimum_retained_fraction"]):
            reasons.append("depth_region_retention_below_minimum")
        output_region = (
            region_from_mask(mask, int(p["maximum_region_runs"])) if p["emit_region"] else NO_VALUE
        )
        outputs = {
            "mask": mask,
            "region": output_region,
            "depth": DepthArray(
                depth, "m", float(p["minimum_target_z_m"]), float(p["maximum_target_z_m"])
            ),
            "source_support": as_mask(source_optics[2], owner[allowed]),
            "occluded_mask": as_mask(size, cells[occluded]),
            "unverified_mask": as_mask(size, cells[unverified]),
            "report": {
                "model": "depth_region_projection_v1",
                "source_camera_frame": source["frame"],
                "target_camera_frame": target["frame"],
                "selected_source_pixels": selected_count,
                "source_depth_pixels": len(samples),
                "invalid_source_depth_pixels": selected_count - len(samples),
                **diagnostics,
                "candidate_target_pixels": len(cells),
                "occluded_target_pixels": int(np.count_nonzero(occluded)),
                "unverified_target_pixels": int(np.count_nonzero(unverified)),
                "retained_target_pixels": len(accepted),
                "retained_source_fraction": retained,
                "target_depth_compared": measured is not None,
            },
        }
        graphics, drawings = {}, {}
        if self.generate_graphics:
            graphics["Coverage"] = cv2.merge(
                (outputs["unverified_mask"], mask, outputs["occluded_mask"])
            )
            if "target_image" in inputs:
                objects = boundaries(mask, 4096)
                if objects:
                    drawings["Reprojected region"] = GraphicDrawing(objects, "target_image")
        return Result(
            outputs,
            payloads={key: target_payload for key in outputs if key != "source_support"},
            graphics=graphics,
            drawings=drawings,
            passed=not reasons,
            score=retained,
            message=f"{len(accepted)}/{selected_count} selected pixels retained",
            fail_reasons=reasons,
        )
