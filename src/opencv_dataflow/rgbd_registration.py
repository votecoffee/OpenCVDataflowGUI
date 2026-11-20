"""Align camera-Z depth and its confidence to a separate color camera grid."""

import numpy as np

from .calibration_revisions import is_full_grid
from .cloud_projection import MAX_PIXELS, METERS, nearest_samples, raster_indices
from .core import Port, Result
from .depth_smoothing import normalize_depth_confidence
from .geometry3d_common import Tool3D, camera_for_image, require_pinhole, rigid
from .images import DepthArray, ImageArray
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, require_same_frame
from .stereo3d import metric_depth
from .tickets import ticket_key

_BAND_SAMPLES = 65_536
_FLOAT32_MAX = float(np.finfo(np.float32).max)


def _grid(model, image, payload):
    k, d, size = require_pinhole(model)
    camera_for_image(model, image)
    if size[0] * size[1] > MAX_PIXELS:
        raise ValidationError("Camera grid exceeds the 8,388,608 pixel registration limit")
    if not is_full_grid(payload, size):
        raise ValidationError("Use the full camera pixel grid for RGB-D registration")
    if payload.get("optical_frame", model["frame"]) != model["frame"]:
        raise ValidationError("Image optical frame does not match the camera model")
    return k, d, size


def _project(depth, source_k, target, rotation, translation):
    target_k, distortion, size = target
    nearest = np.full(size[0] * size[1], np.inf, np.float64)
    owners = np.full(len(nearest), -1, np.int32)
    flat = depth.ravel()
    width = depth.shape[1]
    projected = 0
    for start in range(0, len(flat), _BAND_SAMPLES):
        z_band = flat[start : start + _BAND_SAMPLES]
        indices = np.flatnonzero(np.isfinite(z_band))
        if not len(indices):
            continue
        z = z_band[indices].astype(np.float64)
        source = indices + start
        y, x = np.divmod(source, width)
        xyz = np.column_stack(
            (
                (x - source_k[0, 2]) / source_k[0, 0] * z,
                (y - source_k[1, 2]) / source_k[1, 1] * z,
                z,
            )
        )
        xyz = xyz @ rotation.T + translation
        target_z = xyz[:, 2]
        candidates = np.flatnonzero(
            np.isfinite(xyz).all(axis=1) & (target_z > 0) & (target_z <= _FLOAT32_MAX)
        )
        cells = raster_indices(xyz, candidates, target_k, distortion, size)
        inside = cells >= 0
        samples, cells = candidates[inside], cells[inside]
        projected += len(samples)
        zs = target_z[samples]
        sorted_cells, selected = nearest_samples(samples, cells, zs)
        zs = target_z[selected]
        update = zs < nearest[sorted_cells]
        destinations = sorted_cells[update]
        nearest[destinations] = zs[update]
        owners[destinations] = source[selected[update]]
    return nearest.reshape(size[::-1]), owners.reshape(size[::-1]), projected


class RegisterRGBD(Tool3D):
    type_id, title, subtype = "3d.register_rgbd", "Register RGB-D with confidence", "RGB-D"
    inputs = (
        Port("color"),
        Port("depth", "depth"),
        Port("depth_camera", "data"),
        Port("color_camera", "data"),
        Port("transform", "data", visual_type="pose"),
        Port("confidence", required=False),
    )
    outputs = (
        Port("color"),
        Port("depth", "depth"),
        Port("confidence", required=False),
        Port("valid_mask", visual_type="region"),
        Port("report", "data"),
    )
    defaults = {
        "input_units": "m",
        "units_per_sample": 0.001,
        "minimum_confidence": 0.0,
        "display_max_m": 5.0,
    }
    parameter_choices = {"input_units": ("m", "mm")}
    parameter_ranges = {
        "units_per_sample": NumericRange(1e-9, 1000),
        "minimum_confidence": NumericRange(0, 1),
        "display_max_m": NumericRange(0.001, 1000),
    }
    parameter_help = {
        "input_units": "Units for untagged depth. Tagged depth retains its declared units.",
        "units_per_sample": "Physical units per sample for untagged uint16 depth.",
        "minimum_confidence": "Reject the closest sample below this confidence. Zero is invalid.",
        "display_max_m": "Upper preview bound in meters. Does not clip physical depth.",
    }
    input_help = {
        "depth_camera": "Undistorted pinhole intrinsics on the source depth grid.",
        "color_camera": "Target pinhole intrinsics, including any color lens distortion.",
        "transform": "Rigid pose mapping the depth optical frame to the color optical frame.",
        "confidence": "Float [0,1] or integer full-scale confidence on the depth grid.",
    }
    flow_primary = ("color", "depth")
    payload_reference = "color"
    parallel_safe = preview_safe = process_inputs_readonly = True

    def process(self, inputs):
        p = self.parameters
        color, depth = inputs["color"], inputs["depth"]
        a, b = inputs["depth_camera"], inputs["color_camera"]
        color_payload, depth_payload = (self.input_payloads[key] for key in ("color", "depth"))
        ka, da, _ = _grid(a, depth, depth_payload)
        target = _grid(b, color, color_payload)
        if np.any(da):
            raise ValidationError("Undistort source depth before RGB-D registration")
        pose = inputs["transform"]
        if (
            not isinstance(pose, dict)
            or pose.get("source_frame") != a["frame"]
            or pose.get("target_frame") != b["frame"]
            or pose.get("units") not in METERS
        ):
            raise ValidationError(
                "Pose must map the depth optical frame to the color frame in metric units"
            )
        transform = rigid(pose)
        tickets = {ticket_key(v["ticket"]) for v in self.input_payloads.values() if "ticket" in v}
        if len(tickets) > 1:
            raise ValidationError("RGB-D inputs must refer to the same ticket")

        confidence = inputs.get("confidence")
        minimum = float(p["minimum_confidence"])
        if confidence is not None:
            if confidence.shape != depth.shape:
                raise ValidationError("Confidence dimensions must match depth")
            require_same_frame(
                (confidence, depth), message="Confidence belongs to a different depth grid"
            )
            confidence = normalize_depth_confidence(confidence)
        elif minimum:
            raise ValidationError(
                "Minimum confidence requires the confidence input", "minimum_confidence"
            )

        source_z = metric_depth(depth, p["units_per_sample"], p["input_units"])
        rotation = transform[:3, :3]
        translation = transform[:3, 3] * METERS[pose["units"]]
        nearest, owners, projected = _project(source_z, ka, target, rotation, translation)
        covered = owners >= 0
        valid = covered.copy()
        target_confidence = NO_VALUE
        if confidence is not None:
            scores = np.zeros(nearest.shape, np.float32)
            scores[covered] = confidence.ravel()[owners[covered]]
            valid &= (scores > 0) & (scores >= minimum)
            scores[~valid] = 0
            target_confidence = ImageArray(scores, "gray")
        nearest[~valid] = np.nan
        depth_out = DepthArray(nearest.astype(np.float32), "m", 0, float(p["display_max_m"]))
        covered_count = int(covered.sum())
        valid_count = int(valid.sum())
        coverage = valid_count / valid.size
        report = {
            "model": "registered_rgbd_v1",
            "source_frame": a["frame"],
            "target_frame": b["frame"],
            "depth_units": "m",
            "source_size": list(depth.shape[::-1]),
            "target_size": list(target[2]),
            "projected_samples": projected,
            "occluded_samples": projected - covered_count,
            "confidence_rejected_pixels": covered_count - valid_count,
            "registered_pixels": valid_count,
            "coverage": coverage,
        }
        target_payload = dict(color_payload)
        if "recorded_rgbd" in depth_payload:
            target_payload["recorded_depth_sample"] = depth_payload["recorded_rgbd"]
        return Result(
            {
                "color": color,
                "depth": depth_out,
                "confidence": target_confidence,
                "valid_mask": ImageArray(valid.astype(np.uint8) * 255, "gray"),
                "report": report,
            },
            payloads={name: target_payload for name in ("depth", "confidence", "valid_mask")},
            passed=bool(valid_count),
            score=coverage,
            fail_reasons=[] if valid_count else ["rgbd_no_registered_depth"],
            message=f"{valid_count}/{valid.size} registered depth pixels",
        )
