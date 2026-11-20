"""Lift requested depth pixels into the calibrated camera's metric frame."""

from collections import Counter

import cv2
import numpy as np

from .core import Port, Result
from .geometry3d_common import Tool3D, camera_for_image, cloud_graphics
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import ValuePacket, require_same_frame
from .pointclouds import PointCloud
from .selection import selection_mask
from .tickets import ticket_key


def _requests(value, maximum):
    if not isinstance(value, (list, tuple)) or len(value) > maximum:
        raise ValidationError(f"Provide a list of up to {maximum} [x, y] pairs", "pixel_points")
    if not value:
        return np.empty((0, 2), np.float64)
    try:
        points = np.asarray(value)
        if points.shape == (len(value), 2) and points.dtype.kind in "iuf":
            points = points.astype(np.float64)
            if np.isfinite(points).all():
                return points
    except ValueError, OverflowError:
        pass
    raise ValidationError("Provide a list of finite numeric [x, y] pairs", "pixel_points")


def _lift(raw, pixels, camera, scale):
    if not len(pixels):
        return np.empty((0, 3), np.float32)
    if raw.dtype in (np.uint16, np.float32) and raw.flags.c_contiguous:
        requested = pixels.astype(np.float32).reshape(-1, 1, 2)
        native = cv2.depthTo3dSparse(raw, camera, requested)
        xyz = native.reshape(len(pixels), -1)[:, :3].astype(np.float64)
        # The native unsigned-short path has already converted millimeters to meters.
        factor = scale * (1000 if raw.dtype == np.uint16 else 1)
        return xyz * factor
    # OpenCV's sparse binding takes float32 or uint16. Gather directly for
    # float64 or strided depth so an unsupported view never needs a full-image copy.
    z = raw[pixels[:, 1], pixels[:, 0]].astype(np.float64) * scale
    rays = (pixels - camera[:2, 2]) / np.diag(camera)[:2]
    return np.column_stack((rays * z[:, None], z))


class BackprojectDepthPixels(Tool3D):
    type_id = "3d.backproject_depth_pixels"
    title, subtype = "Backproject depth pixels", "RGB-D"
    description = "Recover metric XYZ for selected pixels in an undistorted depth image."
    payload_reference = "depth"
    parallel_safe = preview_safe = process_inputs_readonly = True
    inputs = (
        Port("depth", "depth"),
        Port("camera", "data"),
        Port("points", "data", required=False, required_when_connected=True),
        Port("mask", required=False, visual_type="region"),
        Port("region", "data", required=False, visual_type="region"),
    )
    outputs = (Port("cloud", "cloud"), Port("count", "integer"), Port("report", "data"))
    defaults = {
        "pixel_points": [[0, 0]],
        "pixel_sampling": "nearest",
        "input_units": "m",
        "units_per_sample": 0.001,
        "output_units": "m",
        "minimum_valid_fraction": 0.0,
        "max_points": 10000,
    }
    parameter_choices = {
        "pixel_sampling": ("nearest", "integer"),
        "input_units": ("m", "mm"),
        "output_units": ("m", "mm"),
    }
    parameter_ranges = {
        "units_per_sample": NumericRange(1e-9, 1000),
        "minimum_valid_fraction": NumericRange(0, 1),
        "max_points": NumericRange(1, 50000, whole=True),
    }
    parameter_help = {
        "pixel_points": "Ordered fallback [x, y] requests when the points port is unwired.",
        "pixel_sampling": "Nearest snaps to the pixel center, with halves rounded upward.",
        "input_units": "Units of untagged float depth or scaled uint16 depth.",
        "units_per_sample": "Length in input units per uint16 count. Floats already have units.",
        "output_units": "Length units of the resulting camera-space XYZ values.",
        "minimum_valid_fraction": "Minimum fraction of all requests that must produce valid XYZ.",
        "max_points": "Maximum requests per run, including duplicates and rejected points.",
    }
    input_help = {
        "depth": "Positive axial camera Z. Zero, nonfinite, and uint16 65535 are invalid.",
        "camera": "Matching undistorted pinhole intrinsics and named optical frame.",
        "points": "List of [x, y] pairs on this depth grid. Duplicates preserve request order.",
        "mask": "Optional raster selection applied at the sampled pixel center.",
        "region": "Optional region intersected with the raster selection.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        _requests(p["pixel_points"], int(p["max_points"]))

    def _check_points_context(self, depth, points):
        depth_payload = self.input_payloads["depth"]
        point_payload = self.input_payloads["points"]
        require_same_frame(
            (depth, ValuePacket(points, point_payload)),
            message="Points and depth use different coordinate frames",
        )
        point_grid = point_payload.get("spatial")
        if point_grid is not None and point_grid["image_size"] != list(depth.shape[::-1]):
            raise ValidationError("Points must refer to the depth pixel grid")
        a, b = depth_payload.get("ticket"), point_payload.get("ticket")
        if a and b and ticket_key(a) != ticket_key(b):
            raise ValidationError("Points and depth must refer to the same ticket")

    def process(self, inputs):
        depth, model, p = inputs["depth"], inputs["camera"], self.parameters
        k, distortion, size = camera_for_image(model, depth)
        if model.get("lens", "pinhole") != "pinhole" or np.any(distortion):
            raise ValidationError("Backprojection requires undistorted pinhole intrinsics")
        camera_frame = model.get("frame", "camera")
        if self.input_payloads["depth"].get("optical_frame", camera_frame) != camera_frame:
            raise ValidationError("Depth optical frame differs from the camera model")

        raw = np.asarray(depth)
        data = inputs.get("points", p["pixel_points"])
        requested = _requests(data, int(p["max_points"]))
        if "points" in inputs:
            self._check_points_context(depth, data)
        units = getattr(depth, "units", p["input_units"])
        if units not in ("m", "mm"):
            raise ValidationError("Depth must have physical units m or mm", "input_units")
        scale = (0.001 if units == "mm" else 1.0) * (
            float(p["units_per_sample"]) if raw.dtype == np.uint16 else 1.0
        )
        if p["output_units"] == "mm":
            scale *= 1000

        # Guard native sampling with a bounds check before converting to float pixels.
        inside = np.all((requested >= 0) & (requested <= np.asarray(size) - 1), axis=1)
        sample = inside.copy()
        statuses = np.full(len(requested), "outside_image", dtype=object)
        if p["pixel_sampling"] == "integer":
            sample &= np.all(requested == np.rint(requested), axis=1)
            statuses[inside & ~sample] = "fractional_pixel"
        source_indices = np.flatnonzero(sample)
        pixels = np.floor(requested[sample] + 0.5).astype(np.int32)
        x, y = pixels.T
        mask = selection_mask(inputs, depth)
        selected = np.ones(len(pixels), bool) if mask is None else mask[y, x] != 0
        statuses[source_indices[~selected]] = "outside_selection"
        samples = raw[y, x]
        valid = np.isfinite(samples) & (samples > 0)
        if raw.dtype == np.uint16:
            valid &= samples != 65535
        statuses[source_indices[selected & ~valid]] = "invalid_depth"

        candidate = selected & valid
        candidates = source_indices[candidate]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            xyz = _lift(raw, pixels[candidate], k, scale).astype(np.float32)
        representable = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0)
        good = candidates[representable]
        statuses[candidates[~representable]] = "unrepresentable_xyz"
        statuses[good] = "ok"
        cloud = PointCloud(
            xyz[representable],
            p["output_units"],
            camera_frame,
            pixels=pixels[candidate][representable],
            source_size=size,
            metadata=self.input_payloads["depth"],
        )

        records = [
            {"input_index": index, "requested_pixel": point, "status": status}
            for index, (point, status) in enumerate(zip(requested.tolist(), statuses.tolist()))
        ]
        for index, (px, py) in zip(source_indices.tolist(), pixels.tolist()):
            records[index].update(sample_pixel=[px, py], pixel_index=py * size[0] + px)
        for offset, (index, point) in enumerate(zip(good.tolist(), cloud.points.tolist())):
            records[index].update(cloud_index=offset, xyz=point)
        count, total = len(cloud.points), len(records)
        fraction = count / total if total else 0.0
        passed = bool(count and fraction >= float(p["minimum_valid_fraction"]))
        reason = "no_valid_depth_points" if not count else "insufficient_valid_depth_points"
        drawings = {}
        if self.generate_graphics and len(pixels):
            objects = tuple(
                graphic_object(
                    "marker", (pixel,), (55, 210, 100) if statuses[index] == "ok" else (35, 75, 245)
                )
                for index, pixel in zip(source_indices[:256], pixels[:256])
            )
            drawings["Depth samples"] = GraphicDrawing(objects, "depth")
        report = {
            "frame": camera_frame,
            "units": cloud.units,
            "image_size": list(size),
            "requested": total,
            "valid": count,
            "valid_fraction": fraction,
            "status_counts": dict(Counter(statuses.tolist())),
            "samples": records,
        }
        return Result(
            {"cloud": cloud, "count": count, "report": report},
            graphics=cloud_graphics(self, cloud),
            drawings=drawings,
            passed=passed,
            score=fraction,
            fail_reasons=[] if passed else [reason],
            message=f"Backprojected {count} of {total} pixel requests in {cloud.units}",
        )
