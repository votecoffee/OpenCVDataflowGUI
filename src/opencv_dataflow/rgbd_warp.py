"""Reproject registered RGB-D using OpenCV's pose-dependent z-buffer."""

import cv2
import numpy as np

from .calibration_revisions import APPLIED, CAPTURE
from .cloud_projection import MAX_PIXELS, METERS
from .core import Port, Result
from .depth_encoding import ENCODING
from .geometry3d_common import Tool3D, camera_for_image, camera_record, rigid
from .images import DepthArray, ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, frame, require_same_frame
from .selection import selection_mask
from .stereo3d import metric_depth
from .tickets import ticket_key


def _rasterize(z, k, transform, selected):
    """Carry the source pixel ID through the native color channel without losing precision."""
    if np.array_equal(transform, np.eye(4)) or not selected.any():
        owners = np.full(z.shape, -1, np.int32)
        owners.flat[selected.ravel()] = np.flatnonzero(selected)
        depth = np.where(selected, z, np.nan)
        return depth, owners
    pixel_ids = np.arange(1, z.size + 1, dtype="<u4").reshape(z.shape)
    source_bytes = pixel_ids.view(np.uint8).reshape(*z.shape, 4)
    depth, warped_bytes, valid = cv2.warpFrame(
        z, source_bytes, selected.astype(np.uint8) * 255, transform, k
    )
    owners = np.ascontiguousarray(warped_bytes).view("<i4").reshape(z.shape).astype(np.int32)
    owners -= 1
    good = (valid != 0) & np.isfinite(depth) & (depth > 0) & (owners >= 0) & (owners < z.size)
    depth[~good] = np.nan
    owners[~good] = -1
    return depth, owners


class WarpRGBDByPose(Tool3D):
    type_id = "3d.warp_rgbd_pose"
    title, subtype = "Warp RGB-D by camera pose", "RGB-D"
    description = "Reproject registered depth and color into a new optical frame."
    payload_reference = "depth"
    flow_primary = ("depth", "color")
    parallel_safe = preview_safe = process_inputs_readonly = True
    inputs = (
        Port("depth", "depth"),
        Port("camera", "data"),
        Port("transform", "data", visual_type="pose"),
        Port("color", required=False, required_when_connected=True),
        Port("mask", required=False, required_when_connected=True, visual_type="region"),
        Port("region", "data", required=False, required_when_connected=True, visual_type="region"),
    )
    outputs = (
        Port("depth", "depth"),
        Port("color", required=False),
        Port("valid_mask", visual_type="region"),
        Port("source_index"),
        Port("camera", "data"),
        Port("report", "data"),
    )
    defaults = {
        "color_alignment": "require_same_grid",
        "input_units": "m",
        "units_per_sample": 0.001,
        "minimum_coverage": 0.0,
        "display_max_m": 5.0,
    }
    parameter_choices = {
        "color_alignment": ("require_same_grid", "assume_registered"),
        "input_units": ("m", "mm"),
    }
    parameter_ranges = {
        "units_per_sample": NumericRange(1e-9, 1000),
        "minimum_coverage": NumericRange(0, 1),
        "display_max_m": NumericRange(0.001, 1000),
    }
    parameter_help = {
        "color_alignment": (
            "Require matching known grids or declare separately stored rasters already registered."
        ),
        "input_units": "Physical units for depth without declared units.",
        "units_per_sample": "Physical units per uint16 sample.",
        "minimum_coverage": "Required fraction of the destination grid with a visible sample.",
        "display_max_m": "Preview upper bound in meters. Does not clip geometric depth.",
    }
    input_help = {
        "depth": "Registered axial camera Z. Positive finite samples can contribute.",
        "camera": "Undistorted pinhole intrinsics with positive focal lengths and principal point.",
        "transform": "Source-optical-to-target-optical rigid pose in meters or millimeters.",
        "color": "Optional registered grayscale, three-channel, or four-channel image.",
        "mask": "Optional uint8 source selection. Excluded samples cannot win the z-buffer.",
        "region": "Optional source region intersected with the mask.",
    }

    def _registered_color(self, inputs, source_frame):
        color = inputs.get("color")
        if color is None:
            return None
        depth = inputs["depth"]
        if (
            color.shape[:2] != depth.shape
            or (color.ndim == 3 and color.shape[2] not in (3, 4))
            or color.dtype not in (np.uint8, np.uint16, np.float32, np.float64)
        ):
            raise ValidationError(
                "Registered color must match the depth grid and use gray, RGB or RGBA"
            )
        color_payload = self.input_payloads["color"]
        if color_payload.get("optical_frame", source_frame) != source_frame:
            raise ValidationError("Color optical frame does not match the source camera")
        if self.parameters["color_alignment"] == "require_same_grid":
            require_same_frame((depth, color), message="Color and depth use different pixel grids")
        return color

    def process(self, inputs):
        depth, camera, pose = inputs["depth"], inputs["camera"], inputs["transform"]
        p = self.parameters
        k, distortion, size = camera_for_image(camera, depth)
        if depth.size > MAX_PIXELS:
            raise ValidationError("RGB-D warp exceeds the 8,388,608 pixel limit")
        if camera.get("lens", "pinhole") != "pinhole" or np.any(distortion):
            raise ValidationError("RGB-D warp requires undistorted pinhole intrinsics")
        principal_and_focal = k[[0, 1, 0, 1], [0, 1, 2, 2]]
        if (principal_and_focal <= np.finfo(float).eps).any():
            raise ValidationError(
                "OpenCV warpFrame requires positive focal lengths and principal point"
            )
        # The camera validator allows small numerical residuals in structural zero entries.
        k = np.array([[k[0, 0], 0, k[0, 2]], [0, k[1, 1], k[1, 2]], [0, 0, 1]], np.float64)
        data = self.input_payloads["depth"]
        source_frame = camera.get("frame", "camera")
        if data.get("optical_frame", source_frame) != source_frame:
            raise ValidationError("Depth optical frame does not match the source camera")
        target_frame = pose.get("target_frame") if isinstance(pose, dict) else None
        if (
            not isinstance(target_frame, str)
            or not target_frame
            or pose.get("source_frame") != source_frame
            or pose.get("units") not in METERS
        ):
            raise ValidationError(
                "Pose must map the source optical frame to a named target in m or mm"
            )
        transform = rigid(pose).copy()
        transform[:3, 3] *= METERS[pose["units"]]
        if ENCODING in data:
            raise ValidationError("Decode tagged sensor codes to metric depth before warping")
        color = self._registered_color(inputs, source_frame)
        tickets = {
            ticket_key(self.input_payloads[name]["ticket"])
            for name in ("depth", "color", "mask", "region")
            if "ticket" in self.input_payloads.get(name, {})
        }
        if len(tickets) > 1:
            raise ValidationError("Depth, color and selections must refer to the same ticket")
        color_payload = self.input_payloads.get("color", {})
        depth_record, color_record = data.get("recorded_rgbd"), color_payload.get("recorded_rgbd")
        if isinstance(depth_record, dict) and isinstance(color_record, dict):
            if any(
                depth_record.get(key) != color_record.get(key) for key in ("clock", "pair_index")
            ):
                raise ValidationError(
                    "Registered depth and color belong to different recorded pairs"
                )

        z = metric_depth(depth, p["units_per_sample"], p["input_units"])
        original = np.isfinite(z)
        if depth.dtype == np.uint16:
            original &= np.asarray(depth) != 65535
        selected = original.copy()
        mask = selection_mask(inputs, depth)
        if mask is not None:
            selected &= mask != 0
        warped_depth, owners = _rasterize(z, k, transform, selected)
        covered = owners >= 0
        destination_color = NO_VALUE
        if color is not None:
            result_color = np.zeros_like(np.asarray(color))
            sy, sx = np.divmod(owners[covered], depth.shape[1])
            result_color[covered] = np.asarray(color)[sy, sx]
            destination_color = ImageArray(result_color, image_format(color))

        count = int(np.count_nonzero(covered))
        coverage = count / z.size
        passed = bool(count and coverage >= float(p["minimum_coverage"]))
        failure = "rgbd_warp_no_visible_depth" if not count else "rgbd_warp_coverage"
        target_payload = {
            key: value for key, value in data.items() if key not in ("pyramid", CAPTURE, APPLIED)
        }
        if "ticket" not in target_payload and "ticket" in color_payload:
            target_payload["ticket"] = color_payload["ticket"]
        target_payload.update(
            spatial=frame(size, f"{self.id}:target"),
            optical_frame=target_frame,
            rgbd_warp={
                "source_frame": source_frame,
                "source_spatial": data.get("spatial"),
                "source_to_target_m": transform.tolist(),
            },
        )
        outputs = {
            "depth": DepthArray(warped_depth, "m", 0, float(p["display_max_m"])),
            "color": destination_color,
            "valid_mask": covered.astype(np.uint8) * 255,
            "source_index": owners,
            "camera": camera_record(k, [], size, "m", target_frame),
            "report": {
                "source_frame": source_frame,
                "target_frame": target_frame,
                "size": list(size),
                "depth_units": "m",
                "valid_source_samples": int(np.count_nonzero(original)),
                "selected_source_samples": int(np.count_nonzero(selected)),
                "visible_destination_samples": count,
                "coverage": coverage,
                "has_color": color is not None,
            },
        }
        payloads = {name: target_payload for name in outputs}
        if color is not None and color_record is not None:
            payloads["color"] = {**target_payload, "recorded_rgbd": color_record}
        return Result(
            outputs,
            payloads=payloads,
            passed=passed,
            score=coverage,
            fail_reasons=[] if passed else [failure],
            message=f"{count}/{z.size} destination pixels with visible depth",
        )
