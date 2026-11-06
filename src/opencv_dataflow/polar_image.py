"""Resample circular image features on a radius-by-angle grid."""

import hashlib
import json
import math

import cv2
import numpy as np

from .core import Port, Result
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import frame, metadata
from .processing import ImageTool, image_value
from .selection import SELECTION_INPUTS, selection_mask

_MAX_AXIS = 32_764  # Inverse warpPolar pads the polar source by two angular rows.
_MAX_PIXELS = 16_777_216
_INTERPOLATION = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos4": cv2.INTER_LANCZOS4,
}
_MODEL = "polar_image_mapping_v1"


def _remapped_frame(size, source_spatial, settings, block_id):
    source = source_spatial if source_spatial["root_id"] is not None else [block_id, source_spatial]
    signature = json.dumps([source, settings], sort_keys=True).encode()
    return frame(size, f"polar-warp:{hashlib.sha256(signature).hexdigest()}")


class PolarImageWarp(ImageTool):
    type_id, title, subtype = "cv.warp_polar", "Polar image warp", "Geometry"
    inputs = (
        *SELECTION_INPUTS,
        Port("center_x", "number", required=False, required_when_connected=True),
        Port("center_y", "number", required=False, required_when_connected=True),
        Port("radius", "number", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("valid_mask", visual_type="region"),
        Port("mapping", "data"),
        Port("valid_fraction", "number"),
    )
    process_inputs_readonly = True
    defaults = {
        "direction": "unwrap",
        "radial_scale": "linear",
        "interpolation": "linear",
        "width": 0,
        "height": 0,
        "center_mode": "image_center",
        "center_x": 0.0,
        "center_y": 0.0,
        "radius": 0.0,
    }
    parameter_choices = {
        "direction": ("unwrap", "rewrap"),
        "radial_scale": ("linear", "semilog"),
        "interpolation": tuple(_INTERPOLATION),
        "center_mode": ("image_center", "custom"),
    }
    parameter_ranges = {
        "width": NumericRange(0, _MAX_AXIS, whole=True),
        "height": NumericRange(0, _MAX_AXIS, whole=True),
        "center_x": NumericRange(-_MAX_AXIS, _MAX_AXIS),
        "center_y": NumericRange(-_MAX_AXIS, _MAX_AXIS),
        "radius": NumericRange(0, _MAX_AXIS),
    }
    parameter_help = {
        "direction": "Unwrap to a radius-by-angle image or rewrap to a Cartesian image.",
        "radial_scale": "Linear uses equal radius steps. Semilog samples densely near the center.",
        "interpolation": "Resampling for image pixels. The validity mask always uses nearest.",
        "width": "Radius columns in Unwrap. Zero chooses the rounded radial extent.",
        "height": "Angle rows in Unwrap. Zero chooses approximately π times the width.",
        "center_mode": "Use the Cartesian image center or custom pixel coordinates.",
        "center_x": "Custom Cartesian X coordinate. A wired value takes precedence.",
        "center_y": "Custom Cartesian Y coordinate. A wired value takes precedence.",
        "radius": "Cartesian radial extent. Zero fits the largest circle around the chosen center.",
    }
    input_help = {
        "image": "Cartesian or polar image. Rewrap reuses a Polar image warp's attached mapping.",
        "mask": "Optional input-grid mask. Filters the validity output.",
        "region": "Optional input-grid region. Filters the validity output.",
    }

    def _manual_mapping(self, inputs, source_size, reverse):
        p = self.parameters
        width, height = int(p["width"]), int(p["height"])
        if reverse and (width == 0 or height == 0):
            raise ValidationError("Set the Cartesian width and height to rewrap an untagged image")
        cartesian_size = [width, height] if reverse else source_size
        center = [
            self.parameter_ranges[axis].check(
                inputs.get(axis, p[axis] if p["center_mode"] == "custom" else (length - 1) / 2),
                axis,
                axis,
            )
            for axis, length in zip(("center_x", "center_y"), cartesian_size)
        ]
        # OpenCV takes the center as Point2f. Keep the recorded center consistent with native code.
        center = np.asarray(center, np.float32).tolist()
        radius = self.parameter_ranges["radius"].check(
            inputs.get("radius", p["radius"]), "Radius", "radius"
        )
        if radius == 0:
            radius = min(
                min(coordinate + 0.5, length - 0.5 - coordinate)
                for coordinate, length in zip(center, cartesian_size)
            )
        if radius <= 0 or (p["radial_scale"] == "semilog" and radius <= 1):
            raise ValidationError("Use a positive radius above one for semilog", "radius")
        if not reverse:
            width = width or max(1, round(radius))
            height = height or max(1, round(math.pi * width))
        return {
            "model": _MODEL,
            "center": center,
            "radius": radius,
            "radial_scale": p["radial_scale"],
            "cartesian_size": cartesian_size,
            "polar_size": source_size if reverse else [width, height],
        }

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source, allow_labels=True)
        source_size = [image.shape[1], image.shape[0]]
        if max(source_size) > _MAX_AXIS or source_size[0] * source_size[1] > _MAX_PIXELS:
            raise ValidationError("Source exceeds the polar warp image limits")
        reverse = self.parameters["direction"] == "rewrap"
        interpolation = self.parameters["interpolation"]
        if image.dtype == np.int32 and interpolation != "nearest":
            raise ValidationError("Signed label images require nearest resampling", "interpolation")

        payload = metadata(source)
        source_spatial = payload.pop("spatial")
        source_pyramid = payload.pop("pyramid", None)
        source_mapping = payload.pop("polar_mapping", None)
        reuse = reverse and source_mapping is not None
        if reuse:
            mapping = source_mapping
            if (
                not isinstance(mapping, dict)
                or mapping.get("model") != _MODEL
                or source_spatial != mapping.get("polar_spatial")
                or source_size != mapping.get("polar_size")
            ):
                raise ValidationError("Rewrap requires the original polar grid")
            if inputs.keys() & {"center_x", "center_y", "radius"}:
                raise ValidationError("Disconnect center and radius to rewrap the attached mapping")
            payload["spatial"] = mapping["cartesian_spatial"]
            if mapping["cartesian_pyramid"] is not None:
                payload["pyramid"] = mapping["cartesian_pyramid"]
        else:
            mapping = self._manual_mapping(inputs, source_size, reverse)
            grid = "cartesian" if reverse else "polar"
            warped_spatial = _remapped_frame(
                mapping[f"{grid}_size"], source_spatial, [grid, mapping], self.id
            )
            mapping["cartesian_spatial"] = warped_spatial if reverse else source_spatial
            mapping["polar_spatial"] = source_spatial if reverse else warped_spatial
            mapping["cartesian_pyramid"] = None if reverse else source_pyramid
            payload["spatial"] = warped_spatial
        if not reverse:
            payload["polar_mapping"] = mapping

        size = tuple(mapping["cartesian_size" if reverse else "polar_size"])
        if len(size) != 2 or min(size) < 1 or max(size) > _MAX_AXIS:
            raise ValidationError("Output exceeds the polar warp axis limits", "width", "height")
        if math.prod(size) > _MAX_PIXELS:
            raise ValidationError("Output exceeds the polar warp image limits", "width", "height")
        center, radius = tuple(mapping["center"]), mapping["radius"]
        flags = cv2.WARP_FILL_OUTLIERS
        if reverse:
            flags |= cv2.WARP_INVERSE_MAP
        if mapping["radial_scale"] == "semilog":
            flags |= cv2.WARP_POLAR_LOG
        valid = selection_mask(inputs, source)
        if valid is None:
            valid = np.full(image.shape[:2], 255, dtype=np.uint8)
        warped = cv2.warpPolar(image, size, center, radius, flags | _INTERPOLATION[interpolation])
        valid = cv2.warpPolar(valid, size, center, radius, flags | cv2.INTER_NEAREST)
        fraction = cv2.countNonZero(valid) / valid.size
        outputs = {
            "image": ImageArray(warped, image_format(source)),
            "valid_mask": valid,
            "mapping": mapping,
            "valid_fraction": fraction,
        }
        return Result(
            outputs,
            payloads={name: payload for name in outputs},
            passed=fraction > 0,
            fail_reasons=[] if fraction > 0 else ["polar_empty_coverage"],
            message=f"{size[0]} × {size[1]} polar warp · {fraction:.1%} valid",
        )
