"""Unwrap an image strip along the normals of an ordered path."""

import hashlib
import json

import cv2
import numpy as np

from .core import Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import ValuePacket, frame, metadata, require_same_frame
from .processing import ImageTool, image_value
from .projection import read_path, sample_path
from .selection import selection_mask
from .workplane import require_workplane_scale

_INTERPOLATION = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR}


class UnwrapPathStrip(ImageTool):
    type_id, title, subtype = "cv.unwrap_path_strip", "Unwrap path strip", "Geometry"
    inputs = (
        Port("image"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
        Port("path", "data", required=False, required_when_connected=True),
        Port("scale", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("valid_mask", visual_type="region"),
        Port("map_x", required=False),
        Port("map_y", required=False),
        Port("sampling", "data"),
    )
    defaults = {
        "vertices": [[20, 30], [65, 42], [110, 30]],
        "closed": False,
        "sample_spacing": 1.0,
        "half_width": 16.0,
        "cross_samples": 33,
        "tangent_radius": 2.0,
        "interpolation": "linear",
    }
    parameter_choices = {"interpolation": tuple(_INTERPOLATION)}
    parameter_ranges = {
        "sample_spacing": NumericRange(0.25),
        "half_width": NumericRange(0.25, 512),
        "cross_samples": NumericRange(3, 513, whole=True, odd=True),
        "tangent_radius": NumericRange(0.25, 512),
    }
    parameter_help = {
        "vertices": "Ordered image-space [x, y] vertices. A wired path takes precedence.",
        "closed": "Include the last-to-first segment and emit the first column only once.",
        "sample_spacing": "Maximum along-path step in source pixels.",
        "half_width": "Maximum distance from the centerline on either side, in source pixels.",
        "cross_samples": "Odd row count. The middle row is the path and the outer rows are "
        "at ±Half width.",
        "tangent_radius": "Arc-length window radius for estimating the local path direction. "
        "For closed paths, use less than half the total length.",
        "interpolation": "Nearest preserves labels and masks. Linear interpolates intensities.",
    }
    input_help = {
        "mask": "Exclude source pixels from the strip. Invalid strip samples are zeroed.",
        "region": "Exclude samples outside this source-grid region. Intersected with the mask.",
        "path": "Ordered [x, y] vertices registered to the source image.",
        "scale": "Optional Rectify workplane scale for this same input grid. The strip geometry "
        "controls still use input pixels.",
    }
    process_inputs_readonly = True
    parallel_safe = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        read_path(p["vertices"], p["closed"], "vertices")

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source, allow_labels=True)
        height, width = image.shape[:2]
        if max(height, width) >= 32767 or not 0 < height * width <= 16_777_216:
            raise ValidationError("Source exceeds the OpenCV path-strip image limits")
        p = self.parameters
        interpolation = _INTERPOLATION[p["interpolation"]]
        if image.dtype == np.int32 and interpolation != cv2.INTER_NEAREST:
            raise ValidationError("Use nearest interpolation with signed labels", "interpolation")
        require_same_frame(
            (source, ValuePacket(None, self.input_payloads.get("path", {}))),
            message="Path vertices must share the image coordinate frame",
        )
        closed = p["closed"]
        vertices = read_path(inputs.get("path", p["vertices"]), closed)
        length, distances, points, normals = sample_path(
            vertices, closed, float(p["sample_spacing"]), tangent_radius=float(p["tangent_radius"])
        )
        rows, cols = int(p["cross_samples"]), len(points)
        if rows * cols > 2_100_000:
            raise ValidationError(
                "Strip exceeds 2.1 million pixels. Increase spacing or reduce rows."
            )
        half_width = float(p["half_width"])
        offsets = np.linspace(-half_width, half_width, rows)[:, None]
        map_x = (points[:, 0] + offsets * normals[:, 0]).astype(np.float32)
        map_y = (points[:, 1] + offsets * normals[:, 1]).astype(np.float32)
        valid = (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
        map_x[~valid] = map_y[~valid] = -1
        selection = selection_mask(inputs, source)
        if selection is not None:
            selected = cv2.remap(selection.astype(np.float32), map_x, map_y, interpolation)
            valid &= selected == 255
            map_x[~valid] = map_y[~valid] = -1
        strip = cv2.remap(image, map_x, map_y, interpolation)
        valid_mask = valid.astype(np.uint8) * 255

        payload = metadata(source)
        source_frame = payload.pop("spatial")
        payload.pop("pyramid", None)
        payload.pop("polar_mapping", None)
        identity = [source_frame, vertices.tolist(), closed, [cols, rows],
                    half_width, float(p["tangent_radius"])]
        if source_frame["root_id"] is None:
            identity.append(self.id)
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        strip_frame = frame([cols, rows], f"path-strip:{digest}")
        payload["spatial"] = strip_frame
        sampling = {
            "model": "path_strip_v1",
            "source_spatial": source_frame,
            "strip_spatial": strip_frame,
            "closed": closed,
            "path_length": length,
            "along_step": length / (cols if closed else cols - 1),
            "cross_step": float(offsets[1, 0] - offsets[0, 0]),
            "center_row": rows // 2,
            "invalid_map_value": -1,
        }
        if "scale" in inputs:
            units, step = require_workplane_scale(
                inputs["scale"], self.input_payloads["image"], self.input_payloads["scale"]
            )
            sampling["physical_scale"] = {"units": units, "units_per_pixel": step}
        drawings = {}
        if self.generate_graphics:
            drawings["Sampling path"] = GraphicDrawing(tuple(
                graphic_object("polyline", points + offset * normals, color, closed=closed)
                for offset, color in ((-half_width, (220, 200, 40)),
                                      (0, (0, 210, 255)),
                                      (half_width, (220, 200, 40)))
            ))
        outputs = {
            "image": ImageArray(strip, image_format(source)),
            "valid_mask": valid_mask, "map_x": map_x, "map_y": map_y, "sampling": sampling,
        }
        fraction = float(valid.mean())
        return Result(
            outputs, payloads=dict.fromkeys(outputs, payload), drawings=drawings,
            passed=fraction > 0, fail_reasons=[] if fraction > 0 else ["path_strip_empty_coverage"],
            message=f"{cols} × {rows} path strip · {fraction:.1%} valid",
        )
