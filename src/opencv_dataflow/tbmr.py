"""Tree-based Morse region features on an explicitly mapped image pyramid."""

import math

import cv2
import numpy as np

from .core import Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .processing import ImageTool, gray_image
from .result_graphics import ACCEPTED
from .selection import SELECTION_INPUTS, contains, selection_mask

MAX_PIXELS = 1_048_576
MAX_PYRAMID_PIXELS = 4_194_304
MAX_MOMENT = (1 << 32) - 1


def pyramid_schedule(width, height, parameters):
    levels = []
    total_pixels = 0
    area_limit = np.float32(parameters["max_area_relative"])
    for octave in range(max(1, int(parameters["n_scales"]))):
        factor = float(parameters["scale_factor"]) ** octave
        w, h = round(width / factor), round(height / factor)
        if min(w, h) < 8:
            break
        if levels and levels[-1][:2] == (w, h):
            continue
        # Native duplicate bookkeeping uses a four-pixel grid with floor-divided axes.
        pad_w, pad_h = (-w) % 4, (-h) % 4
        stride_w, stride_h = w + pad_w, h + pad_h
        pixels = stride_w * stride_h
        max_area = int(area_limit * np.float32(pixels))
        if max_area < int(parameters["min_area"]):
            break
        # The native component tree keeps raw selected-region moments in uint32.
        if max_area * (max(stride_w, stride_h) - 1) ** 2 > MAX_MOMENT:
            raise ValidationError(
                "TBMR area cutoff can overflow native region moments. Reduce maximum area or crop.",
                "max_area_relative",
            )
        total_pixels += pixels
        if total_pixels > MAX_PYRAMID_PIXELS:
            raise ValidationError("TBMR pyramid exceeds its work limit. Reduce scales.", "n_scales")
        levels.append((w, h, pad_w, pad_h, octave))
    if not levels:
        raise ValidationError("Minimum area exceeds the effective TBMR maximum area", "min_area")
    return levels


class TreeMorseRegions(ImageTool):
    type_id, title, subtype = (
        "inspect.tbmr", "Tree-based Morse regions (TBMR)", "Features",
    )
    category = block_type = "Inspection"
    description = "Detect component-tree interest points with measured center and major-axis scale."
    inputs = SELECTION_INPUTS
    outputs = (
        Port("image"),
        Port("keypoints", "data"),
        Port("points", "data"),
        Port("count", "number"),
    )
    payload_reference = "image"
    preview_safe = parallel_safe = process_inputs_readonly = True
    defaults = {
        "min_area": 60,
        "max_area_relative": 0.01,
        "scale_factor": 1.25,
        "n_scales": -1,
        "max_keypoints": 2000,
        "preview_count": 100,
    }
    parameter_ranges = {
        "min_area": NumericRange(0, MAX_PIXELS, whole=True),
        "max_area_relative": NumericRange(1e-6, 1),
        "scale_factor": NumericRange(1.01, 4),
        "max_keypoints": NumericRange(1, 4096, whole=True),
        "preview_count": NumericRange(0, 512, whole=True),
    }
    parameter_help = {
        "min_area": "Native component-tree minimum area in each detector raster's pixels.",
        "max_area_relative": "Native maximum area as a fraction of each detector raster.",
        "scale_factor": "Factor between successive pyramid levels.",
        "n_scales": "-1 selects the native default of one level. Request 1–8 independent levels.",
        "max_keypoints": "Retain up to this many selected keypoints, in level then native order.",
        "preview_count": "Draw this many measured major-axis vectors. Zero hides drawings.",
    }
    input_help = {
        "image": "Grayscale or color converted to 8-bit monochrome for detection.",
        "mask": "Select mapped keypoint centers on the source pixel grid.",
        "region": "Intersect the mask with this source-grid region.",
    }
    output_help = {
        "image": "Original input image or selected pyramid level.",
        "keypoints": "Source-frame center, mapped major-axis diameter and octave. "
        "Axis angle is separate from descriptor orientation.",
        "points": "Keypoint centers in the selected input image's pixel frame.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        scales = p["n_scales"]
        if type(scales) is not int or (scales != -1 and not 1 <= scales <= 8):
            raise ValidationError("Number of scales must be -1 or 1..8", "n_scales")

    def process(self, inputs):
        source = inputs["image"]
        height, width = source.shape[:2]
        if (
            min(width, height) < 8 or max(width, height) > 4096 or width * height > MAX_PIXELS
        ):
            raise ValidationError("TBMR requires 8–4096 pixels per axis and at most 1,048,576 pixels")
        p = self.parameters
        levels = pyramid_schedule(width, height, p)
        create = getattr(getattr(cv2, "xfeatures2d", None), "TBMR_create", None)
        if create is None:
            raise ValidationError("TBMR requires OpenCV contrib xfeatures2d")
        gray = gray_image(source, (np.uint8,), auto_depth=True)
        selected = selection_mask(inputs, source)
        keypoints, level_info = [], []
        truncated = False
        detector = create(int(p["min_area"]), float(p["max_area_relative"]), 1.25, 1)
        if selected is None or selected.any():
            for w, h, pad_w, pad_h, octave in levels:
                if self._cancelled():
                    raise InterruptedError("TBMR detection cancelled")
                raster = gray if octave == 0 else cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
                if pad_w or pad_h:
                    raster = cv2.copyMakeBorder(raster, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
                sx, sy = width / w, height / h
                tx, ty = (sx - 1) / 2, (sy - 1) / 2
                level_info.append({
                    "octave": octave, "image_size": [w, h], "detector_size": [w + pad_w, h + pad_h],
                    "level_to_image": [[sx, 0, tx], [0, sy, ty], [0, 0, 1]],
                })
                # Native TBMR returns level centers via plain KeyPoint on this Python binding.
                for key in detector.detect(raster):
                    x, y = key.pt
                    if not (0 <= x < w and 0 <= y < h and math.isfinite(key.size) and key.size > 0
                            and math.isfinite(key.angle)):
                        continue
                    point = [x * sx + tx, y * sy + ty]
                    if point[0] >= width or point[1] >= height or not contains(selected, point):
                        continue
                    if len(keypoints) == p["max_keypoints"]:
                        truncated = True
                        break
                    vx, vy = sx * math.cos(key.angle), sy * math.sin(key.angle)
                    keypoints.append({
                        "point": point,
                        "size": 2 * key.size * math.hypot(vx, vy),
                        "angle": -1.0,
                        "major_axis_angle_degrees": math.degrees(math.atan2(vy, vx)) % 180,
                        "native_major_semiaxis": float(key.size),
                        "response": float(key.response),
                        "octave": octave,
                    })
                if truncated:
                    break
        drawings = []
        if self.generate_graphics:
            for row in keypoints[: int(p["preview_count"])]:
                x, y = row["point"]
                radius, theta = row["size"] / 2, math.radians(row["major_axis_angle_degrees"])
                dx, dy = radius * math.cos(theta), radius * math.sin(theta)
                drawings.append(
                    graphic_object("polyline", [(x - dx, y - dy), (x + dx, y + dy)], ACCEPTED)
                )
        return Result(
            {
                "image": source,
                "keypoints": {
                    "algorithm": "TBMR", "image_size": [width, height],
                    "keypoints": keypoints, "levels": level_info, "truncated": truncated,
                },
                "points": [row["point"] for row in keypoints],
                "count": len(keypoints),
            },
            drawings={"TBMR major axes": GraphicDrawing(tuple(drawings))} if drawings else {},
            message=f"{len(keypoints)} tree-based interest points",
        )
