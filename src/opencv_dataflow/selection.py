"""Shared image-space regions and explicit selected-output semantics."""

import cv2
import numpy as np

from .core import Port
from .images import image_format, opencv_image

SELECTION_INPUTS = (
    Port("image"),
    Port("mask", required=False),
    Port("region", "data", required=False),
)


def whole(value, name, minimum=-1_000_000, maximum=1_000_000):
    value = float(value)
    if not np.isfinite(value) or value != round(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be a whole pixel in [{minimum}, {maximum}]")
    return int(value)


def rasterize(region, shape, *, _depth=0, _budget=None):
    if _budget is None:
        _budget = [128]
    _budget[0] -= 1
    if _depth > 16 or _budget[0] < 0:
        raise ValueError("Region expression exceeds 16 levels or 128 shapes/operators")
    height, width = shape[:2]
    if (
        not isinstance(region, dict)
        or region.get("model") != "image_region_v1"
        or region.get("image_size") != [width, height]
    ):
        raise ValueError("Region must use image_region_v1 and match image dimensions")
    kind = region["shape"]
    if kind == "composite":
        operation = region.get("operation")
        children = region.get("children")
        if (
            operation not in ("union", "intersection", "difference", "xor", "invert")
            or not isinstance(children, list)
            or len(children) != (1 if operation == "invert" else 2)
        ):
            raise ValueError("Composite region needs a supported operation and one/two children")
        a = rasterize(children[0], shape, _depth=_depth + 1, _budget=_budget)
        if operation == "invert":
            mask = cv2.bitwise_not(a)
        else:
            b = rasterize(children[1], shape, _depth=_depth + 1, _budget=_budget)
            if operation == "difference":
                mask = cv2.bitwise_and(a, cv2.bitwise_not(b))
            else:
                mask = {
                    "union": cv2.bitwise_or,
                    "intersection": cv2.bitwise_and,
                    "xor": cv2.bitwise_xor,
                }[operation](a, b)
        return cv2.bitwise_not(mask) if region.get("invert", False) else mask
    mask = np.zeros((height, width), np.uint8)
    if kind == "rectangle":
        x, y, w, h = [whole(v, "Rectangle") for v in region["rectangle"]]
        if w <= 0 or h <= 0:
            raise ValueError("Rectangle width and height must be positive")
        cv2.rectangle(mask, (x, y), (x + w - 1, y + h - 1), 255, -1)
    elif kind == "circle":
        x, y = [whole(v, "Circle center") for v in region["center"]]
        radius = whole(region["radius"], "Radius", 1)
        cv2.circle(mask, (x, y), radius, 255, -1)
    elif kind == "polygon":
        vertices = np.asarray(region["vertices"], float)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 2
            or not 3 <= len(vertices) <= 10000
            or not np.isfinite(vertices).all()
            or np.any(vertices != np.rint(vertices))
            or np.any(np.abs(vertices) > 1_000_000)
        ):
            raise ValueError("Polygon needs 3..10000 finite whole-pixel [x,y] vertices")
        if cv2.contourArea(vertices.astype(np.float32)) <= 0:
            raise ValueError("Polygon must have nonzero area")
        cv2.fillPoly(mask, [vertices.astype(np.int32)], 255)
    else:
        raise ValueError("Region shape must be rectangle, circle or polygon")
    return cv2.bitwise_not(mask) if region.get("invert", False) else mask


def selection_mask(inputs, image):
    mask = inputs.get("mask")
    if mask is not None:
        if mask.dtype != np.uint8 or mask.ndim != 2 or mask.shape != image.shape[:2]:
            raise ValueError("Mask must be uint8 monochrome with matching image dimensions")
        mask = np.where(mask != 0, 255, 0).astype(np.uint8)
    region = inputs.get("region")
    if region is not None:
        area = rasterize(region, image.shape)
        mask = area if mask is None else cv2.bitwise_and(mask, area)
    return mask


def contains(mask, point):
    if mask is None:
        return True
    x, y = np.rint(point).astype(int)
    return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] != 0


class RegionalOperation:
    """Opt-in write selection; neighborhoods still use the full source image."""

    inputs = SELECTION_INPUTS
    region_outputs = ("image",)
    region_background = "preserve"
    region_source = "image"

    def finalize_result(self, result, inputs):
        source = inputs[self.region_source]
        mask = selection_mask(inputs, source)
        if mask is None:
            return result
        for key in self.region_outputs:
            output = result.outputs[key]
            if output.shape[:2] != source.shape[:2]:
                raise ValueError("Selected output must retain input image dimensions")
            selected = output.copy()
            if self.region_background == "preserve" and output.shape == source.shape:
                background = (
                    source if image_format(source) == image_format(output) else opencv_image(source)
                )
                selected[mask == 0] = np.asarray(background)[mask == 0].astype(output.dtype)
            else:
                selected[mask == 0] = 0
            result.outputs[key] = selected
            for name, graphic in result.graphics.items():
                if graphic is output:
                    result.graphics[name] = selected
        return result
