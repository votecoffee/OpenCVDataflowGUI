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


def region_size(region):
    if not isinstance(region, dict) or region.get("model") != "image_region_v1":
        raise ValueError("Source must be an image_region_v1 region")
    size = region.get("image_size")
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError("Region image_size must be [width,height]")
    width, height = (whole(v, "Region dimension", 1, 16384) for v in size)
    if width * height > 64_000_000:
        raise ValueError("Region transform is limited to 64 million pixels")
    return width, height


def region_matrix(value):
    matrix = np.asarray(value, np.float64)
    if matrix.shape == (2, 3):
        matrix = np.vstack((matrix, [0, 0, 1]))
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Region transform must be a finite 2x3 or 3x3 matrix")
    if np.linalg.matrix_rank(matrix) != 3:
        raise ValueError("Region transform must be invertible")
    return matrix


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
    if kind == "transformed":
        source = region.get("source")
        source_width, source_height = region_size(source)
        region_size(region)
        matrix = region_matrix(region.get("source_to_target"))
        interpolation = region.get("interpolation", "nearest")
        threshold = float(region.get("threshold", 0.5))
        if interpolation not in ("nearest", "linear") or not 0 < threshold <= 1:
            raise ValueError("Region sampling must be nearest/linear with threshold in (0,1]")
        original = rasterize(
            source, (source_height, source_width), _depth=_depth + 1, _budget=_budget
        )
        # Binary selections never inherit image border replication or interpolation colors.
        samples = original if interpolation == "nearest" else original.astype(np.float32) / 255
        warped = cv2.warpPerspective(
            samples,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST if interpolation == "nearest" else cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        cutoff = threshold * 255 if interpolation == "nearest" else threshold
        mask = np.where(warped >= cutoff, 255, 0).astype(np.uint8)
        return cv2.bitwise_not(mask) if region.get("invert", False) else mask
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
