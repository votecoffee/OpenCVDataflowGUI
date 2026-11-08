"""Shared OpenCV enum maps and validation for public wrapper parameters."""

import cv2
import numpy as np

BORDERS = {
    "constant": cv2.BORDER_CONSTANT,
    "replicate": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_REFLECT_101,
    "reflect_101": cv2.BORDER_REFLECT_101,
    "reflect_edge": cv2.BORDER_REFLECT,
}
COLORMAPS = {
    name.removeprefix("COLORMAP_").lower(): getattr(cv2, name)
    for name in dir(cv2)
    if name.startswith("COLORMAP_")
}
THRESHOLD_MODES = {
    "binary": cv2.THRESH_BINARY,
    "binary_inv": cv2.THRESH_BINARY_INV,
    "trunc": cv2.THRESH_TRUNC,
    "tozero": cv2.THRESH_TOZERO,
    "tozero_inv": cv2.THRESH_TOZERO_INV,
}
CONTOUR_APPROXIMATIONS = {
    "none": cv2.CHAIN_APPROX_NONE,
    "simple": cv2.CHAIN_APPROX_SIMPLE,
    "tc89_l1": cv2.CHAIN_APPROX_TC89_L1,
    "tc89_kcos": cv2.CHAIN_APPROX_TC89_KCOS,
}
LINEAR_FILTER_OPERATIONS = ("correlation", "convolution")
OUTPUT_DEPTHS = {
    "same": -1,
    "uint8": cv2.CV_8U,
    "uint16": cv2.CV_16U,
    "int16": cv2.CV_16S,
    "float32": cv2.CV_32F,
    "float64": cv2.CV_64F,
}


def anchor_value(value, width, height):
    values = np.asarray(value)
    if (
        values.shape != (2,)
        or not np.issubdtype(values.dtype, np.number)
        or not np.isfinite(values).all()
    ):
        raise ValueError("Anchor must be a finite whole-pixel [x,y] pair")
    if np.any(values != np.rint(values)):
        raise ValueError("Anchor must contain whole pixels")
    x, y = map(int, values)
    if not (-1 <= x < width and -1 <= y < height):
        raise ValueError("Anchor must be -1 (centered) or inside the kernel")
    return x, y


def linear_filter_anchor(value, width, height, operation):
    x, y = anchor_value(value, width, height)
    if operation == "convolution":
        x = width - 1 - (width // 2 if x == -1 else x)
        y = height - 1 - (height // 2 if y == -1 else y)
    return x, y


def optional_mask(inputs, image, key="mask"):
    if key == "mask":
        from .selection import selection_mask

        return selection_mask(inputs, image)
    mask = inputs.get(key)
    if mask is not None and (
        mask.dtype != np.uint8 or mask.ndim != 2 or mask.shape != image.shape[:2]
    ):
        raise ValueError("Mask must be uint8 monochrome with matching image dimensions")
    return np.asarray(mask) if mask is not None else None


def scalar_value(value):
    values = np.asarray(value, dtype=float).reshape(-1)
    if values.size not in (1, 2, 3, 4) or not np.isfinite(values).all():
        raise ValueError("Border value must have one to four finite channel values")
    return tuple(np.repeat(values, 4)) if len(values) == 1 else tuple(values)
