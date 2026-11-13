"""Explicit pixel encodings and deterministic image depth conversion."""

import copy

import cv2
import numpy as np

COLOR_FORMATS = ("unchanged", "bgr", "rgb", "bgra", "rgba", "gray", "hsv", "lab")
PIXEL_DEPTHS = ("unchanged", "uint8", "uint16", "float32")


class ImageArray(np.ndarray):
    """NumPy image with channel-encoding metadata, retained by snapshot copies."""

    def __new__(cls, image, color_format, metadata=None):
        obj = np.asarray(image).view(cls)
        obj.color_format = color_format
        obj.metadata = copy.deepcopy(
            getattr(image, "metadata", {}) if metadata is None else metadata
        )
        return obj

    def __array_finalize__(self, source):
        self.color_format = getattr(source, "color_format", None)
        self.metadata = copy.deepcopy(getattr(source, "metadata", {}))


class DepthArray(np.ndarray):
    """Physical depth plane with units and a separate display range."""

    def __new__(cls, values, units="m", display_min=0.0, display_max=5.0):
        obj = np.asarray(values, dtype=np.float32).view(cls)
        obj.units, obj.display_min, obj.display_max = units, display_min, display_max
        obj.metadata = copy.deepcopy(getattr(values, "metadata", {}))
        return obj

    def __array_finalize__(self, source):
        self.units = getattr(source, "units", "unknown")
        self.display_min = getattr(source, "display_min", 0.0)
        self.display_max = getattr(source, "display_max", 5.0)
        self.metadata = copy.deepcopy(getattr(source, "metadata", {}))


def image_format(image):
    return getattr(image, "color_format", None) or (
        "gray" if image.ndim == 2 else "bgra" if image.shape[2] == 4 else "bgr"
    )


def convert_depth(image, depth):
    """Integer full scale <-> float [0,1], round-to-nearest and saturate.

    No per-image normalization: identical input values always produce identical
    output values. Unchanged depth preserves finite HDR floating-point samples.
    """
    if depth not in PIXEL_DEPTHS:
        raise ValueError(f"Pixel depth must be one of {PIXEL_DEPTHS}")
    if image.dtype not in (np.uint8, np.uint16, np.float32, np.float64):
        raise ValueError(f"Unsupported source depth: {image.dtype}")
    if not np.isfinite(image).all():
        raise ValueError("Image contains nonfinite pixel values")
    if depth == "unchanged" or image.dtype == np.dtype(depth):
        return np.asarray(image).copy()
    unit = np.asarray(image).astype(np.float64)
    if np.issubdtype(image.dtype, np.integer):
        unit /= np.iinfo(image.dtype).max
    if depth == "float32":
        return unit.astype(np.float32)
    maximum = np.iinfo(np.dtype(depth)).max
    return np.rint(np.clip(unit, 0, 1) * maximum).astype(depth)


def format_image(image, color_format="bgr", pixel_depth="uint8"):
    """Convert decoded OpenCV GRAY/BGR/BGRA pixels to a requested encoding."""
    if color_format not in COLOR_FORMATS:
        raise ValueError(f"Color format must be one of {COLOR_FORMATS}")
    source_format = image_format(image)
    if source_format not in ("gray", "bgr", "bgra"):
        raise ValueError("format_image expects decoded GRAY, BGR or BGRA input")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (3, 4)):
        raise ValueError("Expected a grayscale, three-channel or four-channel image")
    result = convert_depth(image, pixel_depth)
    target = source_format if color_format == "unchanged" else color_format
    if target == source_format:
        return ImageArray(result, target)
    if result.dtype == np.float64:
        if target in ("hsv", "lab"):
            raise ValueError("HSV and Lab color conversion requires uint8 or float32 depth")
        bgr = (
            np.repeat(result[..., None], 3, axis=2) if source_format == "gray" else result[..., :3]
        )
        if target == "gray":
            converted = bgr[..., 0] * 0.114 + bgr[..., 1] * 0.587 + bgr[..., 2] * 0.299
        else:
            converted = bgr[..., ::-1] if target in ("rgb", "rgba") else bgr
            if target in ("bgra", "rgba"):
                alpha = result[..., 3] if source_format == "bgra" else np.ones(result.shape[:2])
                converted = np.dstack((converted, alpha))
        if not np.isfinite(converted).all():
            raise ValueError("Color conversion produced nonfinite values")
        return ImageArray(converted.copy(), target)
    if target in ("hsv", "lab") and result.dtype == np.uint16:
        raise ValueError("HSV and Lab require uint8 or float32; select a supported pixel depth")
    if source_format == "gray":
        bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    elif source_format == "bgra":
        bgr = cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)
    else:
        bgr = result
    if target == "bgr":
        converted = bgr
    elif target == "bgra" and source_format == "gray":
        converted = cv2.cvtColor(result, cv2.COLOR_GRAY2BGRA)
    elif target == "rgba" and source_format == "bgra":
        converted = cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA)
    else:
        code = {
            "rgb": cv2.COLOR_BGR2RGB,
            "bgra": cv2.COLOR_BGR2BGRA,
            "rgba": cv2.COLOR_BGR2RGBA,
            "gray": cv2.COLOR_BGR2GRAY,
            "hsv": cv2.COLOR_BGR2HSV,
            "lab": cv2.COLOR_BGR2Lab,
        }[target]
        converted = cv2.cvtColor(bgr, code)
    return ImageArray(converted, target)


def opencv_image(image):
    """Interpret tagged encodings as canonical OpenCV GRAY/BGR/BGRA pixels.

    This changes channel encoding only; operation-specific depth requirements
    remain explicit and unsupported depths fail in the consuming OpenCV tool.
    """
    encoding = image_format(image)
    if image.dtype == np.float64 and encoding in ("rgb", "rgba"):
        order = (2, 1, 0, 3) if encoding == "rgba" else (2, 1, 0)
        return np.asarray(image)[..., order]
    code = {
        "rgb": cv2.COLOR_RGB2BGR,
        "rgba": cv2.COLOR_RGBA2BGRA,
        "hsv": cv2.COLOR_HSV2BGR,
        "lab": cv2.COLOR_Lab2BGR,
    }.get(encoding)
    return cv2.cvtColor(np.asarray(image), code) if code is not None else np.asarray(image)


def label_color_map(labels, background=0):
    """Make a BGR preview without changing the signed label plane."""
    hsv = np.empty((*labels.shape, 3), dtype=np.uint8)
    hue = np.remainder(np.asarray(labels, dtype=np.int32), 180)
    hue *= 67
    hue %= 180
    hsv[..., 0] = hue
    hsv[..., 1:] = (210, 238)
    colors = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    colors[labels <= 0] = 0
    if background > 0:
        colors[labels == background] = (55, 55, 55)
    colors[labels == -1] = (255, 255, 255)
    return colors


def preview_rgb(image):
    from .payloads import ImagePyramid

    if isinstance(image, ImagePyramid):
        image = image.largest
    if isinstance(image, DepthArray):
        valid = np.isfinite(image) & (image > 0)
        pixels = np.clip(
            (np.nan_to_num(np.asarray(image)) - image.display_min)
            / (image.display_max - image.display_min),
            0,
            1,
        )
        graphic = cv2.applyColorMap(np.rint(pixels * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        graphic[~valid] = 0
        return cv2.cvtColor(graphic, cv2.COLOR_BGR2RGB)
    if image.dtype == np.int32 and image.ndim == 2:
        return cv2.cvtColor(label_color_map(image), cv2.COLOR_BGR2RGB)
    if image.dtype == np.int16:
        # Signed filter samples remain raw in the result; only the view is stretched.
        normalized = cv2.normalize(
            np.asarray(image), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        image = ImageArray(normalized, image_format(image))
    invalid = None
    if image.dtype in (np.float32, np.float64):
        nonfinite = cv2.finiteMask(np.asarray(image)) == 0
        if nonfinite.any():
            invalid = nonfinite
            image = image.copy()
            image[invalid] = 0
    image = opencv_image(image)
    image = convert_depth(image, "uint8")
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB if image.shape[2] == 4 else cv2.COLOR_BGR2RGB)
    if invalid is not None:
        rgb[invalid] = (255, 0, 255)
    return rgb
