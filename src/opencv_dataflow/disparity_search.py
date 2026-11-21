"""Plan pixel-domain stereo search and validate complete matching footprints."""

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry3d_common import image_size, matrix
from .parameter_validation import ValidationError
from .processing import integer
from .selection import selection_mask

_MODEL = "stereo_rectification_v1"


@dataclass
class DisparitySearch:
    minimum: int
    count: int
    support: np.ndarray | None
    rois: tuple[tuple, tuple] | None
    report: dict


def _depth_range(rectification, p, step):
    if rectification is None:
        raise ValidationError("Depth search requires calibrated stereo rectification")
    if rectification.get("units") not in ("m", "mm"):
        raise ValidationError("Depth search requires stereo rectification in meters or millimeters")
    q = matrix(rectification.get("Q"), (4, 4), "Stereo Q")
    if q[2, 3] == 0 or q[3, 2] == 0 or np.any(q[2, :3] != 0) or np.any(q[3, :2] != 0):
        raise ValidationError("Stereo Q must define depth from disparity alone")
    low, high = float(p["minimum_depth"]), float(p["maximum_depth"])
    if not 0 < low < high:
        raise ValidationError(
            "Set positive, increasing depth limits", "minimum_depth", "maximum_depth"
        )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        bounds = np.sort((q[2, 3] / np.asarray([low, high]) - q[3, 3]) / q[3, 2])
    if not np.isfinite(bounds).all() or bounds[0] < -2047 or bounds[1] > 2047:
        raise ValidationError("Depth limits exceed the signed fixed-point disparity range")
    lower, upper = int(np.floor(bounds[0])), int(np.ceil(bounds[1]))
    count = ((upper - lower + step) // step) * step
    minimum = min(lower, 2048 - count)
    return minimum, count, bounds.tolist()


def _read_roi(raw, size):
    if raw is None:
        return (0, 0, *size)
    if (
        not isinstance(raw, (tuple, list))
        or len(raw) != 4
        or any(type(v) is not int or v < 0 for v in raw)
        or raw[0] + raw[2] > size[0]
        or raw[1] + raw[3] > size[1]
    ):
        raise ValidationError("Rectification ROIs must lie inside the stereo image grid")
    return tuple(raw)


def rectification_rois(rectification, size):
    if rectification is None:
        return ((0, 0, *size), (0, 0, *size))
    if (
        not isinstance(rectification, dict)
        or rectification.get("model") != _MODEL
        or image_size(rectification.get("image_size")) != size
    ):
        raise ValidationError("Calibrated rectification must match the stereo image grid")
    p2 = matrix(rectification.get("P2"), (3, 4), "Right projection")
    if abs(p2[1, 3]) > 1e-9 * abs(p2[0, 3]) or p2[0, 3] == 0:
        raise ValidationError("Stereo disparity requires a horizontal rectified baseline")
    return tuple(_read_roi(rectification.get(f"roi_{side}"), size) for side in ("left", "right"))


def _intersect(a, b):
    x, y = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    return (x, y, right - x, bottom - y) if x < right and y < bottom else (0, 0, 0, 0)


def _footprint_mask(shape, native, minimum, maximum, block_size, rois, masks):
    a, b = rois
    r = block_size // 2
    # x_right = x_left - d. The native ROI can include unsupported right-edge
    # columns when the minimum disparity is negative.
    x = max(a[0], b[0] + maximum) + r
    end = min(a[0] + a[2], b[0] + b[2] + minimum) - r
    y = max(a[1], b[1]) + r
    bottom = min(a[1] + a[3], b[1] + b[3]) - r
    roi = _intersect(native, (x, y, max(0, end - x), max(0, bottom - y)))
    x, y, w, h = roi
    support = np.zeros(shape, np.uint8)
    if not w or not h:
        return support, roi
    area = support[y : y + h, x : x + w]
    area.fill(255)
    if masks[0] is not None:
        left = cv2.erode(
            masks[0],
            np.ones((block_size, block_size), np.uint8),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        area &= left[y : y + h, x : x + w]
    if masks[1] is not None:
        sweep = cv2.erode(
            masks[1],
            np.ones((block_size, maximum - minimum + block_size), np.uint8),
            anchor=(0, r),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        first = x - maximum - r
        area &= sweep[y : y + h, first : first + w]
    return support, roi


def supported_disparity_pixels(disparity, left_allowed, right_allowed, *, band_pixels=262_144):
    """Find left samples with two valid interpolation taps at the matching right pixel."""
    width = disparity.shape[1]
    guides, left = disparity.ravel(), left_allowed.ravel()
    bands = []
    for start in range(0, disparity.size, band_pixels):
        band = slice(start, start + band_pixels)
        points = np.flatnonzero(left[band] & np.isfinite(guides[band])) + start
        y, x = np.divmod(points, width)
        xr = x - guides[points].astype(np.float64)
        in_bounds = (xr >= 0) & (xr <= width - 1)
        y, xr, points = y[in_bounds], xr[in_bounds], points[in_bounds]
        low, high = np.floor(xr).astype(np.intp), np.ceil(xr).astype(np.intp)
        supported = right_allowed[y, low] & right_allowed[y, high]
        bands.append(points[supported].astype(np.int32))
    return np.concatenate(bands)


def disparity_search_interval(rec, parameters, width, *, native_matcher=True):
    step = 16 if native_matcher else 1
    lower_count, upper_count = (16, 4096) if native_matcher else (2, 128)
    if parameters["search_mode"] == "depth":
        minimum, count, expected = _depth_range(rec, parameters, step)
    else:
        minimum = integer(parameters["min_disparity"], "Minimum disparity", -4096, 4096)
        count = integer(
            parameters["num_disparities"], "Number disparities", lower_count, upper_count
        )
        expected = None
    maximum = minimum + count - 1
    if native_matcher and (count % step or count >= width):
        raise ValidationError("num_disparities must be a multiple of 16 smaller than image width")
    if not native_matcher and (not lower_count <= count <= upper_count or count >= width):
        raise ValidationError("Binary stereo requires 2..128 disparities and a wider image")
    if minimum < -2047 or maximum > 2047:
        raise ValidationError("Disparity search exceeds the signed fixed-point range")
    return minimum, count, expected


def plan_disparity_search(inputs, parameters, shape, block_size, *, native_matcher=True):
    h, w = shape
    size = (w, h)
    rec = inputs.get("rectification")
    calibrated_rois = rectification_rois(rec, size)
    minimum, count, expected = disparity_search_interval(
        rec, parameters, w, native_matcher=native_matcher
    )
    maximum = minimum + count - 1
    depth_mode = parameters["search_mode"] == "depth"

    report = {
        "model": "stereo_disparity_search_v1",
        "mode": parameters["search_mode"],
        "min_disparity": minimum,
        "num_disparities": count,
        "max_disparity": maximum,
        "expected_disparity_px": expected,
        "depth_limits": None
        if not depth_mode
        else [float(parameters["minimum_depth"]), float(parameters["maximum_depth"])],
        "depth_units": rec["units"] if depth_mode else None,
        "block_size": block_size,
        "support_fraction": None,
        "valid_roi": None,
        "opencv_roi": None,
    }
    has_mask = any(f"{side}_mask" in inputs for side in ("left", "right"))
    if native_matcher and rec is None and not has_mask:
        return DisparitySearch(minimum, count, None, None, report)

    masks, rois = [], []
    for side, roi in zip(("left", "right"), calibrated_rois, strict=True):
        mask = selection_mask({"mask": inputs.get(f"{side}_mask")}, inputs[side])
        if mask is not None:
            roi = _intersect(roi, cv2.boundingRect(mask))
        masks.append(mask)
        rois.append(roi)
    if all(roi[2] and roi[3] for roi in rois):
        native = (
            cv2.getValidDisparityROI(*rois, minimum, count, block_size)
            if native_matcher
            else (0, 0, w, h)
        )
        support, valid_roi = _footprint_mask(
            shape, native, minimum, maximum, block_size, rois, masks
        )
    else:
        native = valid_roi = (0, 0, 0, 0)
        support = np.zeros(shape, np.uint8)
    report.update(
        opencv_roi=list(native) if native_matcher else None,
        valid_roi=list(valid_roi),
        support_fraction=float(np.count_nonzero(support) / support.size),
    )
    return DisparitySearch(minimum, count, support, tuple(rois), report)
