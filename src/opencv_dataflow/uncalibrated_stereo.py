"""Projective stereo rectification from image matches, with explicit valid support."""

import cv2
import numpy as np

from .core import Port, Result
from .epipolar import _EpipolarPairTool
from .epipolar_refinement import sampson_pixels
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, moved_payload
from .processing import image_value
from .stereo3d import STEREO_GRID

_INTERPOLATION = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR}
_SIDES = ("left", "right")
_MAX_PIXELS = 16_777_216
_MAX_SIDE = 8192
_MIN_PAIRS = 8


def _fit_indices(inputs, left, right, fundamental, limit):
    n = len(left)
    selected = inputs.get("inliers", [True] * n)
    if (
        not isinstance(selected, list)
        or len(selected) != n
        or any(type(v) is not bool for v in selected)
    ):
        raise ValidationError("Inliers must be a boolean mask in input pair order")
    score = sampson_pixels(left, right, fundamental)
    keep = np.asarray(selected) & np.isfinite(score)
    if limit:
        keep &= score <= limit
    return np.flatnonzero(keep)


def _spread(xy):
    if len(np.unique(xy, axis=0)) < _MIN_PAIRS:
        return False
    singular = np.linalg.svd(xy - xy.mean(axis=0), compute_uv=False)
    return singular[1] > singular[0] * 1e-6


def _corners(size, margin=0.0):
    w, h = size
    return np.asarray(
        [
            [-margin, -margin, 1],
            [w - 1 + margin, -margin, 1],
            [w - 1 + margin, h - 1 + margin, 1],
            [-margin, h - 1 + margin, 1],
        ],
        dtype=np.float64,
    )


def _oriented_homography(homography, size):
    if homography is None or homography.shape != (3, 3) or not np.isfinite(homography).all():
        return None, "invalid_homography"
    norm = np.max(np.abs(homography))
    if not norm:
        return None, "invalid_homography"
    h = homography / norm
    if np.linalg.matrix_rank(h) != 3:
        return None, "invalid_homography"
    corners = _corners(size, margin=0.5)
    w = corners @ h[2]
    epsilon = 1e-10 * np.max(np.abs(corners) @ np.abs(h[2]))
    if np.min(w) > epsilon:
        return h, None
    if np.max(w) < -epsilon:
        return -h, None
    return None, "projective_horizon_intersects_source"


def _map_points(h, points):
    mapped = np.c_[points, np.ones(len(points))] @ h.T
    return mapped[:, :2] / mapped[:, 2, None]


def _resized_canvas(reference, parameters):
    w, h = reference
    width, height = int(parameters["width"]) or w, int(parameters["height"]) or h
    if max(width, height) > _MAX_SIDE or width * height > _MAX_PIXELS:
        raise ValidationError("Rectified canvas exceeds 8192 pixels per axis or 16,777,216 pixels")
    sx, sy = width / w, height / h
    # Resize both views in the same pixel-center coordinate system.
    return (width, height), np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])


def _support(src_size, inverse, dst_size, interpolation):
    width, height = src_size
    weights = cv2.warpPerspective(
        np.ones((height, width), np.float32),
        inverse,
        dst_size,
        flags=interpolation | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid = weights >= 1.0 - 1e-6
    # At W=0 some native samplers return a row of source (0,0) pixels.
    # Only the positive branch maps back to this source rectangle.
    corners = _corners(dst_size)
    row = inverse[2]
    tolerance = 1e-12 * np.max(np.abs(corners) @ np.abs(row))
    if np.min(corners @ row) <= tolerance:
        xx = np.arange(dst_size[0]) * row[0] + row[2]
        for y, mask_row in enumerate(valid):
            mask_row &= xx + y * row[1] > tolerance
    return valid


def _rectify_raster(source, inverse, size, interpolation, border):
    value = image_value(source)
    fill = (border,) * value.shape[2] if value.ndim == 3 else border
    output = cv2.warpPerspective(
        value,
        inverse,
        size,
        flags=interpolation | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )
    if not np.isfinite(output).all():
        raise ValidationError("Rectification produced nonfinite image samples", "border_value")
    return ImageArray(output, image_format(source))


def _mask(support):
    return ImageArray(support.astype(np.uint8) * 255, "gray")


def _rms(delta):
    return float(np.linalg.norm(delta) / np.sqrt(len(delta)))


class RectifyUncalibratedStereo(_EpipolarPairTool):
    type_id, title, subtype = (
        "3d.rectify_uncalibrated_stereo",
        "Rectify uncalibrated stereo images",
        "Stereo",
    )
    inputs = (
        *_EpipolarPairTool.inputs[:3],
        Port("left_image"),
        Port("right_image"),
        Port("inliers", "data", required=False, required_when_connected=True),
    )
    input_help = {
        **_EpipolarPairTool.input_help,
        "left_image": "Left source grid where the matched points and matrix were computed.",
        "right_image": "Right source grid. Its dimensions define the default common canvas.",
        "inliers": "Optional boolean mask in input pair order, such as from Fundamental matrix.",
    }
    outputs = (
        Port("left", required=False),
        Port("right", required=False),
        Port("left_valid", required=False),
        Port("right_valid", required=False),
        Port("common_valid", required=False),
        Port("points_left", "data", required=False),
        Port("points_right", "data", required=False),
        Port("indices", "data", required=False),
        Port("rectification", "data", required=False),
        Port("report", "data"),
    )
    defaults = {
        "width": 0,
        "height": 0,
        "interpolation": "linear",
        "border_value": 0.0,
        "maximum_input_error_px": 2.0,
        "maximum_vertical_rms_px": 1.5,
        "minimum_common_support": 0.1,
    }
    parameter_choices = {"interpolation": tuple(_INTERPOLATION)}
    parameter_ranges = {
        "width": NumericRange(2, _MAX_SIDE, whole=True, zero_allowed=True),
        "height": NumericRange(2, _MAX_SIDE, whole=True, zero_allowed=True),
        "border_value": NumericRange(),
        "maximum_input_error_px": NumericRange(0, 100_000),
        "maximum_vertical_rms_px": NumericRange(0, 100_000),
        "minimum_common_support": NumericRange(0, 1),
    }
    parameter_help = {
        "width": "Common output width. Zero uses the right image width.",
        "height": "Common output height. Zero uses the right image height.",
        "interpolation": "Nearest or linear sampling. Masks require the whole sampling footprint.",
        "border_value": "Constant fill in each input image's sample units.",
        "maximum_input_error_px": "Filter fit matches by square-root Sampson error. Zero disables.",
        "maximum_vertical_rms_px": "Maximum rectified RMS of the fitted pairs' vertical disparity.",
        "minimum_common_support": "Minimum mutually valid fraction of the rectified canvas.",
    }
    process_inputs_readonly = parallel_safe = True
    payload_reference = "left_image"

    def _failure(self, report, reason, message):
        report["status"] = reason
        outputs = dict.fromkeys((port.name for port in self.outputs), NO_VALUE)
        outputs["report"] = report
        return Result(
            outputs,
            passed=False,
            fail_reasons=[reason],
            message=message,
        )

    def process(self, inputs):
        p = self.parameters
        left, right, f = self._read_pair(inputs)
        views = (left, right)
        a, b = left[0], right[0]
        reference_size = tuple(right[3])
        if any(
            min(size) < 2 or max(size) > _MAX_SIDE or size[0] * size[1] > _MAX_PIXELS
            for _, _, _, size in views
        ):
            raise ValidationError(
                "Source grids require at least 2 pixels per axis and at most 16 Mi pixels"
            )
        size, resize = _resized_canvas(reference_size, p)
        indices = _fit_indices(inputs, a, b, f, float(p["maximum_input_error_px"]))
        report = {
            "model": "uncalibrated_stereo_report_v1",
            "pairs": len(a),
            "fit_pairs": len(indices),
            "fit_indices": indices.tolist(),
            "image_size": list(size),
        }
        if len(indices) < _MIN_PAIRS or not (_spread(a[indices]) and _spread(b[indices])):
            return self._failure(
                report,
                "insufficient_rectification_points",
                "Rectification needs eight distinct noncollinear pairs",
            )
        fit = (a[indices], b[indices])
        # The explicit pixel-domain prefilter above preserves the relation to source indices.
        ok, h1, h2 = cv2.stereoRectifyUncalibrated(*fit, f, reference_size, threshold=0.0)
        if not ok:
            return self._failure(
                report,
                "uncalibrated_stereo_solver_failed",
                "OpenCV could not rectify the selected pairs",
            )
        matrices = []
        for side, homography, view in zip(_SIDES, (h1, h2), views, strict=True):
            h, issue = _oriented_homography(
                None if homography is None else resize @ homography, view[3]
            )
            if issue:
                return self._failure(report, issue, f"{side.title()} warp is singular in its grid")
            matrices.append(h)
        inverses = [np.linalg.inv(h) for h in matrices]

        rectified = [_map_points(h, xy) for h, xy in zip(matrices, fit, strict=True)]
        dy = rectified[0][:, 1] - rectified[1][:, 1]
        vertical_rms = _rms(dy)
        if not np.isfinite(vertical_rms):
            return self._failure(
                report,
                "nonfinite_rectified_points",
                "The rectification could not project finite points",
            )
        report.update(
            input_vertical_rms_px=_rms(fit[0][:, 1] - fit[1][:, 1]),
            vertical_rms_px=vertical_rms,
            vertical_max_px=float(np.max(np.abs(dy))),
            vertical_disparity_px=dy.tolist(),
        )
        if vertical_rms > float(p["maximum_vertical_rms_px"]):
            return self._failure(
                report,
                "rectification_vertical_limit",
                f"Vertical rectification RMS is {vertical_rms:.4g} px",
            )

        interpolation = _INTERPOLATION[p["interpolation"]]
        coverage = [
            _support(view[3], inverse, size, interpolation)
            for view, inverse in zip(views, inverses, strict=True)
        ]
        common = coverage[0] & coverage[1]
        fractions = [float(np.mean(mask)) for mask in (*coverage, common)]
        report.update(
            left_support=fractions[0], right_support=fractions[1], common_support=fractions[2]
        )
        if not fractions[2] or fractions[2] < float(p["minimum_common_support"]):
            return self._failure(
                report,
                "rectification_common_support",
                f"Only {fractions[2]:.1%} of the canvas has common support",
            )

        outputs, payloads = {}, {}
        for i, side in enumerate(_SIDES):
            _, context, source, _ = views[i]
            payload = moved_payload(context, inverses[i], size)
            payload.pop(STEREO_GRID, None)
            outputs[side] = _rectify_raster(
                source, inverses[i], size, interpolation, float(p["border_value"])
            )
            outputs[f"{side}_valid"] = _mask(coverage[i])
            outputs[f"points_{side}"] = rectified[i].tolist()
            for name in (side, f"{side}_valid", f"points_{side}"):
                payloads[name] = payload
        outputs["common_valid"] = _mask(common)
        payloads["common_valid"] = payloads["left"]
        report["status"] = "accepted"
        outputs.update(
            indices=indices.tolist(),
            rectification={
                "model": "uncalibrated_stereo_rectification_v1",
                "H1": matrices[0].tolist(),
                "H2": matrices[1].tolist(),
                "left_size": list(left[3]),
                "right_size": list(right[3]),
                "image_size": list(size),
            },
            report=report,
        )
        return Result(
            outputs,
            payloads=payloads,
            message=(
                f"{len(indices)} pairs, {vertical_rms:.3g} px vertical RMS, "
                f"{fractions[2]:.1%} support"
            ),
        )
