"""Native quasi-dense correspondence with metric stereo grid and support checks."""

import cv2
import numpy as np

from .core import Port, Result
from .disparity_search import rectification_rois
from .geometry3d_common import Tool3D
from .images import ImageArray
from .parameter_validation import NumericRange, ValidationError
from .processing import gray_image
from .selection import selection_mask
from .stereo3d import check_stereo_grids
from .tickets import ticket_key

_MAX_PIXELS = 1_048_576
_MAX_CANDIDATES = 1_200_000_000

_NATIVE_FIELDS = {
    "border_x": "borderX",
    "border_y": "borderY",
    "minimum_correlation": "correlationThreshold",
    "texture_threshold": "textrureThreshold",
    "neighborhood_radius": "neighborhoodSize",
    "gradient_radius": "disparityGradient",
    "lk_template_size": "lkTemplateSize",
    "lk_max_level": "lkPyrLvl",
    "lk_iterations": "lkTermParam1",
    "lk_epsilon": "lkTermParam2",
    "feature_quality": "gftQualityThres",
    "feature_separation": "gftMinSeperationDist",
    "maximum_features": "gftMaxNumFeatures",
}
_INTEGER_PARAMS = {
    "border_x",
    "border_y",
    "neighborhood_radius",
    "gradient_radius",
    "lk_template_size",
    "lk_max_level",
    "lk_iterations",
    "feature_separation",
    "maximum_features",
}


def _native_pairs(left, right, p):
    kind = getattr(getattr(cv2, "stereo", None), "QuasiDenseStereo", None)
    if kind is None:
        raise RuntimeError("Install an OpenCV contrib build with QuasiDenseStereo")
    matcher = cv2.stereo.QuasiDenseStereo.create(left.shape[::-1])
    settings = matcher.Param
    for field, attr in _NATIVE_FIELDS.items():
        value = int(p[field]) if field in _INTEGER_PARAMS else float(p[field])
        setattr(settings, attr, value)
    # Param is returned by value. Assign the modified structure back.
    settings.corrWinSizeX = settings.corrWinSizeY = int(p["correlation_radius"])
    matcher.Param = settings
    try:
        matcher.process(left, right)
    except cv2.error as error:
        # OpenCV 5's LK call can raise on a GFTT image with no features.
        if error.code != -215 or "prevPtsMat.checkVector" not in error.err:
            raise
        return np.empty((0, 4), np.int32), 0
    matches = matcher.getDenseMatches()
    pairs = np.fromiter(
        ((*match.p0, *match.p1) for match in matches),
        dtype=(np.int32, 4),
        count=len(matches),
    )
    w, h = left.shape[::-1]
    if np.any(pairs < 0) or np.any(pairs >= [w, h, w, h]):
        raise RuntimeError("OpenCV returned a correspondence outside the stereo grids")
    return pairs, len(matcher.getSparseMatches())


def _support(inputs, side, roi, shape, radius):
    selected = selection_mask(
        {"mask": inputs.get(f"{side}_mask"), "region": inputs.get(f"{side}_region")},
        inputs[side],
    )
    area = np.zeros(shape, np.uint8)
    x, y, w, h = roi
    bounds = np.s_[y : y + h, x : x + w]
    if selected is None:
        area[bounds] = 255
    else:
        area[bounds] = selected[bounds]
    return cv2.erode(
        area,
        np.ones((2 * radius + 1,) * 2, np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _filter(pairs, left, right, low, high, vertical):
    x, y, u, v = pairs.T
    dx = x - u
    return pairs[
        (left[y, x] != 0)
        & (right[v, u] != 0)
        & (dx >= low)
        & (dx <= high)
        & (np.abs(y - v) <= vertical)
    ]


def _consistent(pairs, reverse, shape, tolerance):
    lookup = np.full((*shape, 2), -1, np.int32)
    lookup[reverse[:, 1], reverse[:, 0]] = reverse[:, 2:]
    left = lookup[pairs[:, 3], pairs[:, 2]]
    return pairs[(left >= 0).all(axis=1) & (np.abs(left - pairs[:, :2]) <= tolerance).all(axis=1)]


def _sample_record(pairs, size, limit):
    count = min(limit, len(pairs))
    chosen = pairs[np.linspace(0, len(pairs) - 1, count, dtype=int)]
    return {
        "model": "quasi_dense_matches_v1",
        "query_size": list(size),
        "train_size": list(size),
        "query_points": chosen[:, :2].tolist(),
        "train_points": chosen[:, 2:].tolist(),
        "total_matches": len(pairs),
    }


class QuasiDenseDisparity(Tool3D):
    type_id, title, subtype = (
        "3d.quasi_dense_disparity",
        "Quasi-dense stereo disparity",
        "Stereo",
    )
    inputs = (
        Port("left"),
        Port("right"),
        Port("rectification", "data", required=False, required_when_connected=True),
        Port("left_mask", required=False, required_when_connected=True, visual_type="region"),
        Port("right_mask", required=False, required_when_connected=True, visual_type="region"),
        Port("left_region", "region", required=False, required_when_connected=True),
        Port("right_region", "region", required=False, required_when_connected=True),
    )
    outputs = (
        Port("disparity"),
        Port("vertical_offset"),
        Port("valid_mask", visual_type="region"),
        Port("support_mask", visual_type="region"),
        Port("matches", "data"),
        Port("report", "data"),
    )
    defaults = {
        "border_x": 15,
        "border_y": 15,
        "correlation_radius": 5,
        "minimum_correlation": 0.5,
        "texture_threshold": 200.0,
        "neighborhood_radius": 5,
        "gradient_radius": 1,
        "lk_template_size": 3,
        "lk_max_level": 3,
        "lk_iterations": 3,
        "lk_epsilon": 0.003,
        "feature_quality": 0.01,
        "feature_separation": 10,
        "maximum_features": 500,
        "min_disparity": -64,
        "max_disparity": 64,
        "maximum_vertical_error_px": 0,
        "left_right_check": True,
        "left_right_tolerance_px": 0,
        "max_reported_pairs": 256,
        "minimum_coverage": 0.0,
    }
    parameter_ranges = {
        "border_x": NumericRange(1, 255, whole=True),
        "border_y": NumericRange(1, 255, whole=True),
        "correlation_radius": NumericRange(1, 16, whole=True),
        "minimum_correlation": NumericRange(0, 0.9999),
        "texture_threshold": NumericRange(0, 255),
        "neighborhood_radius": NumericRange(1, 12, whole=True),
        "gradient_radius": NumericRange(0, 3, whole=True),
        "lk_template_size": NumericRange(3, 51, odd=True),
        "lk_max_level": NumericRange(0, 6, whole=True),
        "lk_iterations": NumericRange(1, 100, whole=True),
        "lk_epsilon": NumericRange(0, 1),
        "feature_quality": NumericRange(0.000001, 1),
        "feature_separation": NumericRange(0, 256, whole=True),
        "maximum_features": NumericRange(1, 10000, whole=True),
        "min_disparity": NumericRange(-2047, 2047, whole=True),
        "max_disparity": NumericRange(-2047, 2047, whole=True),
        "maximum_vertical_error_px": NumericRange(0, 32, whole=True),
        "left_right_tolerance_px": NumericRange(0, 16, whole=True),
        "max_reported_pairs": NumericRange(0, 2048, whole=True),
        "minimum_coverage": NumericRange(0, 1),
    }
    parameter_help = {
        "border_x": "Horizontal guard for native propagation and its correlation patch.",
        "border_y": "Vertical guard for native propagation and its correlation patch.",
        "correlation_radius": "Half-width of the square native correlation window in pixels.",
        "minimum_correlation": "Minimum zero-mean normalized correlation for native propagation.",
        "texture_threshold": "Maximum local intensity jump for a propagation candidate.",
        "neighborhood_radius": "Native propagation neighborhood half-width around each seed.",
        "gradient_radius": "Maximum candidate displacement change from each seed.",
        "lk_template_size": "Odd Lucas–Kanade tracking window used to find sparse seeds.",
        "lk_max_level": "Top Lucas–Kanade pyramid index. Zero uses only the original image.",
        "lk_iterations": "Maximum Lucas–Kanade iterations per level.",
        "lk_epsilon": "Lucas–Kanade convergence threshold in pixels.",
        "feature_quality": "Good Features to Track response relative to the strongest corner.",
        "feature_separation": "Minimum separation between initially selected corners.",
        "maximum_features": "Maximum corner seeds submitted to Lucas–Kanade.",
        "min_disparity": "Smallest accepted signed left X minus right X in pixels.",
        "max_disparity": "Largest accepted signed left X minus right X in pixels.",
        "maximum_vertical_error_px": "Largest vertical mismatch. Use zero for calibrated depth.",
        "left_right_check": "Require agreement from a separate right-to-left propagation pass.",
        "left_right_tolerance_px": "Maximum reverse-coordinate difference on either axis.",
        "max_reported_pairs": "Maximum listed coordinate pairs. Zero leaves the list empty.",
        "minimum_coverage": "Minimum matched fraction of eligible left centers.",
    }
    input_help = {
        "left": "Rectified uint8 left image. Outputs use this pixel grid.",
        "right": "Rectified uint8 right image on the same resolution.",
        "rectification": "Optional matching rectification with left and right support ROIs.",
        "left_mask": "Nonzero native-correlation patch support after propagation on the left.",
        "right_mask": "Nonzero native-correlation patch support after propagation on the right.",
        "left_region": "Optional left selection intersected with Left mask.",
        "right_region": "Optional right selection intersected with Right mask.",
    }
    payload_reference = "left"
    parallel_safe = preview_safe = process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        guard = 1 + sum(
            int(p[key])
            for key in (
                "correlation_radius",
                "neighborhood_radius",
                "gradient_radius",
            )
        )
        if min(int(p["border_x"]), int(p["border_y"])) < guard:
            raise ValidationError(
                f"Each border must be at least {guard} pixels", "border_x", "border_y"
            )
        if int(p["min_disparity"]) > int(p["max_disparity"]):
            raise ValidationError(
                "Disparity limits must be ordered", "min_disparity", "max_disparity"
            )
        if type(p["left_right_check"]) is not bool:
            raise ValidationError("Left-right check must be a boolean", "left_right_check")

    def process(self, inputs):
        p = self.parameters
        left, right = (gray_image(inputs[name], (np.uint8,)) for name in ("left", "right"))
        h, w = left.shape
        size = (w, h)
        if left.shape != right.shape or left.size > _MAX_PIXELS:
            raise ValidationError("Use equal stereo grids with at most 1,048,576 pixels each")
        if w <= 2 * int(p["border_x"]) or h <= 2 * int(p["border_y"]):
            raise ValidationError("The propagation borders must leave an interior in both images")
        check_stereo_grids(inputs, self.input_payloads, size)
        tickets = {
            ticket_key(details["ticket"])
            for name, details in self.input_payloads.items()
            if name != "rectification" and "ticket" in details
        }
        if len(tickets) > 1:
            raise ValidationError("Stereo inputs and selections must refer to the same ticket")
        rois = rectification_rois(inputs.get("rectification"), size)
        left_support, right_support = (
            _support(inputs, side, roi, left.shape, int(p["correlation_radius"]))
            for side, roi in zip(("left", "right"), rois, strict=True)
        )
        eligible = int(np.count_nonzero(left_support))
        can_match = eligible > 0 and bool(right_support.any())
        low, high = int(p["min_disparity"]), int(p["max_disparity"])
        vertical = int(p["maximum_vertical_error_px"])
        sparse = native_count = before_check = 0
        pairs = np.empty((0, 4), np.int32)
        if can_match:
            candidates = left.size
            candidates *= (2 * int(p["neighborhood_radius"]) + 1) ** 2
            candidates *= (2 * int(p["gradient_radius"]) + 1) ** 2
            if candidates * (1 + p["left_right_check"]) > _MAX_CANDIDATES:
                raise ValidationError(
                    "Propagation work limit exceeded. Reduce image or neighborhood size"
                )
            pairs, sparse = _native_pairs(left, right, p)
            native_count = len(pairs)
            pairs = _filter(pairs, left_support, right_support, low, high, vertical)
            before_check = len(pairs)
            if p["left_right_check"] and before_check:
                reverse, _ = _native_pairs(right, left, p)
                reverse = _filter(reverse, right_support, left_support, -high, -low, vertical)
                pairs = _consistent(pairs, reverse, left.shape, int(p["left_right_tolerance_px"]))

        x, y, u, v = pairs.T
        disparity = np.full(left.shape, np.nan, np.float32)
        offset = np.full(left.shape, np.nan, np.float32)
        valid = np.zeros(left.shape, np.uint8)
        disparity[y, x] = x - u
        offset[y, x] = y - v
        valid[y, x] = 255
        coverage = len(pairs) / eligible if eligible else 0.0
        report = {
            "model": "quasi_dense_stereo_v1",
            "native_sparse_candidates": sparse,
            "native_correspondences": native_count,
            "rejected_by_support_or_geometry": native_count - before_check,
            "rejected_by_left_right": before_check - len(pairs),
            "eligible_left_pixels": eligible,
            "matched_pixels": len(pairs),
            "coverage": len(pairs) / left.size,
            "supported_coverage": coverage,
        }
        passed = len(pairs) > 0 and coverage >= float(p["minimum_coverage"])
        if not can_match:
            reason = "quasi_dense_no_support"
        elif len(pairs) == 0:
            reason = "quasi_dense_no_correspondences"
        else:
            reason = "quasi_dense_coverage"
        graphics = {}
        if self.generate_graphics:
            preview = np.zeros(left.shape, np.uint8)
            preview[y, x] = 128 if high == low else np.rint((x - u - low) * 255 / (high - low))
            colored = cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)
            colored[valid == 0] = 0
            graphics["Quasi-dense disparity"] = ImageArray(colored, "bgr")
        return Result(
            {
                "disparity": ImageArray(disparity, "gray"),
                "vertical_offset": ImageArray(offset, "gray"),
                "valid_mask": valid,
                "support_mask": left_support,
                "matches": _sample_record(pairs, size, int(p["max_reported_pairs"])),
                "report": report,
            },
            graphics=graphics,
            score=report["coverage"],
            passed=passed,
            fail_reasons=[] if passed else [reason],
            message=f"{len(pairs)}/{eligible} eligible left pixels matched",
        )
