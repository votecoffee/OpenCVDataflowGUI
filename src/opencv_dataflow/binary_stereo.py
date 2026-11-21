"""Dense integer-disparity block matching with Census signatures and Hamming costs."""

import cv2
import numpy as np

from .core import Port, Result
from .disparity_search import plan_disparity_search
from .geometry3d_common import Tool3D
from .images import ImageArray
from .parameter_validation import NumericRange, ValidationError
from .processing import gray_image
from .stereo3d import check_stereo_grids
from .tickets import ticket_key

_MAX_PIXELS = 8_388_608
_MAX_COMPARISONS = 268_435_456


def _census_offsets(size, variant):
    radius = size // 2
    offsets = [
        (y, x)
        for y in range(-radius, radius + 1)
        for x in range(-radius, radius + 1)
        if (y, x) != (0, 0)
    ]
    if variant == "symmetric":
        return offsets[: len(offsets) // 2]
    if variant == "star":
        return [(y, x) for y, x in offsets if y == 0 or x == 0 or abs(y) == abs(x)]
    return offsets


def _census(image, size, variant="center"):
    radius = size // 2
    h, w = image.shape
    yy, xx = h - 2 * radius, w - 2 * radius
    center = image[radius : h - radius, radius : w - radius]
    if variant == "mean":
        sums = cv2.boxFilter(image, cv2.CV_32S, (size, size), normalize=False)
        center = sums[radius : h - radius, radius : w - radius]

    def shifted(y, x):
        return image[radius + y : radius + y + yy, radius + x : radius + x + xx]

    signature = np.zeros(center.shape, np.uint64)
    for y, x in _census_offsets(size, variant):
        sample = shifted(y, x)
        reference = shifted(-y, -x) if variant == "symmetric" else center
        if variant == "mean":
            sample = np.multiply(sample, size**2, dtype=np.uint16)
        signature <<= 1
        signature |= sample < reference
    result = np.zeros(image.shape, np.uint64)
    result[radius : h - radius, radius : w - radius] = signature
    return result


def _work_area(plan, aggregation):
    _, _, width, height = plan.report["valid_roi"]
    pad = aggregation // 2
    return (width + 2 * pad) * (height + 2 * pad) * plan.count if plan.support.any() else 0


def _match(left, right, plan, aggregation):
    h, w = left.shape
    results = [np.full((h, w), np.nan, np.float32) for _ in range(3)]
    x, y, width, height = plan.report["valid_roi"]
    if not width or not height or not plan.support.any():
        return results

    shape = (height, width)
    pad = aggregation // 2
    ys = slice(y - pad, y + height + pad)
    left_patch = left[ys, x - pad : x + width + pad]
    best, next_best = (np.full(shape, np.inf, np.float32) for _ in range(2))
    winner = np.zeros(shape, np.float32)
    for d in range(plan.minimum, plan.minimum + plan.count):
        corresponding = right[ys, x - pad - d : x + width + pad - d]
        hamming = np.bitwise_count(left_patch ^ corresponding)
        if pad:
            cost = cv2.boxFilter(
                hamming,
                cv2.CV_32F,
                (aggregation, aggregation),
                normalize=False,
                borderType=cv2.BORDER_CONSTANT,
            )[pad : pad + height, pad : pad + width]
        else:
            cost = hamming.astype(np.float32)
        wins = cost < best
        next_best = np.where(wins, best, np.minimum(next_best, cost))
        np.minimum(best, cost, out=best)
        winner[wins] = d

    inside = plan.support[y : y + height, x : x + width] != 0
    for result, selected in zip(results, (winner, best, next_best), strict=True):
        np.copyto(result[y : y + height, x : x + width], selected, where=inside)
    return results


def _confidence(best, second):
    delta = second - best
    result = np.full(best.shape, np.nan, np.float32)
    supported = np.isfinite(best)
    result[supported] = 0
    np.divide(delta, second, out=result, where=supported & (second > 0))
    return result


def _reverse_plan(inputs, p, shape, footprint, plan):
    reversed_inputs = {"left": inputs["right"], "right": inputs["left"]}
    for a, b in (("left_mask", "right_mask"), ("right_mask", "left_mask")):
        if b in inputs:
            reversed_inputs[a] = inputs[b]
    rec = inputs.get("rectification")
    if rec is not None:
        reversed_inputs["rectification"] = {
            **rec,
            "roi_left": rec.get("roi_right"),
            "roi_right": rec.get("roi_left"),
        }
    p = {
        **p,
        "search_mode": "manual",
        "min_disparity": 1 - plan.minimum - plan.count,
        "num_disparities": plan.count,
    }
    return plan_disparity_search(reversed_inputs, p, shape, footprint, native_matcher=False)


def _cross_checked(valid, disparity, reverse, tolerance):
    y, x = np.nonzero(valid)
    right_x = x - disparity[y, x].astype(np.int32)
    counterpart = reverse[y, right_x]
    checked = np.zeros(valid.shape, bool)
    checked[y, x] = np.isfinite(counterpart) & (np.abs(disparity[y, x] + counterpart) <= tolerance)
    return checked


def _census_pair(inputs, payloads):
    left, right = (gray_image(inputs[name], (np.uint8,)) for name in ("left", "right"))
    if left.shape != right.shape:
        raise ValidationError("Rectified stereo images must have the same dimensions")
    if left.size > _MAX_PIXELS:
        raise ValidationError("Binary stereo inputs are limited to 8,388,608 pixels each")
    check_stereo_grids(inputs, payloads, left.shape[::-1])
    tickets = [
        payloads[side].get("ticket")
        for side in ("left", "right", "left_mask", "right_mask")
        if side in inputs
    ]
    if len({ticket_key(ticket) for ticket in tickets if ticket is not None}) > 1:
        raise ValidationError("Stereo input images and masks must refer to the same ticket")
    return left, right


class BinaryStereoDisparity(Tool3D):
    type_id, title, subtype = "3d.binary_disparity", "Binary Census stereo disparity", "Stereo"
    inputs = (
        Port("left"),
        Port("right"),
        Port("rectification", "data", required=False, required_when_connected=True),
        Port("left_mask", required=False, required_when_connected=True, visual_type="region"),
        Port("right_mask", required=False, required_when_connected=True, visual_type="region"),
    )
    outputs = (
        Port("disparity"),
        Port("valid_mask", visual_type="region"),
        Port("support_mask", visual_type="region"),
        Port("confidence"),
        Port("best_cost"),
        Port("second_best_cost"),
        Port("report", "data"),
    )
    defaults = {
        "search_mode": "manual",
        "min_disparity": 0,
        "num_disparities": 32,
        "minimum_depth": 0.1,
        "maximum_depth": 5.0,
        "census_size": "5",
        "aggregation_size": 5,
        "minimum_confidence": 0.05,
        "left_right_check": True,
        "left_right_tolerance_px": 1,
    }
    parameter_choices = {"search_mode": ("manual", "depth"), "census_size": ("3", "5", "7")}
    parameter_ranges = {
        "min_disparity": NumericRange(-2047, 2047, whole=True),
        "num_disparities": NumericRange(2, 128, whole=True),
        "minimum_depth": NumericRange(0),
        "maximum_depth": NumericRange(0),
        "aggregation_size": NumericRange(1, 31, odd=True),
        "minimum_confidence": NumericRange(0, 1),
        "left_right_tolerance_px": NumericRange(0, 2047, whole=True),
    }
    parameter_help = {
        "search_mode": "Use a signed pixel interval or derive one from calibrated depth limits.",
        "min_disparity": "Lowest integer disparity to compare. Left X minus right X.",
        "num_disparities": "Number of consecutive integer disparity candidates.",
        "minimum_depth": "Near limit in the calibrated rectification's physical units.",
        "maximum_depth": "Far limit in the calibrated rectification's physical units.",
        "census_size": "Odd Census neighborhood. Encodes comparisons with its center pixel.",
        "aggregation_size": "Odd square window used to sum Hamming distances.",
        "minimum_confidence": "Minimum relative gap between the two lowest costs. Rejects ties.",
        "left_right_check": "Reject disagreements with a separate right-to-left match.",
        "left_right_tolerance_px": "Allowed disparity sum magnitude across the two directions.",
    }
    input_help = {
        "left": "Rectified left uint8 image, used as the output grid.",
        "right": "Rectified right uint8 image.",
        "rectification": "Optional calibrated ROI data. Required in Depth search mode.",
        "left_mask": "Full source support before the Census and aggregation windows.",
        "right_mask": "Full right-view support throughout the signed search.",
    }
    payload_reference = "left"
    parallel_safe = preview_safe = process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if type(p["left_right_check"]) is not bool:
            raise ValidationError("Left-right check must be a boolean", "left_right_check")

    def process(self, inputs):
        left, right = _census_pair(inputs, self.input_payloads)
        p = self.parameters
        census, aggregation = int(p["census_size"]), int(p["aggregation_size"])
        footprint = census + aggregation - 1
        plan = plan_disparity_search(inputs, p, left.shape, footprint, native_matcher=False)
        reverse = (
            _reverse_plan(inputs, p, left.shape, footprint, plan)
            if p["left_right_check"] and plan.support.any()
            else None
        )
        comparisons = _work_area(plan, aggregation)
        if reverse is not None:
            comparisons += _work_area(reverse, aggregation)
        if comparisons > _MAX_COMPARISONS:
            raise ValidationError(
                "Binary stereo work limit exceeded. Narrow the ROI or search range."
            )

        if plan.support.any():
            a, b = _census(left, census), _census(right, census)
            estimates, best, second = _match(a, b, plan, aggregation)
        else:
            estimates, best, second = (np.full(left.shape, np.nan, np.float32) for _ in range(3))
        confidence = _confidence(best, second)
        threshold = float(p["minimum_confidence"])
        unique = (confidence > 0) & (confidence >= threshold)
        valid = unique
        if reverse is not None and unique.any():
            rev_disp, rev_best, rev_second = _match(b, a, reverse, aggregation)
            rev_conf = _confidence(rev_best, rev_second)
            rev_disp[~((rev_conf > 0) & (rev_conf >= threshold))] = np.nan
            valid = _cross_checked(unique, estimates, rev_disp, int(p["left_right_tolerance_px"]))
        estimates[~valid] = np.nan
        supported = int(np.count_nonzero(plan.support))
        matched = int(np.count_nonzero(valid))
        report = {
            **plan.report,
            "model": "binary_census_stereo_v1",
            "census_size": census,
            "aggregation_size": aggregation,
            "max_cost": (census**2 - 1) * aggregation**2,
            "supported_pixels": supported,
            "minimum_confidence": threshold,
            "left_right_check": p["left_right_check"],
            "left_right_tolerance_px": int(p["left_right_tolerance_px"]),
            "ambiguous_pixels": supported - int(np.count_nonzero(unique)),
            "left_right_rejected_pixels": int(np.count_nonzero(unique)) - matched,
            "matched_pixels": matched,
            "coverage": matched / left.size,
        }
        graphics = {}
        if self.generate_graphics:
            scaled = np.zeros(left.shape, np.uint8)
            scaled[valid] = np.rint((estimates[valid] - plan.minimum) * 255 / (plan.count - 1))
            image = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
            image[~valid] = 0
            graphics["Binary disparity"] = ImageArray(image, "bgr")
        reason = "binary_stereo_no_matches" if supported else "stereo_search_no_support"
        return Result(
            {
                "disparity": ImageArray(estimates, "gray"),
                "valid_mask": valid.astype(np.uint8) * 255,
                "support_mask": plan.support,
                "confidence": ImageArray(confidence, "gray"),
                "best_cost": ImageArray(best, "gray"),
                "second_best_cost": ImageArray(second, "gray"),
                "report": report,
            },
            graphics=graphics,
            passed=bool(matched),
            score=report["coverage"],
            fail_reasons=[] if matched else [reason],
            message=f"{matched}/{supported} supported pixels matched",
        )
