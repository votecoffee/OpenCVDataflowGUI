"""Inspect binary stereo matching costs at selected rectified pixels."""

from collections import Counter

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .binary_stereo import BinaryStereoDisparity, _census, _census_offsets, _census_pair
from .core import Port, Result
from .disparity_search import disparity_search_interval, rectification_rois
from .geometry3d_common import Tool3D
from .graphics_layers import GraphicDrawing, graphic_circle, graphic_object
from .images import ImageArray
from .parameter_validation import NumericRange, ValidationError
from .payloads import ValuePacket, require_same_frame
from .projection import profile_plot
from .result_graphics import ACCEPTED, REJECTED
from .selection import selection_mask

_VARIANTS = ("center", "mean", "symmetric", "star")
_MAX_PROBES = 64
_AMBER = (30, 180, 255)
_SHARED_PARAMETERS = (
    "search_mode",
    "min_disparity",
    "num_disparities",
    "minimum_depth",
    "maximum_depth",
    "census_size",
    "aggregation_size",
    "minimum_confidence",
)


def _probe_points(values):
    if not isinstance(values, (list, tuple, np.ndarray)) or (
        isinstance(values, np.ndarray) and values.ndim == 0
    ):
        raise ValidationError("Use a list of whole-pixel [x, y] probe pairs", "probe_points")
    if len(values) > _MAX_PROBES:
        raise ValidationError(f"Use at most {_MAX_PROBES} pixel probes", "probe_points")
    result = []
    for point in values:
        if (
            not isinstance(point, (list, tuple, np.ndarray))
            or (isinstance(point, np.ndarray) and point.ndim != 1)
            or len(point) != 2
            or any(
                isinstance(v, (bool, np.bool_))
                or not isinstance(v, (int, float, np.integer, np.floating))
                or not 0 <= v <= 1_000_000
                or int(v) != v
                for v in point
            )
        ):
            raise ValidationError("Use whole nonnegative [x, y] probe pixels", "probe_points")
        result.append(tuple(map(int, point)))
    return result


def _footprint_issue(x, y, radius, roi, mask):
    start_x, start_y, width, height = roi
    if (
        x - radius < start_x
        or y - radius < start_y
        or x + radius >= start_x + width
        or y + radius >= start_y + height
    ):
        return "outside_roi"
    if (
        mask is not None
        and not mask[y - radius : y + radius + 1, x - radius : x + radius + 1].all()
    ):
        return "masked"
    return None


def _candidate_costs(left, right, x, y, right_x, census, aggregation, variant):
    margin, pad = census // 2, aggregation // 2
    radius = margin + pad
    yslice = slice(y - radius, y + radius + 1)
    first, last = min(right_x), max(right_x)
    a = _census(left[yslice, x - radius : x + radius + 1], census, variant)
    b = _census(right[yslice, first - radius : last + radius + 1], census, variant)
    a = a[margin:-margin, margin:-margin]
    b = b[margin:-margin, margin:-margin]
    neighborhoods = sliding_window_view(b, (aggregation, aggregation))
    comparisons = neighborhoods[0, np.asarray(right_x) - first]
    return np.bitwise_count(comparisons ^ a).sum(axis=(1, 2)).tolist()


def _measure_probe(left, right, point, disparities, rois, masks, settings):
    x, y = point
    census, aggregation = int(settings["census_size"]), int(settings["aggregation_size"])
    radius = census // 2 + aggregation // 2
    left_issue = _footprint_issue(x, y, radius, rois[0], masks[0])
    candidates = []
    measured = []
    for disparity in disparities:
        right_x = x - disparity
        if left_issue:
            issue = f"left_{left_issue}"
        else:
            right_issue = _footprint_issue(right_x, y, radius, rois[1], masks[1])
            issue = f"right_{right_issue}" if right_issue else None
        row = {"disparity_px": disparity, "right_x": right_x, "cost": None, "unavailable": issue}
        if issue is None:
            measured.append(row)
        candidates.append(row)
    if measured:
        costs = _candidate_costs(
            left,
            right,
            x,
            y,
            [row["right_x"] for row in measured],
            census,
            aggregation,
            settings["census_variant"],
        )
        for row, cost in zip(measured, costs, strict=True):
            row["cost"] = cost
    ranked = sorted(measured, key=lambda row: row["cost"])
    best = ranked[0]["cost"] if ranked else None
    second = ranked[1]["cost"] if len(ranked) >= 2 else None
    relative_gap = None if second is None else (second - best) / second if second else 0.0
    minima = [row["disparity_px"] for row in measured if row["cost"] == best]
    if not measured:
        status = "no_support"
    elif len(measured) < len(candidates):
        status = "partial_support"
    elif not relative_gap or relative_gap < float(settings["minimum_confidence"]):
        status = "ambiguous"
    else:
        status = "matched"
    return {
        "left_xy": [x, y],
        "status": status,
        "supported_candidates": len(measured),
        "minimum_cost": best,
        "next_cost": second,
        "relative_gap": relative_gap,
        "minimum_disparities_px": minima,
        "selected_disparity_px": minima[0] if status == "matched" else None,
        "candidates": candidates,
    }


def _graphics(curves, plot_probe, disparities):
    colors = {
        "matched": ACCEPTED,
        "ambiguous": _AMBER,
        "partial_support": _AMBER,
        "no_support": REJECTED,
    }
    objects = []
    for index, curve in enumerate(curves, 1):
        x, y = curve["left_xy"]
        color = colors[curve["status"]]
        objects.append(graphic_circle(x, y, 4, color, thickness=2))
        objects.append(graphic_object("label", [(x + 5, y)], color, text=str(index)))
    plots = {}
    if plot_probe <= len(curves):
        curve = curves[plot_probe - 1]
        values = np.array([row["cost"] for row in curve["candidates"]], np.float64)
        graph = profile_plot(
            values[:, None],
            np.isfinite(values),
            [f"Probe {plot_probe}"],
            "disparity",
            "Hamming cost",
            distances=np.array(disparities),
        )
        plots[f"Probe {plot_probe} cost curve"] = ImageArray(graph, "bgr")
    return plots, {"Stereo probe pixels": GraphicDrawing(tuple(objects), source="left")}


class InspectBinaryStereoCosts(Tool3D):
    type_id, title, subtype = "3d.binary_stereo_costs", "Inspect binary stereo costs", "Stereo"
    inputs = (
        *BinaryStereoDisparity.inputs,
        Port("points", "data", required=False, required_when_connected=True),
    )
    outputs = (Port("curves", "data"), Port("report", "data"), Port("image"))
    defaults = {
        **{key: BinaryStereoDisparity.defaults[key] for key in _SHARED_PARAMETERS},
        "probe_points": [],
        "census_variant": "center",
        "plot_probe": 1,
    }
    parameter_choices = {
        **BinaryStereoDisparity.parameter_choices,
        "census_variant": _VARIANTS,
    }
    parameter_ranges = {
        **{
            key: BinaryStereoDisparity.parameter_ranges[key]
            for key in _SHARED_PARAMETERS
            if key in BinaryStereoDisparity.parameter_ranges
        },
        "plot_probe": NumericRange(1, _MAX_PROBES, whole=True),
    }
    parameter_help = {
        **{key: BinaryStereoDisparity.parameter_help[key] for key in _SHARED_PARAMETERS},
        "census_variant": "Compare neighbors to center, local mean, opposite pixels or star rays.",
        "probe_points": "Whole [x, y] left pixels. An empty list uses the image center.",
        "plot_probe": "One-based probe to show as a plot in the result graphics.",
    }
    input_help = {
        **BinaryStereoDisparity.input_help,
        "points": "Optional list of whole [x, y] pairs in the left view. Overrides saved probes.",
        "right_mask": "Right support mask. Each candidate is tested separately.",
    }
    payload_reference = "left"
    composes_result_image = True
    parallel_safe = preview_safe = process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        _probe_points(p["probe_points"])

    def process(self, inputs):
        left, right = _census_pair(inputs, self.input_payloads)
        h, w = left.shape
        p = self.parameters
        rectification = inputs.get("rectification")
        rois = rectification_rois(rectification, (w, h))
        minimum, count, expected = disparity_search_interval(
            rectification, p, w, native_matcher=False
        )
        masks = tuple(
            selection_mask({"mask": inputs.get(side + "_mask")}, inputs[side])
            for side in ("left", "right")
        )
        points = _probe_points(inputs.get("points", p["probe_points"]))
        if "points" in inputs:
            require_same_frame(
                [inputs["left"], ValuePacket(points, self.input_payloads.get("points", {}))],
                message="Probe points must use the left pixel frame",
            )
        points = points or [(w // 2, h // 2)]
        if any(x >= w or y >= h for x, y in points):
            raise ValidationError("Probe pixels must lie inside the left image", "probe_points")
        disparities = range(minimum, minimum + count)
        curves = [_measure_probe(left, right, pt, disparities, rois, masks, p) for pt in points]
        counts = Counter(curve["status"] for curve in curves)
        measurements = sum(curve["supported_candidates"] for curve in curves)
        census, aggregation = int(p["census_size"]), int(p["aggregation_size"])
        bits = len(_census_offsets(census, p["census_variant"]))
        report = {
            "model": "binary_stereo_cost_curves_v1",
            "census_variant": p["census_variant"],
            "census_size": census,
            "aggregation_size": aggregation,
            "footprint_size": census + aggregation - 1,
            "bits_per_descriptor": bits,
            "maximum_cost": bits * aggregation**2,
            "minimum_disparity_px": minimum,
            "maximum_disparity_px": disparities[-1],
            "expected_disparity_px": expected,
            "probes": len(curves),
            "matched_probes": counts["matched"],
            "ambiguous_probes": counts["ambiguous"],
            "partial_support_probes": counts["partial_support"],
            "unsupported_probes": counts["no_support"],
            "measured_candidates": measurements,
        }
        plots, drawings = (
            _graphics(curves, int(p["plot_probe"]), disparities)
            if self.generate_graphics
            else ({}, {})
        )
        return Result(
            {"curves": curves, "report": report, "image": inputs["left"]},
            graphics=plots,
            drawings=drawings,
            passed=bool(measurements),
            score=counts["matched"] / len(curves),
            fail_reasons=[] if measurements else ["binary_stereo_no_probe_support"],
            message=f"{counts['matched']}/{len(curves)} probes have a unique supported minimum",
        )
