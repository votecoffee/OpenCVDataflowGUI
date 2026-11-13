"""Selected straight-edge ESF/SFR estimates, not an ISO conformance instrument."""

import math

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .processing import gray_image
from .result_graphics import ACCEPTED, REJECTED
from .selection import SELECTION_INPUTS, selection_mask


class _UnsupportedEdge(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def _crossing(x, y, level, *, falling=False):
    pairs = (
        ((y[:-1] >= level) & (y[1:] <= level) & (y[1:] < y[:-1]))
        if falling
        else ((y[:-1] <= level) & (y[1:] >= level) & (y[1:] > y[:-1]))
    )
    indices = np.flatnonzero(pairs)
    if not indices.size:
        return None
    index = int(indices[0])
    return float(
        x[index] + (level - y[index]) * (x[index + 1] - x[index]) / (y[index + 1] - y[index])
    )


def _fit_edge(gray, selected, p, report):
    ys, xs = np.flatnonzero(selected.any(axis=1)), np.flatnonzero(selected.any(axis=0))
    if not len(xs):
        raise _UnsupportedEdge("edge_support", "The selection contains no edge samples")
    left, top, right, bottom = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    crop = gray[top:bottom, left:right]
    mask = selected[top:bottom, left:right]
    orientation = p["orientation"]
    if orientation == "auto":
        dx = cv2.Sobel(crop, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(crop, cv2.CV_64F, 0, 1, ksize=3)
        orientation = (
            "vertical" if np.abs(dx[mask]).sum() >= np.abs(dy[mask]).sum() else "horizontal"
        )
    if orientation == "horizontal":
        crop, mask = crop.T, mask.T
    report["orientation"] = orientation
    radius, plateau = int(p["half_width"]), int(p["plateau_samples"])
    if crop.shape[1] < 2 * radius + 3:
        raise _UnsupportedEdge("edge_support", "Selection is too narrow for the sampling band")
    points, rows = [], []
    for row in np.flatnonzero(mask.any(axis=1)):
        columns = np.flatnonzero(mask[row])
        lo, hi = int(columns[0]), int(columns[-1]) + 1
        if hi - lo < 2 * radius + 3 or not mask[row, lo:hi].all():
            continue
        profile = crop[row, lo:hi]
        contrast = float(profile[-plateau:].mean() - profile[:plateau].mean())
        if abs(contrast) < float(p["minimum_contrast"]):
            continue
        derivative = (profile[2:] - profile[:-2]) * (1 if contrast > 0 else -1)
        peak = int(np.argmax(derivative)) + 1
        start, end = peak - radius, peak + radius + 1
        if start < 1 or end >= len(profile) - 1:
            continue
        weights = derivative[start - 1 : end - 1]
        total = weights.sum()
        if total <= 0:
            continue
        center = float(np.arange(start, end) @ weights / total) + lo
        if not lo + radius + 1 <= center <= hi - radius - 2:
            continue
        points.append((center, float(row)))
        rows.append(int(row))
    report["valid_rows"] = len(rows)
    report["selected_rows"] = int(mask.any(axis=1).sum())
    if (
        len(rows) < int(p["minimum_rows"])
        or len(rows) < report["selected_rows"] * p["minimum_coverage"]
    ):
        raise _UnsupportedEdge(
            "edge_support", "Insufficient continuous edge rows with contrast and plateaus"
        )
    points = np.asarray(points, np.float64)
    fit = cv2.fitLine(
        points, {"l2": cv2.DIST_L2, "huber": cv2.DIST_HUBER}[p["fit_method"]], 0, 0.01, 0.01
    )
    vx, vy, x0, y0 = fit.ravel().astype(float)
    if abs(vy) < 1e-9:
        raise _UnsupportedEdge("edge_angle", "Edge does not follow the selected orientation")
    slope, intercept = vx / vy, x0 - vx / vy * y0
    slant = math.degrees(math.atan(slope))
    residual = float(np.sqrt(np.mean((points[:, 0] - slope * points[:, 1] - intercept) ** 2)))
    report.update(slant_degrees=slant, fit_rms_pixels=residual)
    if not p["minimum_slant"] <= abs(slant) <= p["maximum_slant"]:
        raise _UnsupportedEdge("edge_angle", "Edge slant is outside the configured range")
    if residual > p["maximum_fit_rms"]:
        raise _UnsupportedEdge(
            "edge_fit", "Edge is curved, ambiguous or too noisy for a straight fit"
        )
    # Signed distance is perpendicular to the fitted line, in current image pixels.
    distances, samples = [], []
    normal = math.sqrt(1 + slope * slope)
    for row in rows:
        center = slope * row + intercept
        lo = math.ceil(center - radius * normal)
        hi = math.floor(center + radius * normal) + 1
        if lo < 0 or hi > crop.shape[1] or not mask[row, lo:hi].all():
            raise _UnsupportedEdge(
                "edge_support", "The fitted sampling band leaves the image or selection"
            )
        columns = np.arange(lo, hi)
        distances.append((columns - center) / normal)
        samples.append(crop[row, lo:hi])
    endpoints = np.array(
        [[slope * rows[0] + intercept, rows[0]], [slope * rows[-1] + intercept, rows[-1]]]
    )
    if orientation == "horizontal":
        endpoints = endpoints[:, ::-1]
    endpoints += [left, top]
    report["edge_endpoints"] = endpoints.tolist()
    return np.concatenate(distances), np.concatenate(samples)


def _curves(distances, samples, p, report):
    radius, oversampling = int(p["half_width"]), int(p["oversampling"])
    bins = 2 * radius * oversampling
    indices = np.floor((distances + radius) * oversampling).astype(int)
    inside = (indices >= 0) & (indices < bins)
    counts = np.bincount(indices[inside], minlength=bins)
    if not counts.all():
        raise _UnsupportedEdge(
            "edge_sampling", "Subpixel phases leave empty ESF bins; use a longer/slanted edge"
        )
    esf = np.bincount(indices[inside], weights=samples[inside], minlength=bins) / counts
    positions = (np.arange(bins) + 0.5) / oversampling - radius
    tail = max(4, bins // 10)
    low, high = float(esf[:tail].mean()), float(esf[-tail:].mean())
    contrast = high - low
    report.update(
        contrast=abs(contrast),
        polarity="rising" if contrast > 0 else "falling",
        minimum_bin_samples=int(counts.min()),
        sampled_pixels=int(inside.sum()),
    )
    if abs(contrast) < p["minimum_contrast"]:
        raise _UnsupportedEdge("edge_contrast", "Edge contrast is below the configured minimum")
    clipped = float(np.mean((samples <= 0) | (samples >= 1)))
    report["clipped_fraction"] = clipped
    if p["reject_clipping"] and clipped > p["maximum_clipped_fraction"]:
        raise _UnsupportedEdge(
            "edge_clipping", "Too many samples are at the luminance range limits"
        )
    esf = (esf - low) / contrast
    half_tail = tail // 2
    drift = max(
        abs(esf[:half_tail].mean() - esf[half_tail:tail].mean()),
        abs(esf[-half_tail:].mean() - esf[-tail:-half_tail].mean()),
    )
    report["plateau_drift"] = float(drift)
    if drift > p["maximum_plateau_drift"]:
        raise _UnsupportedEdge("edge_plateau", "The ESF has insufficient flat plateau support")
    lsf = (esf[2:] - esf[:-2]) * oversampling / 2
    window = {"hann": np.hanning, "hamming": np.hamming, "none": np.ones}[p["window"]](len(lsf))
    weighted = lsf * window
    # Zero padding refines frequency interpolation; it adds no measured bandwidth.
    size = cv2.getOptimalDFTSize(8 * len(lsf))
    padded = np.zeros(size, np.float64)
    padded[: len(lsf)] = weighted
    transform = cv2.dft(padded, flags=cv2.DFT_COMPLEX_OUTPUT).reshape(-1, 2)
    frequency = np.arange(size // 2 + 1) * oversampling / size
    magnitude = cv2.magnitude(
        transform[: len(frequency), 0].copy(), transform[: len(frequency), 1].copy()
    ).ravel()
    if magnitude[0] <= 1e-12:
        raise _UnsupportedEdge(
            "edge_response", "The line-spread function has no usable DC response"
        )
    response = magnitude / magnitude[0]
    keep = frequency < 0.5
    # Always include native Nyquist, even if it lies between DFT samples.
    response = np.r_[response[keep], np.interp(0.5, frequency, response)]
    frequency = np.r_[frequency[keep], 0.5]
    if p["derivative_correction"]:
        # Central differences have response sin(2*pi*f*dx)/(2*pi*f*dx).
        response /= np.sinc(2 * frequency / oversampling)
    pitch = float(p["sample_pitch"])
    metrics = {}
    for percent in (50, 10):
        value = _crossing(frequency, response, percent / 100, falling=True)
        metrics[f"mtf{percent}_cycles_per_pixel"] = value
        metrics[f"mtf{percent}"] = None if value is None else value / pitch
        metrics[f"mtf{percent}_status"] = "above_nyquist" if value is None else "measured"
    lower, upper = _crossing(positions, esf, 0.1), _crossing(positions, esf, 0.9)
    width = None if lower is None or upper is None else upper - lower
    metrics["edge_width_10_90_pixels"] = width
    metrics["edge_width_10_90"] = None if width is None else width * pitch
    report.update(
        metrics,
        nyquist=0.5 / pitch,
        esf={
            "positions_pixels": positions.tolist(),
            "normalized_values": esf.tolist(),
            "sample_counts": counts.tolist(),
        },
        lsf={
            "positions_pixels": positions[1:-1].tolist(),
            "values": lsf.tolist(),
            "windowed_values": weighted.tolist(),
        },
        sfr={
            "cycles_per_pixel": frequency.tolist(),
            "frequencies": (frequency / pitch).tolist(),
            "response": response.tolist(),
        },
    )


def _plot(report):
    image = np.full((390, 760, 3), 24, np.uint8)
    panels = (
        (
            "Edge spread (input pixels)",
            report["esf"]["positions_pixels"],
            report["esf"]["normalized_values"],
            -0.1,
            1.1,
        ),
        (
            f"SFR (cycles/{report['length_unit']})",
            report["sfr"]["frequencies"],
            report["sfr"]["response"],
            0,
            max(1.1, max(report["sfr"]["response"])),
        ),
    )
    for index, (title, x, y, lo, hi) in enumerate(panels):
        left, right = 50 + index * 375, 355 + index * 375
        top, bottom = 56, 320
        cv2.putText(
            image, title, (left - 20, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1
        )
        cv2.rectangle(image, (left, top), (right, bottom), (100, 100, 100), 1)
        px = left + (np.asarray(x) - x[0]) / (x[-1] - x[0]) * (right - left)
        py = bottom - (np.asarray(y) - lo) / (hi - lo) * (bottom - top)
        points = np.column_stack((px, np.clip(py, top, bottom))).astype(np.int32)
        cv2.polylines(image, [points], False, (90, 220, 240), 1, cv2.LINE_AA)
        for value in (0, 0.5, 1):
            cy = round(bottom - (value - lo) / (hi - lo) * (bottom - top))
            cv2.putText(
                image,
                f"{value:g}",
                (left - 30, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (180, 180, 180),
                1,
            )
        for value, cx in ((x[0], left), (x[-1], right - 35)):
            cv2.putText(
                image,
                f"{value:.3g}",
                (cx, bottom + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (180, 180, 180),
                1,
            )
    mtf = report["mtf50"]
    text = f"MTF50: {mtf:.4g}" if mtf is not None else f"MTF50 > {report['nyquist']:.4g} (Nyquist)"
    cv2.putText(
        image,
        text + f" cycles/{report['length_unit']}",
        (45, 375),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
    )
    return image


class MeasureSlantedEdge(Block):
    type_id, title = "inspect.slanted_edge", "Measure slanted-edge sharpness"
    category, block_type, subtype = "Inspection", "Inspection", "Measurements"
    inputs = SELECTION_INPUTS
    outputs = (
        Port("mtf50", "number", required=False),
        Port("mtf10", "number", required=False),
        Port("measurements", "data"),
        Port("image"),
    )
    defaults = {
        "orientation": "auto",
        "half_width": 16,
        "oversampling": "4",
        "fit_method": "huber",
        "minimum_rows": 32,
        "minimum_coverage": 0.8,
        "minimum_contrast": 0.1,
        "minimum_slant": 1.0,
        "maximum_slant": 15.0,
        "maximum_fit_rms": 0.35,
        "plateau_samples": 4,
        "maximum_plateau_drift": 0.01,
        "window": "hann",
        "derivative_correction": True,
        "gamma": 1.0,
        "reject_clipping": True,
        "maximum_clipped_fraction": 0.01,
        "sample_pitch": 1.0,
        "length_unit": "pixel",
        "check_threshold": False,
        "minimum_mtf50": 0.15,
    }
    parameter_choices = {
        "orientation": ("auto", "vertical", "horizontal"),
        "oversampling": ("2", "4", "8"),
        "fit_method": ("huber", "l2"),
        "window": ("hann", "hamming", "none"),
        "length_unit": ("pixel", "mm", "um", "in"),
    }
    parameter_ranges = {
        "half_width": NumericRange(4, 128, whole=True),
        "minimum_rows": NumericRange(8, 4096, whole=True),
        "minimum_coverage": NumericRange(0.1, 1),
        "minimum_contrast": NumericRange(0.001, 1),
        "minimum_slant": NumericRange(0.1, 15),
        "maximum_slant": NumericRange(0.1, 30),
        "maximum_fit_rms": NumericRange(0.01, 5),
        "plateau_samples": NumericRange(2, 32, whole=True),
        "maximum_plateau_drift": NumericRange(0.001, 1),
        "gamma": NumericRange(0.1, 5),
        "maximum_clipped_fraction": NumericRange(0, 1),
        "sample_pitch": NumericRange(1e-9, 1e9),
        "minimum_mtf50": NumericRange(0, 1e9),
    }
    parameter_help = {
        "orientation": "Select one near-vertical or near-horizontal straight edge in the ROI.",
        "half_width": (
            "Perpendicular sampling distance on each side, in current input pixels; "
            "include flat plateaus."
        ),
        "oversampling": "ESF bins per pixel. A long slanted edge must populate every bin.",
        "fit_method": "Native fitLine distance metric for the scanline gradient centroids.",
        "minimum_rows": "Minimum number of complete, contrasting scanlines along the edge.",
        "minimum_coverage": "Fraction of selected scanlines required to contain a supported edge.",
        "minimum_contrast": (
            "Minimum plateau difference after range normalization and gamma, on [0,1]."
        ),
        "minimum_slant": (
            "Absolute angle from vertical/horizontal; avoid axis-aligned edges "
            "with no subpixel diversity."
        ),
        "maximum_slant": "Maximum absolute slant from the chosen orientation, in degrees.",
        "maximum_fit_rms": "Maximum scanline centroid residual, in current input pixels.",
        "plateau_samples": (
            "Samples at each selected scanline end used to find edge contrast/polarity."
        ),
        "maximum_plateau_drift": "Allowed end-plateau drift as a fraction of edge contrast.",
        "window": "Taper the line-spread function before the DFT; keep fixed for comparisons.",
        "derivative_correction": (
            "Compensate the central-difference frequency response through native Nyquist."
        ),
        "gamma": (
            "Raise normalized luminance to this power before measuring. "
            "1 assumes linear light; this is not an sRGB decoder."
        ),
        "reject_clipping": (
            "Fail when luminance samples at 0 or full scale exceed the configured fraction."
        ),
        "maximum_clipped_fraction": "Maximum fraction at normalized luminance range limits.",
        "sample_pitch": (
            "Length per current input pixel at the measurement plane. No calibration is inferred."
        ),
        "length_unit": (
            "Frequency outputs are cycles per this unit; sample pitch uses the same unit."
        ),
        "check_threshold": (
            "Apply an inclusive minimum MTF50, expressed in cycles per selected length unit."
        ),
        "minimum_mtf50": (
            "Inclusive sharpness limit. A crossing above Nyquist is reported as a bound, "
            "not invented."
        ),
    }
    input_help = {
        "image": (
            "Gray/color uint8 or uint16, or float32 on [0,1]. Original output image is preserved."
        ),
        "mask": (
            "Optional edge support mask; intersects the region. "
            "Sampling bands need continuous support."
        ),
        "region": (
            "Optional image-space ROI around one straight edge, with both plateaus included."
        ),
    }
    process_inputs_readonly = True
    parallel_safe = True

    def process(self, inputs):
        p, source = self.parameters, inputs["image"]
        if p["minimum_slant"] > p["maximum_slant"]:
            raise ValidationError(
                "Minimum slant must not exceed maximum slant", "minimum_slant", "maximum_slant"
            )
        if p["plateau_samples"] > p["half_width"]:
            raise ValidationError(
                "Plateau samples must not exceed the sampling half width", "plateau_samples"
            )
        if min(source.shape[:2]) < 8 or np.prod(source.shape[:2]) > 16_777_216:
            raise ValidationError(
                "Slanted-edge measurement needs at least 8 x 8 and at most 16,777,216 pixels"
            )
        gray = gray_image(source).astype(np.float64)
        if source.dtype.kind == "u":
            gray /= np.iinfo(source.dtype).max
        if gray.min() < 0 or gray.max() > 1:
            raise ValidationError("Floating luminance must be in [0,1]")
        gray **= float(p["gamma"])
        mask = selection_mask(inputs, source)
        selected = np.ones(gray.shape, bool) if mask is None else mask != 0
        report = {
            "model": "slanted_edge_v1",
            "valid": False,
            "sample_pitch": p["sample_pitch"],
            "length_unit": p["length_unit"],
            "frequency_unit": f"cycles/{p['length_unit']}",
            "parameters": dict(p),
            "measurement_grid": "current_input_pixels",
        }
        reason = None
        try:
            distances, samples = _fit_edge(gray, selected, p, report)
            _curves(distances, samples, p, report)
            report["valid"] = True
            value = report["mtf50"]
            bound = report["nyquist"] if value is None else value
            if p["check_threshold"] and bound < p["minimum_mtf50"]:
                reason = (
                    "sharpness_beyond_measurement_band"
                    if value is None
                    else "sharpness_below_minimum"
                )
            message = (
                f"MTF50 {value:.4g}" if value is not None else f"MTF50 > {bound:.4g} (Nyquist)"
            )
            message += f" cycles/{p['length_unit']}"
        except _UnsupportedEdge as exc:
            reason, message = exc.code, str(exc)
        report["failure_reason"] = reason
        result = Result(
            {
                "image": source,
                "measurements": report,
                "mtf50": report.get("mtf50") if report.get("mtf50") is not None else NO_VALUE,
                "mtf10": report.get("mtf10") if report.get("mtf10") is not None else NO_VALUE,
            },
            passed=reason is None,
            message=message,
            fail_reasons=[] if reason is None else [reason],
            score=(
                min(1.0, (report["mtf50_cycles_per_pixel"] or 0.5) / 0.5)
                if report["valid"]
                else None
            ),
            event_data={
                key: report.get(key)
                for key in (
                    "valid",
                    "failure_reason",
                    "frequency_unit",
                    "mtf50",
                    "mtf50_status",
                    "mtf50_cycles_per_pixel",
                    "edge_width_10_90_pixels",
                )
            },
        )
        if self.generate_graphics:
            color = ACCEPTED if reason is None else REJECTED
            objects = [graphic_object("label", ((8, 22),), color, text=message)]
            if "edge_endpoints" in report:
                objects.append(graphic_object("polyline", report["edge_endpoints"], color))
                endpoints = np.asarray(report["edge_endpoints"])
                extent = p["half_width"] / math.cos(math.radians(report["slant_degrees"]))
                offset = np.array(
                    (extent, 0) if report["orientation"] == "vertical" else (0, extent)
                )
                band = np.vstack(
                    (endpoints + offset, (endpoints - offset)[::-1], endpoints[:1] + offset)
                )
                objects.append(graphic_object("polyline", band, (200, 180, 70)))
            result.drawings["slanted edge"] = GraphicDrawing(tuple(objects))
        if self.capture_graphics and report["valid"]:
            result.graphics["edge spread and SFR"] = _plot(report)
        return result
