"""Bounded, raw-intensity scikit-image global, multi-level and local thresholds."""

import math
import warnings

import numpy as np

from .core import Block, Port, Result
from .images import image_format, label_color_map
from .parameter_validation import NumericRange, ValidationError
from .processing import image_value
from .selection import selection_mask
from .skimage_regions import INSTALL


def filters_module():
    try:
        from skimage import filters
    except ImportError as exc:
        raise RuntimeError(INSTALL) from exc
    return filters


def samples(inputs):
    source = inputs["image"]
    array = image_value(source)
    if not 0 < array.shape[0] * array.shape[1] <= 2_100_000 or max(array.shape[:2]) > 16384:
        raise ValidationError("Thresholding supports up to 2.1M pixels and 16384 pixels per axis")
    if np.max(np.abs(array.astype(np.float64))) > 1e12:
        raise ValidationError("Threshold intensities must be finite and within +/-1e12")
    if array.ndim == 3:
        encoding = image_format(source)
        if encoding not in ("rgb", "bgr", "rgba", "bgra"):
            raise ValidationError("Convert HSV/Lab to grayscale or extract a channel first")
        weights = (0.299, 0.587, 0.114) if encoding.startswith("rgb") else (0.114, 0.587, 0.299)
        array = np.einsum("ijk,k->ij", array[..., :3].astype(np.float64), weights)
    scope = selection_mask(inputs, source)
    active = np.ones(array.shape, bool) if scope is None else scope != 0
    if not active.any():
        raise ValidationError("Threshold learning/output selection is empty")
    return array, active, array[active]


def histogram(values, bins):
    low, high = float(values.min()), float(values.max())
    if low == high:
        return np.array([values.size]), np.array([low])
    # Exact integer levels when they fit; otherwise explicitly bin even uint16 data.
    # Native integer histograms ignore nbins, which can make Multi-Otsu unbounded.
    if values.dtype.kind in "iu" and high - low + 1 <= bins:
        edges = np.arange(low, high + 2, dtype=float) - 0.5
    else:
        edges = bins
    counts, edges = np.histogram(values, bins=edges)
    return counts, (edges[:-1] + edges[1:]) / 2


def binary(image, threshold, active, polarity):
    selected = image > threshold if polarity == "bright" else image <= threshold
    return (selected & active).astype(np.uint8) * 255


class ThresholdBase(Block):
    category = block_type = "Image processing"
    subtype = "Segmentation"
    inputs = (
        Port("image"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    process_inputs_readonly = parallel_safe = True
    input_help = {
        "image": "Gray or RGB/BGR(A), uint8/uint16/int16/float32/float64. Raw units are retained; "
        "color uses weighted luminance without rescaling. Up to 2.1M pixels.",
        "mask": "Global/Multi-Otsu learn from selected pixels; local neighborhoods use the full "
        "image. All masks are zero outside selection; Multi-Otsu labels use -1 there.",
        "region": "Rectangle/circle/polygon selection, intersected with a wired mask.",
    }


class MultiOtsu(ThresholdBase):
    type_id, title = "skimage.multi_otsu", "Multi-Otsu threshold"
    outputs = (
        Port("mask"),
        Port("labels"),
        Port("preview"),
        Port("thresholds", "data"),
        Port("statistics", "data"),
    )
    defaults = {"classes": 3, "bins": 256, "selected_classes": [1, 2]}
    parameter_ranges = {
        "classes": NumericRange(2, 8, whole=True),
        "bins": NumericRange(8, 1024, whole=True),
    }
    parameter_help = {
        "classes": "Number of ordered intensity classes (0 is darkest). Work is limited to "
        "2 million candidate bin partitions; reduce bins for larger class counts.",
        "bins": "Histogram bins, including for uint16. Integer levels are exact if their range "
        "fits; otherwise cutoffs are bin centers in original units. More bins costs more work.",
        "selected_classes": "Class IDs combined into the binary mask; an empty list selects none. "
        "Labels retain all classes, with -1 outside scope. Equality joins the upper class.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        ids = p["selected_classes"]
        if (
            not isinstance(ids, list)
            or len(ids) > 8
            or any(type(i) is not int or not 0 <= i < p["classes"] for i in ids)
        ):
            raise ValidationError("Select class IDs from 0 through classes-1", "selected_classes")

    def process(self, inputs):
        filters = filters_module()
        image, active, values = samples(inputs)
        p = self.parameters
        counts, centers = histogram(values, int(p["bins"]))
        occupied = np.count_nonzero(counts)
        if occupied < p["classes"]:
            raise ValidationError("Not enough occupied intensity bins for the requested classes")
        if (
            occupied > p["classes"]
            and math.comb(len(counts) - 1, int(p["classes"]) - 1) > 2_000_000
        ):
            raise ValidationError("Multi-Otsu work budget exceeded; reduce bins or classes")
        thresholds = filters.threshold_multiotsu(classes=int(p["classes"]), hist=(counts, centers))
        labels = np.digitize(image, thresholds).astype(np.int32)
        labels[~active] = -1
        mask = np.isin(labels, p["selected_classes"]).astype(np.uint8) * 255
        totals = np.bincount(labels[active], minlength=int(p["classes"])).tolist()
        return Result(
            {
                "mask": mask,
                "labels": labels,
                "preview": label_color_map(labels + 1),
                "thresholds": thresholds.tolist(),
                "statistics": {
                    "model": "skimage_multi_otsu_v1",
                    "thresholds": thresholds.tolist(),
                    "class_counts": totals,
                    "selected_pixels": int(active.sum()),
                    "histogram": {"counts": counts.tolist(), "centers": centers.tolist()},
                    "class_rule": "digitize, equality to upper class",
                    "outside_label": -1,
                },
            },
            message=f"{p['classes']} intensity classes · thresholds {thresholds.tolist()}",
        )


class GlobalThreshold(ThresholdBase):
    type_id, title = "skimage.global_threshold", "Advanced global threshold"
    outputs = (Port("mask"), Port("threshold", "number"), Port("statistics", "data"))
    defaults = {
        "method": "yen",
        "polarity": "bright",
        "bins": 256,
        "maximum_iterations": 1000,
        "li_tolerance": 0.0,
        "li_initial": "mean",
        "li_initial_value": 127.0,
        "li_quantile": 0.5,
        "isodata_threshold": "lowest",
    }
    parameter_choices = {
        "method": ("otsu", "yen", "li", "isodata", "minimum", "triangle", "mean"),
        "polarity": ("bright", "dark"),
        "li_initial": ("mean", "median", "quantile", "value"),
        "isodata_threshold": ("lowest", "middle", "highest"),
    }
    parameter_ranges = {
        "bins": NumericRange(8, 1024, whole=True),
        "maximum_iterations": NumericRange(1, 10000, whole=True),
        "li_tolerance": NumericRange(0, 1e12),
        "li_initial_value": NumericRange(-1e12, 1e12),
        "li_quantile": NumericRange(0, 1),
    }
    parameter_help = {
        "method": "Histogram or raw-sample global estimator. Algorithm failures are reported; "
        "there is no silent fallback to another method.",
        "polarity": "Bright selects image > threshold; dark selects image <= threshold.",
        "bins": "Bounded histogram bins for Otsu, Yen, Isodata, Minimum, Triangle. Exact integer "
        "levels when the range fits. Mean and Li use the selected raw samples.",
        "maximum_iterations": "Minimum smoothing iterations or Li threshold updates; reaching "
        "the limit raises a block error instead of publishing an unconverged result.",
        "li_tolerance": "Li convergence tolerance in raw intensity units; 0 uses native automatic.",
        "li_initial": "Li start: sample mean, median, quantile or an explicit intensity.",
        "li_initial_value": "Explicit Li starting intensity, strictly inside the sample range.",
        "li_quantile": "Quantile (0..1) for the Li starting estimate.",
        "isodata_threshold": "Select lowest, middle or highest of all valid Isodata cutoffs. "
        "All candidates remain in statistics.",
    }

    def process(self, inputs):
        filters = filters_module()
        image, active, values = samples(inputs)
        p = self.parameters
        method = p["method"]
        counts, centers = histogram(
            values.astype(np.float64) if method == "triangle" else values, int(p["bins"])
        )
        iterations, candidates = [], []
        if method == "otsu" and values.min() == values.max():
            # Native Otsu short-circuits a uniform image before building its histogram.
            threshold = values.flat[0]
        elif method == "mean":
            threshold = filters.threshold_mean(values)
        elif method == "li":
            initial = {
                "mean": None,
                "median": np.median,
                "quantile": lambda a: np.quantile(a, p["li_quantile"]),
                "value": p["li_initial_value"],
            }[p["li_initial"]]

            def step(value):
                iterations.append(float(value))
                if len(iterations) > p["maximum_iterations"] + 1:
                    raise ValidationError("Li did not converge within maximum_iterations")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                threshold = filters.threshold_li(
                    values.astype(np.float64),
                    tolerance=float(p["li_tolerance"]) or None,
                    initial_guess=initial,
                    iter_callback=step,
                )
        elif method == "triangle":
            # Native Triangle has no histogram input; float input makes nbins effective.
            threshold = filters.threshold_triangle(values.astype(np.float64), nbins=int(p["bins"]))
        elif method == "isodata":
            candidates = filters.threshold_isodata(hist=(counts, centers), return_all=True).tolist()
            index = {"lowest": 0, "middle": len(candidates) // 2, "highest": -1}[
                p["isodata_threshold"]
            ]
            threshold = candidates[index]
        else:
            kwargs = {"hist": (counts, centers)}
            if method == "minimum":
                kwargs["max_num_iter"] = int(p["maximum_iterations"])
            threshold = getattr(filters, "threshold_" + method)(**kwargs)
        if not np.isfinite(threshold) or not np.isfinite(iterations).all():
            raise ValidationError("Threshold estimator produced a nonfinite result")
        threshold = float(threshold)
        return Result(
            {
                "mask": binary(image, threshold, active, p["polarity"]),
                "threshold": threshold,
                "statistics": {
                    "model": "skimage_global_threshold_v1",
                    "method": method,
                    "threshold": threshold,
                    "candidates": candidates or [threshold],
                    "li_iterations": iterations,
                    "selected_pixels": int(active.sum()),
                    "histogram": {"counts": counts.tolist(), "centers": centers.tolist()},
                },
            },
            message=f"{method.title()} · threshold {threshold:g}",
        )


class LocalThreshold(ThresholdBase):
    type_id, title = "skimage.local_threshold", "Advanced local threshold"
    outputs = (Port("mask"), Port("threshold_map"), Port("statistics", "data"))
    defaults = {
        "method": "sauvola",
        "polarity": "bright",
        "window_yx": [31, 31],
        "offset": 0.0,
        "border": "reflect",
        "border_value": 0.0,
        "sigma_yx": [0.0, 0.0],
        "k": 0.2,
        "sauvola_range": 128.0,
    }
    parameter_choices = {
        "method": ("gaussian", "mean", "median", "niblack", "sauvola"),
        "polarity": ("bright", "dark"),
        "border": ("reflect", "constant", "nearest", "mirror", "wrap"),
    }
    parameter_ranges = {
        "offset": NumericRange(-1e12, 1e12),
        "border_value": NumericRange(-1e12, 1e12),
        "k": NumericRange(-10, 10),
        "sauvola_range": NumericRange(1e-12, 1e12),
    }
    parameter_help = {
        "method": "Local Gaussian/mean/median, or native scikit-image Niblack/Sauvola. "
        "Generic Python neighborhood callables belong in user blocks.",
        "polarity": "Bright selects image > local threshold; dark uses <=.",
        "window_yx": "Odd [height, width] from 1 to 255 pixels. Median has a bounded "
        "pixel-window work limit. Neighborhoods always use the full input image.",
        "offset": "Subtract from the computed threshold, in original intensity units.",
        "border": "Border mode for Gaussian/mean/median only; Niblack/Sauvola use native reflect.",
        "border_value": "Constant border intensity for Gaussian/mean/median.",
        "sigma_yx": "Gaussian [row, column] sigmas, or [0, 0] for native window-derived values.",
        "k": "Niblack subtracts k times local deviation (OpenCV's sign convention differs). "
        "Sauvola uses mean * (1 + k * (std/range - 1)).",
        "sauvola_range": "Positive standard-deviation reference in original units; 128 for "
        "ordinary uint8, often 0.5 for normalized [0,1] images. No implicit rescaling.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        window = p["window_yx"]
        if (
            not isinstance(window, (list, tuple))
            or len(window) != 2
            or any(type(v) is not int or not 1 <= v <= 255 or v % 2 != 1 for v in window)
        ):
            raise ValidationError("Window must be two odd pixel counts in 1..255", "window_yx")
        sigma = p["sigma_yx"]
        if (
            not isinstance(sigma, (list, tuple))
            or len(sigma) != 2
            or any(
                type(v) not in (int, float) or not np.isfinite(v) or not 0 <= v <= 128
                for v in sigma
            )
            or (any(v == 0 for v in sigma) and any(v != 0 for v in sigma))
        ):
            raise ValidationError("Use two positive sigmas <=128, or [0, 0]", "sigma_yx")

    def process(self, inputs):
        filters = filters_module()
        image, active, _ = samples(inputs)
        p = self.parameters
        method = p["method"]
        image = image.astype(np.float64)
        window = tuple(p["window_yx"])
        if method == "median" and image.size * math.prod(window) > 100_000_000:
            raise ValidationError("Median pixel-window work exceeds 100M; reduce window or image")
        if method in ("niblack", "sauvola"):
            kwargs = {"window_size": window, "k": float(p["k"])}
            if method == "sauvola":
                kwargs["r"] = float(p["sauvola_range"])
            threshold = getattr(filters, "threshold_" + method)(image, **kwargs) - float(
                p["offset"]
            )
        else:
            threshold = filters.threshold_local(
                image,
                block_size=window,
                method=method,
                offset=float(p["offset"]),
                mode=p["border"],
                cval=float(p["border_value"]),
                param=tuple(p["sigma_yx"]) if any(p["sigma_yx"]) else None,
            )
        if not np.isfinite(threshold).all():
            raise ValidationError("Local threshold map contains nonfinite values")
        output = threshold.copy()
        output[~active] = 0
        return Result(
            {
                "mask": binary(image, threshold, active, p["polarity"]),
                "threshold_map": output,
                "statistics": {
                    "model": "skimage_local_threshold_v1",
                    "method": method,
                    "selected_pixels": int(active.sum()),
                    "threshold_min": float(threshold[active].min()),
                    "threshold_max": float(threshold[active].max()),
                    "threshold_mean": float(threshold[active].mean()),
                    "outside_threshold": 0,
                    "neighborhoods": "full_image",
                },
            },
            message=f"{method.title()} · {window[0]} × {window[1]} neighborhood",
        )
