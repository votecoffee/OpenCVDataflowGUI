"""Compare histogram outputs using OpenCV's per-channel metrics."""

from statistics import mean

import cv2
import numpy as np

from .core import Block, Port, Result
from .histogram import HISTOGRAM_CHANNELS
from .images import image_format
from .parameter_validation import ValidationError
from .processing import finite, image_value
from .selection import SELECTION_INPUTS, selection_mask

METRICS = {
    "correlation": cv2.HISTCMP_CORREL,
    "chi_square": cv2.HISTCMP_CHISQR,
    "intersection": cv2.HISTCMP_INTERSECT,
    "bhattacharyya": cv2.HISTCMP_BHATTACHARYYA,
    "chi_square_alt": cv2.HISTCMP_CHISQR_ALT,
    "kl_divergence": cv2.HISTCMP_KL_DIV,
}


def histogram_channels(histogram):
    if not isinstance(histogram, dict) or not isinstance(histogram.get("channels"), list):
        raise ValueError("Connect a histogram output")
    channels = {}
    for channel in histogram["channels"]:
        label = channel["channel"]
        counts = np.asarray(channel["counts"], np.float64)
        edges = np.asarray(channel["edges"], np.float64)
        if (
            not isinstance(label, str)
            or label in channels
            or counts.ndim != 1
            or not 2 <= counts.size <= 4096
            or edges.shape != (counts.size + 1,)
            or not np.isfinite(counts).all()
            or np.any(counts < 0)
            or counts.max() <= 0
            or not np.isfinite(edges).all()
            or np.any(edges[1:] <= edges[:-1])
        ):
            raise ValueError(
                "Histograms require unique channels, ordered edges and nonempty counts"
            )
        channels[label] = (counts, edges)
    if not channels:
        raise ValueError("Histogram has no channels")
    return channels


def earth_movers_distance(a, a_edges, b, b_edges, normalize):
    occupied_a, occupied_b = a > 0, b > 0
    if max(np.count_nonzero(occupied_a), np.count_nonzero(occupied_b)) > 512:
        raise ValidationError("Earth Mover's Distance supports up to 512 occupied bins per channel")
    if not normalize:
        mass_scale = max(float(a.max()), float(b.max()))
        if not np.isclose((a / mass_scale).sum(), (b / mass_scale).sum(), rtol=1e-12, atol=0):
            raise ValidationError("Earth Mover's Distance needs equal counts or Normalize enabled")

    origin = min(float(a_edges[0]), float(b_edges[0]))
    span = max(float(a_edges[-1]), float(b_edges[-1])) - origin
    if not np.isfinite(span) or span <= 0:
        raise ValidationError("Histogram bin coordinates exceed the supported range")
    scale = span if span < 1e-10 or span > 1e10 else 1.0

    def signature(counts, edges, occupied):
        shifted = (edges - origin) / scale
        centers = (shifted[:-1] * 0.5 + shifted[1:] * 0.5)[occupied]
        weights = counts[occupied] / counts.max()
        weights /= weights.sum()
        return centers, np.column_stack((weights, centers)).astype(np.float32)

    a_centers, a_signature = signature(a, a_edges, occupied_a)
    b_centers, b_signature = signature(b, b_edges, occupied_b)
    if (
        np.any(a_signature[:, 0] == 0)
        or np.any(b_signature[:, 0] == 0)
        or np.unique(np.r_[a_signature[:, 1], b_signature[:, 1]]).size
        != np.unique(np.r_[a_centers, b_centers]).size
    ):
        raise ValidationError("Occupied bins or weights are too close for OpenCV's EMD precision")
    distance, _, _ = cv2.EMD(a_signature, b_signature, cv2.DIST_L1)
    return float(distance) * scale


class CompareHistograms(Block):
    type_id, title = "inspect.compare_histograms", "Compare histograms"
    category, block_type, subtype = "Inspection", "Inspection", "Measurements"
    inputs = (Port("reference", "data"), Port("sample", "data"))
    outputs = (Port("value", "number"), Port("measurements", "data"))
    defaults = {
        "method": "bhattacharyya",
        "normalize": True,
        "reduction": "mean",
        "check_threshold": False,
        "threshold": 0.1,
    }
    parameter_choices = {
        "method": (*METRICS, "earth_movers_distance"),
        "reduction": ("mean", "worst"),
    }
    parameter_help = {
        "normalize": "Compare relative counts. EMD requires equal counts when disabled.",
        "threshold": "Minimum for correlation/intersection. Maximum for the distance metrics.",
    }

    def process(self, inputs):
        reference = histogram_channels(inputs["reference"])
        sample = histogram_channels(inputs["sample"])
        if reference.keys() != sample.keys():
            raise ValueError("Histograms must have the same channels")
        method = self.parameters["method"]
        similarity = method in ("correlation", "intersection")
        measurements = {}
        for channel, (a, edges) in reference.items():
            b, other_edges = sample[channel]
            if method == "earth_movers_distance":
                value = earth_movers_distance(
                    a, edges, b, other_edges, self.parameters["normalize"]
                )
            else:
                if not np.array_equal(edges, other_edges):
                    raise ValueError("Histogram bin edges differ. Set matching manual ranges.")
                if self.parameters["normalize"]:
                    a, b = a / a.sum(), b / b.sum()
                value = cv2.compareHist(a.astype(np.float32), b.astype(np.float32), METRICS[method])
            if not np.isfinite(value):
                raise ValueError("Histogram metric produced a nonfinite result")
            measurements[channel] = float(value)
        values = list(measurements.values())
        if self.parameters["reduction"] == "worst":
            value = min(values) if similarity else max(values)
        elif method == "earth_movers_distance":
            value = mean(values)
        else:
            value = float(np.mean(values))
        check = self.parameters["check_threshold"]
        threshold = finite(self.parameters["threshold"], "Threshold") if check else None
        passed = not check or (value >= threshold if similarity else value <= threshold)
        details = {"method": method, "channels": measurements}
        if method == "earth_movers_distance":
            details["units"] = "sample"
        return Result(
            {"value": value, "measurements": details},
            passed=passed,
            message=f"Histogram {method.replace('_', ' ')} = {value:.6g}",
            fail_reasons=[] if passed else ["histogram_out_of_tolerance"],
        )


class HistogramBackProjection(Block):
    type_id, title = "inspect.histogram_backproject", "Histogram backprojection"
    category, block_type, subtype = "Inspection", "Inspection", "Measurements"
    inputs = (*SELECTION_INPUTS, Port("histogram", "data"))
    outputs = (Port("image"),)
    defaults = {"channel": "auto", "normalization": "peak", "scale": 1.0}
    parameter_choices = {
        "channel": (
            "auto",
            *dict.fromkeys(c for names in HISTOGRAM_CHANNELS.values() for c in names),
        ),
        "normalization": ("peak", "probability", "counts"),
    }
    parameter_help = {
        "channel": "Select the modeled channel. Auto requires exactly one compatible channel.",
        "normalization": "Divide counts by their peak or sum, or keep the raw counts.",
        "scale": "Multiplier applied to the resulting bin weights.",
    }

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source)
        labels = HISTOGRAM_CHANNELS[image_format(source)]
        channels = histogram_channels(inputs["histogram"])
        channel = self.parameters["channel"]
        if channel == "auto":
            common = [label for label in labels if label in channels]
            if len(common) != 1:
                raise ValueError("Select a histogram channel or provide a single compatible plane")
            channel = common[0]
        if channel not in channels or (image.ndim == 3 and channel not in labels):
            raise ValueError("Histogram channel must be present in the model and color image")
        plane = image if image.ndim == 2 else image[..., labels.index(channel)]
        counts, edges = channels[channel]
        low, high = float(edges[0]), float(edges[-1])
        width = (high - low) / len(counts)
        limit = np.finfo(np.float32).max
        if (
            max(abs(low), abs(high)) > limit
            or not np.isfinite(width)
            or not np.allclose(np.diff(edges), width, rtol=1e-5, atol=0)
            or not np.all(np.diff(edges.astype(np.float32).astype(float)) > 0)
        ):
            raise ValueError("Backprojection requires uniform float32-representable bin edges")
        mode = self.parameters["normalization"]
        if mode in ("peak", "probability"):
            counts = counts / counts.max()
            if mode == "probability":
                counts /= counts.sum()
        scale = finite(self.parameters["scale"], "Scale", 0)
        if counts.max() > limit or counts.max() * scale > limit:
            raise ValueError("Backprojection weights exceed the float32 output range")
        in_range = (plane >= low) & (plane <= high)
        samples = np.where(in_range, plane, low).astype(np.float32)
        projection = cv2.calcBackProject(
            [samples], [0], counts.astype(np.float32), [low, high], scale
        )
        # Histogram includes the rightmost endpoint. OpenCV uses a half-open interval.
        projection[in_range & (samples == np.float32(high))] = counts[-1] * scale
        selection = selection_mask(inputs, image)
        if selection is not None:
            in_range &= selection != 0
        projection[~in_range] = 0
        return Result({"image": projection}, message=f"{channel} histogram backprojection")
