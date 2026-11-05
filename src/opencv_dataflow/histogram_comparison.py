"""Compare histogram outputs using OpenCV's per-channel metrics."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .processing import finite

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
    parameter_choices = {"method": tuple(METRICS), "reduction": ("mean", "worst")}
    parameter_help = {
        "normalize": "Divide each histogram by its sample count before comparison.",
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
        else:
            value = float(np.mean(values))
        check = self.parameters["check_threshold"]
        threshold = finite(self.parameters["threshold"], "Threshold") if check else None
        passed = not check or (value >= threshold if similarity else value <= threshold)
        return Result(
            {"value": value, "measurements": {"method": method, "channels": measurements}},
            passed=passed,
            message=f"Histogram {method.replace('_', ' ')} = {value:.6g}",
            fail_reasons=[] if passed else ["histogram_out_of_tolerance"],
        )
