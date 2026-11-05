"""Selected image statistics with native extrema and reusable result markers."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .histogram import CHANNEL_CHOICES, HISTOGRAM_CHANNELS
from .images import image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .processing import image_value
from .selection import SELECTION_INPUTS, selection_mask

METRICS = ("minimum", "maximum", "mean", "standard_deviation", "rms", "sum", "nonzero_count")
PERCENTILE_RANGE = NumericRange(0, 100)


def channel_summary(plane, mask, selected, percentiles, method):
    if selected is not None and not selected.any():
        return {
            **dict.fromkeys(METRICS),
            "nonzero_count": 0,
            "minimum_location": None,
            "maximum_location": None,
            "percentiles": [
                {"percentile": percentile, "value": None} for percentile in percentiles
            ],
        }
    minimum, maximum, min_point, max_point = cv2.minMaxLoc(plane, mask=mask)
    samples = plane if selected is None else plane[selected]
    samples = np.asarray(samples, dtype=np.float64).reshape(-1, 1)
    maximum_magnitude = max(abs(minimum), abs(maximum))
    scale = (
        maximum_magnitude if 0 < maximum_magnitude < 1e-150 or maximum_magnitude > 1e150 else 1.0
    )
    values = samples if scale == 1 else samples / scale
    total = cv2.sumElems(values)[0]
    mean = total / len(values)
    # Center the samples before native variance accumulation to preserve low contrast.
    _, deviation = cv2.meanStdDev(values - mean)
    rms = np.sqrt(cv2.norm(values, cv2.NORM_L2SQR) / len(values))
    summary = {
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean * scale,
        "standard_deviation": float(deviation[0, 0]) * scale,
        "rms": float(rms) * scale,
        "sum": total * scale,
        "nonzero_count": cv2.countNonZero(samples),
        "minimum_location": list(min_point),
        "maximum_location": list(max_point),
        "percentiles": [],
    }
    if percentiles:
        quantiles = np.percentile(values, percentiles, method=method)
        summary["percentiles"] = [
            {"percentile": percentile, "value": float(quantile) * scale}
            for percentile, quantile in zip(percentiles, quantiles, strict=True)
        ]
    if not all(np.isfinite(summary[key]) for key in METRICS) or any(
        not np.isfinite(entry["value"]) for entry in summary["percentiles"]
    ):
        raise ValidationError("Image statistics overflowed. Reduce the sample range")
    return summary


class ImageSummaryStatistics(Block):
    type_id, title, category = "cv.image_statistics", "Image summary statistics", "Image processing"
    block_type, subtype = "Image processing", "Statistics"
    inputs = SELECTION_INPUTS
    outputs = (
        Port("summary", "data"),
        Port("selected_count", "number"),
        Port("value", "number", required=False),
        Port("minimum_location", "data", required=False),
        Port("maximum_location", "data", required=False),
    )
    defaults = {
        "channel": "auto",
        "statistic": "mean",
        "percentiles": [],
        "value_percentile": 50.0,
        "percentile_method": "linear",
        "include_alpha": False,
    }
    parameter_choices = {
        "channel": ("auto", *CHANNEL_CHOICES),
        "statistic": (*METRICS, "percentile"),
        "percentile_method": ("linear", "lower", "higher", "midpoint", "nearest"),
    }
    parameter_ranges = {"value_percentile": PERCENTILE_RANGE}
    parameter_help = {
        "channel": "Channel used for scalar and location outputs. Auto uses the first channel.",
        "statistic": "Statistic emitted on Value. Summary contains every computed statistic.",
        "percentiles": "Optional list of up to 16 percentile levels between 0 and 100.",
        "value_percentile": "Percentile level sent to Value when the statistic is Percentile.",
        "percentile_method": "How to choose or interpolate between adjacent ordered samples.",
        "include_alpha": "Include the alpha channel in the summary for four-channel input.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        levels = p["percentiles"]
        if not isinstance(levels, list) or len(levels) > 16:
            raise ValidationError("Percentiles must be a list of up to 16 levels", "percentiles")
        if any(type(level) not in (int, float) for level in levels):
            raise ValidationError("Percentile levels must be numeric", "percentiles")
        for level in levels:
            PERCENTILE_RANGE.check(level, "Percentile", "percentiles")

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source)
        labels = HISTOGRAM_CHANNELS[image_format(source)]
        p = self.parameters
        if len(labels) == 4 and not p["include_alpha"]:
            labels = labels[:3]
        channel = labels[0] if p["channel"] == "auto" else p["channel"]
        if channel not in labels:
            raise ValidationError("Selected channel is unavailable in the image", "channel")
        statistic = p["statistic"]
        value_percentile = float(p["value_percentile"])
        percentiles = list(dict.fromkeys(p["percentiles"]))
        if statistic == "percentile" and value_percentile not in percentiles:
            percentiles.append(value_percentile)
        mask = selection_mask(inputs, source)
        selected = None if mask is None else mask != 0
        count = image.shape[0] * image.shape[1] if selected is None else int(selected.sum())
        planes = image[..., None] if image.ndim == 2 else image
        channels = [
            {
                "channel": label,
                **channel_summary(
                    planes[..., index], mask, selected, percentiles, p["percentile_method"]
                ),
            }
            for index, label in enumerate(labels)
        ]
        current = channels[labels.index(channel)]
        if statistic == "percentile":
            value = next(
                entry["value"]
                for entry in current["percentiles"]
                if entry["percentile"] == value_percentile
            )
        else:
            value = current[statistic]
        minimum, maximum = current["minimum_location"], current["maximum_location"]
        drawings = {}
        if self.generate_graphics and minimum is not None:
            drawings["minimum"] = GraphicDrawing(
                (graphic_object("marker", [minimum], (250, 150, 80)),)
            )
            drawings["maximum"] = GraphicDrawing(
                (graphic_object("marker", [maximum], (85, 85, 250)),)
            )
        return Result(
            {
                "summary": {
                    "channel": channel,
                    "image_size": [image.shape[1], image.shape[0]],
                    "selected_count": count,
                    "channels": channels,
                },
                "selected_count": count,
                "value": value if value is not None else NO_VALUE,
                "minimum_location": minimum if minimum is not None else NO_VALUE,
                "maximum_location": maximum if maximum is not None else NO_VALUE,
            },
            drawings=drawings,
            message=f"{count} selected pixels · channel {channel}",
        )
