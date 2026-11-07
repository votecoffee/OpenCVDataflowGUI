"""Per-channel numeric histograms with a result graphic."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .images import image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .selection import SELECTION_INPUTS, selection_mask

HISTOGRAM_CHANNELS = {
    "gray": ("Gray",),
    "bgr": ("B", "G", "R"),
    "rgb": ("R", "G", "B"),
    "bgra": ("B", "G", "R", "A"),
    "rgba": ("R", "G", "B", "A"),
    "hsv": ("H", "S", "V"),
    "lab": ("L", "a", "b"),
}
CHANNEL_CHOICES = tuple(
    dict.fromkeys(label for labels in HISTOGRAM_CHANNELS.values() for label in labels)
)
THRESHOLD_STATISTICS = ("lower_threshold", "upper_threshold", "mean", "median")
STATISTIC_COLORS = {
    "lower_threshold": (55, 195, 248),
    "upper_threshold": (55, 195, 248),
    "mean": (125, 235, 110),
    "median": (245, 130, 225),
}


def channel_statistics(samples, lower, upper):
    names = ("minimum", "maximum", "standard_deviation", *THRESHOLD_STATISTICS)
    if not samples.size:
        return dict.fromkeys(names)
    minimum, maximum = float(samples.min()), float(samples.max())
    scale = max(abs(minimum), abs(maximum))
    values = samples if samples.dtype.kind == "u" else samples.astype(np.float64, copy=False)
    if 0 < scale < 1e-150 or scale > 1e150:
        values = values / scale
    else:
        scale = 1.0
    mean = float(np.mean(values, dtype=np.float64)) * scale
    deviation = float(np.std(values, dtype=np.float64)) * scale
    low, median, high = (
        float(value) * scale
        for value in np.percentile(
            values, (lower, 50, upper), method="linear", overwrite_input=True
        )
    )
    return {
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean,
        "median": median,
        "standard_deviation": deviation,
        "lower_threshold": low,
        "upper_threshold": high,
    }


def _draw_statistics_table(graphic, channels, colors, lower, upper, header_y):
    columns = (
        ("Channel", None),
        ("Mean", "mean"),
        ("Median", "median"),
        ("Std dev", "standard_deviation"),
        ("Min", "minimum"),
        ("Max", "maximum"),
        (f"P{lower:.5g}", "lower_threshold"),
        (f"P{upper:.5g}", "upper_threshold"),
    )
    text_color = (212, 212, 212)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for index, (label, statistic) in enumerate(columns):
        x = 22 + index * 71
        cv2.putText(
            graphic, label, (x, header_y), font, 0.4, STATISTIC_COLORS.get(statistic, text_color), 1
        )
        for row, (channel, color) in enumerate(zip(channels, colors, strict=True), start=1):
            y = header_y + row * 17
            if statistic is None:
                cv2.putText(graphic, channel["channel"], (x, y), font, 0.4, color, 1)
            else:
                value = channel["statistics"][statistic]
                text = "n/a" if value is None else f"{value:.4g}"
                if len(text) > 10:
                    text = f"{value:.3g}"
                cv2.putText(graphic, text, (x, y), font, 0.4, text_color, 1)


def histogram_plot(channels, selected, lower, upper, show_statistics):
    graphic = np.full((300, 600, 3), 25, np.uint8)
    primary_colors = {"B": (240, 150, 70), "G": (90, 220, 90), "R": (80, 80, 245)}
    fallback = ((200, 200, 200), (220, 180, 90), (180, 90, 220))
    colors = [
        primary_colors.get(channel["channel"], fallback[index % len(fallback)])
        for index, channel in enumerate(channels)
    ]
    edges = selected["edges"]
    statistics = selected["statistics"]
    left, right = edges[0], edges[-1]
    header_y = 282 - 17 * len(channels)
    axis_y = header_y - 13 if show_statistics else 260
    font = cv2.FONT_HERSHEY_SIMPLEX

    def position(value):
        return round(36 + 544 * ((value - left) / (right - left)))

    if show_statistics and statistics["mean"] is not None:
        low = max(left, statistics["lower_threshold"])
        high = min(right, statistics["upper_threshold"])
        if low <= high:
            graphic[46:axis_y + 1, position(low) : position(high) + 1] = (34, 37, 44)
    for index, (channel, color) in enumerate(zip(channels, colors, strict=True)):
        counts = np.asarray(channel["counts"])
        x = 36 + (np.arange(len(counts)) + 0.5) * (544 / len(counts))
        y = axis_y - counts.astype(float) / max(int(counts.max()), 1) * (axis_y - 55)
        points = np.column_stack((x, y)).astype(np.int32)
        cv2.polylines(graphic, [points], False, color, 1)
        cv2.putText(graphic, channel["channel"], (40 + index * 110, 24), font, 0.5, color, 1)
    cv2.line(graphic, (36, axis_y), (580, axis_y), (150, 150, 150), 1)
    if show_statistics:
        for name, color in STATISTIC_COLORS.items():
            value = statistics[name]
            if value is not None and left <= value <= right:
                x = position(value)
                cv2.line(graphic, (x, 46), (x, axis_y - 1), color, 1)
        cv2.putText(
            graphic, f"Markers: {selected['channel']}", (450, 24),
            font, 0.4, (212, 212, 212), 1,
        )
        _draw_statistics_table(graphic, channels, colors, lower, upper, header_y)
    return graphic


class Histogram(Block):
    type_id, title, category = "inspect.histogram", "Histogram", "Image processing"
    block_type, subtype = "Image processing", "Statistics"
    inputs = SELECTION_INPUTS
    outputs = (
        Port("histogram", "data"),
        Port("plot"),
        Port("threshold", "number", required=False),
        *(Port(name, "number", required=False) for name in THRESHOLD_STATISTICS),
    )
    defaults = {
        "bins": 256,
        "range_mode": "automatic",
        "range_min": 0.0,
        "range_max": 1.0,
        "include_alpha": False,
        "lower_percentile": 5.0,
        "upper_percentile": 95.0,
        "statistics_channel": "auto",
        "threshold_statistic": "median",
        "show_statistics": True,
    }
    parameter_choices = {
        "range_mode": ("automatic", "manual"),
        "statistics_channel": ("auto", *CHANNEL_CHOICES),
        "threshold_statistic": THRESHOLD_STATISTICS,
    }
    parameter_ranges = {
        "bins": NumericRange(2, 4096, whole=True),
        "lower_percentile": NumericRange(0, 100),
        "upper_percentile": NumericRange(0, 100),
    }
    parameter_help = {
        "lower_percentile": "Lower-tail cutoff. Pair with upper percentile for a middle range.",
        "upper_percentile": "Cutoff for the upper tail, expressed as a percentile from the bottom.",
        "statistics_channel": "Channel for scalar outputs and plot markers. Auto uses the first.",
        "threshold_statistic": "Value sent on the threshold output.",
        "show_statistics": "Show all channel statistics and mark the selected channel on the plot.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = ["histogram", "plot"]

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if float(p["lower_percentile"]) > float(p["upper_percentile"]):
            raise ValidationError(
                "Lower percentile must not exceed upper percentile",
                "lower_percentile",
                "upper_percentile",
            )
        if p["range_mode"] == "manual":
            low = NumericRange().check(p["range_min"], "Range minimum", "range_min")
            high = NumericRange().check(p["range_max"], "Range maximum", "range_max")
            if high <= low:
                raise ValidationError(
                    "Histogram range must have max > min", "range_min", "range_max"
                )

    def process(self, inputs):
        image = inputs["image"]
        encoding = image_format(image)
        bins = int(self.parameters["bins"])
        if image.ndim == 3 and image.shape[2] not in (3, 4):
            raise ValidationError("Histogram requires grayscale, three or four channels")
        labels = HISTOGRAM_CHANNELS[encoding]
        planes = [image] if image.ndim == 2 else [image[..., i] for i in range(image.shape[2])]
        if len(planes) == 4 and not self.parameters["include_alpha"]:
            planes, labels = planes[:3], labels[:3]
        selected_label = self.parameters["statistics_channel"]
        if selected_label == "auto":
            selected_label = labels[0]
        if selected_label not in labels:
            raise ValidationError(
                f"Statistics channel {selected_label} is unavailable in this image"
            )
        lower, upper = (
            float(self.parameters[name]) for name in ("lower_percentile", "upper_percentile")
        )
        channels = []
        selection = selection_mask(inputs, image)
        for label, plane in zip(labels, planes, strict=True):
            valid = np.isfinite(plane)
            if selection is not None:
                valid &= selection != 0
            finite = np.asarray(plane)[valid]
            mode = self.parameters["range_mode"]
            if mode == "manual":
                low, high = float(self.parameters["range_min"]), float(self.parameters["range_max"])
            elif mode == "automatic":
                if np.issubdtype(image.dtype, np.unsignedinteger):
                    low, high = 0.0, float(np.iinfo(image.dtype).max) + 1
                    if encoding == "hsv" and label == "H":
                        high = 180.0
                elif finite.size:
                    low, high = float(finite.min()), float(finite.max())
                    if low == high:
                        high = low + max(1.0, abs(low) * 0.01)
                else:
                    low, high = 0.0, 1.0
            else:
                raise ValueError("Range mode must be automatic or manual")
            if not np.isfinite([low, high]).all() or high <= low:
                raise ValidationError("Histogram range must be finite with max > min")
            counts, edges = np.histogram(finite, bins=bins, range=(low, high))
            channels.append(
                {
                    "channel": label,
                    "counts": counts.tolist(),
                    "edges": edges.tolist(),
                    "finite_samples": int(finite.size),
                    "outside_range": int(finite.size - counts.sum()),
                    "statistics": channel_statistics(finite, lower, upper),
                }
            )
        selected = channels[labels.index(selected_label)]
        graphic = histogram_plot(
            channels, selected, lower, upper, self.parameters["show_statistics"]
        )
        statistics = selected["statistics"]
        scalars = {
            name: statistics[name] if statistics[name] is not None else NO_VALUE
            for name in THRESHOLD_STATISTICS
        }
        return Result(
            {
                "histogram": {
                    "color_format": encoding,
                    "bins": bins,
                    "channels": channels,
                    "lower_percentile": lower,
                    "upper_percentile": upper,
                    "statistics_channel": selected_label,
                },
                "plot": graphic,
                "threshold": scalars[self.parameters["threshold_statistic"]],
                **scalars,
            },
            graphics={"histogram": graphic},
            payloads={"plot": {}},
            message=f"{len(channels)} channels · {bins} bins",
        )
