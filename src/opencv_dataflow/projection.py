"""Row and column intensity profiles using native array reductions."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .histogram import CHANNEL_CHOICES, HISTOGRAM_CHANNELS
from .images import image_format
from .parameter_validation import ValidationError
from .payloads import NO_VALUE
from .processing import image_value
from .selection import SELECTION_INPUTS, selection_mask

REDUCTIONS = {
    "sum": cv2.REDUCE_SUM,
    "mean": cv2.REDUCE_AVG,
    "minimum": cv2.REDUCE_MIN,
    "maximum": cv2.REDUCE_MAX,
    "sum_of_squares": cv2.REDUCE_SUM2,
}
CHANNEL_COLORS = {"B": (245, 155, 75), "G": (80, 220, 90), "R": (90, 90, 245)}
FALLBACK_COLORS = ((220, 220, 220), (210, 180, 85), (185, 95, 230), (90, 220, 225))


def profile_plot(values, valid, labels, axis, statistic):
    graphic = np.full((300, 640, 3), 25, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    neutral = (175, 175, 175)
    title = f"{statistic.replace('_', ' ').capitalize()} by {axis}"
    cv2.putText(graphic, title, (18, 23), font, 0.55, neutral, 1, cv2.LINE_AA)
    if not valid.any():
        cv2.putText(graphic, "No selected pixels", (18, 55), font, 0.55, neutral, 1, cv2.LINE_AA)
        return graphic

    left, right, top, bottom = 88, 615, 42, 248
    cv2.line(graphic, (left, top), (left, bottom), neutral)
    cv2.line(graphic, (left, bottom), (right, bottom), neutral)
    indices = np.flatnonzero(valid)
    x = left + indices * (right - left) / max(len(values) - 1, 1)
    observed = values[valid]
    low, high = float(observed.min()), float(observed.max())
    scale = max(abs(low), abs(high)) or 1.0
    if low == high:
        y = np.full(observed.shape, (top + bottom) / 2)
    else:
        y = bottom - (observed / scale - low / scale) / (high / scale - low / scale) * (
            bottom - top
        )
    # Break a trace when a whole row or column was outside the selection.
    splits = np.flatnonzero(np.diff(indices) != 1) + 1
    for channel, label in enumerate(labels):
        color = CHANNEL_COLORS.get(label, FALLBACK_COLORS[channel])
        points = np.rint(np.column_stack((x, y[:, channel]))).astype(np.int32)
        for segment in np.split(points, splits):
            if len(segment) == 1:
                cv2.circle(graphic, tuple(segment[0]), 2, color, -1)
            else:
                cv2.polylines(graphic, [segment], False, color, 1, cv2.LINE_AA)
        cv2.putText(graphic, label, (left + channel * 105, 289), font, 0.48, color, 1)
    cv2.putText(graphic, f"{high:.4g}", (8, top + 4), font, 0.42, neutral, 1)
    cv2.putText(graphic, f"{low:.4g}", (8, bottom), font, 0.42, neutral, 1)
    cv2.putText(graphic, "0", (left, 268), font, 0.42, neutral, 1)
    last = str(len(values) - 1)
    cv2.putText(graphic, last, (right - 8 * len(last), 268), font, 0.42, neutral, 1)
    return graphic


class IntensityProfile(Block):
    type_id, title, category = "cv.intensity_profile", "Intensity profile", "Image processing"
    block_type, subtype = "Image processing", "Statistics"
    inputs = SELECTION_INPUTS
    outputs = (
        Port("profile", "data"),
        Port("selected_count", "number"),
        Port("points", "data", required=False),
    )
    defaults = {
        "axis": "columns",
        "statistic": "mean",
        "include_alpha": False,
        "extrema_tie": "first",
        "points_channel": "auto",
    }
    parameter_choices = {
        "axis": ("rows", "columns"),
        "statistic": tuple(REDUCTIONS),
        "extrema_tie": ("first", "last"),
        "points_channel": ("auto", *CHANNEL_CHOICES),
    }
    parameter_help = {
        "axis": "Each output entry summarizes one row or one column in the input pixel frame.",
        "statistic": "Reduce selected samples in raw image units, separately for each channel.",
        "include_alpha": "Include alpha as its own profile on four-channel images.",
        "extrema_tie": "For minimum or maximum, choose the first or last matching pixel.",
        "points_channel": "Channel for Points in minimum or maximum mode. Auto uses the first.",
    }

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source)
        labels = HISTOGRAM_CHANNELS[image_format(source)]
        p = self.parameters
        if len(labels) == 4 and not p["include_alpha"]:
            image = image[..., :3]
            labels = labels[:3]
        dim = 0 if p["axis"] == "columns" else 1
        statistic = p["statistic"]
        mask = selection_mask(inputs, source)
        signal = image.astype(np.float64, copy=False)
        reduction = REDUCTIONS[statistic]
        if mask is None:
            count = np.full(image.shape[1 - dim], image.shape[dim], dtype=np.int64)
        else:
            selected = mask != 0
            count = np.count_nonzero(selected, axis=dim)
            fill = {"minimum": np.inf, "maximum": -np.inf}.get(statistic, 0.0)
            signal = np.where(selected if image.ndim == 2 else selected[..., None], signal, fill)
            if statistic == "mean":
                reduction = cv2.REDUCE_SUM
        values = cv2.reduce(signal, dim, reduction, dtype=cv2.CV_64F).reshape(-1, len(labels))
        if mask is not None and statistic == "mean":
            values /= np.maximum(count, 1)[:, None]
        valid = count > 0
        if not np.isfinite(values[valid]).all():
            raise ValidationError("Intensity profile overflowed. Reduce the sample range")
        profile = {
            "axis": p["axis"],
            "statistic": statistic,
            "image_size": [source.shape[1], source.shape[0]],
            "sample_counts": count.tolist(),
            "channels": [
                {
                    "channel": label,
                    "values": [
                        value if keep else None
                        for value, keep in zip(values[:, channel].tolist(), valid, strict=True)
                    ],
                }
                for channel, label in enumerate(labels)
            ],
        }
        points = NO_VALUE
        if statistic in ("minimum", "maximum"):
            requested = labels[0] if p["points_channel"] == "auto" else p["points_channel"]
            if requested not in labels:
                raise ValidationError(
                    "Points channel is unavailable in the image", "points_channel"
                )
            arg_reduce = cv2.reduceArgMin if statistic == "minimum" else cv2.reduceArgMax
            planes = signal[..., None] if signal.ndim == 2 else signal
            last = p["extrema_tie"] == "last"
            for channel, record in enumerate(profile["channels"]):
                indices = arg_reduce(planes[..., channel], dim, lastIndex=last).ravel().tolist()
                locations = [
                    ([i, index] if dim == 0 else [index, i]) if keep else None
                    for i, (index, keep) in enumerate(zip(indices, valid, strict=True))
                ]
                record["locations"] = locations
                if record["channel"] == requested:
                    points = [point for point in locations if point is not None]
        graphics = {}
        if self.generate_graphics:
            graphics["profile"] = profile_plot(values, valid, labels, p["axis"], statistic)
        return Result(
            {"profile": profile, "selected_count": int(count.sum()), "points": points},
            graphics=graphics,
        )
