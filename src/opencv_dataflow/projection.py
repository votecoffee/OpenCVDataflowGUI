"""Row and column intensity profiles using native array reductions."""

import math

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .histogram import CHANNEL_CHOICES, HISTOGRAM_CHANNELS
from .images import image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, ValuePacket, require_same_frame
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


def profile_plot(values, valid, labels, axis, statistic, *, distances=None):
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
    last = str(len(values) - 1) if distances is None else f"{distances[-1]:.4g} px"
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


_PATH_INTERPOLATION = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR}


def read_path(value, closed, *fields):
    try:
        points = np.asarray(value, np.float64)
    except (TypeError, ValueError, OverflowError):
        raise ValidationError("Path requires finite [x, y] vertices", *fields) from None
    minimum = 3 if closed else 2
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or not minimum <= len(points) <= 4096
        or not np.isfinite(points).all()
        or np.max(np.abs(points)) > 1_000_000
        or len(np.unique(points, axis=0)) < minimum
    ):
        raise ValidationError(
            f"Path requires {minimum} or more distinct [x, y] vertices and at most 4096 total",
            *fields,
        )
    return points


class PathIntensityProfile(Block):
    type_id, title = "cv.path_intensity_profile", "Path intensity profile"
    category, block_type, subtype = "Image processing", "Image processing", "Statistics"
    inputs = (*SELECTION_INPUTS, Port("path", "data", required=False))
    outputs = (
        Port("profile", "data"),
        Port("points", "data"),
        Port("path_length", "number"),
        Port("selected_count", "number"),
    )
    defaults = {
        "vertices": [[8, 8], [48, 8], [48, 32]],
        "closed": False,
        "sample_spacing": 1.0,
        "interpolation": "linear",
        "include_alpha": False,
    }
    parameter_choices = {"interpolation": tuple(_PATH_INTERPOLATION)}
    parameter_ranges = {"sample_spacing": NumericRange(0.25)}
    parameter_help = {
        "vertices": "Ordered [x, y] pixel vertices. A connected path overrides these vertices.",
        "closed": "Sample the closing segment. The first sample is emitted once.",
        "sample_spacing": "Maximum arc-length spacing between samples, in pixels.",
        "include_alpha": "Include alpha as its own profile on four-channel images.",
    }
    input_help = {"path": "Ordered [x, y] point list in the image's pixel coordinate frame."}

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        read_path(p["vertices"], p["closed"], "vertices")

    def process(self, inputs):
        source = inputs["image"]
        height, width = source.shape[:2]
        if max(height, width) >= 32767:
            raise ValidationError("Path remapping requires image axes below 32767 pixels")
        image = image_value(source)
        payload = self.input_payloads["image"]
        path_payload = self.input_payloads.get("path", {})
        require_same_frame(
            (ValuePacket(None, payload), ValuePacket(None, path_payload)),
            message="Path vertices must share the image coordinate frame",
        )
        p = self.parameters
        vertices = read_path(inputs.get("path", p["vertices"]), p["closed"])
        chain = np.vstack((vertices, vertices[0])) if p["closed"] else vertices
        deltas = np.diff(chain, axis=0)
        lengths = np.hypot(deltas[:, 0], deltas[:, 1])
        active = lengths > 0
        lengths = lengths[active]
        ends = lengths.cumsum()
        length = float(ends[-1])
        intervals = max(1, math.ceil(length / float(p["sample_spacing"])))
        count = intervals if p["closed"] else intervals + 1
        if count > 8192:
            raise ValidationError("Path exceeds 8192 samples. Increase Sample spacing.")
        distances = np.linspace(0, length, count, endpoint=not p["closed"])
        indices = np.minimum(np.searchsorted(ends, distances, side="right"), len(ends) - 1)
        fraction = (distances - (ends - lengths)[indices]) / lengths[indices]
        points = chain[:-1][active][indices] + deltas[active][indices] * fraction[:, None]

        points[0] = vertices[0]
        if not p["closed"]:
            points[-1] = vertices[-1]
        valid = np.all((points >= 0) & (points <= [width - 1, height - 1]), axis=1)
        maps = np.where(valid[:, None], points, 0).astype(np.float32)[None, ...]
        method = _PATH_INTERPOLATION[p["interpolation"]]
        mask = selection_mask(inputs, source)
        if mask is not None:
            selected = cv2.remap(mask.astype(np.float32) / 255, maps, None, method)
            valid &= selected.ravel() == 1
        labels = HISTOGRAM_CHANNELS[image_format(source)]
        if len(labels) == 4 and not p["include_alpha"]:
            image, labels = image[..., :3], labels[:3]
        if method == cv2.INTER_LINEAR and image.dtype.kind in "ui":
            image = image.astype(np.float32)
        values = cv2.remap(image, maps, None, method).reshape(count, len(labels))
        if not np.isfinite(values[valid]).all():
            raise ValidationError("Path sampling overflowed. Reduce the image sample range.")
        selected_count = int(np.count_nonzero(valid))
        locations = points.tolist()
        outputs = {
            "profile": {
                "closed": p["closed"],
                "path_length": length,
                "distances": distances.tolist(),
                "points": locations,
                "channels": [
                    {
                        "channel": name,
                        "values": [
                            value if keep else None
                            for value, keep in zip(values[:, index].tolist(), valid, strict=True)
                        ],
                    }
                    for index, name in enumerate(labels)
                ],
            },
            "points": locations,
            "path_length": length,
            "selected_count": selected_count,
        }
        graphics, drawings = {}, {}
        if self.generate_graphics:
            graphics["profile"] = profile_plot(
                values, valid, labels, "path distance", "intensity", distances=distances
            )
            outline = graphic_object("polyline", vertices, (0, 210, 255), closed=p["closed"])
            drawings["sampling path"] = GraphicDrawing((outline,))
        return Result(
            outputs,
            payloads=dict.fromkeys(outputs, payload),
            graphics=graphics,
            drawings=drawings,
            message=f"{selected_count} of {count} path samples selected",
        )
