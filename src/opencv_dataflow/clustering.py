"""OpenCV K-means for numeric samples and image pixels."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, frame, require_same_frame
from .processing import image_value
from .selection import SELECTION_INPUTS, selection_mask

INITIALIZATION = {
    "kmeans_plus_plus": cv2.KMEANS_PP_CENTERS,
    "random": cv2.KMEANS_RANDOM_CENTERS,
}
TERMINATION = {
    "both": cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS,
    "iterations": cv2.TERM_CRITERIA_MAX_ITER,
    "epsilon": cv2.TERM_CRITERIA_EPS,
}
MAX_CLUSTERS = 32
MAX_IMAGE_PIXELS = 1_000_000


def cluster_colors(count):
    hsv = np.zeros((1, count, 3), dtype=np.uint8)
    hsv[0, :, 0] = np.arange(count) * 67 % 180
    hsv[0, :, 1:] = (200, 238)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0]


class KMeans(Block):
    defaults = {
        "clusters": 3,
        "initialization": "kmeans_plus_plus",
        "attempts": 3,
        "termination": "both",
        "max_iterations": 50,
        "epsilon": 0.01,
    }
    parameter_choices = {
        "initialization": tuple(INITIALIZATION),
        "termination": tuple(TERMINATION),
    }
    parameter_ranges = {
        "clusters": NumericRange(1, MAX_CLUSTERS, whole=True),
        "attempts": NumericRange(1, 20, whole=True),
        "max_iterations": NumericRange(2, 100, whole=True),
        "epsilon": NumericRange(0),
    }
    parameter_help = {
        "clusters": "Must be no greater than the number of samples used in this run.",
        "attempts": "Returns the fit with the lowest sum of squared distances.",
        "max_iterations": "Used by the both and iterations modes.",
        "epsilon": "Center movement tolerance in feature units, used by both and epsilon modes.",
    }

    def fit(self, samples, initial=None):
        p = self.parameters
        count = int(p["clusters"])
        if len(samples) < count:
            raise ValidationError(f"Need at least {count} samples for {count} clusters")
        if not np.isfinite(samples).all() or samples.min() < -1e15 or samples.max() > 1e15:
            raise ValidationError("K-means samples must be finite and within ±1e15")
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        flags = INITIALIZATION[p["initialization"]]
        if initial is not None:
            if (
                initial.shape != (len(samples),)
                or initial.dtype.kind not in "iu"
                or np.any(initial < 0)
                or np.any(initial >= count)
            ):
                raise ValidationError(
                    "Initial labels need one integer in [0, clusters) per fitted sample"
                )
            initial = np.ascontiguousarray(initial, dtype=np.int32).reshape(-1, 1)
            flags |= cv2.KMEANS_USE_INITIAL_LABELS
        if len(samples) == 1:
            # OpenCV interprets a one-row scalar matrix as one feature per column.
            return 0.0, np.zeros(1, np.int32), samples
        criteria = (TERMINATION[p["termination"]], int(p["max_iterations"]), float(p["epsilon"]))
        compactness, labels, centers = cv2.kmeans(
            samples, count, initial, criteria, int(p["attempts"]), flags
        )
        order = sorted(range(count), key=lambda index: tuple(centers[index]))
        ranks = np.empty(count, dtype=np.int32)
        ranks[order] = np.arange(count)
        return float(compactness), ranks[labels.ravel()], centers[order]


def sample_plot(samples, labels, centers, x_axis, y_axis):
    canvas = np.full((480, 720, 3), 249, np.uint8)
    palette = cluster_colors(len(centers))
    dimensions = [x_axis] if y_axis is None else [x_axis, y_axis]
    points = samples[:, dimensions].astype(np.float32).astype(np.float64)
    projected = np.asarray(centers[:, dimensions], dtype=np.float64)
    low, high = points.min(axis=0), points.max(axis=0)
    scale = np.where(high > low, high - low, 1)
    points = (points - low) / scale
    projected = (projected - low) / scale
    np.clip(projected, 0, 1, out=projected)
    points[:, high == low] = 0.5
    projected[:, high == low] = 0.5

    def pixels(values):
        y = np.full(len(values), 207) if y_axis is None else 382 - values[:, 1] * 350
        return np.rint(np.column_stack((66 + values[:, 0] * 618, y))).astype(int)

    cv2.rectangle(canvas, (52, 20), (698, 394), (165, 165, 165), 1)
    indexes = np.linspace(0, len(points) - 1, min(len(points), 8000), dtype=int)
    for point, label in zip(pixels(points[indexes]), labels[indexes], strict=True):
        cv2.circle(canvas, tuple(point), 3, tuple(map(int, palette[label])), -1)
    for index, point in enumerate(pixels(projected)):
        x, y = point
        cv2.drawMarker(canvas, (x, y), (25, 25, 25), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(
            canvas, str(index), (x - 18, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (25, 25, 25), 1
        )
    axes = f"X: feature {x_axis}  [{low[0]:.4g}, {high[0]:.4g}]"
    if y_axis is not None:
        axes += f"    Y: feature {y_axis}  [{low[1]:.4g}, {high[1]:.4g}]"
    cv2.putText(canvas, axes, (52, 423), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 1)
    cv2.putText(
        canvas,
        "Crosses mark centers. Numbers are cluster IDs.",
        (52, 451),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (65, 65, 65),
        1,
    )
    return canvas


class KMeansSamples(KMeans):
    type_id, title = "ml.kmeans", "K-means samples"
    category, block_type, subtype = "ML", "ML", "Clustering"
    inputs = (
        Port("samples", "data", required=False),
        Port("initial_labels", "data", required=False),
    )
    outputs = (
        Port("labels", "data"),
        Port("centers", "data"),
        Port("counts", "data"),
        Port("compactness", "number"),
        Port("plot", required=False),
    )
    defaults = {**KMeans.defaults, "samples": [], "plot_x": 0, "plot_y": -1, "draw_plot": True}
    parameter_ranges = {
        **KMeans.parameter_ranges,
        "plot_x": NumericRange(0, 127, whole=True),
        "plot_y": NumericRange(-1, 127, whole=True),
    }
    parameter_help = {
        **KMeans.parameter_help,
        "samples": "One feature vector per row. The samples input overrides this table.",
        "plot_x": "Zero-based feature index for the plot's horizontal axis.",
        "plot_y": "-1 chooses the first other feature or a one-dimensional strip.",
    }
    input_help = {"initial_labels": "Optional list of zero-based labels for the first attempt."}

    def process(self, inputs):
        p = self.parameters
        try:
            samples = np.asarray(inputs.get("samples", p["samples"]))
        except (TypeError, ValueError) as exc:
            raise ValidationError("Samples must be a rectangular numeric table") from exc
        if (
            samples.ndim != 2
            or not 1 <= samples.shape[0] <= 100_000
            or not 1 <= samples.shape[1] <= 128
            or samples.size > 2_000_000
            or samples.dtype.kind not in "iuf"
        ):
            raise ValidationError(
                "Use a numeric table with 1–100,000 rows, 1–128 features and at most 2M values"
            )
        try:
            initial = np.asarray(inputs["initial_labels"]) if "initial_labels" in inputs else None
        except (TypeError, ValueError) as exc:
            raise ValidationError("Initial labels must be a one-dimensional integer list") from exc
        x_axis, y_axis = int(p["plot_x"]), int(p["plot_y"])
        dimensions = samples.shape[1]
        if p["draw_plot"] and (x_axis >= dimensions or y_axis >= dimensions or x_axis == y_axis):
            raise ValidationError("Plot axes must be distinct, available feature indices")
        compactness, labels, centers = self.fit(samples, initial)
        if y_axis < 0:
            y_axis = (1 if x_axis == 0 else 0) if dimensions > 1 else None
        plot = sample_plot(samples, labels, centers, x_axis, y_axis) if p["draw_plot"] else NO_VALUE
        payloads = {}
        if plot is not NO_VALUE:
            payloads["plot"] = {"spatial": frame([plot.shape[1], plot.shape[0]], f"{self.id}:plot")}
        return Result(
            {
                "labels": labels.tolist(),
                "centers": centers.tolist(),
                "counts": np.bincount(labels, minlength=len(centers)).tolist(),
                "compactness": compactness,
                "plot": plot,
            },
            payloads=payloads,
            message=f"{len(samples)} samples grouped into {len(centers)} clusters",
        )


class KMeansImage(KMeans):
    type_id, title = "inspect.kmeans_image", "K-means image pixels"
    category, block_type, subtype = "Inspection", "Inspection", "Segmentation"
    inputs = (*SELECTION_INPUTS, Port("initial_labels", required=False))
    outputs = (
        Port("labels"),
        Port("cluster_map"),
        Port("quantized"),
        Port("centers", "data"),
        Port("counts", "data"),
        Port("compactness", "number"),
    )
    input_help = {
        "initial_labels": "Optional single-channel integer label image for the first attempt."
    }

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source, (np.uint8, np.uint16, np.float32, np.float64))
        if image.shape[0] * image.shape[1] > MAX_IMAGE_PIXELS:
            raise ValidationError(
                "K-means image is limited to 1 million pixels; use a smaller level"
            )
        mask = selection_mask(inputs, image)
        selected = np.ones(image.shape[:2], dtype=bool) if mask is None else mask != 0
        channels = image[..., None] if image.ndim == 2 else image[..., :3]
        initial = inputs.get("initial_labels")
        if initial is not None:
            if initial.shape != selected.shape:
                raise ValidationError("Initial labels must match the image dimensions")
            require_same_frame([source, initial])
            initial = initial[selected]
        compactness, labels, centers = self.fit(channels[selected], initial)
        label_image = np.full(selected.shape, -1, dtype=np.int16)
        label_image[selected] = labels
        cluster_map = np.zeros((*selected.shape, 3), dtype=np.uint8)
        cluster_map[selected] = cluster_colors(len(centers))[labels]
        quantized = image.copy()
        values = centers[labels]
        if image.dtype.kind == "u":
            values = np.clip(np.rint(values), 0, np.iinfo(image.dtype).max)
        if image.ndim == 2:
            quantized[selected] = values[:, 0]
        else:
            quantized[..., :3][selected] = values
        return Result(
            {
                "labels": label_image,
                "cluster_map": cluster_map,
                "quantized": ImageArray(quantized, image_format(source)),
                "centers": centers.tolist(),
                "counts": np.bincount(labels, minlength=len(centers)).tolist(),
                "compactness": compactness,
            },
            message=f"{len(labels)} pixels grouped into {len(centers)} clusters",
        )
