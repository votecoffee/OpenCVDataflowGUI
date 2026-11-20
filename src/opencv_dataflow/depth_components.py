"""Label foreground surfaces and measure their visible camera-frame footprint."""

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .collection_tools import collection_result
from .core import Port
from .geometry3d_common import Tool3D
from .graphics_layers import GraphicDrawing, graphic_box, graphic_object
from .images import label_color_map
from .label_measurement import MAX_LABEL_PIXELS, attach_label_source, measure_regions
from .objects import collection
from .parameter_validation import NumericRange, ValidationError
from .ptcloud3d import depth_intrinsics
from .result_graphics import ACCEPTED
from .selection import selection_mask
from .stereo3d import metric_depth


def _depth_connected_labels(values, candidate, connectivity, step):
    nodes = np.full(values.shape, -1, np.int32)
    pixel_count = int(np.count_nonzero(candidate))
    nodes[candidate] = np.arange(pixel_count, dtype=np.int32)
    selectors = [
        (np.s_[:, :-1], np.s_[:, 1:]),
        (np.s_[:-1, :], np.s_[1:, :]),
    ]
    if connectivity == 8:
        selectors.extend(
            [
                (np.s_[:-1, :-1], np.s_[1:, 1:]),
                (np.s_[:-1, 1:], np.s_[1:, :-1]),
            ]
        )
    rows, columns = [], []
    for a, b in selectors:
        # Each permitted undirected edge is stored once, including diagonal edges.
        paired = candidate[a] & candidate[b]
        paired &= np.abs(values[a] - values[b]) <= step
        rows.append(nodes[a][paired])
        columns.append(nodes[b][paired])
    starts, ends = np.concatenate(rows), np.concatenate(columns)
    edges = csr_matrix(
        (np.ones(len(starts), np.uint8), (starts, ends)), shape=(pixel_count, pixel_count)
    )
    count, components = connected_components(edges, directed=False)
    labels = np.zeros(values.shape, np.int32)
    labels[candidate] = components + 1
    return count + 1, labels


def _label_foreground(values, candidate, connectivity, step):
    if not candidate.any():
        return 1, np.zeros(values.shape, np.int32)
    if step:
        return _depth_connected_labels(values, candidate, connectivity, step)
    return cv2.connectedComponentsWithAlgorithm(
        candidate.astype(np.uint8), connectivity, cv2.CV_32S, cv2.CCL_SAUF
    )


def _retained_labels(labels, candidate, count, p):
    areas = np.bincount(labels[candidate], minlength=count)
    keep = areas >= int(p["minimum_pixels"])
    maximum = int(p["maximum_pixels"])
    if maximum:
        keep &= areas <= maximum
    ordered = np.flatnonzero(keep)
    if len(ordered) > int(p["maximum_components"]):
        raise ValidationError(
            "Too many components to measure. Tighten the selection or size range.",
            "maximum_components",
        )
    first = np.full(count, labels.size, np.int64)
    np.minimum.at(first, labels[candidate], np.flatnonzero(candidate))
    ordered = ordered[np.argsort(first[ordered])]
    mapping = np.zeros(count, np.int32)
    mapping[ordered] = np.arange(1, len(ordered) + 1, dtype=np.int32)
    return mapping[labels]


def _component_geometry(labels, depth, k, rows):
    """Accumulate visible sample-center and projected pixel-footprint measurements."""
    if not rows:
        return
    indices = np.flatnonzero(labels)
    y, x = np.divmod(indices, labels.shape[1])
    ids = labels.flat[indices]
    z = depth.flat[indices].astype(np.float64)
    count = len(rows) + 1
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        dx, dy = z / fx, z / fy
        px, py = (x - cx) * dx, (y - cy) * dy
        area = np.bincount(ids, weights=dx * dy, minlength=count)
        axes = ((px, dx / 2), (py, dy / 2), (z, 0))
        low = np.full((count, 3), np.inf)
        high = np.full((count, 3), -np.inf)
        summed = np.empty((count, 3), np.float64)
        for i, (axis, half_width) in enumerate(axes):
            np.minimum.at(low[:, i], ids, axis - half_width)
            np.maximum.at(high[:, i], ids, axis + half_width)
            summed[:, i] = np.bincount(ids, weights=axis, minlength=count)
        extent = high[1:] - low[1:]
    if not all(np.isfinite(a).all() for a in (low[1:], high[1:], summed[1:], extent, area[1:])):
        raise ValidationError("Metric component dimensions exceed the numeric range")
    for row, dimensions in zip(rows, extent, strict=True):
        i, metrics = row["id"], row["metrics"]
        center = summed[i] / metrics["area"]
        metrics.update(
            width_m=float(dimensions[0]),
            height_m=float(dimensions[1]),
            depth_span_m=float(dimensions[2]),
            minimum_depth_m=float(low[i, 2]),
            maximum_depth_m=float(high[i, 2]),
            mean_depth_m=float(center[2]),
            projected_area_m2=float(area[i]),
        )
        row["geometry"].update(
            camera_bounds_min_m=low[i].tolist(),
            camera_bounds_max_m=high[i].tolist(),
            camera_centroid_m=center.tolist(),
        )


def _component_drawing(rows, limit):
    graphics = []
    for row in rows[:limit]:
        x, y, w, h = row["geometry"]["bounds"]
        width = 1000 * row["metrics"]["width_m"]
        height = 1000 * row["metrics"]["height_m"]
        label = f"{row['id']}  {width:.3g} x {height:.3g} mm"
        graphics.extend(
            (
                graphic_box(x, y, w, h, ACCEPTED),
                graphic_object("label", ((x, y),), ACCEPTED, text=label),
            )
        )
    return GraphicDrawing(tuple(graphics), source="depth")


class MeasureDepthComponents(Tool3D):
    type_id, title, subtype = "3d.depth_components", "Measure depth components", "Depth"
    inputs = (
        Port("depth", "depth"),
        Port("camera", "data"),
        Port("mask", required=False, visual_type="region", required_when_connected=True),
        Port("region", "data", required=False, visual_type="region", required_when_connected=True),
    )
    outputs = (
        Port("collection", "data"),
        Port("count", "integer"),
        Port("points", "data"),
        Port("labels"),
        Port("mask", visual_type="region"),
        Port("candidate_mask", visual_type="region"),
        Port("report", "data"),
    )
    defaults = {
        "minimum_depth_m": 0.001,
        "maximum_depth_m": 2.0,
        "maximum_depth_step_m": 0.02,
        "connectivity": "8",
        "minimum_pixels": 8,
        "maximum_pixels": 0,
        "maximum_components": 256,
        "preview_count": 25,
        "input_units": "m",
        "units_per_sample": 0.001,
    }
    parameter_choices = {"connectivity": ("4", "8"), "input_units": ("m", "mm")}
    parameter_ranges = {
        "minimum_depth_m": NumericRange(0, 100_000),
        "maximum_depth_m": NumericRange(1e-6, 100_000),
        "maximum_depth_step_m": NumericRange(0, 100_000),
        "minimum_pixels": NumericRange(1, MAX_LABEL_PIXELS, whole=True),
        "maximum_pixels": NumericRange(0, MAX_LABEL_PIXELS, whole=True),
        "maximum_components": NumericRange(1, 10_000, whole=True),
        "preview_count": NumericRange(0, 250, whole=True),
        "units_per_sample": NumericRange(1e-9, 1000),
    }
    parameter_help = {
        "minimum_depth_m": "Inclusive near cutoff for positive camera Z.",
        "maximum_depth_m": "Inclusive far cutoff for positive camera Z.",
        "maximum_depth_step_m": "Join neighbors within this Z difference. Zero uses spatial only.",
        "connectivity": "Four orthogonal neighbors or eight neighbors including diagonals.",
        "minimum_pixels": "Minimum retained pixels per component.",
        "maximum_pixels": "Maximum retained pixels per component. Zero disables this limit.",
        "maximum_components": "Stop measurement if more components pass the pixel-count limits.",
        "preview_count": "Maximum component boxes and labels to draw. Zero disables drawing.",
        "input_units": "Physical units for untagged depth.",
        "units_per_sample": "Physical units per raw uint16 sample.",
    }
    input_help = {
        "depth": "Positive camera Z. Use foreground depth or Mask for a tilted scene.",
        "camera": "Undistorted pinhole intrinsics for this depth grid.",
        "mask": "Optional uint8 selection, such as classified foreground from a support plane.",
        "region": "Optional source-grid selection intersected with Mask.",
    }
    payload_reference = "depth"
    parallel_safe = preview_safe = process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if float(p["minimum_depth_m"]) > float(p["maximum_depth_m"]):
            raise ValidationError(
                "Minimum depth must not exceed maximum depth", "minimum_depth_m", "maximum_depth_m"
            )
        lower, upper = int(p["minimum_pixels"]), int(p["maximum_pixels"])
        if upper and upper < lower:
            raise ValidationError(
                "Maximum pixels must be at least minimum pixels", "minimum_pixels", "maximum_pixels"
            )

    def process(self, inputs):
        p = self.parameters
        depth, model = inputs["depth"], inputs["camera"]
        if depth.size > MAX_LABEL_PIXELS or max(depth.shape) > 16384:
            raise ValidationError("Depth must fit the 2.1 million pixel and 16,384 axis limits")
        if model.get("units") not in ("m", "mm"):
            raise ValidationError("Camera must declare metric units")
        k, _ = depth_intrinsics(model, depth)
        z = metric_depth(depth, p["units_per_sample"], p["input_units"])
        selected = selection_mask(inputs, depth)
        candidate = (z >= float(p["minimum_depth_m"])) & (z <= float(p["maximum_depth_m"]))
        if selected is not None:
            candidate &= selected != 0
        count, initial = _label_foreground(
            z, candidate, int(p["connectivity"]), float(p["maximum_depth_step_m"])
        )
        labels = _retained_labels(initial, candidate, count, p)
        foreground = labels != 0
        items = measure_regions(labels, foreground, None)
        _component_geometry(labels, z, k, items)
        context = self.input_payloads["depth"]
        result_collection = collection("boxes", items, labels, context)
        result_collection["camera_frame"] = model.get("frame", "camera")
        attach_label_source(result_collection, labels)
        report = {
            "model": "depth_components_v1",
            "camera_frame": result_collection["camera_frame"],
            "units": "m",
            "candidate_pixels": int(np.count_nonzero(candidate)),
            "retained_pixels": int(np.count_nonzero(foreground)),
            "candidate_components": count - 1,
            "retained_components": len(items),
        }
        result = collection_result(
            result_collection,
            context,
            labels=labels,
            mask=foreground.astype(np.uint8) * 255,
            candidate_mask=candidate.astype(np.uint8) * 255,
            report=report,
        )
        result.message = f"{len(items)} depth components, {report['retained_pixels']} pixels"
        if not items:
            result.passed = False
            result.fail_reasons = ["depth_components_empty"]
        if self.generate_graphics:
            result.graphics["Depth components"] = label_color_map(labels)
            if items and p["preview_count"]:
                result.drawings["Measured components"] = _component_drawing(
                    items, int(p["preview_count"])
                )
        return result
