"""Move strip-space diagnostic outlines and anchors to the camera image."""

from dataclasses import replace

import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, layer_transform
from .parameter_validation import NumericRange, ValidationError
from .tickets import ticket_key

_MAX_SAMPLES = 131_072
_ANCHORS = ("label", "marker", "text")


def _dense_points(points, spacing, budget):
    if len(points) > budget:
        raise ValidationError("Strip graphics exceed the sample budget. Increase Sample spacing.")
    if len(points) == 1:
        return points
    deltas = np.diff(points, axis=0)
    counts = np.maximum(1, np.ceil(np.hypot(deltas[:, 0], deltas[:, 1]) / spacing)).astype(int)
    total = int(counts.sum())
    if total + 1 > budget:
        raise ValidationError("Strip graphics exceed the sample budget. Increase Sample spacing.")
    segments = np.repeat(np.arange(len(deltas)), counts)
    first = np.r_[0, counts.cumsum()[:-1]]
    fractions = (np.arange(total) - first[segments]) / counts[segments]
    return np.vstack((points[segments] + fractions[:, None] * deltas[segments], points[-1]))


def _sample_map(points, map_x, map_y, valid_map):
    height, width = map_x.shape
    inside = np.all((points >= 0) & (points <= [width - 1, height - 1]), axis=1)
    x, y = np.where(inside[:, None], points, 0).T
    left, top = np.floor(x).astype(int), np.floor(y).astype(int)
    right, bottom = np.minimum(left + 1, width - 1), np.minimum(top + 1, height - 1)
    fx, fy = x - left, y - top
    weights = np.column_stack(((1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy))
    rows = np.column_stack((top, top, bottom, bottom))
    cols = np.column_stack((left, right, left, right))
    valid = inside & np.all(valid_map[rows, cols] | (weights == 0), axis=1)
    mapped = np.column_stack((
        np.sum(map_x[rows, cols] * weights, axis=1),
        np.sum(map_y[rows, cols] * weights, axis=1),
    ))
    return mapped, valid


def _map_graphic(item, transform, map_x, map_y, valid_map, spacing, budget):
    if item.kind not in (*_ANCHORS, "polyline") or not item.points:
        return (), 0, False
    outline = item.kind == "polyline"
    points = item.points if outline else item.points[:1]
    if len(points) > budget:
        raise ValidationError("Strip graphics exceed the sample budget. Increase Sample spacing.")
    points = np.asarray(points, np.float64)
    projected = points @ transform[:, :2].T + transform[:, 2]
    divisor = projected[:, 2]
    if not (np.all(divisor > 1e-12) or np.all(divisor < -1e-12)):
        return (), 0, False
    coords = projected[:, :2] / divisor[:, None]
    if not np.isfinite(coords).all() or np.any(np.abs(coords) > 1_000_000):
        return (), 0, False
    loop = outline and item.closed and len(coords) > 1
    if loop:
        coords = np.vstack((coords, coords[0]))
    dense = _dense_points(coords, spacing, budget)
    mapped, valid = _sample_map(dense, map_x, map_y, valid_map)
    if not valid.any():
        return (), len(dense), False

    def replacement(samples, *, closed):
        return replace(item, points=tuple(map(tuple, samples.tolist())), closed=closed)

    if valid.all():
        samples = mapped[:-1] if loop else mapped
        return (replacement(samples, closed=item.closed),), len(dense), False

    # Do not join across an invalid part of the map or close clipped outlines.
    changes = np.diff(np.r_[0, valid, 0])
    starts, stops = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    parts = tuple(
        replacement(mapped[first:stop], closed=stop - first == 1)
        for first, stop in zip(starts, stops, strict=True)
    )
    return parts, len(dense), True


class MapStripGraphics(Block):
    type_id, title = "inspect.map_strip_graphics", "Map strip graphics to source"
    category, block_type, subtype = "Image processing", "Image processing", "Compositing"
    inputs = (
        Port("image"), Port("map_x"), Port("map_y"),
        Port("sampling", "data"), Port("graphics", "graphics"),
    )
    outputs = (Port("counts", "data"),)
    payload_reference = "image"
    process_inputs_readonly = True
    parallel_safe = True
    defaults = {"sample_spacing": 0.5}
    parameter_ranges = {"sample_spacing": NumericRange(0.25, 1)}
    parameter_help = {
        "sample_spacing": "Maximum spacing in strip pixels when projecting outline segments.",
    }
    input_help = {
        "image": "Original image used to unwrap the strip.",
        "map_x": "Source X coordinates from Unwrap path strip.",
        "map_y": "Source Y coordinates from the same strip.",
        "sampling": "Unwrap path strip sampling data for frame validation.",
        "graphics": "Strip result graphics. Maps outlines, labels, text and point markers.",
    }

    def process(self, inputs):
        image, sampling = inputs["image"], inputs["sampling"]
        source_frame = self.input_payloads["image"].get("spatial")
        if (
            not isinstance(sampling, dict) or sampling.get("model") != "path_strip_v1"
            or sampling.get("source_spatial") != source_frame
        ):
            raise ValidationError("Sampling data must describe the current source image frame")
        strip = sampling.get("strip_spatial")
        if not isinstance(strip, dict) or any(
            self.input_payloads[name].get("spatial") != strip for name in ("map_x", "map_y")
        ):
            raise ValidationError("Both coordinate maps must share the sampling strip frame")
        map_x, map_y = (inputs[name] for name in ("map_x", "map_y"))
        if (
            map_x.ndim != 2 or map_y.shape != map_x.shape
            or map_x.dtype != np.float32 or map_y.dtype != np.float32
            or not 0 < map_x.size <= 2_100_000
            or list(map_x.shape[::-1]) != strip.get("image_size")
            or not np.isfinite(map_x).all() or not np.isfinite(map_y).all()
        ):
            raise ValidationError("Use matching finite float32 coordinate maps on the strip grid")
        tickets = {
            ticket_key(payload["ticket"])
            for name, payload in self.input_payloads.items()
            if name != "graphics" and "ticket" in payload
        }
        if len(tickets) > 1:
            raise ValidationError("Source, coordinate maps and sampling must share one ticket")
        expected_ticket = next(iter(tickets), None)
        fallback_ticket = self.input_payloads["graphics"].get("ticket")
        height, width = image.shape[:2]
        valid_map = (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
        layers = inputs["graphics"].layers
        if len(layers) > 128 or sum(len(layer.objects) for layer in layers) > 4096:
            raise ValidationError("Strip graphics are limited to 128 layers and 4096 objects")
        spacing = float(self.parameters["sample_spacing"])
        budget = _MAX_SAMPLES
        drawings = {}
        counts = {"mapped": 0, "clipped": 0, "skipped": 0}
        for layer in layers:
            ticket = layer.payload.get("ticket", fallback_ticket)
            transform = layer_transform(layer, {"spatial": strip})
            if transform is None or (
                expected_ticket is not None and ticket is not None
                and ticket_key(ticket) != expected_ticket
            ):
                counts["skipped"] += len(layer.objects)
                continue
            objects = []
            for item in layer.objects:
                parts, used, clipped = _map_graphic(
                    item, transform, map_x, map_y, valid_map, spacing, budget
                )
                budget -= used
                counts["mapped" if parts else "skipped"] += 1
                counts["clipped"] += int(clipped)
                objects.extend(parts)
            if objects:
                existing = drawings.get(layer.name)
                drawings[layer.name] = GraphicDrawing(
                    (existing.objects if existing else ()) + tuple(objects)
                )
        return Result(
            {"counts": counts}, drawings=drawings,
            message=f"Mapped {counts['mapped']} objects · {counts['clipped']} clipped · "
                    f"{counts['skipped']} skipped",
        )
