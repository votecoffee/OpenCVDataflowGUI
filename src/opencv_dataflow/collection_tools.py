"""Detection, filtering, stable sorting and rendering of measured object collections."""

import copy
from functools import cmp_to_key

import cv2
import numpy as np

from .core import Block, Port, Result
from .features import overlay_image
from .graphics_layers import GraphicDrawing, graphic_box, graphic_object
from .objects import METRICS, collection, line_item, validate_collection
from .payloads import (
    ValuePacket,
    image_with_payload,
    moved_payload,
    require_same_frame,
    validate_payload,
)
from .processing import ConnectedComponents, finite, gray_image, integer
from .result_graphics import ACCEPTED, REJECTED, review_legend
from .selection import SELECTION_INPUTS, contains, selection_mask


def read_collection(block, inputs, port="collection"):
    data = copy.deepcopy(validate_collection(inputs[port]))
    payload = {**data.get("context", {}), **block.input_payloads.get(port, {})}
    validate_payload(payload)
    if payload.get("spatial", {}).get("image_size", data["image_size"]) != data["image_size"]:
        raise ValueError("Collection spatial dimensions do not match its image_size")
    data["context"] = {
        key: copy.deepcopy(payload[key]) for key in ("spatial", "pyramid") if key in payload
    }
    return data, payload


def metric_value(item, name, reference):
    if name != "distance":
        return item["metrics"].get(name)
    x, y = item["metrics"].get("center_x"), item["metrics"].get("center_y")
    return (
        float(np.hypot(x - reference[0], y - reference[1]))
        if x is not None and y is not None
        else None
    )


def reference_point(value):
    point = np.asarray(value, float)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("Reference point must be a finite [x,y] pair")
    return point


def subset(data, items):
    output = copy.deepcopy(data)
    output["items"], output["count"] = copy.deepcopy(items), len(items)
    return output


def collection_result(data, payload, **extra):
    points = [
        [item["metrics"]["center_x"], item["metrics"]["center_y"]]
        for item in data["items"]
        if item["metrics"].get("center_x") is not None
        and item["metrics"].get("center_y") is not None
    ]
    outputs = {"collection": data, "count": data["count"], "points": points, **extra}
    return Result(
        outputs,
        payloads={key: payload for key in outputs},
        message=f"{data['count']} {data['kind']}; coordinates in working image",
    )


def draw_points(value, origin):
    points = np.asarray(value, float)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or not len(points)
        or not np.isfinite(points).all()
        or np.any(np.abs(points - origin) > 10_000_000)
    ):
        raise ValueError(
            "Geometry needs nonempty finite Nx2 points within drawing coordinate limits"
        )
    return np.rint(points - origin).astype(np.int32)


class FindBlobs(ConnectedComponents):
    type_id, title = "inspect.blobs", "Find blobs"
    defaults = {**ConnectedComponents.defaults, "check_count": False}

    def process(self, inputs):
        result = super().process(inputs)
        if not self.parameters["check_count"]:
            result.passed, result.score = True, None
            result.message = f"{result.outputs['count']} blobs; coordinates in working image"
        return result


class FindLineSegments(Block):
    type_id, title, category = "inspect.line_segments", "Find line segments (LSD)", "Inspection"
    block_type, subtype = "Inspection", "Features"
    inputs = SELECTION_INPUTS
    outputs = (
        Port("collection", "data"),
        Port("segments", "data"),
        Port("count", "number"),
        Port("overlay"),
    )
    defaults = {
        "refinement": "standard",
        "scale": 0.8,
        "sigma_scale": 0.6,
        "quantization": 2.0,
        "angle_tolerance": 22.5,
        "log_epsilon": 0.0,
        "density_threshold": 0.7,
        "bins": 1024,
    }
    parameter_choices = {"refinement": ("none", "standard", "advanced")}
    parameter_help = {
        "log_epsilon": "-log10(NFA) threshold; used only with advanced refinement.",
        "scale": "Detection image scale in (0,1]. Returned coordinates use the input image.",
    }

    def process(self, inputs):
        image = gray_image(inputs["image"], (np.uint8,))
        p = self.parameters
        scale, sigma, quant, angle, density = (
            finite(p[k], k, 0)
            for k in (
                "scale",
                "sigma_scale",
                "quantization",
                "angle_tolerance",
                "density_threshold",
            )
        )
        if not 0 < scale <= 1 or sigma <= 0 or not 0 < angle < 180 or density >= 1:
            raise ValueError("Invalid LSD scale, sigma, angle tolerance or density threshold")
        refine = {
            "none": cv2.LSD_REFINE_NONE,
            "standard": cv2.LSD_REFINE_STD,
            "advanced": cv2.LSD_REFINE_ADV,
        }[p["refinement"]]
        detector = cv2.createLineSegmentDetector(
            refine,
            scale,
            sigma,
            quant,
            angle,
            finite(p["log_epsilon"], "Log epsilon"),
            density,
            integer(p["bins"], "Bins", 1, 100000),
        )
        found, widths, precisions, nfa = detector.detect(image)
        # OpenCV 5 returns scalar-per-line measurements as 1D vectors.
        widths, precisions, nfa = (
            values.reshape(-1) if values is not None else None
            for values in (widths, precisions, nfa)
        )
        selection = selection_mask(inputs, image)
        items, segments = [], []
        overlay = overlay_image(inputs["image"])
        for index, segment in enumerate(found.reshape(-1, 4) if found is not None else []):
            if not contains(selection, segment[:2]) or not contains(selection, segment[2:]):
                continue
            items.append(
                line_item(
                    segment,
                    index,
                    widths[index] if widths is not None else None,
                    precisions[index] if precisions is not None else None,
                    nfa[index] if nfa is not None else None,
                )
            )
            segments.append(segment.tolist())
            a, b = np.rint(segment.reshape(2, 2)).astype(int)
            cv2.line(overlay, tuple(a), tuple(b), (80, 230, 140), 1)
        return Result(
            {
                "collection": collection("lines", items, image, self.input_payloads["image"]),
                "segments": segments,
                "count": len(items),
                "overlay": overlay,
            },
            graphics={"lines": overlay},
            drawings={"lines": GraphicDrawing(tuple(
                graphic_object("polyline", (s[:2], s[2:]), (80, 230, 140), thickness=1)
                for s in segments
            ))} if self.generate_graphics and segments else {},
            message=f"{len(items)} line segments",
        )


class FilterCollection(Block):
    type_id, title, category = "collection.filter", "Filter object collection", "Inspection"
    block_type, subtype = "Inspection", "Collections"
    inputs = (
        Port("collection", "data"),
        Port("mask", required=False),
        Port("region", "data", required=False),
    )
    outputs = (
        Port("collection", "data"),
        Port("count", "number"),
        Port("points", "data"),
        Port("rejected", "data"),
        Port("rejected_count", "number"),
    )
    defaults = {
        "metric": "area",
        "use_minimum": True,
        "minimum": 0.0,
        "include_minimum": True,
        "use_maximum": False,
        "maximum": 0.0,
        "include_maximum": True,
        "invert_range": False,
        "missing": "reject",
        "reference_point": [0.0, 0.0],
    }
    parameter_choices = {"metric": METRICS, "missing": ("reject", "keep", "error")}
    parameter_help = {
        "reference_point": "[x,y] in reported collection coordinates; used for distance.",
        "missing": "Undefined metrics (e.g. Hu moments on lines) use this policy.",
        "invert_range": "Invert numeric range only; the optional region/mask still intersects it.",
    }

    def process(self, inputs):
        data, payload = read_collection(self, inputs)
        p = self.parameters
        minimum, maximum = finite(p["minimum"], "Minimum"), finite(p["maximum"], "Maximum")
        if p["use_minimum"] and p["use_maximum"] and maximum < minimum:
            raise ValueError("Maximum must be at least minimum")
        reference = reference_point(p["reference_point"])
        width, height = data["image_size"]
        dummy = np.empty((height, width), np.uint8)
        origin = np.asarray(data.get("origin", [0, 0]))
        local_payload = moved_payload(
            payload, np.array([[1, 0, origin[0]], [0, 1, origin[1]], [0, 0, 1]]), [width, height]
        )
        image = image_with_payload(dummy, local_payload)
        selection = selection_mask({**inputs, "image": image}, image)
        accepted, rejected = [], []
        for item in data["items"]:
            value = metric_value(item, p["metric"], reference)
            if value is None:
                if p["missing"] == "error":
                    raise ValueError(f"Item {item['id']} has no {p['metric']} metric")
                keep = p["missing"] == "keep"
            else:
                lower = value >= minimum if p["include_minimum"] else value > minimum
                upper = value <= maximum if p["include_maximum"] else value < maximum
                keep = (not p["use_minimum"] or lower) and (not p["use_maximum"] or upper)
                if p["invert_range"]:
                    keep = not keep
            if selection is not None:
                center = [item["metrics"].get("center_x"), item["metrics"].get("center_y")]
                keep = (
                    keep and None not in center and contains(selection, np.asarray(center) - origin)
                )
            (accepted if keep else rejected).append(item)
        result = collection_result(
            subset(data, accepted),
            payload,
            rejected=subset(data, rejected),
            rejected_count=len(rejected),
        )
        if self.generate_graphics:
            scale = min(1.0, 1200 / max(width, height))
            shape = (max(1, round(height * scale)), max(1, round(width * scale)), 3)
            graphic = np.full(shape, 24, np.uint8)
            for items, color in ((rejected, REJECTED), (accepted, ACCEPTED)):
                for item in items:
                    metrics = item["metrics"]
                    x, y, w, h = (metrics.get(name) for name in ("x", "y", "width", "height"))
                    if None in (x, y, w, h):
                        continue
                    bounds = (np.array([[x, y], [x + w - 1, y + h - 1]]) - origin) * scale
                    start, end = np.rint(np.clip(bounds, -10000, 10000)).astype(int)
                    cv2.rectangle(graphic, tuple(start), tuple(end), color, 1)
            review_legend(graphic, len(accepted), len(rejected))
            result.graphics["collection filter"] = graphic
        return result


class SortCollection(Block):
    type_id, title, category = "collection.sort", "Sort object collection", "Inspection"
    block_type, subtype = "Inspection", "Collections"
    inputs = (Port("collection", "data"),)
    outputs = (Port("collection", "data"), Port("count", "number"), Port("points", "data"))
    defaults = {
        "metric": "area",
        "direction": "descending",
        "secondary_metric": "none",
        "secondary_direction": "ascending",
        "missing": "last",
        "reference_point": [0.0, 0.0],
    }
    parameter_choices = {
        "metric": METRICS,
        "direction": ("ascending", "descending"),
        "secondary_metric": ("none", *METRICS),
        "secondary_direction": ("ascending", "descending"),
        "missing": ("last", "first", "error"),
    }

    def process(self, inputs):
        data, payload = read_collection(self, inputs)
        p, reference = self.parameters, reference_point(self.parameters["reference_point"])
        criteria = [(p["metric"], p["direction"])]
        if p["secondary_metric"] != "none":
            criteria.append((p["secondary_metric"], p["secondary_direction"]))
        if p["missing"] == "error" and any(
            metric_value(item, metric, reference) is None
            for item in data["items"]
            for metric, _ in criteria
        ):
            raise ValueError("Collection has undefined sort metrics")

        def compare(a, b):
            for metric, direction in criteria:
                x, y = metric_value(a, metric, reference), metric_value(b, metric, reference)
                if x is None and y is None:
                    continue
                if x is None or y is None:
                    return (-1 if p["missing"] == "first" else 1) * (1 if x is None else -1)
                result = (x > y) - (x < y)
                if result:
                    return result if direction == "ascending" else -result
            return 0

        return collection_result(
            subset(data, sorted(data["items"], key=cmp_to_key(compare))), payload
        )


class DrawCollection(Block):
    type_id, title, category = "collection.draw", "Draw object collection", "Inspection"
    block_type, subtype = "Inspection", "Collections"
    inputs = (Port("image"), Port("collection", "data"))
    outputs = (Port("overlay"), Port("mask"), Port("count", "number"))
    defaults = {
        "color": [80, 230, 140],
        "thickness": 2,
        "draw_bounds": True,
        "draw_centers": True,
        "labels": "rank",
    }
    parameter_choices = {"labels": ("rank", "id", "none")}

    def process(self, inputs):
        data, payload = read_collection(self, inputs)
        image = inputs["image"]
        if data["image_size"] != [image.shape[1], image.shape[0]]:
            raise ValueError(
                "Collection dimensions differ from target image; select the matching image level"
            )
        origin = np.asarray(data.get("origin", [0, 0]))
        local = moved_payload(
            payload, np.array([[1, 0, origin[0]], [0, 1, origin[1]], [0, 0, 1]]), data["image_size"]
        )
        try:
            require_same_frame([image, ValuePacket(image, local)])
        except ValueError as exc:
            raise ValueError(
                "Collection coordinate frame differs from target image; use the matching frame"
            ) from exc
        color = np.asarray(self.parameters["color"], float)
        if (
            color.shape != (3,)
            or not np.isfinite(color).all()
            or np.any(color != np.rint(color))
            or np.any((color < 0) | (color > 255))
        ):
            raise ValueError("Color requires three BGR byte values")
        color = tuple(map(int, color))
        thickness = integer(self.parameters["thickness"], "Thickness", 1, 20)
        overlay, mask = overlay_image(image), np.zeros(image.shape[:2], np.uint8)
        draw_objects = self.generate_graphics
        objects = []
        for rank, item in enumerate(data["items"], 1):
            geometry, metrics = item["geometry"], item["metrics"]
            if data["kind"] == "blobs":
                contours = [draw_points(c, origin).reshape(-1, 1, 2) for c in geometry["contours"]]
                cv2.drawContours(mask, contours, -1, 255, -1)
                cv2.drawContours(overlay, contours, -1, color, thickness)
                if draw_objects:
                    objects.extend(
                        graphic_object(
                            "polyline", contour.reshape(-1, 2), color,
                            closed=True, thickness=thickness,
                        )
                        for contour in contours
                    )
            else:
                points = draw_points(geometry["points"], origin)
                if data["kind"] == "lines":
                    if points.shape != (2, 2):
                        raise ValueError("Line geometry needs two endpoints")
                    cv2.line(mask, tuple(points[0]), tuple(points[1]), 255, thickness)
                    cv2.line(overlay, tuple(points[0]), tuple(points[1]), color, thickness)
                else:
                    contours = [points.reshape(-1, 1, 2)]
                    cv2.drawContours(mask, contours, -1, 255, -1)
                    cv2.drawContours(overlay, contours, -1, color, thickness)
                if draw_objects:
                    objects.append(graphic_object(
                        "polyline", points, color,
                        closed=data["kind"] != "lines", thickness=thickness,
                    ))
            x, y = np.rint([metrics["x"], metrics["y"]]).astype(int) - origin
            if self.parameters["draw_bounds"]:
                w, h = round(metrics["width"]), round(metrics["height"])
                cv2.rectangle(overlay, (x, y), (x + w - 1, y + h - 1), color, 1)
                if draw_objects:
                    objects.append(graphic_box(x, y, w, h, color))
            if self.parameters["draw_centers"]:
                center = np.rint([metrics["center_x"], metrics["center_y"]]).astype(int) - origin
                cv2.drawMarker(overlay, tuple(center), color, cv2.MARKER_CROSS, 8, 1)
                if draw_objects:
                    objects.append(graphic_object("marker", [center], color))
            if self.parameters["labels"] != "none":
                label = str(rank if self.parameters["labels"] == "rank" else item["id"])
                position = (int(x), max(12, int(y)))
                cv2.putText(
                    overlay,
                    label,
                    position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                )
                if draw_objects:
                    objects.append(graphic_object("text", [position], color, text=label))
        return Result(
            {"overlay": overlay, "mask": mask, "count": data["count"]},
            graphics={"objects": overlay},
            drawings={"objects": GraphicDrawing(tuple(objects))} if objects else {},
            message=f"{data['count']} drawn {data['kind']}",
        )


COLLECTION_BLOCKS = (FindBlobs, FindLineSegments, FilterCollection, SortCollection, DrawCollection)
