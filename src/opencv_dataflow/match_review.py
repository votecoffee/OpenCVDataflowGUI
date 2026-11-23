"""Side-by-side diagnostics for paired points, segments, and geometric inliers."""

import numpy as np

from .core import Block, Port, Result
from .features import overlay_image
from .graphics_layers import GraphicDrawing, draw_object, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import ValuePacket, frame, require_same_frame
from .result_graphics import ACCEPTED, REJECTED

_UNCLASSIFIED = (60, 220, 255)


def paired_points(data, query_size, train_size, *, geometry="points"):
    if not isinstance(data, dict):
        raise ValidationError("Connect a paired feature match record")
    points = []
    for role, size in (("query", query_size), ("train", train_size)):
        if data.get(f"{role}_size") is not None and data[f"{role}_size"] != size:
            raise ValidationError(f"{role.title()} image size differs from the match record")
        raw = data.get(f"{role}_{geometry}")
        if not isinstance(raw, (list, tuple)) or len(raw) > 100_000:
            raise ValidationError(
                f"{role.title()} {geometry} must be a list of up to 100,000 pairs"
            )
        try:
            array = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError(
                f"{role.title()} {geometry} must contain finite [x, y] pairs"
            ) from exc
        point_shape = (2,) if geometry == "points" else (2, 2)
        if len(raw) == 0:
            array = array.reshape(0, *point_shape)
        if (
            array.shape != (len(raw), *point_shape)
            or not np.isfinite(array).all()
            or np.any(array < 0)
            or np.any(array >= size)
        ):
            raise ValidationError(
                f"{role.title()} {geometry} must be finite and inside their image"
            )
        points.append(array)
    if len(points[0]) != len(points[1]):
        raise ValidationError("Query and train point counts must match")
    return points


def fit_inliers(fit, count, query_size, train_size):
    if not isinstance(fit, dict):
        raise ValidationError("Connect a geometric fit that includes its inlier mask")
    for field, size in (("source_image_size", query_size), ("target_image_size", train_size)):
        if fit.get(field) is not None and fit[field] != size:
            raise ValidationError("Geometric fit image sizes differ from the match record")
    mask = fit.get("inlier_mask")
    if not isinstance(mask, (list, tuple)) or len(mask) != count or any(
        type(item) is not bool for item in mask
    ):
        raise ValidationError("Fit must contain one boolean inlier entry per match")
    return np.asarray(mask, dtype=bool)


class ReviewFeatureMatches(Block):
    type_id, title = "inspect.review_matches", "Review feature matches"
    category, block_type, subtype = "Inspection", "Inspection", "Registration"
    inputs = (
        Port("query_image"), Port("train_image"), Port("matches", "data"),
        Port("fit", "data", required=False, required_when_connected=True),
    )
    outputs = (Port("overlay"), Port("canvas"), Port("info", "data"))
    defaults = {"show": "all", "max_pairs": 160, "gap": 16, "line_width": 1, "endpoints": True}
    parameter_choices = {"show": ("all", "inliers", "outliers")}
    parameter_ranges = {
        "max_pairs": NumericRange(1, 1000, whole=True),
        "gap": NumericRange(0, 128, whole=True),
        "line_width": NumericRange(1, 8, whole=True),
    }
    parameter_help = {
        "show": "Filter displayed pairs by the optional geometric fit.",
        "max_pairs": "Maximum number of pairs to draw, in the incoming match order.",
        "gap": "Pixels between the two images.",
        "line_width": "Width of the lines in pixels.",
        "endpoints": "Mark both feature locations on each displayed pair.",
    }
    input_help = {
        "query_image": "Image used by the query feature detector, displayed on the left.",
        "train_image": "Image used by the train feature detector, displayed on the right.",
        "matches": "Paired Query and Train points in image pixels, with optional line endpoints.",
        "fit": "Optional geometric fit made from this same ordered match list.",
    }
    composes_result_image = True
    process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if type(p["endpoints"]) is not bool:
            raise ValidationError("Endpoints must be boolean", "endpoints")

    def process(self, inputs):
        query, train = inputs["query_image"], inputs["train_image"]
        query_size, train_size = [query.shape[1], query.shape[0]], [train.shape[1], train.shape[0]]
        p = self.parameters
        offset = query_size[0] + int(p["gap"])
        size = [offset + train_size[0], max(query_size[1], train_size[1])]
        if max(size) > 16384 or size[0] * size[1] > 16_000_000:
            raise ValidationError(
                "Match review canvas exceeds 16 million pixels or 16,384 per axis"
            )
        matches = inputs["matches"]
        query_points, train_points = paired_points(matches, query_size, train_size)
        count = len(query_points)
        segments = None
        if "query_segments" in matches or "train_segments" in matches:
            segments = paired_points(matches, query_size, train_size, geometry="segments")
            if len(segments[0]) != count:
                raise ValidationError("Line match endpoints must align with the matched midpoints")
            for role, image in (("query", query), ("train", train)):
                source_context = matches.get(f"{role}_context", {})
                require_same_frame(
                    (image, ValuePacket(None, source_context)),
                    message=f"{role.title()} line segments belong to another image frame",
                )
        mask = None
        if "fit" in inputs:
            mask = fit_inliers(inputs["fit"], count, query_size, train_size)
        show = p["show"]
        if show != "all" and mask is None:
            raise ValidationError("Connect a geometric fit to display inliers or outliers", "show")
        indexes = np.arange(count)
        if show != "all":
            indexes = indexes[mask if show == "inliers" else ~mask]
        indexes = indexes[:int(p["max_pairs"])]
        canvas = np.full((size[1], size[0], 3), 30, np.uint8)
        canvas[:query_size[1], :query_size[0]] = overlay_image(query)
        canvas[:train_size[1], offset:] = overlay_image(train)
        layers = {}
        line_width = int(p["line_width"])
        for index in indexes:
            if mask is None:
                label, color = "Unclassified matches", _UNCLASSIFIED
            elif mask[index]:
                label, color = "Inlier matches", ACCEPTED
            else:
                label, color = "Outlier matches", REJECTED
            start = query_points[index]
            end = train_points[index] + (offset, 0)
            objects = layers.setdefault(label, [])
            objects.append(graphic_object("polyline", (start, end), color, thickness=line_width))
            if segments is not None:
                objects.extend((
                    graphic_object("polyline", segments[0][index], color, thickness=line_width + 1),
                    graphic_object(
                        "polyline", segments[1][index] + (offset, 0),
                        color, thickness=line_width + 1,
                    ),
                ))
            if p["endpoints"]:
                objects.extend(graphic_object("marker", (point,), color) for point in (start, end))
        overlay = canvas.copy()
        identity = np.eye(3)
        for objects in layers.values():
            for obj in objects:
                draw_object(overlay, obj, identity)
        path = getattr(self, "_event_path", self.id)
        context = {"spatial": frame(size, f"{path}:matches")}
        ticket = self.input_payloads["query_image"].get("ticket")
        if ticket is not None:
            context["ticket"] = ticket
        inliers = None if mask is None else int(mask.sum())
        info = {
            "pairs": count, "shown": int(indexes.size), "inliers": inliers,
            "outliers": None if inliers is None else count - inliers,
            "train_origin": [offset, 0], "canvas_size": size,
        }
        return Result(
            {"overlay": overlay, "canvas": canvas, "info": info},
            payloads={"canvas": context, "overlay": context, "info": context},
            drawings={
                label: GraphicDrawing(tuple(objects), source="canvas", from_output=True)
                for label, objects in layers.items()
            } if self.generate_graphics else {},
            message=f"{indexes.size}/{count} feature pairs shown",
        )
