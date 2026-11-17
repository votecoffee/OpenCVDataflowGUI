"""Project ticketed source measurements onto an explicitly owned panorama grid."""

import copy

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .panorama_geometry import project_points, validate_layout
from .parameter_validation import NumericRange, ValidationError
from .payloads import ValuePacket, require_same_frame
from .selection import contains
from .tickets import ticket_key


class PanoramaPoints(Block):
    type_id, title = "cv.panorama_points", "Map source points into panorama"
    category, block_type, subtype = "Image processing", "Image processing", "Panoramas"
    inputs = (
        Port("layout", "data"),
        Port("points", "data"),
        Port("source_label", "text", required=False, required_when_connected=True),
        Port("mask", required=False, visual_type="region"),
        Port("source_mask", required=False, visual_type="region"),
    )
    outputs = (
        Port("points", "data"),
        Port("indices", "data"),
        Port("valid", "data"),
        Port("count", "number"),
    )
    defaults = {
        "source_label": "",
        "source_index": -1,
        "outside": "filter",
        "ticket_policy": "matching_source",
        "minimum_points": 1,
        "maximum_graphics_points": 256,
    }
    parameter_choices = {
        "outside": ("filter", "error"),
        "ticket_policy": ("matching_source", "ignore"),
    }
    parameter_ranges = {
        "minimum_points": NumericRange(0, 100000, whole=True),
        "source_index": NumericRange(-1, 7, whole=True),
        "maximum_graphics_points": NumericRange(0, 4096, whole=True),
    }
    parameter_help = {
        "source_label": "Optional source label. Input overrides label, then input index/anchor.",
        "source_index": "Original input index, or -1 for the anchor; label overrides.",
        "outside": "Filter points outside source/projection/mask, or reject them.",
        "ticket_policy": "Require known current source tickets, or explicitly ignore them.",
        "minimum_points": "Minimum retained points; zero permits an empty set.",
        "maximum_graphics_points": "Bound accepted/rejected mask markers on the panorama grid.",
    }
    input_help = {
        "layout": "Current rotating panorama layout; nonlinear grid identity is checked.",
        "points": "Up to 100000 finite source [x,y] points; compatible crop frames are converted.",
        "source_label": "Optional source label overriding the configured label.",
        "mask": "Optional panorama uint8 support, typically the stitcher's Coverage output.",
        "source_mask": "Optional full-source uint8 selection, before projection.",
    }
    output_help = {
        "indices": "Original input indices retained in Points.",
        "valid": "One boolean per input point, including points filtered out.",
    }
    preview_safe = parallel_safe = process_inputs_readonly = True
    payload_reference = "layout"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_inputs = ["layout", "points", "mask"]
        self.description = self.description or "Carry source measurements into a panorama grid."
        self.documentation = self.documentation or (
            "<p>Map pinhole source points through a rotating panorama layout. Known crop frames "
            "are converted; source tickets and nonlinear layout identity are checked. Source "
            "and panorama masks can filter results. See docs/rotating-panorama.md.</p>"
        )

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not isinstance(p["source_label"], str) or len(p["source_label"]) > 1024:
            raise ValidationError("Source label must be text up to 1024 characters", "source_label")
        for name in ("minimum_points", "source_index", "maximum_graphics_points"):
            if type(p[name]) is not int:
                raise ValidationError("Use typed integer point settings", name)

    def process(self, inputs):
        layout = validate_layout(inputs["layout"])
        p = self.parameters
        by_index = next(
            (c["label"] for c in layout["cameras"] if c["index"] == p["source_index"]), None
        )
        label = inputs.get(
            "source_label",
            p["source_label"] or (layout["anchor_label"] if p["source_index"] == -1 else by_index),
        )
        if not isinstance(label, str):
            raise ValidationError("Source label must be text")
        camera = next((c for c in layout["cameras"] if c["label"] == label), None)
        if camera is None:
            raise ValidationError("Source label is not retained in the panorama")
        points = np.asarray(inputs["points"], np.float64)
        if points.size == 0:
            points = points.reshape(0, 2)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) > 100000
            or not np.isfinite(points).all()
        ):
            raise ValidationError("Supply up to 100000 finite [x,y] source points")
        payload = self.input_payloads["points"]
        source = camera.get("source_spatial")
        actual = payload.get("spatial")
        if (
            source
            and actual
            and source.get("root_id") is not None
            and actual.get("root_id") is not None
        ):
            if source["root_id"] != actual["root_id"] or source["root_size"] != actual["root_size"]:
                raise ValidationError("Points and panorama camera have different coordinate roots")
            mapping = np.linalg.solve(
                np.asarray(source["to_root"], float), np.asarray(actual["to_root"], float)
            )
            points = (
                cv2.perspectiveTransform(points.reshape(-1, 1, 2), mapping).reshape(-1, 2)
                if len(points)
                else points
            )
        elif actual and actual.get("image_size") != camera["source_size"]:
            raise ValidationError("Unknown point frames must use full source-image pixels")
        expected_ticket = camera.get("source_ticket")
        current = payload.get("ticket")
        if (
            p["ticket_policy"] == "matching_source"
            and expected_ticket is not None
            and current is not None
            and ticket_key(expected_ticket) != ticket_key(current)
        ):
            raise ValidationError("Points and source-camera tickets differ")
        transformed, valid = project_points(
            points, camera, layout["projection"], layout["scale"], layout["origin"]
        )
        valid &= (points >= 0).all(axis=1) & (points <= np.asarray(camera["source_size"]) - 1).all(
            axis=1
        )
        valid &= (
            np.isfinite(transformed).all(axis=1)
            & (transformed >= 0).all(axis=1)
            & (transformed <= np.asarray(layout["image_size"]) - 1).all(axis=1)
        )
        geometric = valid.copy()
        source_mask = inputs.get("source_mask")
        if source_mask is not None:
            if source_mask.dtype != np.uint8 or source_mask.shape != tuple(
                camera["source_size"][::-1]
            ):
                raise ValidationError("Source mask must match the full source grid")
            require_same_frame(
                (source_mask, ValuePacket(source_mask, {"spatial": source})),
                message="Source mask uses a different coordinate grid",
            )
            valid &= np.array(
                [
                    contains(source_mask, point) if ok else False
                    for point, ok in zip(points, valid, strict=True)
                ],
                bool,
            )
        mask = inputs.get("mask")
        if mask is not None:
            if mask.dtype != np.uint8 or mask.shape != tuple(layout["image_size"][::-1]):
                raise ValidationError("Panorama mask must be uint8 and match the canvas")
            require_same_frame(
                (mask, ValuePacket(mask, {"spatial": layout["spatial"]})),
                message="Panorama mask uses a different coordinate grid",
            )
            valid &= np.array(
                [
                    contains(mask, point) if ok else False
                    for point, ok in zip(transformed, valid, strict=True)
                ],
                bool,
            )
        if not valid.all() and p["outside"] == "error":
            raise ValidationError("Some source points lack valid projected support", "outside")
        indices = np.flatnonzero(valid).tolist()
        passed = len(indices) >= p["minimum_points"]
        context = copy.deepcopy(self.input_payloads["layout"])
        context["spatial"] = copy.deepcopy(layout["spatial"])
        context.pop("pyramid", None)
        context["panorama_point_source"] = {"label": label, "ticket": current}
        drawings = {}
        if self.generate_graphics and p["maximum_graphics_points"]:
            selected = np.flatnonzero(geometric)[: p["maximum_graphics_points"]]
            objects = []
            for accepted, color in [(True, (70, 220, 140)), (False, (60, 90, 240))]:
                chosen = selected[valid[selected] == accepted]
                if len(chosen):
                    objects.append(graphic_object("marker", transformed[chosen], color))
            if objects:
                drawings["Projected source points"] = GraphicDrawing(
                    tuple(objects), "points", from_output=True
                )
        return Result(
            {
                "points": transformed[valid].tolist(),
                "indices": indices,
                "valid": valid.tolist(),
                "count": len(indices),
            },
            payloads={
                name: copy.deepcopy(context) for name in ("points", "indices", "valid", "count")
            },
            passed=passed,
            drawings=drawings,
            fail_reasons=[] if passed else ["panorama.too_few_projected_points"],
            message=f"{len(indices)}/{len(points)} points projected from {label}",
        )
