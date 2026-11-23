"""Rank object proposals from same-frame edge strengths and normal angles."""

import math

import cv2
import numpy as np

from .core import Port, Result
from .graphics_layers import GraphicDrawing, graphic_box
from .images import convert_depth
from .objects import collection
from .parameter_validation import NumericRange, ValidationError
from .payloads import require_same_frame
from .processing import ImageTool, image_value
from .region_proposals import box_item
from .selection import SELECTION_INPUTS, contains, selection_mask

_MAX_SEARCH_WINDOWS = 1_000_000


def check_search_size(width, height, alpha, area, aspect_ratio):
    """Estimate the native sliding-window count before OpenCV allocates candidates."""
    initial_side = math.sqrt(area)
    scale_step = math.sqrt(1 / alpha)
    aspect_step = (1 + alpha) / (2 * alpha)
    stride_ratio = (1 - alpha) / (1 + alpha)
    scale_count = math.ceil(math.log(max(width, height) / initial_side) / math.log(scale_step))
    aspect_radius = int(math.log(aspect_ratio) / math.log(aspect_step**2))
    windows = 0
    for scale in range(scale_count):
        side = initial_side * scale_step**scale
        for aspect_index in range(-aspect_radius, aspect_radius + 1):
            aspect = aspect_step**aspect_index
            box_width, box_height = int(side * aspect), int(side / aspect)
            step_x = max(2, int(box_width * stride_ratio))
            step_y = max(2, int(box_height * stride_ratio))
            count_x = max(0, math.ceil(1 + (width - box_width) / step_x))
            count_y = max(0, math.ceil(1 + (height - box_height) / step_y))
            windows += count_x * count_y
            if windows > _MAX_SEARCH_WINDOWS:
                raise ValidationError(
                    "EdgeBoxes search exceeds one million estimated windows. "
                    "Increase the starting area, lower alpha, or reduce the image size.",
                    "min_box_area",
                    "alpha",
                )


class EdgeBoxProposals(ImageTool):
    type_id, title, subtype = "inspect.edge_boxes", "EdgeBoxes proposals", "Segmentation"
    category = block_type = "Inspection"
    description = "Rank candidate object rectangles from aligned edge strengths and orientations."
    inputs = (
        Port("edge_map"),
        Port("orientation"),
        Port("image", required=False, required_when_connected=True),
        *SELECTION_INPUTS[1:],
    )
    outputs = (
        Port("image"),
        Port("collection", "data"),
        Port("proposals", "data"),
        Port("count", "number"),
    )
    payload_reference = "edge_map"
    preview_safe = parallel_safe = process_inputs_readonly = True
    defaults = {
        "alpha": 0.65,
        "beta": 0.75,
        "eta": 1.0,
        "min_score": 0.01,
        "max_boxes": 200,
        "edge_min_mag": 0.1,
        "edge_merge_threshold": 0.5,
        "cluster_min_mag": 0.5,
        "max_aspect_ratio": 3.0,
        "min_box_area": 1000.0,
        "gamma": 2.0,
        "kappa": 1.5,
        "angle_units": "auto",
        "preview_count": 12,
    }
    parameter_choices = {"angle_units": ("auto", "radians", "degrees")}
    parameter_ranges = {
        "alpha": NumericRange(0.05, 0.95),
        "beta": NumericRange(0.01, 1),
        "eta": NumericRange(0.1, 1),
        "min_score": NumericRange(1e-6, 1e9),
        "max_boxes": NumericRange(1, 2048, whole=True),
        "edge_min_mag": NumericRange(0, 1e6),
        "edge_merge_threshold": NumericRange(0.001, 10),
        "cluster_min_mag": NumericRange(0, 1e6),
        "max_aspect_ratio": NumericRange(1, 20),
        "min_box_area": NumericRange(16, 2_097_152),
        "gamma": NumericRange(0.01, 16),
        "kappa": NumericRange(0.01, 8),
        "preview_count": NumericRange(0, 100, whole=True),
    }
    parameter_help = {
        "alpha": "Sliding-window search density. Higher values try more windows.",
        "beta": "Initial intersection-over-union suppression threshold. Values above 0.99 "
        "disable native suppression while retaining the output limit.",
        "eta": "Multiply the suppression threshold after accepting boxes while it is above 0.5.",
        "min_score": "Native minimum edge-enclosure score.",
        "max_boxes": "Maximum native candidates retained before the optional center filter.",
        "edge_min_mag": "Minimum individual edge strength used in clustering.",
        "edge_merge_threshold": "Angular difference budget when joining neighboring edge pixels.",
        "cluster_min_mag": "Minimum accumulated edge strength for a segment.",
        "max_aspect_ratio": "Longest-to-shortest side ratio used in the window search.",
        "min_box_area": "Starting search window area in square pixels. Refinement can shrink it.",
        "gamma": "Sensitivity of affinity between nearby edge clusters.",
        "kappa": "Sensitivity of the proposal score to window size.",
        "angle_units": "Auto reads the orientation payload, or uses radians for an untagged map.",
        "preview_count": "Draw only this many top-ranked proposals. Zero hides the boxes.",
    }
    input_help = {
        "edge_map": "Monochrome nonnegative strengths. Integer maps use full-scale conversion "
        "to [0,1]. Floating maps keep their native scale.",
        "orientation": "Aligned float map of edge-normal angles. Directed angles are folded "
        "modulo half a turn.",
        "image": "Optional aligned image for the overlays. Defaults to the edge map.",
        "mask": "Keep only proposals centered inside this mask.",
        "region": "Keep only proposals centered inside this region and any wired mask.",
    }
    output_help = {
        "image": "Unmodified review image. Proposal graphics are drawable objects.",
        "collection": "Boxes in the edge map's pixel frame, sortable by Strength.",
        "proposals": "Native rank, rectangle [x, y, width, height] and edge-enclosure score.",
    }

    def process(self, inputs):
        source = inputs["edge_map"]
        direction = inputs["orientation"]
        review = inputs.get("image", source)
        height, width = source.shape[:2]
        if min(height, width) < 8 or max(height, width) > 4096 or height * width > 2_097_152:
            raise ValidationError(
                "EdgeBoxes requires 8–4096 pixels per axis and at most 2,097,152 pixels"
            )
        if source.ndim != 2 or direction.shape != source.shape or review.shape[:2] != source.shape:
            raise ValidationError("EdgeBoxes needs matching monochrome maps and review image size")
        require_same_frame(
            (source, direction, review), message="EdgeBoxes inputs use different frames"
        )
        strengths = image_value(source, (np.uint8, np.uint16, np.float32, np.float64))
        angles = image_value(direction, (np.float32, np.float64))
        if strengths.min() < 0 or strengths.max() > np.finfo(np.float32).max / (2 * source.size):
            raise ValidationError("Edge strengths must be nonnegative and sum safely in float32")
        strengths = convert_depth(strengths, "float32")

        p = self.parameters
        units = p["angle_units"]
        if units == "auto":
            units = self.input_payloads["orientation"].get("angle_units", "radians")
        if units not in ("radians", "degrees"):
            raise ValidationError(
                "EdgeBoxes orientation units must be radians or degrees", "angle_units"
            )
        half_turn = 180.0 if units == "degrees" else float(np.float32(np.pi))
        low, high = float(angles.min()), float(angles.max())
        if low < -2 * half_turn or high > 2 * half_turn:
            raise ValidationError("EdgeBoxes normal angles must be within one turn of zero")
        if low < 0 or high > half_turn:
            angles = np.remainder(angles, half_turn)
        if units == "degrees":
            angles = np.deg2rad(angles)
        angles = np.asarray(angles, dtype=np.float32)
        selection = selection_mask({**inputs, "image": source}, source)

        create = getattr(getattr(cv2, "ximgproc", None), "createEdgeBoxes", None)
        if create is None:
            raise ValidationError("EdgeBoxes proposals requires OpenCV contrib ximgproc")
        limit = int(p["max_boxes"])
        rectangles, scores = (), None
        if strengths.max() > float(p["edge_min_mag"]) and (
            selection is None or cv2.countNonZero(selection)
        ):
            check_search_size(
                width,
                height,
                float(p["alpha"]),
                float(p["min_box_area"]),
                float(p["max_aspect_ratio"]),
            )
            detector = create(
                alpha=float(p["alpha"]),
                beta=float(p["beta"]),
                eta=float(p["eta"]),
                minScore=float(p["min_score"]),
                maxBoxes=limit,
                edgeMinMag=float(p["edge_min_mag"]),
                edgeMergeThr=float(p["edge_merge_threshold"]),
                clusterMinMag=float(p["cluster_min_mag"]),
                maxAspectRatio=float(p["max_aspect_ratio"]),
                minBoxArea=float(p["min_box_area"]),
                gamma=float(p["gamma"]),
                kappa=float(p["kappa"]),
            )
            rectangles, scores = detector.getBoundingBoxes(strengths, angles)

        items, proposals = [], []
        ranked_scores = scores.reshape(-1)[:limit] if scores is not None else ()
        for rank, (bounds, score) in enumerate(
            zip(rectangles[:limit], ranked_scores, strict=True), 1
        ):
            x, y, w, h = map(int, bounds)
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
                raise ValidationError("OpenCV returned an invalid EdgeBoxes rectangle")
            if not np.isfinite(score):
                raise ValidationError("OpenCV returned a nonfinite EdgeBoxes score")
            item = box_item(rank, x, y, x + w, y + h)
            if not contains(selection, (item["metrics"]["center_x"], item["metrics"]["center_y"])):
                continue
            item["metrics"]["strength"] = float(score)
            items.append(item)
            proposals.append({"rank": rank, "bounds": [x, y, w, h], "score": float(score)})

        objects = ()
        if self.generate_graphics:
            objects = tuple(
                graphic_box(*proposal["bounds"], (55, 210, 255))
                for proposal in proposals[: int(p["preview_count"])]
            )
        return Result(
            {
                "image": review,
                "collection": collection("boxes", items, source, self.input_payloads["edge_map"]),
                "proposals": proposals,
                "count": len(items),
            },
            drawings={
                "EdgeBoxes proposals": GraphicDrawing(objects, source="image", from_output=True)
            }
            if objects
            else {},
            message=f"{len(items)} ranked object proposals",
        )
