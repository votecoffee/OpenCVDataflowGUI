"""Locate translated template shapes by voting on grayscale edges."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .features import overlay_image
from .graphics_layers import GraphicDrawing, draw_object, graphic_box, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .poses import make_pose
from .processing import gray_image
from .result_graphics import ACCEPTED
from .selection import SELECTION_INPUTS, selection_mask, whole

_MAX_VOTES = 150_000_000


def _template_anchor(value):
    try:
        if not isinstance(value, (list, tuple)):
            raise TypeError
        x, y = value
        return whole(x, "Anchor x", -1), whole(y, "Anchor y", -1)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValidationError(
            "Use a whole-pixel [x, y] template anchor", "template_anchor"
        ) from exc


def _detect(detector, template, template_edges, anchor, image, edges):
    detector.setTemplate(
        edges=template_edges,
        dx=cv2.Sobel(template, cv2.CV_32F, 1, 0),
        dy=cv2.Sobel(template, cv2.CV_32F, 0, 1),
        templCenter=anchor,
    )
    return detector.detect(
        edges=edges,
        dx=cv2.Sobel(image, cv2.CV_32F, 1, 0),
        dy=cv2.Sobel(image, cv2.CV_32F, 0, 1),
    )


def _ranked_centers(positions, votes, shape, selection, distance, maximum):
    if positions is None:
        return []
    centers = positions.reshape(-1, 4)[:, :2]
    counts = votes.reshape(-1, 3)[:, 0]
    pixels = np.rint(centers).astype(np.intp)
    height, width = shape
    x, y = pixels.T
    allowed = np.flatnonzero((x >= 0) & (x < width) & (y >= 0) & (y < height))
    if selection is not None:
        allowed = allowed[selection[y[allowed], x[allowed]] != 0]
    order = np.lexsort((centers[allowed, 0], centers[allowed, 1], -counts[allowed]))
    accepted = []
    for index in allowed[order]:
        cx, cy = map(float, centers[index])
        if any((cx - ax) ** 2 + (cy - ay) ** 2 < distance**2 for ax, ay, _ in accepted):
            continue
        accepted.append((cx, cy, int(counts[index])))
        if len(accepted) == maximum:
            break
    return accepted


class FindHoughTemplate(Block):
    type_id, title, category = "inspect.hough_template", "Find Hough template", "Inspection"
    block_type, subtype = "Inspection", "Matching"
    inputs = (*SELECTION_INPUTS, Port("template"))
    outputs = (
        Port("matches", "data"),
        Port("poses", "data"),
        Port("count", "number"),
        Port("overlay", required=False),
    )
    composes_result_image = True
    defaults = {
        "template_anchor": [-1, -1],
        "canny_low": 40.0,
        "canny_high": 100.0,
        "levels": 360,
        "dp": 1.0,
        "minimum_votes": 25,
        "min_center_distance": 10.0,
        "minimum_matches": 1,
        "maximum_matches": 25,
        "draw_overlay": True,
    }
    parameter_ranges = {
        "canny_low": NumericRange(0),
        "canny_high": NumericRange(0),
        "levels": NumericRange(1, 720, whole=True),
        "dp": NumericRange(1, 16),
        "minimum_votes": NumericRange(2, _MAX_VOTES, whole=True),
        "min_center_distance": NumericRange(0, 10_000),
        "minimum_matches": NumericRange(0, 64, whole=True),
        "maximum_matches": NumericRange(1, 64, whole=True),
    }
    parameter_help = {
        "template_anchor": "Template pose origin. [-1, -1] uses the integer center.",
        "levels": "Number of gradient orientation bins used for voting.",
        "dp": "Accumulator cell spacing in source pixels. Larger values use a coarser grid.",
        "minimum_votes": "Minimum accepted center vote count.",
        "min_center_distance": "Minimum distance between retained anchors after selection.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        anchor = _template_anchor(p["template_anchor"])
        if -1 in anchor and anchor != (-1, -1):
            raise ValidationError("Set both anchor coordinates or use [-1, -1]", "template_anchor")
        if p["canny_low"] >= p["canny_high"]:
            raise ValidationError("Canny low must be below Canny high", "canny_low", "canny_high")
        if p["minimum_matches"] > p["maximum_matches"]:
            raise ValidationError(
                "Minimum matches must not exceed maximum matches",
                "minimum_matches",
                "maximum_matches",
            )

    def process(self, inputs):
        p = self.parameters
        source = inputs["image"]
        image = gray_image(source, (np.uint8,), auto_depth=True)
        template = gray_image(inputs["template"], (np.uint8,), auto_depth=True)
        height, width = template.shape
        if min(height, width) < 3 or height > image.shape[0] or width > image.shape[1]:
            raise ValidationError(
                "Use a template of at least 3 × 3 pixels that fits the search image"
            )
        if image.size > 8_000_000 or template.size > 1_000_000:
            raise ValidationError("Crop the search image or template to fit the Hough pixel limit")
        anchor = _template_anchor(p["template_anchor"])
        if anchor == (-1, -1):
            anchor = (width // 2, height // 2)
        if not (0 <= anchor[0] < width and 0 <= anchor[1] < height):
            raise ValidationError("The anchor must be inside the template", "template_anchor")
        selection = selection_mask(inputs, image)
        low, high = float(p["canny_low"]), float(p["canny_high"])
        template_edges = cv2.Canny(template, low, high)
        edge_count = cv2.countNonZero(template_edges)
        if not edge_count:
            raise ValidationError("Template has no edges at the selected Canny thresholds")
        edges = cv2.Canny(image, low, high)
        if cv2.countNonZero(edges) * edge_count > _MAX_VOTES:
            raise ValidationError(
                "Hough search is too large. Crop inputs or increase Canny thresholds."
            )

        detector = cv2.createGeneralizedHoughBallard()
        detector.setLevels(int(p["levels"]))
        detector.setDp(float(p["dp"]))
        detector.setVotesThreshold(int(p["minimum_votes"]) - 1)
        positions, votes = _detect(detector, template, template_edges, anchor, image, edges)
        centers = _ranked_centers(
            positions,
            votes,
            image.shape,
            selection,
            float(p["min_center_distance"]),
            int(p["maximum_matches"]),
        )
        return self._result(source, centers, anchor, width, height)

    def _result(self, source, centers, anchor, width, height):
        p = self.parameters
        size = [source.shape[1], source.shape[0]]
        payload = self.input_payloads.get("image", {})
        matches = []
        objects = []
        for x, y, votes in centers:
            matrix = np.eye(3)
            matrix[:2, 2] = (x, y)
            pose = make_pose(matrix, size, payload)
            bounds = [x - anchor[0], y - anchor[1], width, height]
            matches.append({"center": [x, y], "bounds": bounds, "votes": votes, "pose": pose})
            if p["draw_overlay"] or self.generate_graphics:
                objects.extend(
                    (
                        graphic_box(*bounds, ACCEPTED),
                        graphic_object("marker", ((x, y),), ACCEPTED),
                        graphic_object(
                            "text", ((bounds[0], bounds[1] - 3),), ACCEPTED, text=str(votes)
                        ),
                    )
                )
        overlay = overlay_image(source) if p["draw_overlay"] else NO_VALUE
        if overlay is not NO_VALUE:
            identity = np.eye(3)
            for item in objects:
                draw_object(overlay, item, identity)
        count = len(matches)
        passed = count >= int(p["minimum_matches"])
        return Result(
            {
                "matches": matches,
                "poses": [m["pose"] for m in matches],
                "count": count,
                "overlay": overlay,
            },
            drawings={"Hough matches": GraphicDrawing(tuple(objects))}
            if self.generate_graphics
            else {},
            passed=passed,
            message=f"Found {count} Hough template matches",
            fail_reasons=[] if passed else ["hough_match_count_below_minimum"],
        )
